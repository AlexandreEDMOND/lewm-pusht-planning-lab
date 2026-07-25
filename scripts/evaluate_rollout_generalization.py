#!/usr/bin/env python3
"""Evaluate LeWM t=35 rollouts on every held-out PushT episode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import cv2
import hdf5plugin  # noqa: F401
import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torchmetrics.functional.image import structural_similarity_index_measure

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "le-wm"))

from evaluate_decoder_rollouts import (  # noqa: E402
    BLOCK_SIZE,
    CONTEXT_FRAMES,
    OFFICIAL_ACTION_BLOCKS,
    decode_pixels,
    pack_action_blocks,
    rollout_indices,
)
from train_structured_decoder import (  # noqa: E402
    StructuredStateDecoder,
    circular_error,
    decode_state,
    render_states,
)
from train_visual_decoder import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    VisualDecoder,
    encode_images,
    foreground_mask,
    label_panel,
    load_config,
    seed_everything,
)
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402


PUSH_POSITION_THRESHOLD_PX = 2.0
PUSH_ANGLE_THRESHOLD_DEG = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def configure_determinism(seed: int) -> None:
    seed_everything(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def uniform_episode_starts(
    episodes: list[int],
    lengths: np.ndarray,
    seed: int,
    horizon: int = OFFICIAL_ACTION_BLOCKS * BLOCK_SIZE,
) -> dict[int, int]:
    """Choose one valid window per episode without looking at states or outputs."""
    rng = np.random.default_rng(seed)
    starts = {}
    for episode in episodes:
        maximum = int(lengths[episode]) - horizon - 1
        if maximum < 0:
            raise ValueError(f"Episode {episode} is too short for horizon {horizon}")
        starts[episode] = int(rng.integers(0, maximum + 1))
    return starts


def has_agent_block_contact(environment: PushT, state: np.ndarray) -> bool:
    """Query exact Pymunk overlap after restoring a recorded state."""
    environment._set_state(state)
    block_shapes = set(environment.block.shapes)
    for agent_shape in environment.agent.shapes:
        if any(query.shape in block_shapes for query in environment.space.shape_query(agent_shape)):
            return True
    return False


def transition_label(
    contact: bool,
    block_displacement_px: float,
    block_angle_displacement_deg: float,
) -> str:
    effective_push = (
        block_displacement_px >= PUSH_POSITION_THRESHOLD_PX
        or block_angle_displacement_deg >= PUSH_ANGLE_THRESHOLD_DEG
    )
    if effective_push:
        return "effective_push"
    if contact:
        return "contact_no_motion"
    return "free_motion"


def encode_in_batches(
    images: np.ndarray,
    world_model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(images), batch_size):
        outputs.append(
            encode_images(images[start : start + batch_size], world_model, device)
        )
    return torch.cat(outputs)


def decode_in_batches(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    batch_size: int = 64,
) -> tuple[torch.Tensor, np.ndarray]:
    tensors, arrays = [], []
    for start in range(0, len(latents), batch_size):
        tensor, array = decode_pixels(decoder, latents[start : start + batch_size])
        tensors.append(tensor.cpu())
        arrays.append(array)
    return torch.cat(tensors), np.concatenate(arrays)


def per_frame_image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psnr_values, ssim_values, iou_values = [], [], []
    for start in range(0, len(target), batch_size):
        pred = prediction[start : start + batch_size].cuda(non_blocking=True)
        real = target[start : start + batch_size].cuda(non_blocking=True)
        mse = F.mse_loss(pred, real, reduction="none").mean(dim=(1, 2, 3))
        psnr_values.append((-10.0 * torch.log10(mse.clamp_min(1e-12))).cpu())
        ssim_values.append(
            structural_similarity_index_measure(
                pred, real, data_range=1.0, reduction="none"
            ).cpu()
        )
        target_mask = foreground_mask(real)
        prediction_mask = foreground_mask(pred)
        intersection = (target_mask * prediction_mask).sum(dim=(1, 2, 3))
        union = (
            ((target_mask + prediction_mask) > 0)
            .float()
            .sum(dim=(1, 2, 3))
            .clamp_min(1)
        )
        iou_values.append((intersection / union).cpu())
    return (
        torch.cat(psnr_values).numpy(),
        torch.cat(ssim_values).numpy(),
        torch.cat(iou_values).numpy(),
    )


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def summarize_rows(rows: list[dict]) -> dict:
    predicted = [row for row in rows if not row["is_context"]]
    terminal = [row for row in rows if row["model_step"] == OFFICIAL_ACTION_BLOCKS]
    keys = (
        "latent_mse",
        "conv_prediction_psnr_db",
        "conv_prediction_ssim",
        "conv_prediction_foreground_iou",
        "agent_prediction_error_px",
        "block_prediction_error_px",
        "block_prediction_angle_error_deg",
        "block_excess_error_px",
        "block_excess_angle_error_deg",
    )

    def summarize(group: list[dict]) -> dict:
        return {key: quantiles([row[key] for row in group]) for key in keys}

    by_category = {}
    for category in ("free_motion", "contact_no_motion", "effective_push"):
        group = [row for row in predicted if row["transition_category"] == category]
        by_category[category] = summarize(group) if group else {"count": 0}
    latent = np.asarray([row["latent_mse"] for row in terminal])
    block = np.asarray([row["block_prediction_error_px"] for row in terminal])
    angle = np.asarray([row["block_prediction_angle_error_deg"] for row in terminal])
    rho_block = spearmanr(latent, block)
    rho_angle = spearmanr(latent, angle)
    return {
        "predicted_frames": summarize(predicted),
        "terminal_frames": summarize(terminal),
        "by_transition_category": by_category,
        "terminal_correlations": {
            "latent_mse_vs_block_error_spearman_rho": float(rho_block.statistic),
            "latent_mse_vs_block_error_pvalue": float(rho_block.pvalue),
            "latent_mse_vs_angle_error_spearman_rho": float(rho_angle.statistic),
            "latent_mse_vs_angle_error_pvalue": float(rho_angle.pvalue),
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_horizon(
    rows: list[dict], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = np.arange(OFFICIAL_ACTION_BLOCKS + 1)
    medians, lower, upper = [], [], []
    for step in steps:
        values = np.asarray([row[key] for row in rows if row["model_step"] == step])
        medians.append(np.quantile(values, 0.5))
        lower.append(np.quantile(values, 0.1))
        upper.append(np.quantile(values, 0.9))
    return steps * BLOCK_SIZE, np.asarray(medians), np.asarray(lower), np.asarray(upper)


def plot_horizon_line(axis: plt.Axes, rows: list[dict], key: str, label: str) -> None:
    time, median, lower, upper = aggregate_horizon(rows, key)
    line = axis.plot(time, median, label=label)[0]
    axis.fill_between(time, lower, upper, alpha=0.14, color=line.get_color())


def render_horizon_curves(rows: list[dict], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plot_horizon_line(axes[0, 0], rows, "latent_mse", "MSE latent")
    axes[0, 0].set(title="Dérive latente", ylabel="MSE")
    plot_horizon_line(
        axes[0, 1], rows, "conv_ceiling_psnr_db", "Plafond encode→decode"
    )
    plot_horizon_line(
        axes[0, 1], rows, "conv_prediction_psnr_db", "Rollout prédit"
    )
    axes[0, 1].set(title="Fidélité visuelle", ylabel="PSNR (dB)")
    axes[0, 1].legend()
    plot_horizon_line(
        axes[0, 2], rows, "conv_prediction_foreground_iou", "IoU rollout"
    )
    axes[0, 2].set(title="Géométrie visible", ylabel="IoU premier plan")
    plot_horizon_line(
        axes[1, 0], rows, "agent_prediction_error_px", "Pousseur"
    )
    plot_horizon_line(
        axes[1, 0], rows, "block_prediction_error_px", "Bloc T"
    )
    axes[1, 0].set(title="Erreur physique absolue", ylabel="pixels")
    axes[1, 0].legend()
    plot_horizon_line(
        axes[1, 1], rows, "block_prediction_angle_error_deg", "Angle T"
    )
    axes[1, 1].set(title="Orientation physique", ylabel="degrés")
    plot_horizon_line(
        axes[1, 2], rows, "block_excess_error_px", "Excès position"
    )
    plot_horizon_line(
        axes[1, 2], rows, "block_excess_angle_error_deg", "Excès angle"
    )
    axes[1, 2].set(
        title="Erreur ajoutée par la dynamique", ylabel="erreur rollout − plafond"
    )
    axes[1, 2].legend()
    for axis in axes.flat:
        axis.axvspan(-0.5, 10.5, color="#2ca02c", alpha=0.08)
        axis.axvline(12.5, color="#d2691e", linestyle=":", linewidth=1)
        axis.set_xlabel("actions bas niveau depuis t=0")
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Généralisation LeWM — 128 épisodes de test (médiane, bande P10–P90)",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def render_category_plot(rows: list[dict], output: Path) -> None:
    predicted = [row for row in rows if not row["is_context"]]
    categories = ("free_motion", "contact_no_motion", "effective_push")
    labels = ("Libre", "Contact\nsans mouvement", "Poussée\neffective")
    metrics = (
        ("latent_mse", "MSE latent"),
        ("block_prediction_error_px", "Erreur T (px)"),
        ("block_prediction_angle_error_deg", "Erreur angle T (°)"),
        ("conv_prediction_psnr_db", "PSNR rollout (dB)"),
    )
    figure, axes = plt.subplots(1, 4, figsize=(17, 5), constrained_layout=True)
    for axis, (key, title) in zip(axes, metrics):
        values = [
            [row[key] for row in predicted if row["transition_category"] == category]
            for category in categories
        ]
        parts = axis.violinplot(values, showmedians=True, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.55)
        axis.set_xticks((1, 2, 3), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if key == "latent_mse":
            axis.set_yscale("log")
        elif key in (
            "block_prediction_error_px",
            "block_prediction_angle_error_deg",
        ):
            axis.set_yscale("symlog", linthresh=1.0)
        for index, group in enumerate(values, 1):
            axis.text(
                index,
                axis.get_ylim()[1],
                f"n={len(group)}",
                ha="center",
                va="top",
                fontsize=8,
            )
    figure.suptitle(
        "Erreurs selon l'interaction physique réelle",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def render_terminal_calibration(rows: list[dict], output: Path) -> None:
    terminal = [row for row in rows if row["model_step"] == OFFICIAL_ACTION_BLOCKS]
    latent = np.asarray([row["latent_mse"] for row in terminal])
    block = np.asarray([row["block_prediction_error_px"] for row in terminal])
    angle = np.asarray([row["block_prediction_angle_error_deg"] for row in terminal])
    categories = [row["transition_category"] for row in terminal]
    colors = {
        "free_motion": "#4c78a8",
        "contact_no_motion": "#f2cf5b",
        "effective_push": "#e45756",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for category, color in colors.items():
        mask = np.asarray([value == category for value in categories])
        axes[0].scatter(latent[mask], block[mask], s=28, alpha=0.7, color=color, label=category)
        axes[1].scatter(latent[mask], angle[mask], s=28, alpha=0.7, color=color, label=category)
    for axis, target, title, ylabel in (
        (axes[0], block, "Position du T", "erreur (px)"),
        (axes[1], angle, "Orientation du T", "erreur (°)"),
    ):
        rho = spearmanr(latent, target)
        axis.set(
            xlabel="MSE latente terminale",
            ylabel=ylabel,
            title=f"{title} — ρ={rho.statistic:.2f}, p={rho.pvalue:.2g}",
        )
        axis.set_xscale("log")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "Le signal latent est-il calibré avec l'erreur physique à t=35 ?",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def select_representatives(rows: list[dict]) -> list[tuple[str, dict]]:
    terminal = [row for row in rows if row["model_step"] == OFFICIAL_ACTION_BLOCKS]
    ordered = sorted(
        terminal,
        key=lambda row: row["block_prediction_error_px"]
        + 2.0 * row["block_prediction_angle_error_deg"],
    )
    return [
        ("best", ordered[0]),
        ("median", ordered[len(ordered) // 2]),
        ("worst", ordered[-1]),
    ]


def save_representative_outputs(
    representatives: list[tuple[str, dict]],
    episodes: list[int],
    starts: dict[int, int],
    target_images: np.ndarray,
    ceiling_images: np.ndarray,
    prediction_images: np.ndarray,
    predicted_states: np.ndarray,
    output_dir: Path,
) -> list[dict]:
    episode_to_index = {episode: index for index, episode in enumerate(episodes)}
    artifacts = []
    contact_rows = []
    for rank, row in representatives:
        batch_index = episode_to_index[int(row["episode"])]
        start = batch_index * (OFFICIAL_ACTION_BLOCKS + 1)
        end = start + OFFICIAL_ACTION_BLOCKS + 1
        structured = render_states(predicted_states[start:end])
        frames = []
        for model_step in range(OFFICIAL_ACTION_BLOCKS + 1):
            flat = start + model_step
            status = "contexte" if model_step < CONTEXT_FRAMES else "prédit"
            panels = (
                label_panel(target_images[flat], f"Réel t={model_step * BLOCK_SIZE}"),
                label_panel(ceiling_images[flat], "Plafond encode→decode"),
                label_panel(prediction_images[flat], f"Rollout — {status}"),
                label_panel(structured[model_step], "État physique décodé"),
            )
            frames.append(np.concatenate(panels, axis=1))
        path = output_dir / f"generalization_{rank}_episode_{int(row['episode']):05d}.gif"
        imageio.mimsave(path, frames, duration=500, loop=0)
        artifacts.append(
            {
                "rank": rank,
                "episode": int(row["episode"]),
                "local_start": starts[int(row["episode"])],
                "terminal_block_error_px": row["block_prediction_error_px"],
                "terminal_angle_error_deg": row[
                    "block_prediction_angle_error_deg"
                ],
                "terminal_latent_mse": row["latent_mse"],
                "gif": path.name,
            }
        )
        contact_rows.append((rank, frames[-1]))

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    for axis, (rank, frame) in zip(axes, contact_rows):
        axis.imshow(frame)
        axis.axis("off")
        axis.set_title(rank)
    figure.suptitle(
        "Cas représentatifs à t=35 : meilleur, médian et pire",
        fontsize=15,
        fontweight="bold",
    )
    contact_sheet = output_dir / "generalization_representatives.png"
    figure.savefig(contact_sheet, dpi=170)
    plt.close(figure)
    return artifacts


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_determinism(config["seed"])
    device = torch.device("cuda")
    stablewm_home = Path(os.environ["STABLEWM_HOME"])
    dataset_path = stablewm_home / "pusht_expert_train.h5"
    checkpoint_path = stablewm_home / "pusht" / "lewm_object.ckpt"
    output_dir = Path(config["output"]["directory"]) / "generalization"
    output_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir = Path(config["output"]["directory"])
    split_metadata = json.loads(
        (decoder_dir / "cache" / "split_metadata.json").read_text()
    )
    episodes = [int(value) for value in split_metadata["test_episodes"]]

    world_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    world_model = world_model.to(device).eval().requires_grad_(False)
    pixel_saved = torch.load(
        decoder_dir / "visual_decoder_best.pt", map_location=device, weights_only=True
    )
    pixel_decoder = VisualDecoder(**pixel_saved["decoder_config"]).to(device)
    pixel_decoder.load_state_dict(pixel_saved["state_dict"])
    pixel_decoder.eval().requires_grad_(False)
    structured_saved = torch.load(
        decoder_dir / "structured_decoder_best.pt",
        map_location=device,
        weights_only=True,
    )
    structured_decoder = StructuredStateDecoder().to(device)
    structured_decoder.load_state_dict(structured_saved["state_dict"])
    structured_decoder.eval().requires_grad_(False)

    target_image_groups, target_state_groups, action_groups = [], [], []
    transition_metadata: list[list[dict]] = []
    with h5py.File(dataset_path, "r") as h5:
        offsets = h5["ep_offset"][:]
        lengths = h5["ep_len"][:]
        starts = uniform_episode_starts(
            episodes, lengths, config["seed"] + 101
        )
        all_actions = h5["action"][:].astype(np.float32)
        valid_actions = all_actions[~np.isnan(all_actions).any(axis=1)]
        action_mean = valid_actions.mean(axis=0, dtype=np.float64).astype(np.float32)
        action_std = valid_actions.std(
            axis=0, ddof=1, dtype=np.float64
        ).astype(np.float32)
        environment = PushT(resolution=224, render_mode="rgb_array")
        environment.reset(seed=0)
        try:
            for episode in episodes:
                offset = int(offsets[episode])
                local_start = starts[episode]
                local_frames, local_actions = rollout_indices(
                    local_start, OFFICIAL_ACTION_BLOCKS
                )
                target_image_groups.append(h5["pixels"][offset + local_frames])
                target_state_groups.append(
                    h5["state"][offset + local_frames].astype(np.float32)
                )
                raw_actions = h5["action"][offset + local_actions].astype(np.float32)
                action_groups.append(
                    pack_action_blocks(raw_actions, action_mean, action_std)
                )
                raw_states = h5["state"][
                    offset + local_start : offset + local_start + 36
                ].astype(np.float32)
                contacts = [
                    has_agent_block_contact(environment, state)
                    for state in raw_states
                ]
                episode_transitions = [
                    {
                        "category": "initial",
                        "contact": bool(contacts[0]),
                        "block_displacement_px": 0.0,
                        "block_angle_displacement_deg": 0.0,
                    }
                ]
                for step in range(1, OFFICIAL_ACTION_BLOCKS + 1):
                    before = raw_states[(step - 1) * BLOCK_SIZE]
                    after = raw_states[step * BLOCK_SIZE]
                    block_displacement = float(
                        np.linalg.norm(after[2:4] - before[2:4])
                    )
                    angle = float(
                        np.degrees(
                            abs(
                                (
                                    (after[4] - before[4] + np.pi)
                                    % (2 * np.pi)
                                )
                                - np.pi
                            )
                        )
                    )
                    contact = bool(
                        any(
                            contacts[
                                (step - 1) * BLOCK_SIZE : step * BLOCK_SIZE + 1
                            ]
                        )
                    )
                    episode_transitions.append(
                        {
                            "category": transition_label(
                                contact, block_displacement, angle
                            ),
                            "contact": contact,
                            "block_displacement_px": block_displacement,
                            "block_angle_displacement_deg": angle,
                        }
                    )
                transition_metadata.append(episode_transitions)
        finally:
            environment.close()

    target_images = np.concatenate(target_image_groups)
    target_states = np.concatenate(target_state_groups)
    action_sequences = np.stack(action_groups)
    real_latents = encode_in_batches(target_images, world_model, device)
    context_images = np.stack(
        [images[:CONTEXT_FRAMES] for images in target_image_groups]
    )
    context = (
        torch.from_numpy(context_images)
        .to(device)
        .permute(0, 1, 4, 2, 3)
        .float()
        .div_(255)
    )
    context = (context - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    rollout = world_model.rollout(
        {"pixels": context[:, None]},
        torch.from_numpy(action_sequences).to(device)[:, None],
        history_size=CONTEXT_FRAMES,
    )
    predicted_latents = rollout["predicted_emb"][:, 0]
    predicted_flat = predicted_latents.reshape(-1, predicted_latents.shape[-1])
    context_mse = float(
        F.mse_loss(
            predicted_latents[:, :CONTEXT_FRAMES],
            real_latents.reshape(len(episodes), -1, 192)[:, :CONTEXT_FRAMES],
        )
    )
    if context_mse > 1e-6:
        raise RuntimeError(f"Context alignment failed: MSE={context_mse}")

    ceiling_tensor, ceiling_images = decode_in_batches(pixel_decoder, real_latents)
    prediction_tensor, prediction_images = decode_in_batches(
        pixel_decoder, predicted_flat
    )
    target_tensor = (
        torch.from_numpy(target_images).permute(0, 3, 1, 2).float().div_(255)
    )
    ceiling_psnr, ceiling_ssim, ceiling_iou = per_frame_image_metrics(
        ceiling_tensor, target_tensor
    )
    prediction_psnr, prediction_ssim, prediction_iou = per_frame_image_metrics(
        prediction_tensor, target_tensor
    )

    real_states_decoded = decode_state(structured_decoder(real_latents)).cpu()
    predicted_states_decoded = decode_state(structured_decoder(predicted_flat)).cpu()
    target_state_tensor = torch.from_numpy(target_states)
    latent_mse = (
        F.mse_loss(predicted_flat, real_latents, reduction="none")
        .mean(dim=1)
        .cpu()
        .numpy()
    )

    def position_error(prediction: torch.Tensor, target: torch.Tensor, start: int) -> np.ndarray:
        return (
            torch.linalg.vector_norm(
                prediction[:, start : start + 2] - target[:, start : start + 2],
                dim=1,
            )
            .cpu()
            .numpy()
        )

    agent_ceiling = position_error(real_states_decoded, target_state_tensor, 0)
    agent_prediction = position_error(predicted_states_decoded, target_state_tensor, 0)
    block_ceiling = position_error(real_states_decoded, target_state_tensor, 2)
    block_prediction = position_error(predicted_states_decoded, target_state_tensor, 2)
    angle_ceiling = (
        torch.rad2deg(
            circular_error(real_states_decoded[:, 4], target_state_tensor[:, 4])
        )
        .cpu()
        .numpy()
    )
    angle_prediction = (
        torch.rad2deg(
            circular_error(
                predicted_states_decoded[:, 4], target_state_tensor[:, 4]
            )
        )
        .cpu()
        .numpy()
    )

    rows = []
    episode_rows = []
    frame_count = OFFICIAL_ACTION_BLOCKS + 1
    for episode_index, episode in enumerate(episodes):
        first = episode_index * frame_count
        for model_step in range(frame_count):
            flat = first + model_step
            transition = transition_metadata[episode_index][model_step]
            rows.append(
                {
                    "episode": episode,
                    "local_start": starts[episode],
                    "model_step": model_step,
                    "low_level_time": model_step * BLOCK_SIZE,
                    "is_context": model_step < CONTEXT_FRAMES,
                    "transition_category": transition["category"],
                    "ground_truth_contact": transition["contact"],
                    "ground_truth_block_displacement_px": transition[
                        "block_displacement_px"
                    ],
                    "ground_truth_block_angle_displacement_deg": transition[
                        "block_angle_displacement_deg"
                    ],
                    "latent_mse": float(latent_mse[flat]),
                    "conv_ceiling_psnr_db": float(ceiling_psnr[flat]),
                    "conv_prediction_psnr_db": float(prediction_psnr[flat]),
                    "conv_ceiling_ssim": float(ceiling_ssim[flat]),
                    "conv_prediction_ssim": float(prediction_ssim[flat]),
                    "conv_ceiling_foreground_iou": float(ceiling_iou[flat]),
                    "conv_prediction_foreground_iou": float(prediction_iou[flat]),
                    "agent_ceiling_error_px": float(agent_ceiling[flat]),
                    "agent_prediction_error_px": float(agent_prediction[flat]),
                    "block_ceiling_error_px": float(block_ceiling[flat]),
                    "block_prediction_error_px": float(block_prediction[flat]),
                    "block_ceiling_angle_error_deg": float(angle_ceiling[flat]),
                    "block_prediction_angle_error_deg": float(
                        angle_prediction[flat]
                    ),
                    "block_excess_error_px": float(
                        block_prediction[flat] - block_ceiling[flat]
                    ),
                    "block_excess_angle_error_deg": float(
                        angle_prediction[flat] - angle_ceiling[flat]
                    ),
                }
            )
        predicted_episode = rows[-5:]
        terminal = rows[-1]
        episode_rows.append(
            {
                "episode": episode,
                "local_start": starts[episode],
                "mean_predicted_latent_mse": float(
                    np.mean([row["latent_mse"] for row in predicted_episode])
                ),
                "mean_predicted_block_error_px": float(
                    np.mean(
                        [row["block_prediction_error_px"] for row in predicted_episode]
                    )
                ),
                "terminal_latent_mse": terminal["latent_mse"],
                "terminal_block_error_px": terminal[
                    "block_prediction_error_px"
                ],
                "terminal_block_angle_error_deg": terminal[
                    "block_prediction_angle_error_deg"
                ],
                "terminal_transition_category": terminal["transition_category"],
            }
        )

    metrics_path = output_dir / "generalization_frame_metrics.csv"
    episode_path = output_dir / "generalization_episode_metrics.csv"
    write_csv(metrics_path, rows)
    write_csv(episode_path, episode_rows)
    render_horizon_curves(rows, output_dir / "generalization_horizon_curves.png")
    render_category_plot(rows, output_dir / "generalization_categories.png")
    render_terminal_calibration(rows, output_dir / "generalization_calibration.png")
    representatives = select_representatives(rows)
    representative_artifacts = save_representative_outputs(
        representatives,
        episodes,
        starts,
        target_images,
        ceiling_images,
        prediction_images,
        predicted_states_decoded.numpy(),
        output_dir,
    )
    summary = summarize_rows(rows)
    result = {
        "protocol": {
            "episode_count": len(episodes),
            "split_seed": split_metadata["seed"],
            "window_seed": config["seed"] + 101,
            "window_sampling": (
                "one uniform valid t=0..35 window per held-out episode; "
                "independent of states and model outputs"
            ),
            "context_frames": CONTEXT_FRAMES,
            "action_block": BLOCK_SIZE,
            "times": list(range(0, 36, 5)),
            "push_thresholds": {
                "block_position_px": PUSH_POSITION_THRESHOLD_PX,
                "block_angle_deg": PUSH_ANGLE_THRESHOLD_DEG,
            },
            "category_precedence": (
                "effective_push if block motion crosses a threshold; otherwise "
                "contact_no_motion if exact Pymunk overlap occurs; otherwise free_motion"
            ),
            "context_alignment_mse": context_mse,
        },
        "sources": {
            "dataset": str(dataset_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
            "episodes": episodes,
            "starts": {str(key): value for key, value in starts.items()},
        },
        "summary": summary,
        "representatives": representative_artifacts,
        "artifacts": {
            "frame_metrics": metrics_path.name,
            "episode_metrics": episode_path.name,
            "horizon_curves": "generalization_horizon_curves.png",
            "categories": "generalization_categories.png",
            "calibration": "generalization_calibration.png",
            "representatives": "generalization_representatives.png",
        },
    }
    result_path = output_dir / "generalization_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
