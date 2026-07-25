#!/usr/bin/env python3
"""Evaluate LeWM open-loop latent rollouts with visual and physical decoders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

# Required by PyTorch for deterministic CUDA matrix multiplications.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import cv2
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "le-wm"))

from train_structured_decoder import (  # noqa: E402
    StructuredStateDecoder,
    circular_error,
    decode_state,
    render_states,
)
from train_transformer_decoder import TransformerVisualDecoder  # noqa: E402
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


BLOCK_SIZE = 5
CONTEXT_FRAMES = 3
OFFICIAL_ACTION_BLOCKS = 7
STRESS_ACTION_BLOCKS = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="Skip the 18-transition/90-action stress test.",
    )
    return parser.parse_args()


def rollout_indices(
    local_start: int,
    action_blocks: int,
    block_size: int = BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned observation and low-level-action indices."""
    if local_start < 0 or action_blocks < 1 or block_size < 1:
        raise ValueError("start, action_blocks and block_size must define a positive rollout")
    frame_indices = local_start + np.arange(action_blocks + 1) * block_size
    action_indices = local_start + np.arange(action_blocks * block_size)
    return frame_indices.astype(np.int64), action_indices.astype(np.int64)


def pack_action_blocks(
    actions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """Apply the training z-score and flatten five 2-D actions per model step."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"Expected actions shaped (N, 2), got {actions.shape}")
    if len(actions) % block_size:
        raise ValueError("The number of actions must be divisible by block_size")
    normalized = (actions - mean.astype(np.float32)) / std.astype(np.float32)
    return normalized.reshape(len(actions) // block_size, block_size * 2)


def choose_dynamic_start(states: np.ndarray, action_blocks: int) -> int:
    """Pick a deterministic high-motion window without inspecting model outputs."""
    horizon = action_blocks * BLOCK_SIZE
    max_start = len(states) - horizon - 1
    if max_start < 0:
        raise ValueError(
            f"Episode of length {len(states)} is shorter than the {horizon}-action rollout"
        )
    best_start = 0
    best_score = -np.inf
    for start in range(max_start + 1):
        sampled = states[start : start + horizon + 1 : BLOCK_SIZE]
        agent_path = np.linalg.norm(np.diff(sampled[:, :2], axis=0), axis=1).sum()
        block_path = np.linalg.norm(np.diff(sampled[:, 2:4], axis=0), axis=1).sum()
        endpoint = np.linalg.norm(sampled[-1, 2:4] - sampled[0, 2:4])
        angles = np.unwrap(sampled[:, 4])
        angle_path = np.abs(np.diff(angles)).sum()
        # Block motion is the most informative PushT event; agent motion breaks ties.
        score = 4.0 * block_path + 2.0 * endpoint + 8.0 * angle_path + 0.1 * agent_path
        if score > best_score:
            best_start, best_score = start, score
    return best_start


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_models(
    output_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, VisualDecoder, TransformerVisualDecoder, StructuredStateDecoder]:
    world_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    world_model = world_model.to(device).eval().requires_grad_(False)

    pixel_saved = torch.load(
        output_dir / "visual_decoder_best.pt", map_location=device, weights_only=True
    )
    pixel_decoder = VisualDecoder(**pixel_saved["decoder_config"]).to(device)
    pixel_decoder.load_state_dict(pixel_saved["state_dict"])
    pixel_decoder.eval().requires_grad_(False)

    transformer_saved = torch.load(
        output_dir / "transformer_decoder_best.pt",
        map_location=device,
        weights_only=True,
    )
    transformer_decoder = TransformerVisualDecoder(
        **transformer_saved["decoder_config"]
    ).to(device)
    transformer_decoder.load_state_dict(transformer_saved["state_dict"])
    transformer_decoder.eval().requires_grad_(False)

    structured_saved = torch.load(
        output_dir / "structured_decoder_best.pt",
        map_location=device,
        weights_only=True,
    )
    structured_decoder = StructuredStateDecoder().to(device)
    structured_decoder.load_state_dict(structured_saved["state_dict"])
    structured_decoder.eval().requires_grad_(False)
    return world_model, pixel_decoder, transformer_decoder, structured_decoder


def decode_pixels(decoder: torch.nn.Module, latents: torch.Tensor) -> tuple[torch.Tensor, np.ndarray]:
    prediction = decoder(latents).clamp(0, 1)
    uint8 = prediction.mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return prediction, uint8


def frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mse = F.mse_loss(prediction, target, reduction="none").mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    ssim = torch.stack(
        [
            structural_similarity_index_measure(
                prediction[index : index + 1],
                target[index : index + 1],
                data_range=1.0,
            )
            for index in range(len(target))
        ]
    )
    target_mask = foreground_mask(target)
    prediction_mask = foreground_mask(prediction)
    intersection = (target_mask * prediction_mask).sum(dim=(1, 2, 3))
    union = ((target_mask + prediction_mask) > 0).float().sum(dim=(1, 2, 3)).clamp_min(1)
    return (
        psnr.detach().cpu().numpy(),
        ssim.detach().cpu().numpy(),
        (intersection / union).detach().cpu().numpy(),
    )


def draw_status_label(image: np.ndarray, text: str, status: str) -> np.ndarray:
    panel = label_panel(image, text)
    color = (38, 145, 65) if status == "contexte" else (210, 105, 30)
    cv2.rectangle(panel, (panel.shape[1] - 82, 3), (panel.shape[1] - 5, 22), color, -1)
    cv2.putText(
        panel,
        status,
        (panel.shape[1] - 78, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def save_gif(
    path: Path,
    targets: np.ndarray,
    ceiling: np.ndarray,
    prediction: np.ndarray,
    structured: np.ndarray,
) -> None:
    frames = []
    for index in range(len(targets)):
        time_step = index * BLOCK_SIZE
        status = "contexte" if index < CONTEXT_FRAMES else "prédit"
        panels = (
            draw_status_label(targets[index], f"Réel — t={time_step}", status),
            draw_status_label(ceiling[index], "Encode → decode (plafond)", status),
            draw_status_label(prediction[index], "Rollout latent → image", status),
            draw_status_label(structured[index], "Rollout latent → état", status),
        )
        frames.append(np.concatenate(panels, axis=1))
    imageio.mimsave(path, frames, duration=500, loop=0)


def save_transformer_gif(
    path: Path,
    targets: np.ndarray,
    ceiling: np.ndarray,
    prediction: np.ndarray,
) -> None:
    frames = []
    for index in range(len(targets)):
        time_step = index * BLOCK_SIZE
        status = "contexte" if index < CONTEXT_FRAMES else "prédit"
        panels = (
            draw_status_label(targets[index], f"Réel — t={time_step}", status),
            draw_status_label(ceiling[index], "Transformer : latent réel", status),
            draw_status_label(prediction[index], "Transformer : rollout", status),
        )
        frames.append(np.concatenate(panels, axis=1))
    imageio.mimsave(path, frames, duration=500, loop=0)


@torch.inference_mode()
def evaluate_one_rollout(
    h5: h5py.File,
    episode: int,
    action_blocks: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    world_model: torch.nn.Module,
    pixel_decoder: VisualDecoder,
    transformer_decoder: TransformerVisualDecoder,
    structured_decoder: StructuredStateDecoder,
    output_dir: Path,
    protocol_name: str,
) -> tuple[list[dict], dict]:
    device = next(world_model.parameters()).device
    offset = int(h5["ep_offset"][episode])
    length = int(h5["ep_len"][episode])
    episode_states = h5["state"][offset : offset + length].astype(np.float32)
    local_start = choose_dynamic_start(episode_states, action_blocks)
    local_frames, local_actions = rollout_indices(local_start, action_blocks)
    global_frames = offset + local_frames
    global_actions = offset + local_actions
    target_images = h5["pixels"][global_frames]
    target_states = h5["state"][global_frames].astype(np.float32)
    raw_actions = h5["action"][global_actions].astype(np.float32)
    action_sequence = pack_action_blocks(raw_actions, action_mean, action_std)

    real_latents = encode_images(target_images, world_model, device)
    context = (
        torch.from_numpy(target_images[:CONTEXT_FRAMES])
        .to(device)
        .permute(0, 3, 1, 2)
        .float()
        .div_(255)
    )
    context = (context - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
    info = {"pixels": context[None, None]}
    action_tensor = torch.from_numpy(action_sequence).to(device)[None, None]
    rollout = world_model.rollout(info, action_tensor, history_size=CONTEXT_FRAMES)
    predicted_latents = rollout["predicted_emb"][0, 0]
    expected_frames = action_blocks + 1
    if len(predicted_latents) != expected_frames:
        raise RuntimeError(
            f"World model returned {len(predicted_latents)} frames, expected {expected_frames}"
        )
    context_error = float(
        F.mse_loss(predicted_latents[:CONTEXT_FRAMES], real_latents[:CONTEXT_FRAMES])
    )
    # CUDA kernels can differ very slightly when the same images are encoded in
    # batches of 3 and 19; this guards alignment, not bitwise batch invariance.
    if context_error > 1e-6:
        raise RuntimeError(f"Context encoding mismatch: MSE={context_error}")

    target_tensor = (
        torch.from_numpy(target_images).to(device).permute(0, 3, 1, 2).float().div_(255)
    )
    pixel_ceiling_tensor, pixel_ceiling = decode_pixels(pixel_decoder, real_latents)
    pixel_prediction_tensor, pixel_prediction = decode_pixels(
        pixel_decoder, predicted_latents
    )
    transformer_ceiling_tensor, transformer_ceiling = decode_pixels(
        transformer_decoder, real_latents
    )
    transformer_prediction_tensor, transformer_prediction = decode_pixels(
        transformer_decoder, predicted_latents
    )

    structured_output = structured_decoder(predicted_latents)
    predicted_states = decode_state(structured_output).cpu().numpy()
    structured_images = render_states(predicted_states)

    ceiling_psnr, ceiling_ssim, ceiling_iou = frame_metrics(
        pixel_ceiling_tensor, target_tensor
    )
    prediction_psnr, prediction_ssim, prediction_iou = frame_metrics(
        pixel_prediction_tensor, target_tensor
    )
    transformer_ceiling_psnr, transformer_ceiling_ssim, transformer_ceiling_iou = (
        frame_metrics(transformer_ceiling_tensor, target_tensor)
    )
    transformer_prediction_psnr, transformer_prediction_ssim, transformer_prediction_iou = (
        frame_metrics(transformer_prediction_tensor, target_tensor)
    )
    latent_mse = (
        F.mse_loss(predicted_latents, real_latents, reduction="none")
        .mean(dim=1)
        .cpu()
        .numpy()
    )
    state_prediction = torch.from_numpy(predicted_states)
    state_target = torch.from_numpy(target_states)
    agent_error = torch.linalg.vector_norm(
        state_prediction[:, :2] - state_target[:, :2], dim=1
    ).numpy()
    block_error = torch.linalg.vector_norm(
        state_prediction[:, 2:4] - state_target[:, 2:4], dim=1
    ).numpy()
    angle_error = torch.rad2deg(
        circular_error(state_prediction[:, 4], state_target[:, 4])
    ).numpy()

    prefix = f"episode_{episode:05d}_{protocol_name}"
    gif_path = output_dir / f"{prefix}.gif"
    transformer_gif_path = output_dir / f"{prefix}_transformer.gif"
    save_gif(gif_path, target_images, pixel_ceiling, pixel_prediction, structured_images)
    save_transformer_gif(
        transformer_gif_path,
        target_images,
        transformer_ceiling,
        transformer_prediction,
    )

    rows = []
    for index in range(expected_frames):
        rows.append(
            {
                "protocol": protocol_name,
                "episode": int(episode),
                "local_start": int(local_start),
                "model_step": index,
                "low_level_time": index * BLOCK_SIZE,
                "is_context": index < CONTEXT_FRAMES,
                "latent_mse": float(latent_mse[index]),
                "conv_ceiling_psnr_db": float(ceiling_psnr[index]),
                "conv_prediction_psnr_db": float(prediction_psnr[index]),
                "conv_ceiling_ssim": float(ceiling_ssim[index]),
                "conv_prediction_ssim": float(prediction_ssim[index]),
                "conv_ceiling_foreground_iou": float(ceiling_iou[index]),
                "conv_prediction_foreground_iou": float(prediction_iou[index]),
                "transformer_ceiling_psnr_db": float(transformer_ceiling_psnr[index]),
                "transformer_prediction_psnr_db": float(
                    transformer_prediction_psnr[index]
                ),
                "transformer_ceiling_ssim": float(transformer_ceiling_ssim[index]),
                "transformer_prediction_ssim": float(
                    transformer_prediction_ssim[index]
                ),
                "transformer_ceiling_foreground_iou": float(
                    transformer_ceiling_iou[index]
                ),
                "transformer_prediction_foreground_iou": float(
                    transformer_prediction_iou[index]
                ),
                "agent_error_px": float(agent_error[index]),
                "block_error_px": float(block_error[index]),
                "block_angle_error_deg": float(angle_error[index]),
            }
        )
    provenance = {
        "episode": int(episode),
        "episode_length": length,
        "local_start": int(local_start),
        "global_start": int(offset + local_start),
        "frames": global_frames.tolist(),
        "low_level_action_range": [
            int(global_actions[0]),
            int(global_actions[-1]),
        ],
        "raw_actions_sha256": hashlib.sha256(raw_actions.tobytes()).hexdigest(),
        "gif": gif_path.name,
        "transformer_gif": transformer_gif_path.name,
    }
    return rows, provenance


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def aggregate_by_step(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = sorted({int(row["model_step"]) for row in rows})
    means, standard_errors = [], []
    for step in steps:
        values = np.asarray(
            [float(row[key]) for row in rows if int(row["model_step"]) == step]
        )
        means.append(values.mean())
        standard_errors.append(values.std(ddof=0) / np.sqrt(len(values)))
    return np.asarray(steps), np.asarray(means), np.asarray(standard_errors)


def plot_line(
    axis: plt.Axes,
    rows: list[dict],
    key: str,
    label: str,
    *,
    linestyle: str = "-",
) -> None:
    steps, mean, error = aggregate_by_step(rows, key)
    time = steps * BLOCK_SIZE
    line = axis.plot(time, mean, label=label, linestyle=linestyle)[0]
    axis.fill_between(time, mean - error, mean + error, alpha=0.16, color=line.get_color())


def render_curves(rows: list[dict], output: Path, title: str) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plot_line(axes[0, 0], rows, "latent_mse", "MSE latent")
    axes[0, 0].set(title="Erreur du world model", ylabel="MSE")

    plot_line(axes[0, 1], rows, "conv_ceiling_psnr_db", "Plafond conv", linestyle="--")
    plot_line(axes[0, 1], rows, "conv_prediction_psnr_db", "Rollout conv")
    plot_line(
        axes[0, 1],
        rows,
        "transformer_prediction_psnr_db",
        "Rollout Transformer",
    )
    axes[0, 1].set(title="Fidélité image", ylabel="PSNR (dB)")
    axes[0, 1].legend(fontsize=8)

    plot_line(axes[0, 2], rows, "conv_ceiling_ssim", "Plafond conv", linestyle="--")
    plot_line(axes[0, 2], rows, "conv_prediction_ssim", "Rollout conv")
    plot_line(
        axes[0, 2],
        rows,
        "transformer_prediction_ssim",
        "Rollout Transformer",
    )
    axes[0, 2].set(title="Structure image", ylabel="SSIM")
    axes[0, 2].legend(fontsize=8)

    plot_line(axes[1, 0], rows, "conv_ceiling_foreground_iou", "Plafond conv", linestyle="--")
    plot_line(axes[1, 0], rows, "conv_prediction_foreground_iou", "Rollout conv")
    axes[1, 0].set(title="Géométrie du premier plan", ylabel="IoU")
    axes[1, 0].legend(fontsize=8)

    plot_line(axes[1, 1], rows, "agent_error_px", "Pousseur")
    plot_line(axes[1, 1], rows, "block_error_px", "Bloc T")
    axes[1, 1].set(title="Erreur physique en position", ylabel="pixels")
    axes[1, 1].legend(fontsize=8)

    plot_line(axes[1, 2], rows, "block_angle_error_deg", "Orientation du T")
    axes[1, 2].set(title="Erreur physique en orientation", ylabel="degrés")

    for axis in axes.flat:
        axis.axvspan(-0.5, 2 * BLOCK_SIZE + 0.5, color="#2ca02c", alpha=0.08)
        axis.axvline(2.5 * BLOCK_SIZE, color="#d2691e", linestyle=":", linewidth=1)
        axis.set_xlabel("actions bas niveau depuis t=0")
        axis.grid(alpha=0.25)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def render_contact_sheet(
    rows: list[dict],
    provenances: list[dict],
    output_dir: Path,
    protocol_name: str,
) -> None:
    selected_steps = (0, 2, max(int(row["model_step"]) for row in rows))
    figure, axes = plt.subplots(
        len(provenances),
        len(selected_steps),
        figsize=(12, 3.5 * len(provenances)),
        constrained_layout=True,
    )
    for row_index, provenance in enumerate(provenances):
        frames = imageio.mimread(output_dir / provenance["gif"])
        for column, step in enumerate(selected_steps):
            # Crop the main GIF's latent-rollout panel (third of four).
            prediction = np.asarray(frames[step])[:, 2 * 224 : 3 * 224, :3]
            axes[row_index, column].imshow(prediction)
            axes[row_index, column].axis("off")
            axes[row_index, column].set_title(f"t={step * BLOCK_SIZE}")
            if column == 0:
                axes[row_index, column].set_ylabel(
                    f"épisode {provenance['episode']}\ndépart {provenance['local_start']}"
                )
    figure.suptitle(
        f"Rollouts LeWM décodés — {protocol_name}",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_dir / f"rollout_{protocol_name}_contact_sheet.png", dpi=170)
    plt.close(figure)


def summarize(rows: list[dict]) -> dict:
    predicted = [row for row in rows if not row["is_context"]]
    terminal_step = max(int(row["model_step"]) for row in rows)
    terminal = [row for row in rows if int(row["model_step"]) == terminal_step]

    def mean(group: list[dict], key: str) -> float:
        return float(np.mean([float(row[key]) for row in group]))

    return {
        "predicted_frames": {
            "latent_mse": mean(predicted, "latent_mse"),
            "conv_ceiling_psnr_db": mean(predicted, "conv_ceiling_psnr_db"),
            "conv_prediction_psnr_db": mean(predicted, "conv_prediction_psnr_db"),
            "conv_prediction_ssim": mean(predicted, "conv_prediction_ssim"),
            "conv_prediction_foreground_iou": mean(
                predicted, "conv_prediction_foreground_iou"
            ),
            "transformer_prediction_psnr_db": mean(
                predicted, "transformer_prediction_psnr_db"
            ),
            "transformer_prediction_ssim": mean(
                predicted, "transformer_prediction_ssim"
            ),
            "agent_error_px": mean(predicted, "agent_error_px"),
            "block_error_px": mean(predicted, "block_error_px"),
            "block_angle_error_deg": mean(predicted, "block_angle_error_deg"),
        },
        "terminal_frame": {
            "low_level_time": terminal_step * BLOCK_SIZE,
            "latent_mse": mean(terminal, "latent_mse"),
            "conv_prediction_psnr_db": mean(terminal, "conv_prediction_psnr_db"),
            "conv_prediction_ssim": mean(terminal, "conv_prediction_ssim"),
            "conv_prediction_foreground_iou": mean(
                terminal, "conv_prediction_foreground_iou"
            ),
            "agent_error_px": mean(terminal, "agent_error_px"),
            "block_error_px": mean(terminal, "block_error_px"),
            "block_angle_error_deg": mean(terminal, "block_angle_error_deg"),
        },
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("Rollout evaluation requires CUDA.")
    device = torch.device("cuda")
    stablewm_home = Path(os.environ["STABLEWM_HOME"])
    dataset_path = stablewm_home / "pusht_expert_train.h5"
    checkpoint_path = stablewm_home / "pusht" / "lewm_object.ckpt"
    output_dir = Path(config["output"]["directory"])
    cache_dir = output_dir / "cache"
    metadata = json.loads((cache_dir / "split_metadata.json").read_text())
    episodes = [int(value) for value in metadata["qualitative_episodes"]]
    models = load_models(output_dir, checkpoint_path, device)

    with h5py.File(dataset_path, "r") as h5:
        all_actions = h5["action"][:].astype(np.float32)
        valid_actions = all_actions[~np.isnan(all_actions).any(axis=1)]
        action_mean = valid_actions.mean(axis=0, dtype=np.float64).astype(np.float32)
        # torch.std defaults to the Bessel-corrected estimator used during LeWM training.
        action_std = valid_actions.std(axis=0, ddof=1, dtype=np.float64).astype(np.float32)
        protocols = [("official_8frames", OFFICIAL_ACTION_BLOCKS)]
        if not args.official_only:
            protocols.append(("stress_18steps", STRESS_ACTION_BLOCKS))
        result_protocols = {}
        for protocol_name, action_blocks in protocols:
            print(
                f"Evaluating {protocol_name}: {action_blocks} model steps, "
                f"{action_blocks * BLOCK_SIZE} low-level actions"
            )
            rows: list[dict] = []
            provenances: list[dict] = []
            for episode in episodes:
                episode_rows, provenance = evaluate_one_rollout(
                    h5,
                    episode,
                    action_blocks,
                    action_mean,
                    action_std,
                    *models,
                    output_dir,
                    protocol_name,
                )
                rows.extend(episode_rows)
                provenances.append(provenance)
                print(
                    f"  episode={episode} start={provenance['local_start']} "
                    f"gif={provenance['gif']}"
                )
            metrics_path = output_dir / f"rollout_{protocol_name}_metrics.csv"
            curves_path = output_dir / f"rollout_{protocol_name}_curves.png"
            write_rows(metrics_path, rows)
            render_curves(
                rows,
                curves_path,
                (
                    "Protocole officiel : t=0,5,…,35"
                    if protocol_name == "official_8frames"
                    else "Stress test : 18 transitions latentes / 90 actions"
                ),
            )
            render_contact_sheet(rows, provenances, output_dir, protocol_name)
            result_protocols[protocol_name] = {
                "action_blocks": action_blocks,
                "low_level_actions": action_blocks * BLOCK_SIZE,
                "frames": action_blocks + 1,
                "summary": summarize(rows),
                "episodes": provenances,
                "metrics_csv": metrics_path.name,
                "curves": curves_path.name,
                "contact_sheet": f"rollout_{protocol_name}_contact_sheet.png",
            }

    results = {
        "protocol": {
            "context_frames": CONTEXT_FRAMES,
            "frame_skip": BLOCK_SIZE,
            "action_block": BLOCK_SIZE,
            "selection": (
                "Deterministic highest-motion ground-truth window per held-out episode; "
                "model outputs are never inspected for selection."
            ),
            "action_normalization": {
                "mean": action_mean.tolist(),
                "std_unbiased": action_std.tolist(),
            },
        },
        "sources": {
            "dataset": str(dataset_path.resolve()),
            "world_model_checkpoint": str(checkpoint_path.resolve()),
            "world_model_sha256": file_sha256(checkpoint_path),
            "split_seed": int(metadata["seed"]),
            "episodes": episodes,
        },
        "protocols": result_protocols,
        "interpretation": (
            "The independently decoded real latent is the visualization ceiling. "
            "The gap between that ceiling and the decoded rollout isolates world-model drift."
        ),
    }
    result_path = output_dir / "rollout_evaluation_results.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: value["summary"] for key, value in result_protocols.items()}, indent=2))
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
