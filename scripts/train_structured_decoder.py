#!/usr/bin/env python3
"""Decode LeWM latents into PushT physical state and render exact scene geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401 - registers HDF5 compression filters
import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "le-wm"))

from train_visual_decoder import (  # noqa: E402
    VisualDecoder,
    encode_images,
    heatmap_difference,
    label_panel,
    load_config,
    seed_everything,
)
from stable_worldmodel.envs.pusht.env import PushT  # noqa: E402


WORLD_SIZE = 512.0


class StructuredStateDecoder(nn.Module):
    """Predict agent xy, block xy, and block orientation from a global latent."""

    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 512,
        depth: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(latent_dim)]
        input_dim = latent_dim
        for _ in range(depth):
            layers.extend((nn.Linear(input_dim, hidden_dim), nn.GELU()))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 6))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        raw = self.network(latent)
        positions = torch.sigmoid(raw[:, :4])
        orientation = F.normalize(raw[:, 4:6], dim=1, eps=1e-6)
        return torch.cat((positions, orientation), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def encode_state_target(state: torch.Tensor) -> torch.Tensor:
    positions = state[:, :4] / WORLD_SIZE
    angle = state[:, 4]
    return torch.cat((positions, torch.sin(angle)[:, None], torch.cos(angle)[:, None]), dim=1)


def decode_state(output: torch.Tensor) -> torch.Tensor:
    positions = output[:, :4] * WORLD_SIZE
    angle = torch.atan2(output[:, 4], output[:, 5]).remainder(2 * torch.pi)
    velocity = torch.zeros((len(output), 2), device=output.device)
    return torch.cat((positions, angle[:, None], velocity), dim=1)


def circular_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = (prediction - target).abs().remainder(2 * torch.pi)
    return torch.minimum(difference, 2 * torch.pi - difference)


def load_split(cache_dir: Path, h5: h5py.File, split: str) -> TensorDataset:
    latents = torch.from_numpy(
        np.asarray(np.load(cache_dir / f"{split}_latents.npy", mmap_mode="r")).copy()
    ).float()
    indices = np.load(cache_dir / f"{split}_indices.npy")
    states = torch.from_numpy(h5["state"][indices].astype(np.float32))
    return TensorDataset(latents, states)


@torch.inference_mode()
def evaluate(
    decoder: StructuredStateDecoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    agent_errors = []
    block_errors = []
    angle_errors = []
    losses = []
    for latent, state in loader:
        latent = latent.to(device)
        state = state.to(device)
        output = decoder(latent)
        target = encode_state_target(state)
        losses.append(F.mse_loss(output, target, reduction="none").mean(dim=1).cpu())
        prediction = decode_state(output)
        agent_errors.append(torch.linalg.vector_norm(prediction[:, :2] - state[:, :2], dim=1).cpu())
        block_errors.append(torch.linalg.vector_norm(prediction[:, 2:4] - state[:, 2:4], dim=1).cpu())
        angle_errors.append(circular_error(prediction[:, 4], state[:, 4]).cpu())
    agent = torch.cat(agent_errors)
    block = torch.cat(block_errors)
    angle = torch.cat(angle_errors)
    return {
        "loss": float(torch.cat(losses).mean()),
        "agent_position_mae_px": float(agent.mean()),
        "agent_position_median_px": float(agent.median()),
        "block_position_mae_px": float(block.mean()),
        "block_position_median_px": float(block.median()),
        "block_angle_mae_deg": float(torch.rad2deg(angle).mean()),
        "block_angle_median_deg": float(torch.rad2deg(angle).median()),
        "agent_within_5px": float((agent < 5).float().mean()),
        "block_within_5px": float((block < 5).float().mean()),
        "angle_within_5deg": float((torch.rad2deg(angle) < 5).float().mean()),
    }


def train(
    cache_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    config: dict,
) -> tuple[StructuredStateDecoder, list[dict[str, float]], dict[str, float]]:
    device = torch.device("cuda")
    settings = config["structured_training"]
    model = StructuredStateDecoder(
        latent_dim=config["decoder"]["latent_dim"],
        **config["structured_decoder"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    with h5py.File(dataset_path, "r") as h5:
        train_data = load_split(cache_dir, h5, "train")
        validation_data = load_split(cache_dir, h5, "validation")
        test_data = load_split(cache_dir, h5, "test")
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_data,
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=settings["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_data, batch_size=settings["batch_size"], shuffle=False
    )
    checkpoint = output_dir / "structured_decoder_best.pt"
    best_loss = math.inf
    stale_epochs = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for latent, state in train_loader:
            latent = latent.to(device)
            state = state.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(latent)
            target = encode_state_target(state)
            position_loss = F.mse_loss(output[:, :4], target[:, :4])
            orientation_loss = F.mse_loss(output[:, 4:], target[:, 4:])
            loss = position_loss + orientation_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(latent)
            count += len(latent)
        model.eval()
        metrics = evaluate(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / count,
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} val={metrics['loss']:.6f} "
            f"agent={metrics['agent_position_mae_px']:.2f}px "
            f"block={metrics['block_position_mae_px']:.2f}px "
            f"angle={metrics['block_angle_mae_deg']:.2f}deg"
        )
        if metrics["loss"] < best_loss - 1e-6:
            best_loss = metrics["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation": metrics,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= settings["patience"]:
                break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    test_metrics = evaluate(model, test_loader, device)
    test_metrics.update(
        {
            "best_epoch": saved["epoch"],
            "training_seconds": time.perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    with (output_dir / "structured_training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=history[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history)
    return model, history, test_metrics


def render_curves(history: list[dict[str, float]], output: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    series = (
        ("validation_agent_position_mae_px", "Pousseur : erreur position", "pixels"),
        ("validation_block_position_mae_px", "T : erreur position", "pixels"),
        ("validation_block_angle_mae_deg", "T : erreur angulaire", "degrés"),
    )
    for axis, (key, title, unit) in zip(axes, series):
        axis.plot(epochs, [row[key] for row in history])
        axis.set(xlabel="époque", ylabel=unit, title=title)
        axis.grid(alpha=0.25)
    figure.suptitle("Décodeur structuré latent → état PushT", fontweight="bold")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def render_states(states: np.ndarray) -> np.ndarray:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    environment = PushT(resolution=224, render_mode="rgb_array")
    environment.reset(seed=0)
    frames = []
    try:
        for state in states:
            environment._set_state(state)
            frames.append(environment.render())
    finally:
        environment.close()
    return np.asarray(frames)


@torch.inference_mode()
def render_comparisons(
    dataset_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    cache_dir: Path,
    decoder: StructuredStateDecoder,
    config: dict,
) -> list[dict]:
    device = next(decoder.parameters()).device
    world_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    world_model = world_model.to(device).eval().requires_grad_(False)
    pixel_saved = torch.load(
        output_dir / "visual_decoder_best.pt", map_location=device, weights_only=True
    )
    pixel_decoder = VisualDecoder(**pixel_saved["decoder_config"]).to(device)
    pixel_decoder.load_state_dict(pixel_saved["state_dict"])
    pixel_decoder.eval()
    metadata = json.loads((cache_dir / "split_metadata.json").read_text())
    episodes = np.asarray(metadata["qualitative_episodes"])
    steps = config["data"]["qualitative_steps"]
    results = []
    grid_rows = []
    with h5py.File(dataset_path, "r") as h5:
        offsets = h5["ep_offset"][:]
        lengths = h5["ep_len"][:]
        for episode in episodes:
            count = min(steps, int(lengths[episode]))
            indices = offsets[episode] + np.arange(count)
            target_images = h5["pixels"][indices]
            target_states = h5["state"][indices].astype(np.float32)
            latents = encode_images(target_images, world_model, device)
            state_output = decoder(latents)
            predicted_states = decode_state(state_output).cpu().numpy()
            structured_images = render_states(predicted_states)
            pixel_images = (
                pixel_decoder(latents)
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            frames = []
            for step in range(count):
                frames.append(
                    np.concatenate(
                        (
                            label_panel(target_images[step], f"Réel — pas {step + 1}/{count}"),
                            label_panel(pixel_images[step], "Décodeur pixel CLS"),
                            label_panel(structured_images[step], "État prédit + rendu exact"),
                            label_panel(
                                heatmap_difference(target_images[step], structured_images[step]),
                                "Erreur structurée ×4",
                            ),
                        ),
                        axis=1,
                    )
                )
            gif_path = output_dir / f"episode_{int(episode):05d}_structured_18_steps.gif"
            imageio.mimsave(gif_path, frames, duration=300, loop=0)
            selected = min(count - 1, count // 2)
            grid_rows.append(
                (
                    target_images[selected],
                    pixel_images[selected],
                    structured_images[selected],
                    heatmap_difference(target_images[selected], structured_images[selected]),
                )
            )
            state_prediction_tensor = torch.from_numpy(predicted_states)
            state_target_tensor = torch.from_numpy(target_states)
            agent_error = torch.linalg.vector_norm(
                state_prediction_tensor[:, :2] - state_target_tensor[:, :2], dim=1
            )
            block_error = torch.linalg.vector_norm(
                state_prediction_tensor[:, 2:4] - state_target_tensor[:, 2:4], dim=1
            )
            angle_error = circular_error(
                state_prediction_tensor[:, 4], state_target_tensor[:, 4]
            )
            results.append(
                {
                    "episode": int(episode),
                    "agent_position_mae_px": float(agent_error.mean()),
                    "block_position_mae_px": float(block_error.mean()),
                    "block_angle_mae_deg": float(torch.rad2deg(angle_error).mean()),
                    "gif": gif_path.name,
                }
            )

    figure, axes = plt.subplots(
        len(grid_rows), 4, figsize=(12, 3 * len(grid_rows)), constrained_layout=True
    )
    titles = (
        "Image réelle",
        "Décodeur pixel CLS",
        "État prédit + rendu exact",
        "Erreur structurée ×4",
    )
    for row, panels in enumerate(grid_rows):
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])
            if column == 0:
                axes[row, column].set_ylabel(f"épisode {int(episodes[row])}")
    figure.suptitle(
        "Deux stratégies de décodage du latent global LeWM",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_dir / "structured_reconstruction_comparison.png", dpi=180)
    plt.close(figure)
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    stablewm_home = Path(os.environ["STABLEWM_HOME"])
    output_dir = Path(config["output"]["directory"])
    cache_dir = output_dir / "cache"
    dataset_path = stablewm_home / "pusht_expert_train.h5"
    checkpoint_path = stablewm_home / "pusht" / "lewm_object.ckpt"
    if not (cache_dir / "split_metadata.json").is_file():
        raise FileNotFoundError("Run train_visual_decoder.py first to create the split cache.")
    decoder, history, metrics = train(cache_dir, dataset_path, output_dir, config)
    qualitative = render_comparisons(
        dataset_path,
        checkpoint_path,
        output_dir,
        cache_dir,
        decoder,
        config,
    )
    render_curves(history, output_dir / "structured_training_curves.png")
    result = {
        "protocol": "frozen LeWM CLS embedding -> physical PushT state -> exact simulator renderer",
        "test_metrics": metrics,
        "qualitative_metrics": qualitative,
        "artifacts": {
            "checkpoint": "structured_decoder_best.pt",
            "curves": "structured_training_curves.png",
            "comparison": "structured_reconstruction_comparison.png",
        },
        "limitation": (
            "This renderer is PushT-specific. Autoregressive predicted latents still need "
            "to be evaluated separately from real encoded latents."
        ),
    }
    (output_dir / "structured_feasibility_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(f"Structured results saved to {output_dir}")


if __name__ == "__main__":
    main()
