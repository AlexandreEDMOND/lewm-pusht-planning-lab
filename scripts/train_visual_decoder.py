#!/usr/bin/env python3
"""Train and evaluate a visual decoder on frozen LeWM PushT embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import hdf5plugin  # noqa: F401 - registers the HDF5 compression filters
import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image import structural_similarity_index_measure

matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Split:
    train_episodes: np.ndarray
    validation_episodes: np.ndarray
    test_episodes: np.ndarray


class VisualDecoder(nn.Module):
    """Decode one 192-D global LeWM embedding into a 224x224 RGB image."""

    def __init__(self, latent_dim: int = 192, base_channels: int = 256) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.stem = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, base_channels * 7 * 7),
            nn.GELU(),
        )
        channels = (base_channels, 192, 128, 64, 32, 16)
        blocks = []
        for input_channels, output_channels in zip(channels, channels[1:]):
            groups = min(8, output_channels)
            blocks.extend(
                (
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(input_channels, output_channels, 3, padding=1),
                    nn.GroupNorm(groups, output_channels),
                    nn.SiLU(),
                    nn.Conv2d(output_channels, output_channels, 3, padding=1),
                    nn.GroupNorm(groups, output_channels),
                    nn.SiLU(),
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Sequential(nn.Conv2d(channels[-1], 3, 3, padding=1), nn.Sigmoid())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.stem(latent)
        features = features.view(-1, self.base_channels, 7, 7)
        return self.output(self.blocks(features))


class CachedFrameDataset(Dataset):
    """Read cached uint8 images and float32 latents through memory maps."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        self.images = np.load(cache_dir / f"{split}_images.npy", mmap_mode="r")
        self.latents = np.load(cache_dir / f"{split}_latents.npy", mmap_mode="r")
        if len(self.images) != len(self.latents):
            raise ValueError(f"Mismatched cached arrays for {split}")

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.from_numpy(np.asarray(self.latents[index]).copy())
        image = torch.from_numpy(np.asarray(self.images[index]).copy())
        return latent, image.float().div_(255.0).permute(2, 0, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--epochs", type=int, help="Override the configured epoch count")
    parser.add_argument("--train-frames", type=int, help="Override the training frame count")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    text = os.path.expandvars(path.read_text())
    config = yaml.safe_load(text)
    if "${" in text:
        unresolved = [line for line in text.splitlines() if "${" in line]
        raise ValueError(f"Unresolved environment variable in config: {unresolved}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_episode_split(count: int, config: dict, seed: int) -> Split:
    requested = (
        config["train_episodes"]
        + config["validation_episodes"]
        + config["test_episodes"]
    )
    if requested > count:
        raise ValueError(f"Requested {requested} episodes but the dataset contains {count}")
    episodes = np.random.default_rng(seed).permutation(count)[:requested]
    train_end = config["train_episodes"]
    validation_end = train_end + config["validation_episodes"]
    return Split(
        train_episodes=np.sort(episodes[:train_end]),
        validation_episodes=np.sort(episodes[train_end:validation_end]),
        test_episodes=np.sort(episodes[validation_end:]),
    )


def sample_frame_indices(
    episode_ids: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    count: int,
    seed: int,
) -> np.ndarray:
    """Sample frames evenly across episodes, then sort for efficient HDF5 reads."""
    rng = np.random.default_rng(seed)
    per_episode = math.ceil(count / len(episode_ids))
    selected: list[np.ndarray] = []
    for episode in episode_ids:
        length = int(lengths[episode])
        take = min(per_episode, length)
        local = rng.choice(length, size=take, replace=False)
        selected.append(offsets[episode] + local)
    indices = np.concatenate(selected)
    if len(indices) > count:
        indices = rng.choice(indices, size=count, replace=False)
    return np.sort(indices.astype(np.int64))


def select_diverse_episodes(
    candidates: np.ndarray,
    initial_states: np.ndarray,
    count: int,
) -> np.ndarray:
    """Deterministic farthest-point sampling over standardized initial states."""
    states = initial_states[candidates].astype(np.float64)
    scale = states.std(axis=0)
    scale[scale < 1e-6] = 1.0
    states = (states - states.mean(axis=0)) / scale
    chosen = [int(np.argmax(np.linalg.norm(states, axis=1)))]
    minimum_distance = np.full(len(states), np.inf)
    while len(chosen) < count:
        latest = states[chosen[-1]]
        distance = np.square(states - latest).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[chosen] = -1
        chosen.append(int(np.argmax(minimum_distance)))
    return candidates[np.asarray(chosen)]


def batched(items: np.ndarray, size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


@torch.inference_mode()
def encode_and_cache_split(
    h5: h5py.File,
    indices: np.ndarray,
    model: nn.Module,
    device: torch.device,
    cache_dir: Path,
    split: str,
    batch_size: int = 256,
) -> None:
    image_path = cache_dir / f"{split}_images.npy"
    latent_path = cache_dir / f"{split}_latents.npy"
    index_path = cache_dir / f"{split}_indices.npy"
    images_out = np.lib.format.open_memmap(
        image_path, mode="w+", dtype=np.uint8, shape=(len(indices), 224, 224, 3)
    )
    latents_out = np.lib.format.open_memmap(
        latent_path, mode="w+", dtype=np.float32, shape=(len(indices), 192)
    )
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    cursor = 0
    for frame_indices in batched(indices, batch_size):
        images = h5["pixels"][frame_indices]
        batch = torch.from_numpy(images).to(device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        normalized = (batch - mean) / std
        output = model.encoder(normalized, interpolate_pos_encoding=True)
        cls = output.last_hidden_state[:, 0]
        latent = model.projector(cls)
        end = cursor + len(images)
        images_out[cursor:end] = images
        latents_out[cursor:end] = latent.float().cpu().numpy()
        cursor = end
    images_out.flush()
    latents_out.flush()
    np.save(index_path, indices)


def build_cache(
    dataset_path: Path,
    checkpoint_path: Path,
    cache_dir: Path,
    config: dict,
    rebuild: bool,
) -> tuple[Split, np.ndarray]:
    required = [
        cache_dir / f"{split}_{kind}.npy"
        for split in ("train", "validation", "test")
        for kind in ("images", "latents", "indices")
    ]
    metadata_path = cache_dir / "split_metadata.json"
    if not rebuild and metadata_path.is_file() and all(path.is_file() for path in required):
        metadata = json.loads(metadata_path.read_text())
        split = Split(
            train_episodes=np.asarray(metadata["train_episodes"]),
            validation_episodes=np.asarray(metadata["validation_episodes"]),
            test_episodes=np.asarray(metadata["test_episodes"]),
        )
        return split, np.asarray(metadata["qualitative_episodes"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parents[1] / "third_party" / "le-wm"
    sys.path.insert(0, str(source_dir))
    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval().requires_grad_(False)

    data_config = config["data"]
    with h5py.File(dataset_path, "r") as h5:
        offsets = h5["ep_offset"][:]
        lengths = h5["ep_len"][:]
        split = make_episode_split(len(offsets), data_config, config["seed"])
        initial_indices = offsets.astype(np.int64)
        initial_states_by_episode = h5["state"][initial_indices]
        qualitative = select_diverse_episodes(
            split.test_episodes,
            initial_states_by_episode,
            data_config["qualitative_episodes"],
        )
        frame_counts = {
            "train": data_config["train_frames"],
            "validation": data_config["validation_frames"],
            "test": data_config["test_frames"],
        }
        episode_sets = {
            "train": split.train_episodes,
            "validation": split.validation_episodes,
            "test": split.test_episodes,
        }
        for split_name, episode_ids in episode_sets.items():
            indices = sample_frame_indices(
                episode_ids,
                offsets,
                lengths,
                frame_counts[split_name],
                config["seed"] + {"train": 1, "validation": 2, "test": 3}[split_name],
            )
            print(f"Caching {split_name}: {len(indices)} frames")
            encode_and_cache_split(
                h5, indices, model, device, cache_dir, split_name
            )

    metadata = {
        "seed": config["seed"],
        "dataset": str(dataset_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "train_episodes": split.train_episodes.tolist(),
        "validation_episodes": split.validation_episodes.tolist(),
        "test_episodes": split.test_episodes.tolist(),
        "qualitative_episodes": qualitative.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    del model
    torch.cuda.empty_cache()
    return split, qualitative


def image_edges(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    dx = gray[..., :, 1:] - gray[..., :, :-1]
    dy = gray[..., 1:, :] - gray[..., :-1, :]
    return F.pad(dx.abs(), (0, 1, 0, 0)) + F.pad(dy.abs(), (0, 0, 0, 1))


def foreground_mask(image: torch.Tensor) -> torch.Tensor:
    """Mask non-white scene content, including the gray arena border."""
    distance_from_white = (1.0 - image).abs().mean(dim=1, keepdim=True)
    saturation = image.max(dim=1, keepdim=True).values - image.min(
        dim=1, keepdim=True
    ).values
    return ((distance_from_white > 0.055) | (saturation > 0.06)).float()


def reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_weight: float,
    edge_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = foreground_mask(target)
    absolute = (prediction - target).abs()
    pixel = absolute.mean()
    foreground = (absolute * mask).sum() / (mask.sum() * 3 + 1e-6)
    edge = (image_edges(prediction) - image_edges(target)).abs().mean()
    total = pixel + foreground_weight * foreground + edge_weight * edge
    return total, {
        "loss": total.detach(),
        "pixel_l1": pixel.detach(),
        "foreground_l1": foreground.detach(),
        "edge_l1": edge.detach(),
    }


@torch.inference_mode()
def evaluate(
    decoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    foreground_weight: float,
    edge_weight: float,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = 0
    for latent, target in loader:
        latent = latent.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction = decoder(latent)
        _, components = reconstruction_loss(
            prediction, target, foreground_weight, edge_weight
        )
        mse = F.mse_loss(prediction, target)
        psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
        ssim = structural_similarity_index_measure(
            prediction, target, data_range=1.0
        )
        target_mask = foreground_mask(target)
        predicted_mask = foreground_mask(prediction)
        intersection = (target_mask * predicted_mask).sum()
        union = ((target_mask + predicted_mask) > 0).float().sum().clamp_min(1)
        metrics = {
            **components,
            "mse": mse,
            "psnr_db": psnr,
            "ssim": ssim,
            "foreground_iou": intersection / union,
        }
        batch = len(latent)
        count += batch
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value.cpu()) * batch
    return {key: value / count for key, value in totals.items()}


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=history[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history)


def render_training_curves(history: list[dict[str, float]], output: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, title, axis in (
        ("validation_loss", "Perte de validation", axes[0, 0]),
        ("validation_psnr_db", "PSNR (dB)", axes[0, 1]),
        ("validation_ssim", "SSIM", axes[1, 0]),
        ("validation_foreground_iou", "IoU du premier plan", axes[1, 1]),
    ):
        axis.plot(epochs, [row[key] for row in history], marker="o", markersize=3)
        axis.set(xlabel="époque", title=title)
        axis.grid(alpha=0.25)
    figure.suptitle("Décodeur visuel LeWM — courbes de faisabilité", fontweight="bold")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def train_decoder(
    cache_dir: Path,
    output_dir: Path,
    config: dict,
) -> tuple[VisualDecoder, list[dict[str, float]], dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_config = config["decoder"]
    train_config = config["training"]
    decoder = VisualDecoder(**decoder_config).to(device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and train_config["precision"] == "fp16"
    )
    loader_args = {
        "batch_size": train_config["batch_size"],
        "num_workers": train_config["num_workers"],
        "pin_memory": device.type == "cuda",
        "persistent_workers": train_config["num_workers"] > 0,
    }
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        CachedFrameDataset(cache_dir, "train"),
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_args,
    )
    validation_loader = DataLoader(
        CachedFrameDataset(cache_dir, "validation"),
        shuffle=False,
        **loader_args,
    )
    test_loader = DataLoader(
        CachedFrameDataset(cache_dir, "test"),
        shuffle=False,
        **loader_args,
    )
    autocast_dtype = torch.bfloat16 if train_config["precision"] == "bf16" else torch.float16
    history: list[dict[str, float]] = []
    best_loss = math.inf
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "visual_decoder_best.pt"
    training_started = time.perf_counter()

    for epoch in range(1, train_config["epochs"] + 1):
        decoder.train()
        running_loss = 0.0
        samples = 0
        for latent, target in train_loader:
            latent = latent.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=device.type == "cuda",
            ):
                prediction = decoder(latent)
                loss, _ = reconstruction_loss(
                    prediction,
                    target,
                    train_config["foreground_weight"],
                    train_config["edge_weight"],
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu()) * len(latent)
            samples += len(latent)

        decoder.eval()
        validation = evaluate(
            decoder,
            validation_loader,
            device,
            train_config["foreground_weight"],
            train_config["edge_weight"],
        )
        row = {
            "epoch": epoch,
            "train_loss": running_loss / samples,
            **{f"validation_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train={row['train_loss']:.4f} "
            f"val={validation['loss']:.4f} psnr={validation['psnr_db']:.2f} "
            f"ssim={validation['ssim']:.3f} fg_iou={validation['foreground_iou']:.3f}"
        )
        if validation["loss"] < best_loss - 1e-4:
            best_loss = validation["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": decoder.state_dict(),
                    "decoder_config": decoder_config,
                    "epoch": epoch,
                    "validation": validation,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_config["patience"]:
                print(f"Early stopping after epoch {epoch}")
                break

    saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
    decoder.load_state_dict(saved["state_dict"])
    decoder.eval()
    test_metrics = evaluate(
        decoder,
        test_loader,
        device,
        train_config["foreground_weight"],
        train_config["edge_weight"],
    )
    test_metrics.update(
        {
            "best_epoch": saved["epoch"],
            "training_seconds": time.perf_counter() - training_started,
            "parameters": sum(parameter.numel() for parameter in decoder.parameters()),
            "peak_gpu_memory_gib": (
                torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            ),
        }
    )
    write_history(output_dir / "training_history.csv", history)
    render_training_curves(history, output_dir / "training_curves.png")
    return decoder, history, test_metrics


@torch.inference_mode()
def encode_images(
    images: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    batch = torch.from_numpy(images).to(device).permute(0, 3, 1, 2).float().div_(255)
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    output = model.encoder((batch - mean) / std, interpolate_pos_encoding=True)
    return model.projector(output.last_hidden_state[:, 0])


def heatmap_difference(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    error = np.abs(target.astype(np.float32) - prediction.astype(np.float32)).mean(axis=2)
    error = np.clip(error * 4, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(error, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 25), (255, 255, 255), -1)
    cv2.putText(
        panel,
        label,
        (7, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    return panel


@torch.inference_mode()
def render_qualitative_outputs(
    dataset_path: Path,
    checkpoint_path: Path,
    cache_dir: Path,
    output_dir: Path,
    decoder: VisualDecoder,
    qualitative_episodes: np.ndarray,
    steps: int,
) -> list[dict]:
    source_dir = Path(__file__).resolve().parents[1] / "third_party" / "le-wm"
    sys.path.insert(0, str(source_dir))
    device = next(decoder.parameters()).device
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device).eval().requires_grad_(False)
    train_latents = torch.from_numpy(
        np.asarray(np.load(cache_dir / "train_latents.npy", mmap_mode="r")).copy()
    ).to(device)
    train_images = np.load(cache_dir / "train_images.npy", mmap_mode="r")
    results: list[dict] = []
    grid_rows = []

    with h5py.File(dataset_path, "r") as h5:
        offsets = h5["ep_offset"][:]
        lengths = h5["ep_len"][:]
        for row, episode in enumerate(qualitative_episodes):
            count = min(steps, int(lengths[episode]))
            indices = offsets[episode] + np.arange(count)
            targets = h5["pixels"][indices]
            latents = encode_images(targets, model, device)
            predictions = (
                decoder(latents).clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
            )
            distance = torch.cdist(latents.float(), train_latents.float())
            nearest_indices = distance.argmin(dim=1).cpu().numpy()
            nearest = np.asarray(train_images[nearest_indices])
            frames = []
            for step in range(count):
                panels = (
                    label_panel(targets[step], f"Réel — pas {step + 1}/{count}"),
                    label_panel(predictions[step], "Décodage du latent réel"),
                    label_panel(heatmap_difference(targets[step], predictions[step]), "Erreur ×4"),
                )
                frames.append(np.concatenate(panels, axis=1))
            gif_path = output_dir / f"episode_{int(episode):05d}_reconstruction_18_steps.gif"
            imageio.mimsave(gif_path, frames, duration=300, loop=0)

            selected_step = min(count - 1, count // 2)
            grid_rows.append(
                (
                    targets[selected_step],
                    predictions[selected_step],
                    heatmap_difference(targets[selected_step], predictions[selected_step]),
                    nearest[selected_step],
                )
            )
            target_tensor = torch.from_numpy(targets).permute(0, 3, 1, 2).float().div(255)
            prediction_tensor = torch.from_numpy(predictions).permute(0, 3, 1, 2).float().div(255)
            results.append(
                {
                    "episode": int(episode),
                    "start_index": int(indices[0]),
                    "steps": int(count),
                    "pixel_l1": float(F.l1_loss(prediction_tensor, target_tensor)),
                    "psnr_db": float(
                        -10
                        * torch.log10(
                            F.mse_loss(prediction_tensor, target_tensor).clamp_min(1e-12)
                        )
                    ),
                    "ssim": float(
                        structural_similarity_index_measure(
                            prediction_tensor, target_tensor, data_range=1.0
                        )
                    ),
                    "gif": gif_path.name,
                }
            )

    figure, axes = plt.subplots(
        len(grid_rows), 4, figsize=(12, 3 * len(grid_rows)), constrained_layout=True
    )
    column_titles = ("Image réelle", "Décodeur CLS", "Erreur ×4", "Plus proche latent train")
    for row, panels in enumerate(grid_rows):
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(column_titles[column])
            if column == 0:
                axes[row, column].set_ylabel(
                    f"épisode {int(qualitative_episodes[row])}", fontsize=10
                )
    figure.suptitle(
        "Faisabilité du décodage visuel depuis le latent global LeWM",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_dir / "reconstruction_comparison.png", dpi=180)
    plt.close(figure)
    del model, train_latents
    torch.cuda.empty_cache()
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.train_frames is not None:
        config["data"]["train_frames"] = args.train_frames
    seed_everything(config["seed"])

    stablewm_home = Path(os.environ["STABLEWM_HOME"])
    dataset_path = stablewm_home / "pusht_expert_train.h5"
    checkpoint_path = stablewm_home / "pusht" / "lewm_object.ckpt"
    output_dir = Path(config["output"]["directory"])
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("This feasibility protocol requires CUDA.")
    torch.cuda.reset_peak_memory_stats()

    split, qualitative = build_cache(
        dataset_path,
        checkpoint_path,
        cache_dir,
        config,
        args.rebuild_cache,
    )
    decoder, history, test_metrics = train_decoder(cache_dir, output_dir, config)
    qualitative_metrics = render_qualitative_outputs(
        dataset_path,
        checkpoint_path,
        cache_dir,
        output_dir,
        decoder,
        qualitative,
        config["data"]["qualitative_steps"],
    )
    result = {
        "protocol": "frozen LeWM encoder and projector; learned global-CLS visual decoder",
        "config": config,
        "split": {
            "train_episode_count": len(split.train_episodes),
            "validation_episode_count": len(split.validation_episodes),
            "test_episode_count": len(split.test_episodes),
            "qualitative_episodes": qualitative.tolist(),
        },
        "test_metrics": test_metrics,
        "qualitative_metrics": qualitative_metrics,
        "artifacts": {
            "checkpoint": "visual_decoder_best.pt",
            "history": "training_history.csv",
            "curves": "training_curves.png",
            "comparison": "reconstruction_comparison.png",
        },
        "limitations": [
            "This test reconstructs real encoded latents, not autoregressive predicted latents.",
            "The current LeWM checkpoint predicts one latent per block of five low-level actions.",
            "Pixel metrics are secondary to the saved qualitative comparison.",
        ],
    }
    (output_dir / "feasibility_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
