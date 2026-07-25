#!/usr/bin/env python3
"""Train the paper-style query Transformer decoder on frozen LeWM CLS latents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401 - register HDF5 compression filters
import h5py
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "le-wm"))

from train_visual_decoder import (  # noqa: E402
    CachedFrameDataset,
    VisualDecoder,
    encode_images,
    evaluate,
    heatmap_difference,
    load_config,
    reconstruction_loss,
    seed_everything,
)


class QueryDecoderBlock(nn.Module):
    """Cross-attend learned spatial queries to one global CLS memory token."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_memory = self.memory_norm(memory)
        attended, _ = self.cross_attention(
            normalized_queries,
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.mlp(self.mlp_norm(queries))


class TransformerVisualDecoder(nn.Module):
    """Appendix-D decoder: CLS memory, learned patch queries, RGB patch output."""

    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 256,
        image_size: int = 224,
        patch_size: int = 16,
        depth: int = 4,
        heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_queries = self.grid_size**2
        self.latent_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
        )
        self.queries = nn.Parameter(
            torch.randn(1, self.num_queries, hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                QueryDecoderBlock(hidden_dim, heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.patch_projection = nn.Linear(
            hidden_dim, patch_size * patch_size * 3
        )

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        pixels = patches.view(
            batch,
            self.grid_size,
            self.grid_size,
            self.patch_size,
            self.patch_size,
            3,
        )
        return (
            pixels.permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, 3, self.image_size, self.image_size)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        memory = self.latent_projection(latent).unsqueeze(1)
        queries = self.queries.expand(len(latent), -1, -1)
        for block in self.blocks:
            queries = block(queries, memory)
        patches = self.patch_projection(self.output_norm(queries))
        return torch.sigmoid(self.unpatchify(patches))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Reuse the saved best checkpoint and only evaluate/render outputs.",
    )
    return parser.parse_args()


def render_curves(history: list[dict[str, float]], output: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = (
        ("validation_loss", "Perte de validation"),
        ("validation_psnr_db", "PSNR (dB)"),
        ("validation_ssim", "SSIM"),
        ("validation_foreground_iou", "IoU du premier plan"),
    )
    for axis, (key, title) in zip(axes.flat, plots):
        axis.plot(epochs, [row[key] for row in history], marker="o", markersize=3)
        axis.set(xlabel="époque", title=title)
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Décodeur Transformer à requêtes de patches",
        fontweight="bold",
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def train_decoder(
    cache_dir: Path,
    output_dir: Path,
    config: dict,
) -> tuple[TransformerVisualDecoder, list[dict[str, float]], dict[str, float]]:
    device = torch.device("cuda")
    model = TransformerVisualDecoder(**config["transformer_decoder"]).to(device)
    settings = config["transformer_training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    loader_args = {
        "batch_size": settings["batch_size"],
        "num_workers": settings["num_workers"],
        "pin_memory": True,
        "persistent_workers": settings["num_workers"] > 0,
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
    autocast_dtype = (
        torch.bfloat16 if settings["precision"] == "bf16" else torch.float16
    )
    history: list[dict[str, float]] = []
    best_loss = math.inf
    stale_epochs = 0
    checkpoint = output_dir / "transformer_decoder_best.pt"
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for latent, target in train_loader:
            latent = latent.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=autocast_dtype):
                prediction = model(latent)
                loss, _ = reconstruction_loss(
                    prediction,
                    target,
                    settings["foreground_weight"],
                    settings["edge_weight"],
                )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(latent)
            count += len(latent)
        model.eval()
        validation = evaluate(
            model,
            validation_loader,
            device,
            settings["foreground_weight"],
            settings["edge_weight"],
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / count,
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
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "decoder_config": config["transformer_decoder"],
                    "epoch": epoch,
                    "validation": validation,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= settings["patience"]:
                print(f"Early stopping after epoch {epoch}")
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        settings["foreground_weight"],
        settings["edge_weight"],
    )
    test_metrics.update(
        {
            "best_epoch": saved["epoch"],
            "training_seconds": time.perf_counter() - started,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
    )
    with (output_dir / "transformer_training_history.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=history[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history)
    render_curves(history, output_dir / "transformer_training_curves.png")
    return model, history, test_metrics


@torch.inference_mode()
def render_comparison(
    dataset_path: Path,
    checkpoint_path: Path,
    cache_dir: Path,
    output_dir: Path,
    transformer: TransformerVisualDecoder,
) -> list[dict]:
    device = next(transformer.parameters()).device
    world_model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    world_model = world_model.to(device).eval().requires_grad_(False)
    conv_saved = torch.load(
        output_dir / "visual_decoder_best.pt", map_location=device, weights_only=True
    )
    convolutional = VisualDecoder(**conv_saved["decoder_config"]).to(device)
    convolutional.load_state_dict(conv_saved["state_dict"])
    convolutional.eval()
    metadata = json.loads((cache_dir / "split_metadata.json").read_text())
    episodes = np.asarray(metadata["qualitative_episodes"])
    rows = []
    results = []
    with h5py.File(dataset_path, "r") as h5:
        offsets = h5["ep_offset"][:]
        lengths = h5["ep_len"][:]
        for episode in episodes:
            step = min(9, int(lengths[episode]) - 1)
            image = h5["pixels"][offsets[episode] + step : offsets[episode] + step + 1]
            latent = encode_images(image, world_model, device)
            conv = (
                convolutional(latent)[0]
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            decoded = (
                transformer(latent)[0]
                .clamp(0, 1)
                .mul(255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            rows.append((image[0], conv, decoded, heatmap_difference(image[0], decoded)))
            results.append({"episode": int(episode), "frame_step": int(step)})

    figure, axes = plt.subplots(
        len(rows), 4, figsize=(12, 3 * len(rows)), constrained_layout=True
    )
    titles = (
        "Image réelle",
        "Convolution",
        "Transformer",
        "Erreur Transformer ×4",
    )
    for row, panels in enumerate(rows):
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(titles[column])
            if column == 0:
                axes[row, column].set_ylabel(f"épisode {episodes[row]}")
    figure.suptitle(
        "Décodeurs LeWM — même latent, même split de données",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_dir / "transformer_decoder_comparison.png", dpi=180)
    plt.close(figure)
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["transformer_training"]["epochs"] = args.epochs
    seed_everything(config["seed"])
    stablewm_home = Path(os.environ["STABLEWM_HOME"])
    output_dir = Path(config["output"]["directory"])
    cache_dir = output_dir / "cache"
    dataset_path = stablewm_home / "pusht_expert_train.h5"
    checkpoint_path = stablewm_home / "pusht" / "lewm_object.ckpt"
    if not (cache_dir / "split_metadata.json").is_file():
        raise FileNotFoundError("Run train_visual_decoder.py first.")
    if args.render_only:
        saved = torch.load(
            output_dir / "transformer_decoder_best.pt",
            map_location="cuda",
            weights_only=True,
        )
        transformer = TransformerVisualDecoder(**saved["decoder_config"]).cuda()
        transformer.load_state_dict(saved["state_dict"])
        transformer.eval()
        settings = config["transformer_training"]
        test_loader = DataLoader(
            CachedFrameDataset(cache_dir, "test"),
            batch_size=settings["batch_size"],
            shuffle=False,
            num_workers=settings["num_workers"],
            pin_memory=True,
        )
        metrics = evaluate(
            transformer,
            test_loader,
            torch.device("cuda"),
            settings["foreground_weight"],
            settings["edge_weight"],
        )
        metrics.update(
            {
                "best_epoch": saved["epoch"],
                "training_seconds": None,
                "parameters": sum(
                    parameter.numel() for parameter in transformer.parameters()
                ),
                "peak_gpu_memory_gib": None,
            }
        )
    else:
        transformer, _, metrics = train_decoder(cache_dir, output_dir, config)
    qualitative = render_comparison(
        dataset_path,
        checkpoint_path,
        cache_dir,
        output_dir,
        transformer,
    )
    result = {
        "protocol": (
            "LeWM Appendix-D style decoder: one CLS memory token, 196 learned "
            "queries, cross-attention, 16x16 RGB patches"
        ),
        "test_metrics": metrics,
        "qualitative_samples": qualitative,
        "artifacts": {
            "checkpoint": "transformer_decoder_best.pt",
            "curves": "transformer_training_curves.png",
            "comparison": "transformer_decoder_comparison.png",
        },
    }
    (output_dir / "transformer_feasibility_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
