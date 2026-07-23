#!/usr/bin/env python3
"""Render an exportable CEM planning video from a saved Phase 2 trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_ARRAYS = {
    "candidates",
    "costs",
    "elite_indices",
    "elite_costs",
    "mean_before",
    "std_before",
    "mean_after",
    "std_after",
    "predicted_emb",
    "goal_emb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path, help="Phase 2 decision_*.npz")
    parser.add_argument("--rollout", required=True, type=Path, help="PushT rollout_*.mp4")
    parser.add_argument("--output", required=True, type=Path, help="Output MP4 path")
    parser.add_argument("--environment", type=int, default=0, help="Environment index in the trace")
    parser.add_argument("--fps", type=int, default=8, help="Rendered video frame rate")
    return parser.parse_args()


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = REQUIRED_ARRAYS.difference(archive.files)
        if missing:
            raise ValueError(f"Trace lacks required arrays: {sorted(missing)}")
        return {key: archive[key] for key in REQUIRED_ARRAYS}


def pca_projection_basis(predicted_emb: np.ndarray, goal_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic two-dimensional PCA basis to a bounded latent sample."""
    points = predicted_emb.reshape(-1, predicted_emb.shape[-1])
    if len(points) > 6000:
        indices = np.linspace(0, len(points) - 1, num=6000, dtype=int)
        points = points[indices]
    points = np.concatenate((points, goal_emb.reshape(-1, goal_emb.shape[-1])), axis=0)
    centre = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - centre, full_matrices=False)
    return centre, vectors[:2].T


def project(points: np.ndarray, centre: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (points - centre) @ basis


def read_rollout_frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def add_action_panel(ax, trace: dict[str, np.ndarray], iteration: int, environment: int) -> None:
    candidates = trace["candidates"][iteration, environment, :, 0, :2]
    elite_indices = trace["elite_indices"][iteration, environment]
    elites = candidates[elite_indices]
    mean = trace["mean_after"][iteration, environment, 0, :2]

    ax.scatter(candidates[:, 0], candidates[:, 1], s=8, color="#8d99ae", alpha=0.35, label="candidats")
    ax.scatter(elites[:, 0], elites[:, 1], s=18, color="#f4a261", alpha=0.9, label="élites")
    ax.scatter(*mean, s=90, marker="X", color="#e63946", label="μ après mise à jour")
    ax.axhline(0, color="#adb5bd", linewidth=0.8)
    ax.axvline(0, color="#adb5bd", linewidth=0.8)
    extent = max(1.1, float(np.abs(candidates).max()) * 1.1)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
           xlabel="action normalisée x", ylabel="action normalisée y",
           title="Population : première action 2D normalisée")
    ax.legend(loc="upper right", fontsize=7)


def add_cost_panel(ax, trace: dict[str, np.ndarray], iteration: int, environment: int) -> None:
    costs = trace["costs"][:, environment]
    elite_costs = trace["elite_costs"][:, environment]
    iterations = np.arange(1, iteration + 2)
    ax.plot(iterations, costs[: iteration + 1].mean(axis=1), color="#6c757d", label="coût moyen")
    ax.plot(iterations, elite_costs[: iteration + 1].mean(axis=1), color="#f4a261", label="coût élites")
    ax.plot(iterations, elite_costs[: iteration + 1].min(axis=1), color="#2a9d8f", label="meilleur coût")
    ax.set(xlim=(1, costs.shape[0]), xlabel="itération CEM", ylabel="coût latent",
           title="Convergence du coût")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)

    variance_axis = ax.twinx()
    sigma = trace["std_after"][: iteration + 1, environment].mean(axis=(1, 2))
    variance_axis.plot(iterations, sigma, color="#9b5de5", linestyle="--", label="σ moyen")
    variance_axis.set_ylabel("σ moyen", color="#9b5de5")
    variance_axis.tick_params(axis="y", labelcolor="#9b5de5")


def add_latent_panel(
    ax,
    trace: dict[str, np.ndarray],
    iteration: int,
    environment: int,
    centre: np.ndarray,
    basis: np.ndarray,
) -> None:
    rollouts = trace["predicted_emb"][iteration, environment]
    terminal = project(rollouts[:, -1], centre, basis)
    elite_indices = trace["elite_indices"][iteration, environment]
    mean_rollout = project(rollouts[0], centre, basis)  # candidate 0 is μ before update
    best_rollout = project(rollouts[elite_indices[0]], centre, basis)
    goal = project(trace["goal_emb"][iteration, environment, 0], centre, basis)

    ax.scatter(terminal[:, 0], terminal[:, 1], s=7, color="#8d99ae", alpha=0.25, label="terminaux candidats")
    ax.scatter(terminal[elite_indices, 0], terminal[elite_indices, 1], s=18, color="#f4a261", label="terminaux élites")
    ax.plot(mean_rollout[:, 0], mean_rollout[:, 1], "o-", color="#e63946", markersize=3, label="rollout μ")
    ax.plot(best_rollout[:, 0], best_rollout[:, 1], "o-", color="#2a9d8f", markersize=3, label="meilleur rollout")
    ax.scatter(*goal, s=130, marker="*", color="#264653", label="objectif latent")
    ax.set(xlabel="PCA 1", ylabel="PCA 2", title="Rollouts latents (projection PCA)")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=7)


def render(trace: dict[str, np.ndarray], frames: list[np.ndarray], environment: int, output: Path, fps: int) -> None:
    iterations, environments = trace["costs"].shape[:2]
    if not 0 <= environment < environments:
        raise ValueError(f"environment must be in [0, {environments - 1}]")
    if not frames:
        raise ValueError("The rollout video contains no frames.")

    centre, basis = pca_projection_basis(trace["predicted_emb"], trace["goal_emb"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        for iteration in range(iterations):
            figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
            frame_index = round(iteration * (len(frames) - 1) / max(iterations - 1, 1))
            axes[0, 0].imshow(frames[frame_index])
            applied_action = trace["mean_after"][iteration, environment, 0, :2]
            axes[0, 0].set_title(
                f"Rollout PushT réel — frame {frame_index + 1}/{len(frames)} (contexte)\n"
                f"action moyenne normalisée : [{applied_action[0]:+.2f}, {applied_action[1]:+.2f}]"
            )
            axes[0, 0].axis("off")
            add_action_panel(axes[0, 1], trace, iteration, environment)
            add_cost_panel(axes[1, 0], trace, iteration, environment)
            add_latent_panel(axes[1, 1], trace, iteration, environment, centre, basis)
            figure.suptitle(
                f"CEM / MPC — décision tracée, environnement {environment}, itération {iteration + 1}/{iterations}",
                fontsize=15,
                fontweight="bold",
            )
            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba())[..., :3]
            writer.append_data(image)
            plt.close(figure)


def main() -> None:
    args = parse_args()
    trace = load_trace(args.trace)
    frames = read_rollout_frames(args.rollout)
    render(trace, frames, args.environment, args.output, args.fps)
    metadata = {
        "trace": str(args.trace.resolve()),
        "rollout": str(args.rollout.resolve()),
        "output": str(args.output.resolve()),
        "environment": args.environment,
        "fps": args.fps,
        "panels": {
            "environment": "Real PushT rollout frames sampled as context; they are not time-synchronised to CEM iterations.",
            "actions": "Raw first normalized 2D action of candidates, elites and updated mean.",
            "costs": "Raw candidate and elite costs, with mean standard deviation.",
            "latents": "Qualitative PCA projection fitted to saved latent rollouts and goal.",
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Phase 3 video saved to {args.output}")


if __name__ == "__main__":
    main()
