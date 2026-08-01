#!/usr/bin/env python3
"""Render the physical, predicted CEM population for one recorded PushT decision.

The CEM trace contains 300 action sequences and the corresponding latent
rollouts at every iteration.  This renderer decodes those saved latents with
the structured diagnostic decoder and draws the five predicted T poses for
every candidate.  It never steps PushT for discarded candidates: every line is
therefore explicitly a *world-model prediction*, not a simulated trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import hdf5plugin  # noqa: F401 - registers the dataset compression filter
import h5py
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from matplotlib import cm, colors, pyplot as plt
from matplotlib.patches import Circle, Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "le-wm"))

from train_structured_decoder import StructuredStateDecoder, decode_state  # noqa: E402


FRAME_SIZE = 224.0
WORLD_SIZE = 512.0
SCALE = FRAME_SIZE / WORLD_SIZE
T_VERTICES = np.array(
    [
        [-60.0, 30.0], [60.0, 30.0], [60.0, 0.0], [-60.0, 0.0],
        [-15.0, 30.0], [-15.0, 120.0], [15.0, 120.0], [15.0, 30.0],
    ],
    dtype=np.float64,
)
REQUIRED_TRACE_KEYS = {"costs", "elite_indices", "predicted_emb"}


def parse_args() -> argparse.Namespace:
    stable_home = Path(os.environ.get("STABLEWM_HOME", ROOT / ".local" / "stablewm"))
    demo_root = stable_home / "pusht" / "reproducible_cem_demo"
    decoder_dir = stable_home / "pusht" / "visual_decoder_feasibility"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=demo_root)
    parser.add_argument("--environment", type=int, required=True)
    parser.add_argument("--decision", type=int, default=0, choices=(0, 1))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=stable_home / "pusht_expert_train.h5")
    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        default=decoder_dir / "structured_decoder_best.pt",
    )
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def read_npz_with_sidecar(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    with np.load(path) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return arrays, json.loads(sidecar.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trace(trace: dict[str, np.ndarray], environment: int) -> tuple[int, int, int]:
    """Validate the subset needed by this renderer and return its dimensions."""
    missing = REQUIRED_TRACE_KEYS.difference(trace)
    if missing:
        raise ValueError(f"Trace lacks {sorted(missing)}")
    costs = trace["costs"]
    elites = trace["elite_indices"]
    latents = trace["predicted_emb"]
    if costs.ndim != 3 or latents.ndim != 5 or elites.ndim != 3:
        raise ValueError("Unexpected CEM trace rank")
    iterations, environments, population = costs.shape
    if not 0 <= environment < environments:
        raise ValueError(f"environment must be in [0, {environments - 1}]")
    if latents.shape[:3] != (iterations, environments, population):
        raise ValueError("Candidate latents and costs have incompatible shapes")
    if elites.shape[:2] != (iterations, environments):
        raise ValueError("Elite indices and costs have incompatible shapes")
    if not np.isfinite(costs).all() or not np.isfinite(latents).all():
        raise ValueError("Trace contains NaN or Inf")
    selected = elites[:, environment]
    if (selected < 0).any() or (selected >= population).any():
        raise ValueError("Elite indices are outside the candidate population")
    return iterations, population, selected.shape[1]


def cost_normalizer(costs: np.ndarray) -> colors.Normalize:
    """Robust common scale: yellow means low terminal latent cost."""
    lower, upper = np.quantile(costs, [0.02, 0.98])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = float(costs.min()), float(costs.max() + 1e-6)
    return colors.Normalize(vmin=float(lower), vmax=float(upper), clip=True)


def decode_candidate_poses(
    decoder: StructuredStateDecoder,
    latents: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Decode (candidate, latent time, latent dim) into PushT T poses."""
    if latents.ndim != 3:
        raise ValueError(f"Expected (candidate,time,latent), got {latents.shape}")
    candidate_count, time_count, latent_dim = latents.shape
    flattened = torch.from_numpy(latents.reshape(-1, latent_dim)).to(device=device, dtype=torch.float32)
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(flattened), batch_size):
            states = decode_state(decoder(flattened[start : start + batch_size]))
            chunks.append(states[:, 2:5].cpu())
    return torch.cat(chunks).numpy().reshape(candidate_count, time_count, 3)


def load_goal_state(dataset: Path, episode: int, start_step: int, offset: int) -> np.ndarray:
    with h5py.File(dataset, "r") as h5:
        matches = np.flatnonzero(
            (h5["episode_idx"][:] == episode) & (h5["step_idx"][:] == start_step + offset)
        )
        if len(matches) != 1:
            raise RuntimeError(f"No unique goal frame for episode={episode}, step={start_step + offset}")
        return np.asarray(h5["state"][matches[0]], dtype=np.float64)


def t_polygon(pose: np.ndarray) -> np.ndarray:
    x, y, angle = (float(value) for value in pose)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    polygon = T_VERTICES @ rotation.T + np.array([x, y])
    return polygon * SCALE


def add_t(ax, pose: np.ndarray, color: str, linewidth: float, *, dashed: bool = False, zorder: int = 6) -> None:
    ax.add_patch(
        Polygon(
            t_polygon(pose), closed=True, fill=False, edgecolor=color, linewidth=linewidth,
            linestyle=(0, (5, 3)) if dashed else "solid", zorder=zorder,
        )
    )


def draw_population_frame(
    image: np.ndarray,
    start_state: np.ndarray,
    goal_state: np.ndarray,
    poses: np.ndarray,
    costs: np.ndarray,
    elite_indices: np.ndarray,
    history_costs: np.ndarray,
    iteration: int,
    normalizer: colors.Normalize,
    cmap,
) -> np.ndarray:
    figure = plt.figure(figsize=(12.2, 6.8), dpi=100, constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1.0, 0.72])
    ax_scene = figure.add_subplot(grid[:, 0])
    ax_cost = figure.add_subplot(grid[0, 1])
    ax_help = figure.add_subplot(grid[1, 1])

    ax_scene.imshow(image)
    ax_scene.set_xlim(-1, FRAME_SIZE)
    ax_scene.set_ylim(FRAME_SIZE, -1)
    ax_scene.set_xticks([])
    ax_scene.set_yticks([])
    ax_scene.set_title("Départ réel, objectif CEM et futurs prédits du T", fontsize=10)
    add_t(ax_scene, start_state[2:5], "#111827", 1.8)
    add_t(ax_scene, goal_state[2:5], "#16a34a", 1.8, dashed=True)
    ax_scene.add_patch(
        Circle(tuple(start_state[:2] * SCALE), 15.0 * SCALE, fill=False, edgecolor="#1d4ed8", linewidth=1.5, zorder=6)
    )

    points = poses[:, 1:, :2] * SCALE
    for index, path in enumerate(points):
        ax_scene.plot(path[:, 0], path[:, 1], color=cmap(normalizer(costs[index])), alpha=0.31, linewidth=0.55, zorder=2)
    for index in elite_indices:
        path = points[int(index)]
        ax_scene.plot(path[:, 0], path[:, 1], color="#f97316", alpha=0.88, linewidth=1.15, zorder=4)
    best = int(elite_indices[np.argmin(costs[elite_indices])])
    ax_scene.plot(points[best, :, 0], points[best, :, 1], color="#dc2626", linewidth=1.9, zorder=5)
    ax_scene.text(
        4, 14,
        "noir : T réel au départ · vert pointillé : objectif CEM\n"
        "couleur : coût latent (jaune = faible) · orange : 30 élites · rouge : meilleure élite",
        fontsize=7.4, color="#111827", va="top", bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"}, zorder=8,
    )

    iterations = np.arange(1, iteration + 2)
    ax_cost.plot(iterations, history_costs[: iteration + 1].mean(axis=1), color="#64748b", label="coût moyen")
    ax_cost.plot(iterations, history_costs[: iteration + 1].min(axis=1), color="#dc2626", label="meilleure élite")
    ax_cost.axvline(iteration + 1, color="#111827", linewidth=1.0, alpha=0.65)
    ax_cost.set(xlim=(1, history_costs.shape[0]), xlabel="itération CEM", ylabel="distance latente au but", title="Convergence : 300 → 30 élites")
    ax_cost.grid(alpha=0.2)
    ax_cost.legend(fontsize=7)

    ax_help.axis("off")
    phase = "échantillons initiaux aléatoires" if iteration == 0 else "échantillons issus de la distribution CEM mise à jour"
    ax_help.text(
        0.02, 0.93,
        f"Itération {iteration + 1}/{history_costs.shape[0]} — {phase}.\n\n"
        "Chaque courbe contient cinq poses prédites, une par bloc de cinq actions (25 actions au total).\n\n"
        "Les 300 branches ne sont pas exécutées dans le simulateur : elles sont imaginées par LeWM puis décodées pour être lisibles.",
        va="top", fontsize=9, wrap=True,
    )
    colorbar = figure.colorbar(cm.ScalarMappable(norm=normalizer, cmap=cmap), ax=ax_scene, fraction=0.035, pad=0.02)
    colorbar.set_label("coût latent CEM (faible → élevé)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    figure.suptitle("CEM dans l'espace PushT : la population de trajectoires converge vers l'objectif", fontsize=13, fontweight="bold")
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    plt.close(figure)
    return frame


def render(args: argparse.Namespace) -> dict:
    execution, execution_meta = read_npz_with_sidecar(args.raw_root / "raw" / "execution.npz")
    trace, trace_meta = read_npz_with_sidecar(args.raw_root / "traces" / f"decision_{args.decision:04d}.npz")
    iterations, population, elite_count = validate_trace(trace, args.environment)
    if execution["observations"].ndim != 5 or execution["states"].ndim != 3:
        raise ValueError("Execution recording has unexpected observation/state shapes")
    episode = int(execution["episode_ids"][args.environment])
    start_step = int(execution["start_steps"][args.environment])
    action_offsets = np.flatnonzero(np.r_[True, np.diff(execution["decision_index_per_action"][args.environment]) != 0])
    if args.decision >= len(action_offsets):
        raise ValueError("Decision is absent from the factual execution")
    offset = int(action_offsets[args.decision])
    goal_state = load_goal_state(args.dataset, episode, start_step, offset=25)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.decoder_checkpoint, map_location=device, weights_only=True)
    decoder = StructuredStateDecoder().to(device)
    decoder.load_state_dict(saved["state_dict"])
    decoder.eval().requires_grad_(False)
    all_poses = []
    for iteration in range(iterations):
        all_poses.append(
            decode_candidate_poses(decoder, trace["predicted_emb"][iteration, args.environment], device, args.batch_size)
        )
    costs = trace["costs"][:, args.environment]
    elites = trace["elite_indices"][:, args.environment]
    normalizer = cost_normalizer(costs)
    cmap = plt.get_cmap("viridis_r")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        draw_population_frame(
            execution["observations"][args.environment, offset],
            execution["states"][args.environment, offset], goal_state,
            all_poses[iteration], costs[iteration], elites[iteration], costs,
            iteration, normalizer, cmap,
        )
        for iteration in range(iterations)
    ]
    imageio.mimsave(args.output, frames, format="GIF", duration=1.0 / args.fps, loop=0)
    metadata = {
        "schema_version": 1,
        "episode": episode,
        "start_step": start_step,
        "environment": args.environment,
        "decision": args.decision,
        "decision_action_offset": offset,
        "population": population,
        "elite_count": elite_count,
        "iterations": iterations,
        "trajectory_semantics": "predicted latent rollouts decoded by the structured PushT diagnostic; discarded candidates are never simulated",
        "cost": "recorded CEM terminal latent distance to the goal embedding",
        "source_trace": f"traces/decision_{args.decision:04d}.npz",
        "source_execution": "raw/execution.npz",
        "source_trace_sha256": sha256_file(args.raw_root / "traces" / f"decision_{args.decision:04d}.npz"),
        "source_execution_sha256": sha256_file(args.raw_root / "raw" / "execution.npz"),
        "structured_decoder_sha256": sha256_file(args.decoder_checkpoint),
        "trace_schema": trace_meta.get("schema_version"),
        "execution_schema": execution_meta.get("schema_version"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    metadata = render(parse_args())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
