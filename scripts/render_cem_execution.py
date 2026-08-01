#!/usr/bin/env python3
"""Render the factual PushT execution of actions selected by CEM.

Unlike the population visualisation, every frame here comes from PushT after
the chosen action was really sent to the simulator.  It therefore makes the
distinction between imagined CEM branches and the one executed control stream
visible in the README.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decision_offsets(indices: np.ndarray) -> list[int]:
    """Return elementary-action offsets where the selected CEM plan changes."""
    indices = np.asarray(indices)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("decision indices must be a non-empty vector")
    return np.flatnonzero(np.r_[True, np.diff(indices) != 0]).astype(int).tolist()


def draw_frame(
    image: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    offsets: list[int],
    t: int,
    episode: int,
    start_step: int,
) -> np.ndarray:
    figure = plt.figure(figsize=(8.9, 5.8), dpi=100)
    grid = figure.add_gridspec(2, 2, width_ratios=[4.6, 1.75], height_ratios=[4.2, 1.0], wspace=0.12, hspace=0.28)
    ax_scene = figure.add_subplot(grid[0, 0])
    ax_timeline = figure.add_subplot(grid[0, 1])
    ax_caption = figure.add_subplot(grid[1, :])

    ax_scene.imshow(image)
    ax_scene.set_xticks([])
    ax_scene.set_yticks([])
    ax_scene.set_title("Simulation réelle : les actions sélectionnées sont envoyées à PushT", fontsize=10)
    scale = image.shape[0] / 512.0
    # Blue is the actual pusher path; orange is the factual centre path of the
    # T.  Both are taken from simulator states, not from the world-model decode.
    ax_scene.plot(states[: t + 1, 0] * scale, states[: t + 1, 1] * scale, color="#2563eb", linewidth=1.7, label="pousseur réel")
    ax_scene.plot(states[: t + 1, 2] * scale, states[: t + 1, 3] * scale, color="#f97316", linewidth=1.7, label="centre du T réel")
    ax_scene.scatter(states[0, 0] * scale, states[0, 1] * scale, s=32, color="#1d4ed8", edgecolors="white", linewidths=0.7, zorder=5)
    ax_scene.scatter(states[0, 2] * scale, states[0, 3] * scale, s=32, color="#ea580c", edgecolors="white", linewidths=0.7, zorder=5)
    ax_scene.scatter(states[t, 0] * scale, states[t, 1] * scale, s=36, color="#1d4ed8", edgecolors="#111827", linewidths=0.7, zorder=6)
    ax_scene.scatter(states[t, 2] * scale, states[t, 3] * scale, s=36, color="#f97316", edgecolors="#111827", linewidths=0.7, zorder=6)
    ax_scene.legend(loc="lower left", fontsize=8, framealpha=0.82)

    action_count = len(actions)
    active_decision = int(np.searchsorted(offsets, min(t, action_count), side="right") - 1)
    for index, offset in enumerate(offsets):
        end = offsets[index + 1] if index + 1 < len(offsets) else action_count
        color = "#bfdbfe" if index == 0 else "#fed7aa"
        ax_timeline.axhspan(offset, end, color=color, alpha=0.82)
        ax_timeline.text(0.5, (offset + end) / 2, f"plan CEM {index + 1}\nactions {offset}–{end - 1}", ha="center", va="center", fontsize=8)
    ax_timeline.axhline(min(t, action_count), color="#111827", linewidth=2.0)
    ax_timeline.set_ylim(action_count, 0)
    ax_timeline.set_xlim(0, 1)
    ax_timeline.set_xticks([])
    ax_timeline.set_yticks(offsets + [action_count])
    ax_timeline.set_ylabel("action réellement exécutée", fontsize=8)
    ax_timeline.set_title("Replanification", fontsize=9)

    ax_caption.axis("off")
    action_text = "fin de l'exécution" if t == action_count else f"action envoyée à PushT : [{actions[t, 0]:+.2f}, {actions[t, 1]:+.2f}]"
    ax_caption.text(
        0.5, 0.70,
        f"Épisode {episode}, départ {start_step} · t = {t}/{action_count} · plan CEM actif : {active_decision + 1} · {action_text}",
        ha="center", va="center", fontsize=10, fontweight="bold",
    )
    ax_caption.text(
        0.5, 0.28,
        "Bleu : trajectoire réelle du pousseur. Orange : déplacement réel du T. À t=25, CEM reçoit une nouvelle image et choisit un nouveau plan.",
        ha="center", va="center", fontsize=8.5, color="#475569",
    )
    figure.canvas.draw()
    rendered = np.asarray(figure.canvas.buffer_rgba())[..., :3]
    plt.close(figure)
    return rendered


def parse_args() -> argparse.Namespace:
    stable_home = Path(os.environ.get("STABLEWM_HOME", ROOT / ".local" / "stablewm"))
    demo_root = stable_home / "pusht" / "reproducible_cem_demo"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=demo_root)
    parser.add_argument("--environment", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets" / "cem_execution_success.gif")
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execution_path = args.raw_root / "raw" / "execution.npz"
    metadata_path = execution_path.with_suffix(".json")
    with np.load(execution_path) as archive:
        observations = np.asarray(archive["observations"])
        states = np.asarray(archive["states"])
        actions = np.asarray(archive["actions_postprocessed"])
        decision_indices = np.asarray(archive["decision_index_per_action"])
        episodes = np.asarray(archive["episode_ids"])
        starts = np.asarray(archive["start_steps"])
    metadata = json.loads(metadata_path.read_text())
    if not 0 <= args.environment < len(episodes):
        raise ValueError("environment is outside the recorded execution")
    if observations.shape[1] != actions.shape[1] + 1 or states.shape[1] != observations.shape[1]:
        raise ValueError("execution must contain initial state plus one frame per action")
    offsets = decision_offsets(decision_indices[args.environment])
    if offsets != [0, 25]:
        raise ValueError(f"expected the two recorded CEM replans [0, 25], got {offsets}")
    frames = [
        draw_frame(
            observations[args.environment, t], states[args.environment], actions[args.environment],
            offsets, t, int(episodes[args.environment]), int(starts[args.environment]),
        )
        for t in range(actions.shape[1] + 1)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, format="GIF", duration=1.0 / args.fps, loop=0)
    sidecar = {
        "schema_version": 1,
        "episode": int(episodes[args.environment]),
        "start_step": int(starts[args.environment]),
        "success": bool(metadata["evaluation_results"]["episode_successes"][args.environment]),
        "actions_executed": int(actions.shape[1]),
        "decision_action_offsets": offsets,
        "frame_semantics": "each GIF image is the factual PushT observation after t executed CEM actions",
        "path_semantics": "blue pusher and orange T paths are simulator states; no predicted or decoded state is drawn",
        "source_execution": "$STABLEWM_HOME/pusht/reproducible_cem_demo/raw/execution.npz",
        "source_execution_sha256": sha256_file(execution_path),
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
    print(json.dumps(sidecar, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
