#!/usr/bin/env python3
"""Publish a measured CEM convergence benchmark from one recorded MPC call.

The benchmark is deliberately narrow: it reports convergence of the *internal
latent objective*, not a proof that PushT reaches the physical goal.  Its timer
is recorded inside ``TracedCEMSolver`` with CUDA synchronization around every
iteration, excluding trace serialization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

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


def convergence_summary(costs: np.ndarray, iteration_seconds: np.ndarray) -> dict[str, float | int]:
    """Summarize the observed 95%-of-improvement point for a CEM call.

    ``costs`` is one candidate-cost matrix of shape ``(iterations, population)``.
    The returned count is the number of candidate action trajectories evaluated
    through the first iteration whose *running best* cost achieved 95 percent of
    the improvement observed at the final iteration.
    """
    costs = np.asarray(costs, dtype=np.float64)
    iteration_seconds = np.asarray(iteration_seconds, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] == 0 or costs.shape[1] == 0:
        raise ValueError("costs must have shape (iterations, population)")
    if iteration_seconds.shape != (costs.shape[0],):
        raise ValueError("iteration_seconds must provide one value per iteration")
    if not np.isfinite(costs).all() or not np.isfinite(iteration_seconds).all():
        raise ValueError("costs and iteration_seconds must be finite")
    if (iteration_seconds < 0).any():
        raise ValueError("iteration_seconds cannot be negative")

    candidate_best = costs.min(axis=1)
    running_best = np.minimum.accumulate(candidate_best)
    initial = float(running_best[0])
    final = float(running_best[-1])
    threshold = final + 0.05 * (initial - final)
    convergence_index = int(np.flatnonzero(running_best <= threshold)[0])
    elapsed = np.cumsum(iteration_seconds)
    population = int(costs.shape[1])
    return {
        "iterations": int(costs.shape[0]),
        "population": population,
        "candidate_trajectories_total": int(costs.size),
        "initial_best_cost": initial,
        "final_best_cost": final,
        "observed_improvement_fraction": 0.95,
        "convergence_threshold_cost": float(threshold),
        "convergence_iteration": convergence_index + 1,
        "candidate_trajectories_to_convergence": int((convergence_index + 1) * population),
        "compute_seconds_to_convergence": float(elapsed[convergence_index]),
        "compute_seconds_total": float(elapsed[-1]),
    }


def timing_for_environment(metadata: dict, environment: int, iterations: int) -> np.ndarray:
    """Extract synchronized per-iteration timing for exactly one environment."""
    batches = metadata.get("environment_batch_slices")
    timings = metadata.get("cem_iteration_seconds_per_batch")
    if not isinstance(batches, list) or not isinstance(timings, list) or len(batches) != len(timings):
        raise ValueError("trace has no compatible synchronized per-batch timings")
    for bounds, values in zip(batches, timings):
        if bounds == [environment, environment + 1]:
            result = np.asarray(values, dtype=np.float64)
            if result.shape != (iterations,):
                raise ValueError("trace timing length does not match CEM iterations")
            return result
    raise ValueError(
        "benchmark requires batch_size=1 so the measured time belongs to one environment"
    )


def draw_benchmark(summary: dict[str, float | int], costs: np.ndarray, seconds: np.ndarray, output: Path) -> None:
    candidate_best = costs.min(axis=1)
    running_best = np.minimum.accumulate(candidate_best)
    cumulative_candidates = np.arange(1, len(costs) + 1) * costs.shape[1]
    cumulative_seconds = np.cumsum(seconds)
    convergence = int(summary["convergence_iteration"]) - 1

    figure, (ax_cost, ax_time) = plt.subplots(1, 2, figsize=(12.4, 4.7), dpi=150, constrained_layout=True)
    ax_cost.plot(cumulative_candidates, candidate_best, color="#94a3b8", linewidth=1.2, label="meilleure candidate de l'itération")
    ax_cost.plot(cumulative_candidates, running_best, color="#111827", linewidth=2.2, label="meilleur coût cumulé")
    ax_cost.axhline(float(summary["convergence_threshold_cost"]), color="#f97316", linestyle="--", linewidth=1.4, label="seuil 95 % du gain observé")
    ax_cost.scatter([cumulative_candidates[convergence]], [running_best[convergence]], color="#16a34a", s=55, zorder=4)
    ax_cost.set(xlabel="trajectoires candidates imaginées", ylabel="distance latente au but", title="Convergence interne de CEM")
    ax_cost.grid(alpha=0.22)
    ax_cost.legend(fontsize=8)

    ax_time.plot(cumulative_seconds, running_best, color="#111827", linewidth=2.2)
    ax_time.axvline(float(summary["compute_seconds_to_convergence"]), color="#16a34a", linestyle="--", linewidth=1.4)
    ax_time.scatter([cumulative_seconds[convergence]], [running_best[convergence]], color="#16a34a", s=55, zorder=4)
    ax_time.set(xlabel="temps de calcul CEM synchronisé (s)", ylabel="meilleur coût cumulé", title="Même convergence, mesurée sur GPU")
    ax_time.grid(alpha=0.22)

    figure.suptitle(
        f"CEM mesuré : {summary['candidate_trajectories_to_convergence']:,} / "
        f"{summary['candidate_trajectories_total']:,} trajectoires, "
        f"{summary['compute_seconds_to_convergence']:.2f} / {summary['compute_seconds_total']:.2f} s",
        fontsize=13, fontweight="bold",
    )
    figure.text(
        0.5, 0.01,
        "Le seuil décrit la convergence du coût latent interne (95 % de la baisse observée), pas une garantie de réussite physique dans PushT.",
        ha="center", fontsize=8.5, color="#475569",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    stable_home = Path(os.environ.get("STABLEWM_HOME", ROOT / ".local" / "stablewm"))
    root = stable_home / "pusht" / "cem_convergence_benchmark"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=root)
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "assets" / "cem_convergence_benchmark.png")
    parser.add_argument("--metadata", type=Path, default=ROOT / "docs" / "assets" / "cem_convergence_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_path = args.raw_root / "traces" / "decision_0000.npz"
    execution_path = args.raw_root / "raw" / "execution.npz"
    with np.load(trace_path) as archive:
        costs = np.asarray(archive["costs"], dtype=np.float64)
    trace_metadata = json.loads(trace_path.with_suffix(".json").read_text())
    execution_metadata = json.loads(execution_path.with_suffix(".json").read_text())
    if costs.ndim != 3 or not 0 <= args.environment < costs.shape[1]:
        raise ValueError("costs must have shape (iterations, environments, population)")
    iteration_seconds = timing_for_environment(trace_metadata, args.environment, costs.shape[0])
    summary = convergence_summary(costs[:, args.environment], iteration_seconds)
    summary.update(
        {
            "episode": int(np.load(execution_path)["episode_ids"][args.environment]),
            "start_step": int(np.load(execution_path)["start_steps"][args.environment]),
            "success": bool(execution_metadata["evaluation_results"]["episode_successes"][args.environment]),
            "cost_semantics": "terminal LeWM latent distance to the goal embedding",
            "timing_semantics": trace_metadata["cem_timing_semantics"],
            "source_trace": "$STABLEWM_HOME/pusht/cem_convergence_benchmark/traces/decision_0000.npz",
            "source_trace_sha256": sha256_file(trace_path),
            "source_execution": "$STABLEWM_HOME/pusht/cem_convergence_benchmark/raw/execution.npz",
            "source_execution_sha256": sha256_file(execution_path),
        }
    )
    draw_benchmark(summary, costs[:, args.environment], iteration_seconds, args.output)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
