#!/usr/bin/env python3
"""Run fixed CEM evaluations and relate control outcomes to rollout errors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import pointbiserialr, spearmanr
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LEWM_ROOT = ROOT / "third_party" / "le-wm"
BATCH_SIZE = 4
SUBSET_SIZE = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generalization-dir",
        type=Path,
        default=Path(
            os.path.expandvars(
                "$STABLEWM_HOME/pusht/visual_decoder_feasibility/generalization"
            )
        ),
    )
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def select_risk_stratified_subset(rows: list[dict], count: int) -> list[dict]:
    """Cover the full latent-risk rank range without using control outcomes."""
    ordered = sorted(rows, key=lambda row: float(row["terminal_latent_mse"]))
    indices = np.linspace(0, len(ordered) - 1, count)
    indices = np.unique(np.round(indices).astype(int))
    if len(indices) != count:
        raise RuntimeError("Risk-stratified selection produced duplicate ranks")
    return [ordered[index] for index in indices]


def hydra_list(values: list[int]) -> str:
    return "[" + ",".join(str(value) for value in values) + "]"


def run_batches(selection: list[dict], output_dir: Path) -> list[Path]:
    metrics_paths = []
    for batch_index, start in enumerate(range(0, len(selection), BATCH_SIZE)):
        batch = selection[start : start + BATCH_SIZE]
        episodes = [int(row["episode"]) for row in batch]
        starts = [int(row["local_start"]) for row in batch]
        relative_dir = Path("visual_decoder_feasibility") / "generalization" / "control"
        metrics_name = relative_dir / f"batch_{batch_index:02d}_metrics.json"
        results_name = relative_dir / f"batch_{batch_index:02d}_results.txt"
        command = [
            sys.executable,
            "eval.py",
            "--config-name=pusht_generalization_control.yaml",
            f"eval.num_eval={len(batch)}",
            f"eval.fixed_episode_ids={hydra_list(episodes)}",
            f"eval.fixed_start_steps={hydra_list(starts)}",
            f"output.metrics_filename={metrics_name}",
            f"output.filename={results_name}",
        ]
        print(
            f"CEM batch {batch_index + 1}/{int(np.ceil(len(selection) / BATCH_SIZE))}: "
            f"episodes={episodes}"
        )
        subprocess.run(command, cwd=LEWM_ROOT, check=True)
        metrics_paths.append(
            Path(os.environ["STABLEWM_HOME"]) / "pusht" / metrics_name
        )
    return metrics_paths


def load_control_rows(metrics_paths: list[Path]) -> list[dict]:
    rows = []
    for batch_index, path in enumerate(metrics_paths):
        metrics = json.loads(path.read_text())
        count = len(metrics["episode_ids"])
        for index in range(count):
            rows.append(
                {
                    "batch": batch_index,
                    "episode": int(metrics["episode_ids"][index]),
                    "local_start": int(metrics["start_steps"][index]),
                    "success": bool(metrics["episode_successes"][index]),
                    "final_state_distance": float(
                        metrics["final_state_distance_per_episode"][index]
                    ),
                    "final_block_position_error_px": float(
                        metrics["final_block_position_error_px_per_episode"][index]
                    ),
                    "final_block_angle_error_deg": float(
                        metrics["final_block_angle_error_deg_per_episode"][index]
                    ),
                    "best_goal_step": int(
                        metrics["best_goal_step_per_episode"][index]
                    ),
                    "best_normalized_goal_error": float(
                        metrics["best_normalized_goal_error_per_episode"][index]
                    ),
                    "best_position_error_px": float(
                        metrics["best_position_error_px_per_episode"][index]
                    ),
                    "best_block_position_error_px": float(
                        metrics["best_block_position_error_px_per_episode"][index]
                    ),
                    "best_block_angle_error_deg": float(
                        metrics["best_block_angle_error_deg_per_episode"][index]
                    ),
                    "final_elite_latent_cost": float(
                        metrics["final_elite_cost_per_episode"][index]
                    ),
                }
            )
    return rows


def safe_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    result = spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def analyze(joined: list[dict]) -> dict:
    success = np.asarray([bool(row["success"]) for row in joined])
    failure = (~success).astype(np.int64)
    offline_latent = np.asarray(
        [float(row["terminal_latent_mse"]) for row in joined]
    )
    offline_block = np.asarray(
        [float(row["terminal_block_error_px"]) for row in joined]
    )
    offline_angle = np.asarray(
        [float(row["terminal_block_angle_error_deg"]) for row in joined]
    )
    best_goal_error = np.asarray(
        [float(row["best_normalized_goal_error"]) for row in joined]
    )
    best_block = np.asarray(
        [float(row["best_block_position_error_px"]) for row in joined]
    )
    best_angle = np.asarray(
        [float(row["best_block_angle_error_deg"]) for row in joined]
    )
    elite_cost = np.asarray(
        [float(row["final_elite_latent_cost"]) for row in joined]
    )
    auc = (
        float(roc_auc_score(failure, offline_latent))
        if len(np.unique(failure)) == 2
        else None
    )
    point_biserial = (
        pointbiserialr(failure, offline_latent)
        if len(np.unique(failure)) == 2
        else None
    )

    order = np.argsort(offline_latent)
    quartile_indices = np.array_split(order, 4)
    quartiles = []
    for index, indices in enumerate(quartile_indices, 1):
        quartiles.append(
            {
                "quartile": index,
                "count": int(len(indices)),
                "latent_mse_min": float(offline_latent[indices].min()),
                "latent_mse_max": float(offline_latent[indices].max()),
                "success_rate": float(success[indices].mean() * 100),
                "median_best_normalized_goal_error": float(
                    np.median(best_goal_error[indices])
                ),
                "median_best_block_error_px": float(
                    np.median(best_block[indices])
                ),
            }
        )
    return {
        "subset_size": len(joined),
        "success_count": int(success.sum()),
        "failure_count": int(failure.sum()),
        "success_rate": float(success.mean() * 100),
        "binary_failure_prediction": {
            "terminal_latent_mse_roc_auc": auc,
            "terminal_latent_mse_point_biserial_r": (
                float(point_biserial.statistic) if point_biserial else None
            ),
            "terminal_latent_mse_point_biserial_pvalue": (
                float(point_biserial.pvalue) if point_biserial else None
            ),
        },
        "continuous_correlations": {
            "offline_latent_vs_best_goal_error": safe_spearman(
                offline_latent, best_goal_error
            ),
            "offline_latent_vs_best_block_position": safe_spearman(
                offline_latent, best_block
            ),
            "offline_latent_vs_best_block_angle": safe_spearman(
                offline_latent, best_angle
            ),
            "offline_block_error_vs_best_block_position": safe_spearman(
                offline_block, best_block
            ),
            "offline_angle_error_vs_best_block_angle": safe_spearman(
                offline_angle, best_angle
            ),
            "cem_elite_cost_vs_best_goal_error": safe_spearman(
                elite_cost, best_goal_error
            ),
        },
        "latent_risk_quartiles": quartiles,
    }


def render_control_link(joined: list[dict], output: Path) -> None:
    success = np.asarray([bool(row["success"]) for row in joined])
    colors = np.where(success, "#2ca02c", "#d62728")
    latent = np.asarray([float(row["terminal_latent_mse"]) for row in joined])
    offline_block = np.asarray(
        [float(row["terminal_block_error_px"]) for row in joined]
    )
    best_goal_error = np.asarray(
        [float(row["best_normalized_goal_error"]) for row in joined]
    )
    best_block = np.asarray(
        [float(row["best_block_position_error_px"]) for row in joined]
    )
    elite = np.asarray([float(row["final_elite_latent_cost"]) for row in joined])
    order = np.argsort(latent)
    quartiles = np.array_split(order, 4)

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    axes[0, 0].scatter(latent, best_goal_error, c=colors, s=55, alpha=0.82)
    rho = spearmanr(latent, best_goal_error)
    axes[0, 0].set(
        xlabel="MSE latente offline à t=35",
        ylabel="meilleure erreur normalisée au but",
        title=f"Risque offline → contrôle — ρ={rho.statistic:.2f}",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 1].scatter(offline_block, best_block, c=colors, s=55, alpha=0.82)
    rho = spearmanr(offline_block, best_block)
    axes[0, 1].set(
        xlabel="erreur offline du T à t=35 (px)",
        ylabel="meilleure erreur du T au but (px)",
        title=f"Erreur physique offline → contrôle — ρ={rho.statistic:.2f}",
    )
    axes[1, 0].boxplot(
        [best_goal_error[indices] for indices in quartiles],
        tick_labels=("Q1", "Q2", "Q3", "Q4"),
        showmeans=True,
    )
    rates = [success[indices].mean() * 100 for indices in quartiles]
    for index, rate in enumerate(rates, 1):
        axes[1, 0].text(
            index,
            axes[1, 0].get_ylim()[1],
            f"succès {rate:.0f} %",
            ha="center",
            va="top",
            fontsize=9,
        )
    axes[1, 0].set(
        xlabel="quartile de MSE latente offline",
        ylabel="meilleure erreur normalisée au but",
        title="Calibration par niveau de risque",
    )
    axes[1, 1].scatter(elite, best_goal_error, c=colors, s=55, alpha=0.82)
    rho = spearmanr(elite, best_goal_error)
    axes[1, 1].set(
        xlabel="coût latent des élites CEM",
        ylabel="meilleure erreur normalisée au but",
        title=f"Coût interne CEM → contrôle — ρ={rho.statistic:.2f}",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.suptitle(
        "Le risque de rollout prédit-il la performance CEM ?\n"
        "vert = succès, rouge = échec",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.generalization_dir / "control"
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = read_csv(
        args.generalization_dir / "generalization_episode_metrics.csv"
    )
    selection = select_risk_stratified_subset(episode_rows, SUBSET_SIZE)
    selection_path = output_dir / "control_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "method": (
                    "24 evenly spaced ranks after sorting all 128 held-out windows "
                    "by terminal latent MSE; no control outcome used for selection"
                ),
                "episodes": [
                    {
                        "episode": int(row["episode"]),
                        "local_start": int(row["local_start"]),
                        "terminal_latent_mse": float(row["terminal_latent_mse"]),
                    }
                    for row in selection
                ],
            },
            indent=2,
        )
        + "\n"
    )
    expected_paths = [
        output_dir / f"batch_{index:02d}_metrics.json"
        for index in range(int(np.ceil(len(selection) / BATCH_SIZE)))
    ]
    if args.skip_evaluation:
        missing = [path for path in expected_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing control batches: {missing}")
        metrics_paths = expected_paths
    else:
        metrics_paths = run_batches(selection, output_dir)
    control_rows = load_control_rows(metrics_paths)
    offline = {int(row["episode"]): row for row in episode_rows}
    joined = []
    for control in control_rows:
        episode = int(control["episode"])
        source = offline[episode]
        joined.append(
            {
                **control,
                "terminal_latent_mse": float(source["terminal_latent_mse"]),
                "mean_predicted_latent_mse": float(
                    source["mean_predicted_latent_mse"]
                ),
                "terminal_block_error_px": float(
                    source["terminal_block_error_px"]
                ),
                "terminal_block_angle_error_deg": float(
                    source["terminal_block_angle_error_deg"]
                ),
                "offline_terminal_category": source[
                    "terminal_transition_category"
                ],
            }
        )
    joined.sort(key=lambda row: float(row["terminal_latent_mse"]))
    joined_path = output_dir / "control_link_metrics.csv"
    write_csv(joined_path, joined)
    analysis = analyze(joined)
    render_control_link(joined, output_dir / "control_link_analysis.png")
    result = {
        "protocol": {
            "subset": (
                "risk-stratified 24/128 held-out windows selected only from "
                "terminal offline latent MSE"
            ),
            "offline_rollout": (
                "expert action sequence from the same initial episode/start, t=0..35"
            ),
            "control": {
                "planner": "CEM",
                "horizon": 5,
                "action_block": 5,
                "receding_horizon": 5,
                "population": 300,
                "iterations": 30,
                "elites": 30,
                "goal_offset_steps": 25,
                "evaluation_budget": 50,
            },
            "interpretation_limit": (
                "CEM executes optimized actions, not the expert actions used by "
                "the offline rollout. Correlations measure shared task/model "
                "difficulty, not on-policy prediction error."
            ),
        },
        "analysis": analysis,
        "artifacts": {
            "selection": selection_path.name,
            "joined_metrics": joined_path.name,
            "plot": "control_link_analysis.png",
            "batch_metrics": [path.name for path in metrics_paths],
        },
    }
    result_path = output_dir / "control_link_results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(analysis, indent=2))
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    main()
