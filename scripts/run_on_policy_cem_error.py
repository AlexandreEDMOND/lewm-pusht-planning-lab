#!/usr/bin/env python3
"""Measure LeWM error under the exact actions selected by CEM.

This is intentionally a paired diagnostic on the pre-registered 24 stratified
cases, not an estimate of population success.  It runs both MPC frequencies,
then derives two non-interchangeable analyses: selected-decision forecasts and
reconstructed rollouts of the actions actually executed after replanning.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LEWM_ROOT = ROOT / "third_party" / "le-wm"
sys.path[:0] = [str(ROOT / "scripts"), str(LEWM_ROOT)]

from evaluate_decoder_rollouts import decode_pixels, load_models  # noqa: E402
from train_structured_decoder import circular_error, decode_state  # noqa: E402
from train_visual_decoder import heatmap_difference, label_panel, encode_images  # noqa: E402
from on_policy import (  # noqa: E402
    EXECUTION_SCHEMA_VERSION,
    SELECTED_PLAN_SCHEMA_VERSION,
    decision_schedule,
    executed_prediction_blocks,
    normalized_plan_to_blocks,
    read_versioned_npz,
)


BLOCK = 5
HORIZON = 5
CASES = ROOT / "docs/results/rollout_control_link_metrics.csv"
RESULTS = ROOT / "docs/results"
ASSETS = ROOT / "docs/assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run the first paired case only.")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--raw-root", type=Path, default=None)
    return parser.parse_args()


def read_cases(count: int | None = None) -> list[dict]:
    with CASES.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 24:
        raise RuntimeError(f"Expected the registered 24 cases, found {len(rows)}")
    return rows if count is None else rows[:count]


def hydra_list(values: list[int]) -> str:
    return "[" + ",".join(map(str, values)) + "]"


def run_condition(cases: list[dict], receding_horizon: int, raw_root: Path, batch_size: int) -> list[Path]:
    paths = []
    for batch_index, offset in enumerate(range(0, len(cases), batch_size)):
        batch = cases[offset : offset + batch_size]
        batch_dir = raw_root / f"rh{receding_horizon}" / f"batch_{batch_index:02d}"
        raw_dir, plan_dir = batch_dir / "raw", batch_dir / "plans"
        command = [
            sys.executable, "eval.py", "--config-name=pusht_on_policy_cem.yaml",
            f"eval.num_eval={len(batch)}",
            f"eval.fixed_episode_ids={hydra_list([int(row['episode']) for row in batch])}",
            f"eval.fixed_start_steps={hydra_list([int(row['local_start']) for row in batch])}",
            f"plan_config.receding_horizon={receding_horizon}",
            f"solver.selected_plan_dir={plan_dir}",
            f"on_policy_artifact_dir={raw_dir}",
            "on_policy_artifact_filename=execution.npz",
            f"output.filename=on_policy_rh{receding_horizon}_batch{batch_index:02d}.txt",
            f"output.metrics_filename=on_policy_rh{receding_horizon}_batch{batch_index:02d}.json",
        ]
        print(f"rh={receding_horizon}, batch {batch_index + 1}: {[row['episode'] for row in batch]}")
        subprocess.run(command, cwd=LEWM_ROOT, check=True)
        paths.append(raw_dir / "execution.npz")
    return paths


def encode_real_block_latents(observations: np.ndarray, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    # Encode only model-rate frames.  They are the frames to which a latent
    # transition has an unambiguous five-action correspondence.
    images = observations[:, ::BLOCK]
    flat = images.reshape(-1, *images.shape[2:])
    chunks = [encode_images(flat[start : start + 128], model, device).cpu() for start in range(0, len(flat), 128)]
    return torch.cat(chunks).numpy().reshape(*images.shape[:2], -1)


def physical_errors(predicted: np.ndarray, real_latent: np.ndarray, state: np.ndarray, decoder) -> dict[str, float]:
    device = next(decoder.parameters()).device
    with torch.inference_mode():
        predicted_state = decode_state(decoder(torch.from_numpy(predicted[None]).to(device)))[0].cpu()
        ceiling_state = decode_state(decoder(torch.from_numpy(real_latent[None]).to(device)))[0].cpu()
    target = torch.from_numpy(state)
    def metric(value): return float(value.cpu())
    agent = metric(torch.linalg.vector_norm(predicted_state[:2] - target[:2]))
    block = metric(torch.linalg.vector_norm(predicted_state[2:4] - target[2:4]))
    angle = metric(torch.rad2deg(circular_error(predicted_state[4], target[4])))
    ceiling_agent = metric(torch.linalg.vector_norm(ceiling_state[:2] - target[:2]))
    ceiling_block = metric(torch.linalg.vector_norm(ceiling_state[2:4] - target[2:4]))
    ceiling_angle = metric(torch.rad2deg(circular_error(ceiling_state[4], target[4])))
    return {
        "pusher_error_px": agent, "block_error_px": block, "block_angle_error_deg": angle,
        "pusher_decoder_ceiling_px": ceiling_agent, "block_decoder_ceiling_px": ceiling_block,
        "block_angle_decoder_ceiling_deg": ceiling_angle,
        "pusher_excess_error_px": agent - ceiling_agent,
        "block_excess_error_px": block - ceiling_block,
        "block_angle_excess_error_deg": angle - ceiling_angle,
    }


def rollout_actual_actions(model, context: np.ndarray, actions: np.ndarray, device: torch.device) -> np.ndarray:
    blocks = actions.reshape(HORIZON, BLOCK, -1).reshape(HORIZON, -1)
    with torch.inference_mode():
        output = model.rollout(
            {"pixels": torch.from_numpy(context[None, None]).to(device)},
            torch.from_numpy(blocks[None, None]).to(device),
        )
    return output["predicted_emb"][0, 0].cpu().numpy()


def analyse_execution(path: Path, receding_horizon: int, model, decoder, device: torch.device) -> tuple[list[dict], list[dict], list[dict]]:
    execution, metadata = read_versioned_npz(path, EXECUTION_SCHEMA_VERSION)
    observations, states = execution["observations"], execution["states"]
    latents = encode_real_block_latents(observations, model, device)
    frames, episodes, flows = [], [], []
    planning_times = [float(value) for value in metadata["planning_times_seconds_per_batch_mpc_call"]]
    per_environment_total = sum(planning_times) / len(execution["episode_ids"])
    plan_cache = {}
    for env, episode in enumerate(execution["episode_ids"]):
        decisions = execution["decision_index_per_action"][env]
        normalized_actions = execution["actions_normalized"][env].reshape(len(decisions), -1)
        physical_actions = execution["actions_postprocessed"][env].reshape(len(decisions), -1)
        start_step = int(execution["start_steps"][env])
        unique_decisions = list(dict.fromkeys(decisions.tolist()))
        action_plan_match = True
        for decision in unique_decisions:
            plan_path = Path(metadata["selected_plan_files"][str(decision)])
            if plan_path not in plan_cache:
                plan_cache[plan_path] = read_versioned_npz(plan_path, SELECTED_PLAN_SCHEMA_VERSION)[0]
            plan = plan_cache[plan_path]
            chosen = plan["normalized_actions"][env]
            predicted = plan["predicted_emb"][env]
            action_start = int(np.flatnonzero(decisions == decision)[0])
            factual_blocks = executed_prediction_blocks(receding_horizon, HORIZON)
            model_blocks = normalized_plan_to_blocks(chosen, BLOCK, physical_actions.shape[-1])
            executed = normalized_actions[action_start : action_start + receding_horizon * BLOCK]
            action_plan_match &= np.allclose(executed, model_blocks[:receding_horizon].reshape(-1, physical_actions.shape[-1]), atol=2e-6)
            for block in range(HORIZON):
                absolute = action_start + (block + 1) * BLOCK
                if absolute > len(decisions):
                    continue
                factual = bool(block in factual_blocks)
                row = {
                    "analysis": "decision_forecast", "episode": int(episode), "start_step": start_step,
                    "cem_seed": int(metadata["cem_seed"]),
                    "receding_horizon": receding_horizon, "decision_index": int(decision),
                    "decision_action_offset": action_start, "latent_block": block + 1,
                    "horizon_actions": (block + 1) * BLOCK, "branch_executed": factual,
                    "real_frame_index": action_start + (block + 1) * BLOCK,
                }
                if factual:
                    real = latents[env, (action_start // BLOCK) + block + 1]
                    row["latent_mse"] = float(np.mean((predicted[-HORIZON + block] - real) ** 2))
                    row.update(physical_errors(predicted[-HORIZON + block], real, states[env, absolute], decoder))
                else:
                    row["comparison_note"] = "counterfactual branch intentionally excluded from error aggregates"
                frames.append(row)
        episodes.append({
            "episode": int(episode), "start_step": start_step,
            "cem_seed": int(metadata["cem_seed"]), "environment_seed": "unavailable",
            "receding_horizon": receding_horizon, "action_plan_exact_match": bool(action_plan_match),
            "success": bool(metadata["evaluation_results"]["episode_successes"][env]), "decisions": len(unique_decisions),
            "model_rollouts_cem": len(unique_decisions) * 300 * 30,
            "model_rollouts_final_inference": len(unique_decisions),
            "planning_total_seconds_per_environment": per_environment_total,
            "planning_mean_seconds_per_mpc_call_per_environment": (
                float(np.mean(planning_times)) / len(execution["episode_ids"])
            ),
        })
        # Reconstructed factual flow.  A 25-action suffix starting at every MPC
        # decision is evaluated only when it fits in the 50-action rollout.
        for action_start in decision_schedule(len(decisions), receding_horizon, BLOCK):
            if action_start + HORIZON * BLOCK > len(decisions):
                continue
            decision = int(decisions[action_start])
            plan_path = Path(metadata["selected_plan_files"][str(decision)])
            plan = plan_cache[plan_path]
            context = plan["model_pixels"][env]
            predicted = rollout_actual_actions(model, context, normalized_actions[action_start : action_start + HORIZON * BLOCK], device)
            for block in range(HORIZON):
                absolute = action_start + (block + 1) * BLOCK
                real = latents[env, (action_start // BLOCK) + block + 1]
                row = {
                    "analysis": "executed_action_flow", "episode": int(episode), "start_step": start_step,
                    "receding_horizon": receding_horizon, "decision_index": decision,
                    "decision_action_offset": action_start, "latent_block": block + 1,
                    "horizon_actions": (block + 1) * BLOCK, "branch_executed": True,
                    "real_frame_index": absolute,
                    "latent_mse": float(np.mean((predicted[-HORIZON + block] - real) ** 2)),
                }
                row.update(physical_errors(predicted[-HORIZON + block], real, states[env, absolute], decoder))
                flows.append(row)
    return frames, episodes, flows


def validate_complete_paths(paths: dict[int, list[Path]], cases: list[dict]) -> None:
    """Reject missing, duplicate, or non-paired raw executions before analysis."""
    expected = {(int(row["episode"]), int(row["local_start"])) for row in cases}
    if len(expected) != 24:
        raise RuntimeError("The registered case list must contain 24 unique episode/start pairs")
    for rh, condition_paths in paths.items():
        actual: list[tuple[int, int]] = []
        for path in condition_paths:
            execution, _ = read_versioned_npz(path, EXECUTION_SCHEMA_VERSION)
            actual.extend(zip(execution["episode_ids"].tolist(), execution["start_steps"].tolist()))
        if set(actual) != expected or len(actual) != 24:
            raise RuntimeError(f"RH={rh} raw execution is incomplete or duplicated: {len(actual)} rows")


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"n": int(len(array)), "median": float(np.median(array)), "p90": float(np.quantile(array, .90)), "p95": float(np.quantile(array, .95)), "maximum": float(array.max())}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def make_figures(rows: list[dict], episodes: list[dict], assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for rh, colour in ((5, "#3366cc"), (1, "#dc3912")):
        selected = [r for r in rows if r["analysis"] == "decision_forecast" and r["branch_executed"] and r["receding_horizon"] == rh]
        horizon = sorted({r["horizon_actions"] for r in selected})
        axes[0].plot(horizon, [np.median([r["latent_mse"] for r in selected if r["horizon_actions"] == h]) for h in horizon], "o-", color=colour, label=f"RH={rh}")
        axes[1].plot(horizon, [np.median([r["block_error_px"] for r in selected if r["horizon_actions"] == h]) for h in horizon], "o-", color=colour, label=f"RH={rh}")
    axes[0].set(xlabel="horizon (actions)", ylabel="MSE latente", title="Erreur on-policy à la décision")
    axes[1].set(xlabel="horizon (actions)", ylabel="erreur T (px)", title="Erreur physique décodée")
    for axis in axes: axis.grid(alpha=.25); axis.legend()
    figure.savefig(assets_dir / "on_policy_cem_errors_by_horizon.png", dpi=180); plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    for rh, colour in ((5, "#3366cc"), (1, "#dc3912")):
        data = [r for r in rows if r["analysis"] == "decision_forecast" and r["branch_executed"] and r["receding_horizon"] == rh and r["horizon_actions"] == 5]
        axis.scatter([r["latent_mse"] for r in data], [r["block_error_px"] for r in data], c=colour, label=f"RH={rh}", alpha=.75)
    axis.set(xlabel="MSE latente à 5 actions", ylabel="erreur T décodée (px)", title="Erreur latente et erreur de pose (pas le résultat du contrôle)")
    axis.grid(alpha=.25); axis.legend(); figure.savefig(assets_dir / "on_policy_cem_latent_pose_link.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, rh in zip(axes, (5, 1)):
        episode_success = {row["episode"]: row["success"] for row in episodes if row["receding_horizon"] == rh}
        data = [row for row in rows if row["analysis"] == "decision_forecast" and row["receding_horizon"] == rh and row["branch_executed"] and row["horizon_actions"] == 5]
        by_episode = {ep: np.median([r["block_error_px"] for r in data if r["episode"] == ep]) for ep in episode_success}
        for success, color, name in ((True, "#2ca02c", "success"), (False, "#d62728", "failure")):
            vals = [(ep, value) for ep, value in by_episode.items() if episode_success[ep] == success]
            axis.scatter([ep for ep, _ in vals], [value for _, value in vals], c=color, label=f"{name} (n={len(vals)})")
        axis.set(title=f"RH={rh}: error → episode result", xlabel="episode id", ylabel="median factual block error at 5 actions (px)")
        axis.grid(alpha=.25); axis.legend()
    figure.savefig(assets_dir / "on_policy_cem_error_outcome.png", dpi=180); plt.close(figure)


def write_example_gif(raw_paths: list[Path], want_success: bool, pixel_decoder, label: str, assets_dir: Path) -> bool:
    """Render one factual selected-plan rollout; false when that class is absent."""
    for raw_path in raw_paths:
        execution, metadata = read_versioned_npz(raw_path, EXECUTION_SCHEMA_VERSION)
        outcomes = metadata["evaluation_results"]["episode_successes"]
        for env, outcome in enumerate(outcomes):
            if bool(outcome) != want_success:
                continue
            first_plan = Path(metadata["selected_plan_files"]["0"])
            plan, _ = read_versioned_npz(first_plan, SELECTED_PLAN_SCHEMA_VERSION)
            predicted = plan["predicted_emb"][env][-HORIZON:]
            with torch.inference_mode():
                _, images = decode_pixels(pixel_decoder, torch.from_numpy(predicted).to("cuda"))
            panels = []
            for index in range(HORIZON):
                real = execution["observations"][env, (index + 1) * BLOCK]
                panels.append(np.concatenate((
                    label_panel(real, f"Réel t={(index + 1) * BLOCK}"),
                    label_panel(images[index], "Prédit (plan CEM sélectionné)"),
                    label_panel(heatmap_difference(real, images[index]), "Erreur visuelle"),
                ), axis=1))
            imageio.mimsave(assets_dir / f"on_policy_cem_{label}.gif", panels, duration=550, loop=0)
            return True
    return False


def error_outcome_summary(rows: list[dict], episodes: list[dict]) -> dict:
    """Aggregate each requested factual error to one value per episode before AUC."""
    outcome = {(r["receding_horizon"], r["episode"]): bool(r["success"]) for r in episodes}
    specs = (("decision_forecast_5", "decision_forecast", 5), ("decision_forecast_25", "decision_forecast", 25), ("executed_action_flow_25", "executed_action_flow", 25))
    result = {}
    for rh in (5, 1):
        result[f"rh{rh}"] = {}
        for name, analysis, horizon in specs:
            selected = [r for r in rows if r["receding_horizon"] == rh and r["analysis"] == analysis and r["horizon_actions"] == horizon and r["branch_executed"]]
            per_episode = {ep: [r for r in selected if r["episode"] == ep] for key_rh, ep in outcome if key_rh == rh}
            metrics = {}
            for metric in ("latent_mse", "block_error_px", "block_excess_error_px"):
                values = [(outcome[(rh, ep)], float(np.median([r[metric] for r in vals]))) for ep, vals in per_episode.items() if vals]
                success = [value for ok, value in values if ok]; failure = [value for ok, value in values if not ok]
                metrics[metric] = {"episodes": len(values), "success_count": len(success), "failure_count": len(failure), "success_median": float(np.median(success)) if success else None, "failure_median": float(np.median(failure)) if failure else None, "failure_auc": float(roc_auc_score([not ok for ok, _ in values], [value for _, value in values])) if success and failure else None}
            result[f"rh{rh}"][name] = metrics
    return result


def cost_summary(episodes: list[dict], raw_paths: dict[int, list[Path]]) -> dict:
    out = {}
    for rh in (5, 1):
        calls = []
        for path in raw_paths[rh]:
            _, metadata = read_versioned_npz(path, EXECUTION_SCHEMA_VERSION)
            calls.extend(metadata["planning_times_seconds_per_batch_mpc_call"])
        condition = [r for r in episodes if r["receding_horizon"] == rh]
        total = float(sum(calls))
        out[f"rh{rh}"] = {"mpc_batch_calls": len(calls), "decisions_per_episode": int(condition[0]["decisions"]), "cem_rollouts_per_episode": int(condition[0]["model_rollouts_cem"]), "final_plan_inferences_per_episode": int(condition[0]["model_rollouts_final_inference"]), "total_batch_wall_seconds": total, "mean_seconds_per_mpc_batch_call": float(np.mean(calls)), "mean_seconds_per_environment_decision": total / len(condition) / int(condition[0]["decisions"]), "mean_seconds_per_episode": total / len(condition)}
    return out


def main() -> None:
    args = parse_args(); cases = read_cases(1 if args.smoke else None)
    stable = Path(os.environ["STABLEWM_HOME"])
    raw_root = args.raw_root or stable / ("pusht/on_policy_cem_smoke" if args.smoke else "pusht/on_policy_cem")
    results_dir = raw_root / "results" if args.smoke else ROOT / "docs/results"
    assets_dir = raw_root / "assets" if args.smoke else ROOT / "docs/assets"
    if not args.skip_evaluation:
        all_paths = {rh: run_condition(cases, rh, raw_root, args.batch_size) for rh in (5, 1)}
    else:
        all_paths = {rh: sorted((raw_root / f"rh{rh}").glob("batch_*/raw/execution.npz")) for rh in (5, 1)}
    if not args.smoke:
        validate_complete_paths(all_paths, cases)
    config = ROOT / "config/visual_decoder_feasibility.yaml"
    decoder_dir = stable / "pusht/visual_decoder_feasibility"
    model, pixel_decoder, _, decoder = load_models(decoder_dir, stable / "pusht/lewm_object.ckpt", torch.device("cuda"))
    rows, episodes = [], []
    for rh, paths in all_paths.items():
        for path in paths:
            frame_rows, episode_rows, flows = analyse_execution(path, rh, model, decoder, torch.device("cuda"))
            rows.extend(frame_rows); rows.extend(flows); episodes.extend(episode_rows)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "on_policy_cem_frame_metrics.csv", rows)
    write_csv(results_dir / "on_policy_cem_episode_metrics.csv", episodes)
    factual = [r for r in rows if r["analysis"] == "decision_forecast" and r["branch_executed"]]
    aggregates = {f"rh{rh}": {str(h): {key: quantiles([r[key] for r in factual if r["receding_horizon"] == rh and r["horizon_actions"] == h]) for key in ("latent_mse", "pusher_error_px", "block_error_px", "block_angle_error_deg")} for h in sorted({r["horizon_actions"] for r in factual if r["receding_horizon"] == rh})} for rh in (5, 1)}
    by_key = {(r["episode"], r["start_step"]): {} for r in episodes}
    for row in episodes: by_key[(row["episode"], row["start_step"])][row["receding_horizon"]] = row["success"]
    paired = {"both_success": sum(v.get(5) and v.get(1) for v in by_key.values()), "rh5_only": sum(v.get(5) and not v.get(1) for v in by_key.values()), "rh1_only": sum(not v.get(5) and v.get(1) for v in by_key.values()), "both_failure": sum(not v.get(5) and not v.get(1) for v in by_key.values())}
    first_metadata = read_versioned_npz(all_paths[5][0], EXECUTION_SCHEMA_VERSION)[1]
    result = {"protocol": {"checkpoint": "official LeWM", "checkpoint_sha256": first_metadata["checkpoint_sha256"], "cem_seed": 42, "population": 300, "iterations": 30, "elites": 30, "horizon_blocks": 5, "action_block": 5, "goal_actions": 25, "budget_actions": 50, "cases": [{"episode": int(r["episode"]), "start_step": int(r["local_start"])} for r in cases]}, "evaluation_provenance": first_metadata["code_versions"], "postprocessing_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "success": {f"rh{rh}": int(sum(r["success"] for r in episodes if r["receding_horizon"] == rh)) for rh in (5, 1)}, "paired_success": paired, "planning_cost": cost_summary(episodes, all_paths), "decision_forecast_aggregates": aggregates, "error_outcome_association_episode_median": error_outcome_summary(rows, episodes), "limitations": ["The 24 cases are risk-stratified, not a population sample.", "AUCs are descriptive with only three RH=5 failures; no causal claim follows.", "PushT state observations can leave their declared bounds (agent xy and velocity)."], "artifacts": {"frame_metrics": "docs/results/on_policy_cem_frame_metrics.csv", "episode_metrics": "docs/results/on_policy_cem_episode_metrics.csv", "errors_by_horizon": "docs/assets/on_policy_cem_errors_by_horizon.png", "error_outcome": "docs/assets/on_policy_cem_error_outcome.png", "latent_pose_link": "docs/assets/on_policy_cem_latent_pose_link.png", "success_gif": "docs/assets/on_policy_cem_success.gif", "failure_gif": "docs/assets/on_policy_cem_failure.gif", "raw_root": "$STABLEWM_HOME/pusht/on_policy_cem"}}
    (results_dir / "on_policy_cem_results.json").write_text(json.dumps(result, indent=2) + "\n")
    make_figures(rows, episodes, assets_dir)
    flat_raw_paths = [path for paths in all_paths.values() for path in paths]
    if not write_example_gif(flat_raw_paths, True, pixel_decoder, "success", assets_dir):
        print("No success GIF available in this run.")
    if not write_example_gif(flat_raw_paths, False, pixel_decoder, "failure", assets_dir):
        print("No failure GIF available in this run.")
    print(json.dumps(result["planning_cost"], indent=2))


if __name__ == "__main__":
    main()
