#!/usr/bin/env python3
"""Post-process the raw reproducible CEM demo into published artifacts.

Inputs (heavy, under ``$STABLEWM_HOME/pusht/reproducible_cem_demo/``):
  raw/execution.npz + .json          full factual trajectories of the two episodes
  traces/decision_0000..0001.npz     complete CEM population traces (local only)
  plans/selected_plan_0000..0001.npz final selected plans

Outputs (versioned):
  docs/results/cem_demo_manifest.json
  docs/results/cem_demo_episode_metrics.csv
  docs/results/cem_demo_compact/*.npz + *.json
  docs/assets/cem_demo_success.gif, cem_demo_failure.gif, cem_demo_overview.png

The script is deterministic: it sets the same CUDA determinism options as the
decoder evaluation pipeline, never samples, and only reads recorded data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Required by PyTorch for deterministic CUDA matrix multiplications.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "le-wm"))

from cem_demo import (  # noqa: E402
    ACTION_BLOCK,
    BUDGET_ACTIONS,
    CANDIDATE_STEP,
    COMPACT_TRACE_SCHEMA_VERSION,
    DECLARED_STATE_HIGH,
    DECLARED_STATE_LOW,
    DEMO_CASES,
    DEMO_DIR_NAME,
    ELITE_COUNT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DECISION_OFFSETS,
    GOAL_OFFSET_STEPS,
    HORIZON,
    ITERATIONS,
    POPULATION,
    assert_portable,
    bounds_violations,
    build_compact_trace,
    expected_decision_offsets,
    find_case_environment,
    git_provenance,
    plan_matches_executed,
    portable_path,
    record_environment,
    sanitize_value,
    sha256_file,
    validate_compact_trace,
    write_compact_trace,
)
from cem_demo_video import (  # noqa: E402
    DecisionRender,
    EpisodeRender,
    write_episode_gif,
    write_overview_png,
)
from on_policy import (  # noqa: E402
    EXECUTION_SCHEMA_VERSION,
    SELECTED_PLAN_SCHEMA_VERSION,
    circular_error_radians,
    read_versioned_npz,
)
from cem_trace import TRACE_SCHEMA_VERSION  # noqa: E402
from evaluate_decoder_rollouts import load_models  # noqa: E402
from train_structured_decoder import decode_state  # noqa: E402
from train_visual_decoder import encode_images, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "docs" / "results")
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "docs" / "assets")
    parser.add_argument("--lab-root", type=Path, default=ROOT)
    parser.add_argument("--lewm-root", type=Path, default=ROOT / "third_party" / "le-wm")
    parser.add_argument(
        "--skip-gpu", action="store_true",
        help="Skip model encoding/decoding (used only by determinism tests).",
    )
    return parser.parse_args()


def load_demo_inputs(raw_root: Path) -> tuple[dict[str, np.ndarray], dict]:
    execution_path = raw_root / "raw" / "execution.npz"
    if not execution_path.is_file():
        raise FileNotFoundError(f"Missing raw execution at {execution_path}")
    arrays, metadata = read_versioned_npz(execution_path, EXECUTION_SCHEMA_VERSION)
    expected_files = {
        "traces": [raw_root / "traces" / f"decision_{index:04d}.npz" for index in range(2)],
        "plans": [Path(metadata["selected_plan_files"][str(index)]) for index in range(2)],
    }
    for path in expected_files["traces"] + expected_files["plans"]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing raw artifact {path}")
    return arrays, metadata


def verify_protocol(arrays: dict[str, np.ndarray], metadata: dict, lab_root: Path, lewm_root: Path) -> None:
    """Reject provenance or protocol drift before any artifact is written."""
    git_provenance(lab_root, lewm_root, strict=False)
    if metadata["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: {metadata['checkpoint_sha256']}"
        )
    if int(metadata["cem_seed"]) != 42:
        raise RuntimeError(f"Unexpected CEM seed {metadata['cem_seed']}")
    cases = {(int(episode), int(start)) for episode, start in zip(
        arrays["episode_ids"], arrays["start_steps"]
    )}
    if cases != set(DEMO_CASES):
        raise RuntimeError(f"Unexpected cases {sorted(cases)}")
    for environment in range(len(arrays["episode_ids"])):
        offsets = expected_decision_offsets(arrays["decision_index_per_action"][environment])
        if offsets != EXPECTED_DECISION_OFFSETS:
            raise RuntimeError(
                f"Episode {arrays['episode_ids'][environment]} has decision "
                f"offsets {offsets}, expected {EXPECTED_DECISION_OFFSETS}"
            )


def compute_latent_errors(
    model,
    decoder,
    device: torch.device,
    arrays: dict[str, np.ndarray],
    metadata: dict,
    plans: dict[int, dict],
    environment: int,
) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    """Return per-decision real latents and per-block pred-vs-real metrics."""
    observations = arrays["observations"][environment]
    states = arrays["states"][environment]
    real_latents: dict[int, np.ndarray] = {}
    metrics: dict[int, dict] = {}
    with torch.inference_mode():
        for decision, offset in enumerate(EXPECTED_DECISION_OFFSETS):
            indices = list(range(offset, offset + (HORIZON + 1) * ACTION_BLOCK, ACTION_BLOCK))
            real_frames = observations[indices]
            real = encode_images(real_frames, model, device).cpu().numpy()
            real_latents[decision] = real
            predicted = plans[decision]["predicted_emb"][environment][1:]
            real = encode_images(real_frames, model, device).cpu().numpy()
            real_latents[decision] = real
            predicted_states = decode_state(
                decoder(torch.from_numpy(predicted).to(device))
            ).cpu().numpy()
            real_states = decode_state(
                decoder(torch.from_numpy(real[1:]).to(device))
            ).cpu().numpy()
            block_latent_mse = []
            block_position_error = []
            block_angle_error = []
            decoder_ceiling_position = []
            decoder_ceiling_angle = []
            for block in range(HORIZON):
                real_state = states[offset + (block + 1) * ACTION_BLOCK]
                target = np.asarray(real_state[:5])
                block_latent_mse.append(float(np.mean((predicted[block] - real[block + 1]) ** 2)))
                block_position_error.append(
                    float(np.linalg.norm(predicted_states[block, 2:4] - target[2:4]))
                )
                block_angle_error.append(
                    float(np.degrees(circular_error_radians(predicted_states[block, 4], target[4])))
                )
                decoder_ceiling_position.append(
                    float(np.linalg.norm(real_states[block, 2:4] - target[2:4]))
                )
                decoder_ceiling_angle.append(
                    float(np.degrees(circular_error_radians(real_states[block, 4], target[4])))
                )
            metrics[decision] = {
                "factual_blocks": list(range(1, HORIZON + 1)),
                "latent_mse": block_latent_mse,
                "block_position_error_px": block_position_error,
                "block_angle_error_deg": block_angle_error,
                "decoder_ceiling_block_position_error_px": decoder_ceiling_position,
                "decoder_ceiling_block_angle_error_deg": decoder_ceiling_angle,
                "comparison_note": (
                    "predicted: LeWM plan latent decoded by the structured decoder; "
                    "real: simulator ground-truth state"
                ),
            }
    return real_latents, metrics


def compute_goal_error(final_state: np.ndarray, goal_state: np.ndarray) -> tuple[float, float, float]:
    position_error = np.linalg.norm(final_state[:4] - goal_state[:4])
    block_position_error = np.linalg.norm(final_state[2:4] - goal_state[2:4])
    angle_difference = np.abs(final_state[4] - goal_state[4])
    block_angle_error = np.minimum(angle_difference, 2 * np.pi - angle_difference)
    normalized = np.maximum(position_error / 20.0, block_angle_error / (np.pi / 9))
    return float(block_position_error), float(np.degrees(block_angle_error)), float(normalized)


def load_goal_states(dataset_path: Path, cases: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
    """Load the goal state (row at start + goal_offset) from the dataset."""
    import h5py

    goals: dict[tuple[int, int], np.ndarray] = {}
    with h5py.File(dataset_path, "r") as dataset:
        episode_idx = dataset["episode_idx"][:]
        step_idx = dataset["step_idx"][:]
        state = dataset["state"]
        for episode, start in cases:
            goal_step = start + GOAL_OFFSET_STEPS
            matches = np.flatnonzero((episode_idx == episode) & (step_idx == goal_step))
            if len(matches) != 1:
                raise RuntimeError(f"Cannot resolve goal row for episode={episode}, step={goal_step}")
            goals[(episode, start)] = np.asarray(state[matches[0]], dtype=np.float64)
    return goals


def build_episode_records(
    args: argparse.Namespace,
    arrays: dict[str, np.ndarray],
    metadata: dict,
    traces: dict[int, dict],
    plans: dict[int, dict],
    real_latents: dict[int, dict[int, np.ndarray]],
    latent_metrics: dict[int, dict[int, dict]],
    goals: dict[tuple[int, int], np.ndarray],
    stable_home: Path,
) -> tuple[list[dict], list[EpisodeRender], dict]:
    compact_dir = args.results_dir / "cem_demo_compact"
    compact_dir.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance(args.lab_root, args.lewm_root, strict=False)
    episode_rows: list[dict] = []
    episodes_render: list[EpisodeRender] = []
    compact_files: list[dict] = []
    for case in DEMO_CASES:
        environment = find_case_environment(arrays["episode_ids"], arrays["start_steps"], case)
        success = bool(metadata["evaluation_results"]["episode_successes"][environment])
        goal = goals[case]
        offsets = expected_decision_offsets(arrays["decision_index_per_action"][environment])
        block_position, block_angle, normalized = compute_goal_error(
            arrays["states"][environment, -1], goal
        )
        decisions_render = []
        per_decision_planning: list[float] = []
        trace_paths: list[str] = []
        trace_hashes: list[str] = []
        for decision, offset in enumerate(offsets):
            compact_path = compact_dir / (
                f"compact_trace_decision_{decision:04d}_env_{environment}.npz"
            )
            executed = arrays["actions_normalized"][environment, offset : offset + 25]
            plan = plans[decision]["normalized_actions"][environment].reshape(-1, 2)
            exact_match = plan_matches_executed(plan, executed, ACTION_BLOCK, 2)
            if not exact_match:
                raise RuntimeError(
                    f"Selected plan of decision {decision} does not match the "
                    f"executed actions of episode {case[0]}"
                )
            per_decision_planning.append(
                float(metadata["planning_times_seconds_per_batch_mpc_call"][decision])
                / len(arrays["episode_ids"])
            )
            compact_arrays, compact_metadata = build_compact_trace(
                decision_index=decision,
                environment_index=environment,
                episode=case[0],
                start_step=case[1],
                success=success,
                trace=traces[decision],
                plan=plans[decision],
                real_latents=real_latents[environment][decision],
                executed_action_indices=np.arange(BUDGET_ACTIONS, dtype=np.int64),
                executed_actions_normalized=arrays["actions_normalized"][environment],
                executed_actions_postprocessed=arrays["actions_postprocessed"][environment],
                decision_action_offsets=offsets,
                seed=int(metadata["cem_seed"]),
                checkpoint_sha256=metadata["checkpoint_sha256"],
                lab_commit=provenance["lab_commit"],
                lewm_commit=provenance["lewm_commit"],
            )
            compact_metadata["block_metrics"] = latent_metrics[environment][decision]
            compact_metadata["planning_seconds_episode_estimate"] = per_decision_planning[-1]
            compact_metadata["planning_seconds_batch"] = float(
                metadata["planning_times_seconds_per_batch_mpc_call"][decision]
            )
            write_compact_trace(compact_path, compact_arrays, compact_metadata)
            compact_arrays, _ = validate_compact_trace(compact_path, COMPACT_TRACE_SCHEMA_VERSION)
            relative = compact_path.relative_to(args.lab_root)
            compact_files.append({"path": str(relative), "bytes": compact_path.stat().st_size, "sha256": sha256_file(compact_path)})
            trace_paths.append(str(relative))
            trace_hashes.append(compact_files[-1]["sha256"])
            plan_poses = decode_state(
                torch.from_numpy(plans[decision]["predicted_emb"][environment][1:])
            ).numpy()[:, [2, 3, 4]]
            plan_actions = plans[decision]["normalized_actions"][environment].mean(axis=1)
            decisions_render.append(
                DecisionRender(
                    offset=offset,
                    mean_after=compact_arrays["mean_after"],
                    std_after=compact_arrays["std_after"],
                    cost_mean=compact_arrays["cost_stats_mean"],
                    cost_median=compact_arrays["cost_stats_median"],
                    cost_min=compact_arrays["cost_stats_min"],
                    sigma_mean=compact_arrays["std_after"].mean(axis=(1, 2)),
                    plan_actions=plan_actions,
                    plan_poses=plan_poses,
                    block_errors_px=np.asarray(latent_metrics[environment][decision]["block_position_error_px"]),
                    block_latent_mse=np.asarray(latent_metrics[environment][decision]["latent_mse"]),
                    block_angle_errors_deg=np.asarray(latent_metrics[environment][decision]["block_angle_error_deg"]),
                )
            )
        bounds = bounds_violations(arrays["states"][environment])
        episode_rows.append(
            {
                "episode": case[0],
                "start_step": case[1],
                "success": success,
                "decisions": len(offsets),
                "decision_action_offsets": offsets,
                "cem_rollouts_per_episode": len(offsets) * POPULATION * ITERATIONS,
                "final_plan_inferences_per_episode": len(offsets),
                "model_rollouts_total_per_episode": len(offsets) * (POPULATION * ITERATIONS + 1),
                "planning_seconds_per_decision_estimate": per_decision_planning,
                "planning_seconds_episode_estimate": float(np.sum(per_decision_planning)),
                "action_plan_exact_match": True,
                "final_block_position_error_px": block_position,
                "final_block_angle_error_deg": block_angle,
                "final_normalized_goal_error": normalized,
                "compact_trace_decision_0": trace_paths[0],
                "compact_trace_decision_1": trace_paths[1],
                "compact_trace_decision_0_sha256": trace_hashes[0],
                "compact_trace_decision_1_sha256": trace_hashes[1],
                "pusher_x_min": bounds["pusher_xy_min"][0],
                "pusher_x_max": bounds["pusher_xy_max"][0],
                "pusher_y_min": bounds["pusher_xy_min"][1],
                "pusher_y_max": bounds["pusher_xy_max"][1],
                "velocity_x_min": bounds["velocity_min"][0],
                "velocity_x_max": bounds["velocity_max"][0],
                "velocity_y_min": bounds["velocity_min"][1],
                "velocity_y_max": bounds["velocity_max"][1],
                "out_of_bounds_frames": bounds["out_of_bounds_frames"],
                "out_of_bounds_frames_total": bounds["out_of_bounds_frames_total"],
                "first_out_of_bounds_action_index": bounds["first_out_of_bounds_action_index"],
                "out_of_bounds_at_final_frame": bounds["out_of_bounds_at_final_frame"],
                "out_of_bounds_components": bounds["out_of_bounds_components"],
            }
        )
        episodes_render.append(
            EpisodeRender(
                episode=case[0],
                start_step=case[1],
                success=success,
                observations=arrays["observations"][environment],
                states=arrays["states"][environment],
                actions_postprocessed=arrays["actions_postprocessed"][environment],
                goal_state=goals[case],
                final_block_position_error_px=block_position,
                final_block_angle_error_deg=block_angle,
                decisions=decisions_render,
            )
        )
    return episode_rows, episodes_render, {
        "files": compact_files,
        "per_episode_planning_estimates": {
            str(case): per for case, per in zip(DEMO_CASES, episode_rows)
        },
    }


def write_episode_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    path: Path,
    *,
    metadata: dict,
    provenance: dict,
    environment_snapshot: dict,
    episode_rows: list[dict],
    compact_records: dict,
    gif_paths: list[Path],
    overview_path: Path,
    metrics_path: Path,
    raw_root: Path,
    stable_home: Path,
    args: argparse.Namespace,
) -> dict:
    versioned = [
        {"path": str(metrics_path.relative_to(args.lab_root)), "bytes": metrics_path.stat().st_size, "sha256": sha256_file(metrics_path)},
        *compact_records["files"],
    ]
    gifs = []
    for gif in gif_paths:
        relative = str(gif.relative_to(args.lab_root))
        gifs.append({"path": relative, "bytes": gif.stat().st_size, "sha256": sha256_file(gif)})
        versioned.append(gifs[-1])
    versioned.append(
        {
            "path": str(overview_path.relative_to(args.lab_root)),
            "bytes": overview_path.stat().st_size,
            "sha256": sha256_file(overview_path),
        }
    )
    heavy = []
    for candidate in sorted(
        [raw_root / "raw" / "execution.npz", raw_root / "raw" / "execution.json",
         raw_root / "traces" / "decision_0000.npz", raw_root / "traces" / "decision_0000.json",
         raw_root / "traces" / "decision_0001.npz", raw_root / "traces" / "decision_0001.json",
         raw_root / "plans" / "selected_plan_0000.npz", raw_root / "plans" / "selected_plan_0000.json",
         raw_root / "plans" / "selected_plan_0001.npz", raw_root / "plans" / "selected_plan_0001.json",
         raw_root / "run_context.json", raw_root / "cem_demo_results.txt",
         raw_root / "cem_demo_metrics.json"]
    ):
        if candidate.is_file():
            heavy.append(
                {
                    "path": portable_path(candidate, stable_home),
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    plan_offsets = [0, 25]
    manifest = {
        "schema_version": 1,
        "title": "Reproducible end-to-end CEM demo on two PushT episodes",
        "protocol": {
            "checkpoint": "pusht/lewm (official LeWM checkpoint)",
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "seed": 42,
            "population": POPULATION,
            "iterations": ITERATIONS,
            "elites": ELITE_COUNT,
            "horizon_blocks": HORIZON,
            "action_block": ACTION_BLOCK,
            "receding_horizon": 5,
            "goal_offset_steps": GOAL_OFFSET_STEPS,
            "budget_actions": BUDGET_ACTIONS,
            "cost": "official LeWM latent cost at +25 actions (goal latent)",
            "success_definition": (
                "official PushT success computed by stable-worldmodel "
                "evaluate_from_dataset; the demo only reports it"
            ),
            "cases": [{"episode": episode, "start_step": start} for episode, start in DEMO_CASES],
        },
        "provenance": {
            "lab_commit": provenance["lab_commit"],
            "lewm_commit": provenance["lewm_commit"],
            "lab_clean": provenance["lab_clean"],
            "lewm_clean": provenance["lewm_clean"],
            "evaluation_commit": metadata["code_versions"]["lab"],
            "evaluation_lewm_commit": metadata["code_versions"]["lewm"],
            "postprocessing_commit": provenance["lab_commit"],
            "postprocessing_note": (
                "the evaluation commit is read from the raw execution sidecar "
                "recorded by eval.py at run time; the post-processing commit is "
                "the HEAD of this repository when the published artifacts were "
                "generated. They differ only when a correction was committed "
                "after the evaluation run."
            ),
            "seed": 42,
            "hardware": environment_snapshot["gpu"],
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "environment": environment_snapshot,
        "episodes": episode_rows,
        "planning": {
            "mpc_batch_calls": len(metadata["planning_times_seconds_per_batch_mpc_call"]),
            "planning_seconds_per_batch_call": [
                float(value) for value in metadata["planning_times_seconds_per_batch_mpc_call"]
            ],
            "note": (
                "each MPC call plans for both environments in one vectorized batch; "
                "the per-episode estimate divides the call wall time by the number "
                "of environments"
            ),
        },
        "decisions": {
            "per_episode": 2,
            "action_offsets": plan_offsets,
            "iterations_per_decision": ITERATIONS,
            "verification": (
                "offsets verified from the recorded decision_index_per_action "
                "array of the raw execution, not simulated for display"
            ),
        },
        "candidate_selection": {
            "rule": (
                "per iteration, the compact trace keeps the sorted union of the "
                "30 elite indices and the systematic sample 0,10,...,290 of the "
                "300 candidates"
            ),
            "step": CANDIDATE_STEP,
            "reconstruction_note": (
                "candidates outside the kept set cannot be reconstructed from "
                "the compact trace"
            ),
        },
        "compact_schema_version": COMPACT_TRACE_SCHEMA_VERSION,
        "artifacts": {
            "versioned": versioned,
            "heavy": heavy,
            "self_reference": {
                "path": "docs/results/cem_demo_manifest.json",
                "note": (
                    "the manifest cannot contain its own digest; "
                    "scripts/validate_cem_demo.py recomputes it on every validation"
                ),
            },
        },
        "limitations": [
            "The two episodes are communication examples chosen before the run, not a performance sample.",
            "Physical pred-vs-real errors depend on the structured decoder, a diagnostic that knows PushT geometry; it is not part of the CEM cost.",
            "Per-episode planning times are estimates because one MPC call covers both environments.",
            "The gymnasium out-of-declared-bounds warning is measured per episode and reported as a specification inconsistency or a descriptive factor, not a demonstrated cause.",
            "If the rerun changes an episode outcome, the observed outcome is reported and analysed, never forced.",
        ],
        "resolved_config_sanitized": sanitize_value(metadata["resolved_config"], stable_home),
    }
    assert_portable(manifest, "manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Post-processing requires the same CUDA device as the evaluation")
    stable_home = Path(os.environ["STABLEWM_HOME"]).resolve()
    raw_root = args.raw_root or stable_home / "pusht" / DEMO_DIR_NAME
    environment_snapshot = record_environment()
    seed_everything(42)
    torch.use_deterministic_algorithms(True, warn_only=True)

    arrays, metadata = load_demo_inputs(raw_root)
    verify_protocol(arrays, metadata, args.lab_root, args.lewm_root)
    traces = {
        index: read_versioned_npz(raw_root / "traces" / f"decision_{index:04d}.npz", TRACE_SCHEMA_VERSION)[0]
        for index in range(2)
    }
    plans = {
        index: read_versioned_npz(Path(metadata["selected_plan_files"][str(index)]), SELECTED_PLAN_SCHEMA_VERSION)[0]
        for index in range(2)
    }
    goals = load_goal_states(stable_home / "pusht_expert_train.h5", DEMO_CASES)

    device = torch.device("cuda")
    checkpoint = stable_home / "pusht" / "lewm_object.ckpt"
    decoder_dir = stable_home / "pusht" / "visual_decoder_feasibility"
    model, _, _, decoder = load_models(decoder_dir, checkpoint, device)

    real_latents: dict[int, dict[int, np.ndarray]] = {}
    latent_metrics: dict[int, dict[int, dict]] = {}
    for case in DEMO_CASES:
        environment = find_case_environment(arrays["episode_ids"], arrays["start_steps"], case)
        real_latents[environment], latent_metrics[environment] = compute_latent_errors(
            model, decoder, device, arrays, metadata, plans, environment
        )

    episode_rows, episodes_render, compact_records = build_episode_records(
        args, arrays, metadata, traces, plans, real_latents, latent_metrics, goals,
        stable_home,
    )
    metrics_path = args.results_dir / "cem_demo_episode_metrics.csv"
    write_episode_metrics_csv(metrics_path, episode_rows)

    success_episodes = [episode for episode in episodes_render if episode.success]
    failure_episodes = [episode for episode in episodes_render if not episode.success]
    gif_paths: list[Path] = []
    for label, episodes in (("success", success_episodes), ("failure", failure_episodes)):
        if episodes:
            path = args.assets_dir / f"cem_demo_{label}.gif"
            write_episode_gif(episodes[0], path)
            gif_paths.append(path)
        else:
            print(f"No {label} episode in this run; {label} GIF skipped.")
    overview_path = args.assets_dir / "cem_demo_overview.png"
    write_overview_png(episodes_render, overview_path)

    provenance = git_provenance(args.lab_root, args.lewm_root, strict=False)
    manifest_path = args.results_dir / "cem_demo_manifest.json"
    manifest = write_manifest(
        manifest_path,
        metadata=metadata,
        provenance=provenance,
        environment_snapshot=environment_snapshot,
        episode_rows=episode_rows,
        compact_records=compact_records,
        gif_paths=gif_paths,
        overview_path=overview_path,
        metrics_path=metrics_path,
        raw_root=raw_root,
        stable_home=stable_home,
        args=args,
    )
    print(json.dumps(
        {
            "episodes": [
                {"episode": row["episode"], "start_step": row["start_step"], "success": row["success"]}
                for row in episode_rows
            ]
        },
        default=str,
    ))
    for entry in manifest["artifacts"]["versioned"]:
        print(f"versioned {entry['path']} {entry['bytes']} bytes sha256={entry['sha256'][:16]}...")


if __name__ == "__main__":
    main()
