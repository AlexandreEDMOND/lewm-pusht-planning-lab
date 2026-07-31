"""Shared contract of the reproducible CEM demo.

The demo links, for two pre-registered PushT episodes, the real state, the two
CEM decisions taken during the 50 executed actions, the convergence of the CEM
search, the selected plans, the executed actions, the futures predicted by
LeWM, the trajectory really obtained and the final success or failure.

This module contains no environment or model side-effects.  It defines the
schema of the published compact traces and the pure functions used by the
post-processing script, the validation script and the unit tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from on_policy import write_npz_with_metadata  # type: ignore


# --------------------------------------------------------------------------
# Pre-registered protocol.  These are communication examples, not estimates.
# --------------------------------------------------------------------------

DEMO_CASES = [(3876, 16), (1766, 2)]
EXPECTED_CHECKPOINT_SHA256 = (
    "a7f1ae0cfbfad8aca613f737d66d12220fa2a8e345c5b46de8b89496c44ced62"
)
DEMO_SEED = 42
POPULATION = 300
ITERATIONS = 30
ELITE_COUNT = 30
HORIZON = 5
ACTION_BLOCK = 5
ACTION_DIM = ACTION_BLOCK * 2
GOAL_OFFSET_STEPS = 25
BUDGET_ACTIONS = 50
# The official protocol replans at elementary-action offsets 0 and 25.
EXPECTED_DECISION_OFFSETS = [0, 25]

DEMO_DIR_NAME = "reproducible_cem_demo"
ON_POLICY_DIR_NAME = "on_policy_cem"

COMPACT_TRACE_SCHEMA_VERSION = 1
CANDIDATE_STEP = 10
# Row width of the kept-candidate arrays: every 10th candidate (30) plus the
# 30 elite indices, worst case 60, padded with -1 / NaN.
MAX_KEPT_CANDIDATES = POPULATION // CANDIDATE_STEP + ELITE_COUNT

# Declared PushT state space (gymnasium Box in stable-worldmodel 0.0.6).
DECLARED_STATE_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
DECLARED_STATE_HIGH = np.array(
    [512.0, 512.0, 512.0, 512.0, 2 * np.pi, 512.0, 512.0]
)
STATE_SEMANTICS = {
    0: "agent x (px)",
    1: "agent y (px)",
    2: "block x (px)",
    3: "block y (px)",
    4: "block angle (rad)",
    5: "block velocity x (px/s)",
    6: "block velocity y (px/s)",
}

COMPACT_TRACE_UNITS = {
    "normalized_actions": (
        "model action space (z-scores of the dataset actions; WorldModelPolicy "
        "applies the inverse StandardScaler before sending actions to PushT)"
    ),
    "postprocessed_actions": "exact arrays returned by WorldModelPolicy to PushT (action space [-1, 1]^2)",
    "latents": "LeWM CLS-projector embeddings, latent_dim values per latent time",
    "costs": "official LeWM latent cost between predicted and goal embeddings (unitless)",
    "positions": "PushT world pixels, window 512x512",
    "time": "elementary PushT actions (one environment step per action)",
}

COMPACT_TRACE_SEMANTICS = {
    "mean_before": "CEM mean before the iteration's distribution update, (iteration, horizon, action_dim)",
    "std_before": "CEM standard deviation before the update, (iteration, horizon, action_dim)",
    "mean_after": "CEM mean after the update (elite mean), (iteration, horizon, action_dim)",
    "std_after": "CEM standard deviation after the update (elite std, ddof=1)",
    "cost_stats_min": "minimum candidate cost per iteration",
    "cost_stats_median": "median candidate cost per iteration",
    "cost_stats_mean": "mean candidate cost per iteration",
    "cost_stats_p90": "90th percentile candidate cost per iteration",
    "cost_stats_p95": "95th percentile candidate cost per iteration",
    "elite_costs": "costs of the elite candidates, (iteration, elite)",
    "elite_actions": "actions of the elite candidates, (iteration, elite, horizon, action_dim)",
    "elite_terminal_latents": "terminal latent of each elite candidate, (iteration, elite, latent_dim)",
    "kept_candidate_indices": "deterministic candidate indices kept per iteration, -1 padded",
    "kept_candidate_counts": "number of valid indices per iteration row",
    "kept_candidate_costs": "costs of the kept candidates, NaN padded",
    "kept_candidate_actions": "actions of the kept candidates, NaN padded",
    "kept_terminal_latents": "terminal latents of the kept candidates, NaN padded",
    "goal_latent": "latent of the goal image used by the cost",
    "final_plan_actions": "normalized actions of the selected plan (final CEM mean), (horizon, action_dim)",
    "final_plan_latents": "latent rollout of the selected plan, (latent_time, latent_dim), index 0 is the context",
    "final_plan_cost": "latent cost of the selected plan",
    "real_latents": "latents of the real frames at block boundaries, (latent_time, latent_dim), index 0 is the context",
    "executed_action_indices": "elementary action indices executed in this rollout, 0..49",
    "executed_actions_normalized": "normalized actions really sent to PushT, (budget, action_dim)",
    "executed_actions_postprocessed": "physical actions really sent to PushT, (budget, action_dim)",
    "decision_action_offsets": "elementary action offsets at which this episode replans",
}

COMPACT_TRACE_REQUIRED_ARRAYS = frozenset(
    [
        "mean_before",
        "std_before",
        "mean_after",
        "std_after",
        "cost_stats_min",
        "cost_stats_median",
        "cost_stats_mean",
        "cost_stats_p90",
        "cost_stats_p95",
        "elite_costs",
        "elite_actions",
        "elite_terminal_latents",
        "kept_candidate_indices",
        "kept_candidate_counts",
        "kept_candidate_costs",
        "kept_candidate_actions",
        "kept_terminal_latents",
        "goal_latent",
        "final_plan_actions",
        "final_plan_latents",
        "final_plan_cost",
        "real_latents",
        "executed_action_indices",
        "executed_actions_normalized",
        "executed_actions_postprocessed",
        "decision_action_offsets",
    ]
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance(
    lab_root: Path,
    lewm_root: Path,
    strict: bool = False,
    ignore_paths: Iterable[Path | str] = (),
) -> dict[str, Any]:
    """Return exact commits and cleanliness, rejecting dirty provenance.

    ``strict=True`` also rejects untracked files, which is only appropriate
    before any output of this command has been written.  ``ignore_paths``
    (relative to the repository root) are excluded from the tracked-change
    check: the demo manifest records the post-processing HEAD and is therefore
    necessarily rewritten by the command that produces it; its content is
    instead verified by comparing two reruns.
    """
    ignored = {str(Path(path)) for path in ignore_paths}
    result: dict[str, Any] = {}
    for label, root in (("lab", lab_root), ("lewm", lewm_root)):
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip()
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True, text=True, capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            raise RuntimeError(f"Git provenance requires a clean tracked tree: {root}")
        changes: list[str] = []
        for line in status.splitlines():
            path = line[3:]
            if path in ignored:
                continue
            if line.startswith("??") and not strict:
                continue
            changes.append(line)
        if changes:
            raise RuntimeError(f"Refusing dirty provenance for {label} at {root}")
        result[label + "_commit"] = commit
        result[label + "_clean"] = True
    return result


def record_environment() -> dict[str, Any]:
    """Snapshot the Python/PyTorch/CUDA/GPU environment used for the run."""
    try:
        import torch
    except ImportError:  # pragma: no cover - environment-dependent branch
        return {"python": platform.python_version(), "torch": "unavailable"}
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def portable_path(path: Path, stable_home: Path) -> str:
    """Express a path portably as ``$STABLEWM_HOME/...`` when under the home."""
    text = str(path)
    prefix = str(stable_home.resolve())
    if text.startswith(prefix):
        return "$STABLEWM_HOME" + text[len(prefix):]
    if "$STABLEWM_HOME" in text:
        return text
    raise ValueError(f"Path is not portable (outside STABLEWM_HOME): {text}")


def sanitize_value(value: Any, stable_home: Path) -> Any:
    """Recursively rewrite absolute STABLEWM_HOME prefixes in config values."""
    if isinstance(value, str):
        prefix = str(stable_home.resolve())
        return value.replace(prefix, "$STABLEWM_HOME")
    if isinstance(value, dict):
        return {key: sanitize_value(item, stable_home) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item, stable_home) for item in value]
    return value


def assert_portable(value: Any, label: str) -> None:
    """Reject absolute machine paths in published content."""
    text = json.dumps(value, sort_keys=True, default=str)
    if "/home/" in text or str(Path.home()) in text:
        raise ValueError(f"Non-portable path found in {label}")


def keep_candidate_indices(elite_indices: np.ndarray, population: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic kept-candidate selection per CEM iteration.

    The kept set is the sorted union of the elite indices and the systematic
    sample ``0, step, 2*step, ...``.  Rows are padded with -1 up to
    ``MAX_KEPT_CANDIDATES``.  Candidates outside the kept set cannot be
    reconstructed from this compact trace.
    """
    elite_indices = np.asarray(elite_indices, dtype=np.int64)
    iterations = elite_indices.shape[0]
    systematic = np.arange(0, population, step, dtype=np.int64)
    counts = np.empty(iterations, dtype=np.int64)
    kept = np.full((iterations, MAX_KEPT_CANDIDATES), -1, dtype=np.int64)
    for iteration in range(iterations):
        unique = np.unique(np.concatenate((systematic, elite_indices[iteration])))
        counts[iteration] = len(unique)
        kept[iteration, : len(unique)] = unique
    return kept, counts


def expected_decision_offsets(decision_index_per_action: np.ndarray) -> list[int]:
    """Return the elementary action offsets at which this episode replanned."""
    decisions = decision_index_per_action.astype(np.int64)
    starts = np.flatnonzero(np.concatenate(([1], np.diff(decisions) != 0)))
    if len(starts) < 1 or int(decisions[starts[0]]) != int(decisions[0]):
        raise ValueError("Invalid decision index schedule")
    return [int(offset) for offset in starts.tolist()]


def plan_matches_executed(
    plan_actions: np.ndarray,
    executed_normalized: np.ndarray,
    action_block: int,
    action_dim: int,
    atol: float = 2e-6,
) -> bool:
    """Check that the selected plan equals the actions actually executed.

    The first ``len(executed)`` elementary actions of the plan (which with
    ``receding_horizon=5`` is the whole plan) must match the normalized actions
    recorded on the PushT trajectory.
    """
    plan = np.asarray(plan_actions, dtype=np.float64).reshape(-1, action_dim)
    executed = np.asarray(executed_normalized, dtype=np.float64).reshape(-1, action_dim)
    if executed.shape[0] > plan.shape[0]:
        raise ValueError(
            "Executed actions exceed the selected plan length; pass the window "
            "belonging to this decision"
        )
    if executed.shape[0] % action_block != 0:
        raise ValueError("Executed action count must be a multiple of action_block")
    return bool(np.allclose(executed, plan[: executed.shape[0]], atol=atol, rtol=0))


def bounds_violations(states: np.ndarray) -> dict[str, Any]:
    """Describe the PushT/Gymnasium out-of-declared-bounds observations.

    This is a specification inconsistency or a descriptive factor, never a
    demonstrated cause of the outcome.
    """
    states = np.asarray(states, dtype=np.float64)
    outside = (states < DECLARED_STATE_LOW) | (states > DECLARED_STATE_HIGH)
    per_frame = outside.any(axis=1)
    first = int(np.flatnonzero(per_frame)[0]) if per_frame.any() else None
    components = sorted(
        {int(index) for index in np.flatnonzero(outside.any(axis=0))}
    )
    return {
        "pusher_xy_min": [float(value) for value in states[:, :2].min(axis=0)],
        "pusher_xy_max": [float(value) for value in states[:, :2].max(axis=0)],
        "velocity_min": [float(value) for value in states[:, 5:7].min(axis=0)],
        "velocity_max": [float(value) for value in states[:, 5:7].max(axis=0)],
        "out_of_bounds_frames": int(per_frame.sum()),
        "out_of_bounds_frames_total": int(len(states)),
        "first_out_of_bounds_action_index": first,
        "out_of_bounds_at_final_frame": bool(per_frame[-1]),
        "out_of_bounds_components": components,
        "declared_low": [float(value) for value in DECLARED_STATE_LOW],
        "declared_high": [float(value) for value in DECLARED_STATE_HIGH],
        "note": (
            "The declared gymnasium state space bounds the velocity components to "
            "[0, 512] while the simulator records negative velocities.  This is "
            "reported as a specification inconsistency or a descriptive factor, "
            "not as a demonstrated cause of the episode outcome."
        ),
    }


def build_compact_trace(
    *,
    decision_index: int,
    environment_index: int,
    episode: int,
    start_step: int,
    success: bool,
    trace: dict[str, np.ndarray],
    plan: dict[str, np.ndarray],
    real_latents: np.ndarray,
    executed_action_indices: np.ndarray,
    executed_actions_normalized: np.ndarray,
    executed_actions_postprocessed: np.ndarray,
    decision_action_offsets: list[int],
    seed: int,
    checkpoint_sha256: str,
    lab_commit: str,
    lewm_commit: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the compact per-decision arrays and their metadata sidecar."""
    if not (0 <= environment_index < trace["costs"].shape[1]):
        raise ValueError("environment_index out of trace range")
    if episode < 0 or start_step < 0:
        raise ValueError("episode and start_step must be positive")

    iterations, population = trace["costs"].shape[0], trace["costs"].shape[2]
    if iterations != ITERATIONS or population != POPULATION:
        raise ValueError(f"Unexpected trace shape {trace['costs'].shape}")
    costs = trace["costs"][:, environment_index]
    kept_indices, kept_counts = keep_candidate_indices(
        trace["elite_indices"][:, environment_index], population, CANDIDATE_STEP
    )
    batch_indices = np.arange(iterations)[:, None]
    elites = trace["elite_indices"][:, environment_index]

    elite_actions = trace["candidates"][:, environment_index][
        batch_indices, elites
    ]
    elite_terminal = trace["predicted_emb"][:, environment_index][..., -1, :][
        batch_indices, elites
    ]
    kept_actions = np.full(
        (iterations, MAX_KEPT_CANDIDATES, HORIZON, ACTION_DIM), np.nan, dtype=np.float32
    )
    kept_terminal = np.full(
        (iterations, MAX_KEPT_CANDIDATES, trace["predicted_emb"].shape[-1]),
        np.nan, dtype=np.float32,
    )
    kept_costs = np.full((iterations, MAX_KEPT_CANDIDATES), np.nan, dtype=np.float32)
    for iteration in range(iterations):
        count = int(kept_counts[iteration])
        row = kept_indices[iteration, :count]
        kept_actions[iteration, :count] = trace["candidates"][iteration, environment_index, row]
        kept_terminal[iteration, :count] = trace["predicted_emb"][iteration, environment_index, row, -1]
        kept_costs[iteration, :count] = costs[iteration, row]

    arrays: dict[str, np.ndarray] = {
        "mean_before": np.asarray(trace["mean_before"][:, environment_index], dtype=np.float32),
        "std_before": np.asarray(trace["std_before"][:, environment_index], dtype=np.float32),
        "mean_after": np.asarray(trace["mean_after"][:, environment_index], dtype=np.float32),
        "std_after": np.asarray(trace["std_after"][:, environment_index], dtype=np.float32),
        "cost_stats_min": costs.min(axis=1).astype(np.float32),
        "cost_stats_median": np.median(costs, axis=1).astype(np.float32),
        "cost_stats_mean": costs.mean(axis=1).astype(np.float32),
        "cost_stats_p90": np.quantile(costs, 0.90, axis=1).astype(np.float32),
        "cost_stats_p95": np.quantile(costs, 0.95, axis=1).astype(np.float32),
        "elite_costs": np.asarray(trace["elite_costs"][:, environment_index], dtype=np.float32),
        "elite_actions": np.asarray(elite_actions, dtype=np.float32),
        "elite_terminal_latents": np.asarray(elite_terminal, dtype=np.float32),
        "kept_candidate_indices": kept_indices,
        "kept_candidate_counts": kept_counts,
        "kept_candidate_costs": kept_costs,
        "kept_candidate_actions": np.asarray(kept_actions, dtype=np.float32),
        "kept_terminal_latents": np.asarray(kept_terminal, dtype=np.float32),
        "goal_latent": np.asarray(plan["goal_emb"][environment_index], dtype=np.float32),
        "final_plan_actions": np.asarray(plan["normalized_actions"][environment_index], dtype=np.float32),
        "final_plan_latents": np.asarray(plan["predicted_emb"][environment_index], dtype=np.float32),
        "final_plan_cost": np.asarray(plan["final_cost"][environment_index : environment_index + 1], dtype=np.float32),
        "real_latents": np.asarray(real_latents, dtype=np.float32),
        "executed_action_indices": np.asarray(executed_action_indices, dtype=np.int64),
        "executed_actions_normalized": np.asarray(executed_actions_normalized, dtype=np.float32),
        "executed_actions_postprocessed": np.asarray(executed_actions_postprocessed, dtype=np.float32),
        "decision_action_offsets": np.asarray(decision_action_offsets, dtype=np.int64),
    }
    metadata = {
        "schema_version": COMPACT_TRACE_SCHEMA_VERSION,
        "decision_index": int(decision_index),
        "environment_index": int(environment_index),
        "episode": int(episode),
        "start_step": int(start_step),
        "success": bool(success),
        "seed": int(seed),
        "checkpoint_sha256": checkpoint_sha256,
        "lab_commit": lab_commit,
        "lewm_commit": lewm_commit,
        "protocol": {
            "population": int(population),
            "iterations": int(iterations),
            "elites": int(ELITE_COUNT),
            "horizon": int(HORIZON),
            "action_block": int(ACTION_BLOCK),
            "action_dim": int(ACTION_DIM),
            "latent_dim": int(trace["predicted_emb"].shape[-1]),
            "goal_offset_steps": int(GOAL_OFFSET_STEPS),
            "budget_actions": int(BUDGET_ACTIONS),
            "receding_horizon": int(5),
        },
        "decision_action_offsets": [int(value) for value in decision_action_offsets],
        "candidate_selection": {
            "rule": (
                "sorted union of the elite indices and the systematic sample "
                "0, 10, ..., 290 of the 300 candidates, per CEM iteration"
            ),
            "step": int(CANDIDATE_STEP),
            "max_kept": int(MAX_KEPT_CANDIDATES),
            "reconstruction_note": (
                "candidates outside the kept set cannot be reconstructed from "
                "this compact trace"
            ),
        },
        "units": COMPACT_TRACE_UNITS,
        "semantics": COMPACT_TRACE_SEMANTICS,
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
    }
    return arrays, metadata


def validate_compact_trace(path: Path, expected_schema_version: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a compact trace and reject incomplete or inconsistent content."""
    arrays, metadata = read_compact_trace(path, expected_schema_version)
    missing = COMPACT_TRACE_REQUIRED_ARRAYS.difference(arrays.keys())
    if missing:
        raise ValueError(f"Compact trace lacks required arrays: {sorted(missing)}")
    iterations = arrays["cost_stats_min"].shape[0]
    if iterations != ITERATIONS:
        raise ValueError(f"Expected {ITERATIONS} iterations, got {iterations}")
    if arrays["elite_costs"].shape != (ITERATIONS, ELITE_COUNT):
        raise ValueError(f"Unexpected elite_costs shape {arrays['elite_costs'].shape}")
    if arrays["elite_actions"].shape[1] != ELITE_COUNT:
        raise ValueError("Elite action count does not match the protocol")
    if arrays["mean_before"].shape != (ITERATIONS, HORIZON, ACTION_DIM):
        raise ValueError("Unexpected distribution shape in compact trace")
    offsets = [int(value) for value in arrays["decision_action_offsets"]]
    if offsets != EXPECTED_DECISION_OFFSETS:
        raise ValueError(f"Decision offsets {offsets} != {EXPECTED_DECISION_OFFSETS}")
    if arrays["executed_action_indices"].shape != (BUDGET_ACTIONS,):
        raise ValueError("Executed action indices do not cover the 50-action budget")
    for key in ("kept_candidate_costs", "kept_candidate_actions", "kept_terminal_latents"):
        if arrays[key].shape[1] != MAX_KEPT_CANDIDATES:
            raise ValueError(f"Unexpected kept-candidate width for {key}")
    if "block_metrics" not in metadata:
        raise ValueError("Compact trace sidecar lacks block_metrics")
    if len(metadata["block_metrics"]["factual_blocks"]) != HORIZON:
        raise ValueError("Compact trace block_metrics must cover the five blocks")
    return arrays, metadata


def read_compact_trace(path: Path, expected_schema_version: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read an NPZ/JSON pair and reject incompatible compact schemas."""
    from on_policy import read_versioned_npz  # local import keeps module import light

    return read_versioned_npz(path, expected_schema_version)


def write_compact_trace(path: Path, arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> Path:
    """Atomically persist a compact trace and its versioned sidecar."""
    write_npz_with_metadata(path, arrays, metadata)
    return path


def find_case_environment(
    episode_ids: np.ndarray, start_steps: np.ndarray, case: tuple[int, int]
) -> int:
    """Return the environment index of a (episode, start) case, or raise."""
    matches = np.flatnonzero(
        (np.asarray(episode_ids) == case[0]) & (np.asarray(start_steps) == case[1])
    )
    if len(matches) != 1:
        raise ValueError(f"Case {case} not uniquely present: {len(matches)} matches")
    return int(matches[0])


def markdown_links_valid(docs_dir: Path) -> list[str]:
    """Return broken local Markdown links (file or directory targets)."""
    broken: list[str] = []
    for markdown in sorted(docs_dir.rglob("*.md")):
        for line_number, line in enumerate(markdown.read_text().splitlines(), start=1):
            for target in _markdown_link_targets(line):
                if _is_remote(target):
                    continue
                path = (markdown.parent / target).resolve()
                if not path.exists():
                    broken.append(f"{markdown.relative_to(docs_dir)}:{line_number} -> {target}")
    return broken


def _markdown_link_targets(line: str) -> Iterable[str]:
    import re

    pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    for match in pattern.finditer(line):
        yield match.group(1).split("#", 1)[0]


def _is_remote(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:"))


def record_machine_state() -> dict[str, Any]:
    """Snapshot host and environment facts for the run context file."""
    facts = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda"] = torch.version.cuda
        facts["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:  # pragma: no cover
        facts["torch"] = "unavailable"
    return facts
