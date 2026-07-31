#!/usr/bin/env python3
"""Programmatic final validation of the reproducible CEM demo artifacts.

Checks (all section-10 controls):
- exactly two episodes with the pre-registered (episode, start) pairs;
- exactly two decisions per episode with action offsets {0, 25} verified from
  the raw execution arrays (not from the display);
- 30 CEM iterations per decision, population 300, 30 elites;
- exact match between the selected plans and the executed actions;
- no NaN/Inf in compact traces and published CSV;
- compact traces under 10 MB each and 20 MB total;
- SHA-256 recomputation of every published artifact against the manifest;
- portable paths (no absolute machine path);
- no missing versioned artifact; valid Markdown links;
- clean provenance and ``git diff --check`` in both repositories.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party" / "le-wm"))

from cem_demo import (  # noqa: E402
    ACTION_BLOCK,
    COMPACT_TRACE_SCHEMA_VERSION,
    DEMO_CASES,
    ELITE_COUNT,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_DECISION_OFFSETS,
    ITERATIONS,
    POPULATION,
    assert_portable,
    git_provenance,
    markdown_links_valid,
    plan_matches_executed,
    sha256_file,
    validate_compact_trace,
)
from on_policy import read_versioned_npz  # noqa: E402

MAX_COMPACT_TRACE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_COMPACT_BYTES = 20 * 1024 * 1024
VERSIONED_SUFFIXES = (".csv", ".json", ".png", ".gif", ".npz")


class ValidationError(RuntimeError):
    pass


def check(condition: bool, label: str, failures: list[str], detail: str = "") -> None:
    if not condition:
        failures.append(f"{label}: {detail}")


def validate_finite_arrays(compact_root: Path, failures: list[str]) -> None:
    for path in sorted(compact_root.glob("compact_trace_*.npz")):
        arrays, _ = validate_compact_trace(path, COMPACT_TRACE_SCHEMA_VERSION)
        counts = arrays["kept_candidate_counts"]
        for key, value in arrays.items():
            if not np.issubdtype(value.dtype, np.floating):
                continue
            if key.startswith("kept_"):
                # Columns beyond the per-iteration kept count are documented NaN
                # padding; only the valid region must be finite.
                bad = 0
                for iteration in range(len(counts)):
                    count = int(counts[iteration])
                    valid = value[iteration, :count]
                    bad += int(np.isnan(valid).sum()) + int(np.isinf(valid).sum())
            else:
                bad = int(np.isnan(value).sum()) + int(np.isinf(value).sum())
            if bad:
                failures.append(f"{path.name}[{key}]: {bad} NaN/Inf values")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-root", type=Path, default=ROOT)
    parser.add_argument("--lewm-root", type=Path, default=ROOT / "third_party" / "le-wm")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "docs" / "results")
    parser.add_argument("--assets-dir", type=Path, default=ROOT / "docs" / "assets")
    parser.add_argument("--raw-root", type=Path, default=None)
    args = parser.parse_args()
    failures: list[str] = []

    manifest_path = args.results_dir / "cem_demo_manifest.json"
    metrics_path = args.results_dir / "cem_demo_episode_metrics.csv"
    check(manifest_path.is_file(), "manifest", failures, "missing cem_demo_manifest.json")
    check(metrics_path.is_file(), "metrics", failures, "missing cem_demo_episode_metrics.csv")
    if not manifest_path.is_file():
        print("\n".join(f"FAIL {entry}" for entry in failures) or "OK")
        return 1
    manifest = json.loads(manifest_path.read_text())
    check(
        manifest.get("protocol", {}).get("checkpoint_sha256") == EXPECTED_CHECKPOINT_SHA256,
        "checkpoint", failures, "manifest checkpoint SHA-256 differs from the official one",
    )

    # -- episodes and decisions from the published records
    with metrics_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    check(len(rows) == 2, "episodes", failures, f"expected 2 episodes, got {len(rows)}")
    if len(rows) == 2:
        pairs = [(int(row["episode"]), int(row["start_step"])) for row in rows]
        check(set(pairs) == set(DEMO_CASES), "cases", failures, f"got {pairs}")
        for row in rows:
            check(int(row["decisions"]) == 2, "decisions", failures, f"episode {row['episode']}")
            check(
                [int(value) for value in json.loads(row["decision_action_offsets"])]
                == EXPECTED_DECISION_OFFSETS,
                "offsets", failures, f"episode {row['episode']}",
            )
            check(
                row["action_plan_exact_match"].lower() == "true",
                "plan_actions", failures, f"episode {row['episode']}",
            )

    # -- raw execution: offsets and plan/action match, from data, not display
    compact_root = args.results_dir / "cem_demo_compact"
    compact_files = sorted(compact_root.glob("compact_trace_*.npz"))
    check(len(compact_files) == 4, "compact_coverage", failures, f"expected 4 compact traces, got {len(compact_files)}")
    for compact_path in compact_files:
        arrays, metadata = validate_compact_trace(compact_path, COMPACT_TRACE_SCHEMA_VERSION)
        protocol = metadata["protocol"]
        check(int(protocol["iterations"]) == ITERATIONS, "iterations", failures, compact_path.name)
        check(int(protocol["population"]) == POPULATION, "population", failures, compact_path.name)
        check(int(protocol["elites"]) == ELITE_COUNT, "elites", failures, compact_path.name)
        check(arrays["elite_costs"].shape == (ITERATIONS, ELITE_COUNT), "elite_shape", failures, compact_path.name)
        check(
            int(metadata["decision_index"]) in (0, 1) and metadata["environment_index"] in (0, 1),
            "decision_env", failures, compact_path.name,
        )
    # plan <-> executed actions: the compact trace carries the executed actions
    # and the final plan; replay the exact-match check on every decision window.
    for compact_path in compact_files:
        arrays, metadata = validate_compact_trace(compact_path, COMPACT_TRACE_SCHEMA_VERSION)
        decision_index = int(metadata["decision_index"])
        offset = int(metadata["decision_action_offsets"][decision_index])
        plan = arrays["final_plan_actions"].reshape(-1, 2)
        executed = arrays["executed_actions_normalized"][offset : offset + 25]
        match = plan_matches_executed(plan, executed, ACTION_BLOCK, 2)
        check(match, "plan_executed_match", failures, compact_path.name)

    # -- no NaN/Inf in compact traces and published CSV
    validate_finite_arrays(compact_root, failures)
    for row in rows:
        for key, value in row.items():
            if key in ("compact_trace_decision_0_sha256", "compact_trace_decision_1_sha256"):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                failures.append(f"episode {row['episode']}: non-finite CSV value {key}={value}")

    # -- sizes
    total = sum(path.stat().st_size for path in compact_files)
    for path in compact_files:
        check(path.stat().st_size <= MAX_COMPACT_TRACE_BYTES, "compact_size", failures, f"{path.name}: {path.stat().st_size}")
    check(total <= MAX_TOTAL_COMPACT_BYTES, "compact_total", failures, f"{total} bytes")

    # -- hashes and missing artifacts
    versioned_entries = {entry["path"]: entry for entry in manifest["artifacts"]["versioned"]}
    for path, entry in versioned_entries.items():
        full = (args.lab_root / path).resolve()
        check(full.is_file(), "artifact_missing", failures, path)
        if full.is_file():
            check(entry["bytes"] == full.stat().st_size, "artifact_bytes", failures, path)
            check(sha256_file(full) == entry["sha256"], "artifact_sha256", failures, path)
    manifest_hash = sha256_file(manifest_path)
    print(f"manifest sha256: {manifest_hash}")

    # -- portability
    try:
        assert_portable(manifest, "manifest")
        assert_portable(rows, "episode metrics CSV")
    except ValueError as error:
        failures.append(str(error))

    # -- markdown links
    for broken in markdown_links_valid(args.lab_root / "docs", args.lab_root / "README.md"):
        failures.append(f"markdown link: {broken}")

    # -- provenance and git hygiene
    try:
        git_provenance(args.lab_root, args.lewm_root, strict=False)
    except RuntimeError as error:
        failures.append(f"provenance: {error}")
    recorded_commits = {
        "lab": [manifest["provenance"]["evaluation_commit"], manifest["provenance"]["postprocessing_commit"]],
        "lewm": [manifest["provenance"]["evaluation_lewm_commit"]],
    }
    for label, root in (("lab", args.lab_root), ("lewm", args.lewm_root)):
        for commit in recorded_commits[label]:
            result = subprocess.run(
                ["git", "-C", str(root), "cat-file", "-e", f"{commit}^{{commit}}"],
                capture_output=True, text=True,
            )
            check(result.returncode == 0, f"recorded_commit_{label}", failures, commit)
        result = subprocess.run(
            ["git", "-C", str(root), "diff", "--check"], capture_output=True, text=True
        )
        check(result.returncode == 0, f"git_diff_check_{label}", failures, result.stderr.strip() or result.stdout.strip())

    if failures:
        print("\n".join(f"FAIL: {entry}" for entry in failures))
        return 1
    print(
        f"OK: {len(rows)} episodes, {len(compact_files)} compact traces "
        f"({total} bytes total), {len(versioned_entries)} versioned artifacts verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
