#!/usr/bin/env bash
# Measure one deeper CEM search from a fixed successful PushT state.
#
# This is a planning benchmark, not a population-success evaluation: it runs
# one MPC decision with 300 candidates for 60 CEM iterations and records a
# synchronized duration for every iteration.

set -euo pipefail

source "$(dirname "$0")/_env.sh"

LEWM_ROOT="$LAB_ROOT/third_party/le-wm"
BENCH_ROOT="$STABLEWM_HOME/pusht/cem_convergence_benchmark"

uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/check_phase0.py" --require-cuda --require-assets
mkdir -p "$BENCH_ROOT/raw" "$BENCH_ROOT/traces" "$BENCH_ROOT/plans"

(
  cd "$LEWM_ROOT"
  uv run --project "$LAB_ROOT" python eval.py \
    --config-name=pusht_on_policy_cem.yaml \
    "eval.num_eval=1" \
    "eval.fixed_episode_ids=[3876]" \
    "eval.fixed_start_steps=[16]" \
    "eval.eval_budget=25" \
    "solver.batch_size=1" \
    "solver.num_samples=300" \
    "solver.n_steps=60" \
    "solver.topk=30" \
    "solver.trace_enabled=true" \
    "solver.trace_dir=$BENCH_ROOT/traces" \
    "solver.selected_plan_dir=$BENCH_ROOT/plans" \
    "on_policy_artifact_dir=$BENCH_ROOT/raw" \
    "on_policy_artifact_filename=execution.npz" \
    "output.filename=cem_convergence_benchmark/results.txt" \
    "output.metrics_filename=cem_convergence_benchmark/metrics.json"
)

uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/cem_convergence_benchmark.py" \
  --raw-root "$BENCH_ROOT"
