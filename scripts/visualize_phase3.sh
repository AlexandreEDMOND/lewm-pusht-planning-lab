#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/_env.sh"

trace="${1:-$STABLEWM_HOME/pusht/cem_traces/decision_0000.npz}"
rollout="${2:-$STABLEWM_HOME/pusht/rollout_0.mp4}"
output="${3:-$STABLEWM_HOME/pusht/phase3_cem_decision_0000_env_0.mp4}"

uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/render_cem_trace.py" \
  --trace "$trace" \
  --rollout "$rollout" \
  --output "$output" \
  --environment 0
