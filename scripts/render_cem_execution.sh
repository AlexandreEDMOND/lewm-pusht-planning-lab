#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_env.sh"

uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/render_cem_execution.py" \
  --raw-root "$STABLEWM_HOME/pusht/reproducible_cem_demo" \
  --environment 1 \
  --output "$LAB_ROOT/docs/assets/cem_execution_success.gif"
