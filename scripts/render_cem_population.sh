#!/usr/bin/env bash
# Regenerate the two README GIFs showing physical CEM candidate trajectories.

set -euo pipefail

source "$(dirname "$0")/_env.sh"

raw_root="$STABLEWM_HOME/pusht/reproducible_cem_demo"
renderer="$LAB_ROOT/scripts/render_cem_population.py"

# The recorded vectorized run orders failure (env 0) then success (env 1).
uv run --project "$LAB_ROOT" python "$renderer" \
  --raw-root "$raw_root" --environment 1 --decision 0 \
  --output "$LAB_ROOT/docs/assets/cem_population_success.gif"
uv run --project "$LAB_ROOT" python "$renderer" \
  --raw-root "$raw_root" --environment 0 --decision 0 \
  --output "$LAB_ROOT/research/assets/cem_population_failure.gif"
