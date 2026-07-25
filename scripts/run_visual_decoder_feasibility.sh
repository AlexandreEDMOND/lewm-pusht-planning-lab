#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/_env.sh"

uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/train_visual_decoder.py" \
  --config "$LAB_ROOT/config/visual_decoder_feasibility.yaml" \
  "$@"

uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/train_transformer_decoder.py" \
  --config "$LAB_ROOT/config/visual_decoder_feasibility.yaml"

uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/train_structured_decoder.py" \
  --config "$LAB_ROOT/config/visual_decoder_feasibility.yaml"

uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/evaluate_decoder_rollouts.py" \
  --config "$LAB_ROOT/config/visual_decoder_feasibility.yaml"
