#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/_env.sh"

# Keep stable-pretraining logs and resumable Lightning checkpoints beside the
# dataset/checkpoints chosen for this laboratory run.
export SPT_CACHE_DIR="$STABLEWM_HOME/training"

cd "$LAB_ROOT/third_party/le-wm"
uv run --project "$LAB_ROOT" python train.py --config-name=lewm_phase4_reference.yaml
