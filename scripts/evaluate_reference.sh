#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/_env.sh"

seed="${1:-42}"
num_eval="${2:-5}"

if [[ ! -f "$STABLEWM_HOME/pusht/lewm_object.ckpt" ]]; then
  echo "Missing evaluation checkpoint. Run: bash scripts/download_assets.sh checkpoint" >&2
  exit 1
fi

if [[ ! -f "$STABLEWM_HOME/pusht_expert_train.h5" ]]; then
  echo "Missing PushT dataset. Run: bash scripts/download_assets.sh dataset" >&2
  exit 1
fi

cd "$LAB_ROOT/third_party/le-wm"
uv run --project "$LAB_ROOT" python eval.py --config-name=pusht.yaml \
  policy=pusht/lewm seed="$seed" eval.num_eval="$num_eval"
