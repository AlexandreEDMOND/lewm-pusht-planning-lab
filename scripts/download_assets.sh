#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/_env.sh"

usage() {
  echo "Usage: bash scripts/download_assets.sh {checkpoint|dataset|all}" >&2
  exit 2
}

case "${1:-}" in
  checkpoint)
    download_checkpoint=1
    download_dataset=0
    ;;
  dataset)
    download_checkpoint=0
    download_dataset=1
    ;;
  all)
    download_checkpoint=1
    download_dataset=1
    ;;
  *) usage ;;
esac

mkdir -p "$STABLEWM_HOME"

if [[ "$download_checkpoint" == "1" ]]; then
  uv run --project "$LAB_ROOT" hf download quentinll/lewm-pusht config.json weights.pt \
    --local-dir "$STABLEWM_HOME/hf_pusht"
  uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/convert_checkpoint.py" \
    --cache-dir "$STABLEWM_HOME"
fi

if [[ "$download_dataset" == "1" ]]; then
  command -v zstd >/dev/null || {
    echo "zstd is required to decompress the PushT dataset." >&2
    exit 1
  }
  uv run --project "$LAB_ROOT" hf download quentinll/lewm-pusht pusht_expert_train.h5.zst \
    --repo-type dataset --local-dir "$STABLEWM_HOME"
  zstd --decompress --keep --force "$STABLEWM_HOME/pusht_expert_train.h5.zst"
fi
