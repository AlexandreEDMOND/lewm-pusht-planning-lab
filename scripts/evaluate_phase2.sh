#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/_env.sh"

cd "$LAB_ROOT/third_party/le-wm"
uv run --project "$LAB_ROOT" python eval.py --config-name=pusht_phase2.yaml
