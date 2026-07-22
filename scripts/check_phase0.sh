#!/usr/bin/env bash

set -euo pipefail

source "$(dirname "$0")/_env.sh"
uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/check_phase0.py" "$@"
