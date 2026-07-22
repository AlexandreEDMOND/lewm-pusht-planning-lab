#!/usr/bin/env bash

set -euo pipefail

LAB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_ENV="$LAB_ROOT/config/local.env"

if [[ ! -f "$LOCAL_ENV" ]]; then
  echo "Missing $LOCAL_ENV. Copy config/local.env.example first." >&2
  exit 1
fi

set -a
source "$LOCAL_ENV"
set +a

if [[ -z "${STABLEWM_HOME:-}" ]]; then
  echo "STABLEWM_HOME must be set in $LOCAL_ENV." >&2
  exit 1
fi

if [[ "$STABLEWM_HOME" != /* ]]; then
  STABLEWM_HOME="$LAB_ROOT/$STABLEWM_HOME"
fi

export STABLEWM_HOME
export LOCAL_DATASET_DIR="$STABLEWM_HOME"
