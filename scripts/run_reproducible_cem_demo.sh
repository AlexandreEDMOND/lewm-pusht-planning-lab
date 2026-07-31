#!/usr/bin/env bash
# Reproducible end-to-end CEM demo on the two pre-registered PushT episodes.
#
# Requirements: clean main repository and clean LeWM submodule, CUDA GPU,
# assets under STABLEWM_HOME (dataset, official checkpoint, decoders).
#
# The command:
#   1. verifies the environment and assets (check_phase0);
#   2. refuses dirty provenance;
#   3. records the resolved configuration and machine state;
#   4. runs the official CEM protocol on episodes (3876, start 16) and
#      (1766, start 2) with complete population traces enabled;
#   5. post-processes into compact traces, metrics, manifest, GIFs and PNG;
#   6. runs the programmatic validation and exits non-zero on any failure.
#
# Heavy artifacts stay under $STABLEWM_HOME/pusht/reproducible_cem_demo and the
# command never writes into the existing on-policy directories.

set -euo pipefail

source "$(dirname "$0")/_env.sh"

LEWM_ROOT="$LAB_ROOT/third_party/le-wm"
DEMO_ROOT="$STABLEWM_HOME/pusht/reproducible_cem_demo"

echo "[1/6] Vérification de l'environnement et des assets"
uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/check_phase0.py" --require-cuda --require-assets

echo "[2/6] Refus d'une provenance dirty (dépôt principal et sous-module)"
uv run --project "$LAB_ROOT" python - "$LAB_ROOT" "$LEWM_ROOT" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
sys.path.insert(0, str(Path(sys.argv[1]) / "third_party" / "le-wm"))
from cem_demo import git_provenance, record_machine_state
provenance = git_provenance(Path(sys.argv[1]), Path(sys.argv[2]), strict=True)
print(f"lab={provenance['lab_commit']} lewm={provenance['lewm_commit']}")
PY

mkdir -p "$DEMO_ROOT/raw" "$DEMO_ROOT/traces" "$DEMO_ROOT/plans"

echo "[3/6] Enregistrement de la configuration résolue et du matériel"
uv run --project "$LAB_ROOT" python - "$DEMO_ROOT" "$LAB_ROOT" "$LEWM_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
sys.path.insert(0, str(Path(sys.argv[2]) / "third_party" / "le-wm"))
from cem_demo import DEMO_CASES, EXPECTED_CHECKPOINT_SHA256, git_provenance, record_machine_state
context = {
    "demo_cases": [{"episode": episode, "start_step": start} for episode, start in DEMO_CASES],
    "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    "seed": 42,
    "population": 300,
    "iterations": 30,
    "elites": 30,
    "horizon": 5,
    "action_block": 5,
    "receding_horizon": 5,
    "goal_offset_steps": 25,
    "budget_actions": 50,
    "provenance": git_provenance(Path(sys.argv[2]), Path(sys.argv[3]), strict=True),
    "machine": record_machine_state(),
    "stablewm_home": os.environ["STABLEWM_HOME"],
}
out = Path(sys.argv[1]) / "run_context.json"
out.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n")
print(f"run_context.json écrit ({out.stat().st_size} octets)")
PY

echo "[4/6] Exécution du protocole CEM officiel sur les deux épisodes fixés"
(
  cd "$LEWM_ROOT"
  uv run --project "$LAB_ROOT" python eval.py \
    --config-name=pusht_on_policy_cem.yaml \
    "eval.num_eval=2" \
    "eval.fixed_episode_ids=[3876,1766]" \
    "eval.fixed_start_steps=[16,2]" \
    "solver.trace_enabled=true" \
    "solver.trace_dir=$DEMO_ROOT/traces" \
    "solver.selected_plan_dir=$DEMO_ROOT/plans" \
    "on_policy_artifact_dir=$DEMO_ROOT/raw" \
    "on_policy_artifact_filename=execution.npz" \
    "output.filename=reproducible_cem_demo/cem_demo_results.txt" \
    "output.metrics_filename=reproducible_cem_demo/cem_demo_metrics.json"
)

echo "[5/6] Post-traitement : traces compactes, métriques, manifeste, animations"
uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/postprocess_cem_demo.py" \
  --raw-root "$DEMO_ROOT" \
  --results-dir "$LAB_ROOT/docs/results" \
  --assets-dir "$LAB_ROOT/docs/assets" \
  --lab-root "$LAB_ROOT" \
  --lewm-root "$LEWM_ROOT"

echo "[6/6] Validation programmatique"
uv run --project "$LAB_ROOT" python "$LAB_ROOT/scripts/validate_cem_demo.py" \
  --lab-root "$LAB_ROOT" \
  --lewm-root "$LEWM_ROOT" \
  --results-dir "$LAB_ROOT/docs/results" \
  --assets-dir "$LAB_ROOT/docs/assets" \
  --raw-root "$DEMO_ROOT"

echo "Démo CEM reproductible terminée : $DEMO_ROOT"
