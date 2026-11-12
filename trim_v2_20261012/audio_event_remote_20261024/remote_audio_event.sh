#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-audit}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${EVENT_PROJECT_ROOT:-$HERE/../..}"
RAW="${EVENT_RAW_ROOT:-$ROOT/MSA Datasets/MOSEI/Raw}"
BASE_PYTHON="${PYTHON:-python}"
cd "$HERE"
mkdir -p logs "$HERE/../logs"
case "$MODE" in setup|audit|probe|smoke|pilot|bundle) ;; *) echo "Usage: bash remote_audio_event.sh setup|audit|probe|smoke|pilot|bundle"; exit 2;; esac
exec 9>"$HERE/../logs/suite.lock"
flock -n 9 || { echo "Another TRIM suite is running. Wait for it to finish."; exit 2; }
finish() {
  code=$?
  printf '%s\n' "$code" > "$HERE/logs/${MODE}.exitcode"
  echo "AUDIO_EVENT_EXIT_CODE=$code mode=$MODE"
}
trap finish EXIT
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MPLBACKEND=Agg
export MPLCONFIGDIR="$HERE/.cache/matplotlib" NUMBA_CACHE_DIR="$HERE/.cache/numba"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY="$BASE_PYTHON"
if [[ -x "$HERE/.venv-events/bin/python" ]]; then PY="$HERE/.venv-events/bin/python"; fi
"$PY" -u event_common.py verify
if [[ "$MODE" == setup ]]; then
  "$BASE_PYTHON" -u event_setup.py
  exit 0
fi
if [[ "$MODE" == bundle ]]; then
  "$PY" -u event_summarize.py --out "$HERE/runs/pilot"
  exit 0
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [[ "$MODE" == audit ]]; then
  "$PY" -u _core/preflight_v2.py --root "$ROOT" --datasets MOSEI
  "$PY" -u event_common.py audit --root "$ROOT" --raw-root "$RAW" --out "$HERE/runs/pilot"
  echo "AUDIO_EVENT_AUDIT_COMPLETE paths_only"
  exit 0
fi
if [[ ! -x "$HERE/.venv-events/bin/python" || ! -f "$HERE/assets/Cnn14_mAP=0.431.pth" ]]; then
  echo "Run first: bash remote_audio_event.sh setup"
  exit 2
fi
OUT="$HERE/runs/$MODE"
SMOKE=()
if [[ "$MODE" == smoke ]]; then SMOKE=(--smoke); fi
PARENT=()
if [[ -n "${EVENT_PARENT:-}" ]]; then PARENT=(--parent "$EVENT_PARENT"); fi
if [[ "$MODE" == probe ]]; then
  "$PY" -u event_common.py audit --root "$ROOT" --raw-root "$RAW" --out "$OUT" --probe --sample-count 200
  "$PY" -u event_extract.py --out "$OUT" --device cuda:0
  echo "AUDIO_EVENT_PROBE_COMPLETE available_train_only test_evaluated=False"
  exit 0
fi
"$PY" -u _core/preflight_v2.py --root "$ROOT" --datasets MOSEI --require-cuda
"$PY" -u event_common.py audit --root "$ROOT" --raw-root "$RAW" --out "$OUT" "${SMOKE[@]}"
"$PY" -u event_extract.py --out "$OUT" --device cuda:0
"$PY" -u event_train.py --root "$ROOT" --out "$OUT" --device cuda:0 "${SMOKE[@]}" "${PARENT[@]}"
"$PY" -u event_summarize.py --out "$OUT"
echo "AUDIO_EVENT_SUITE_COMPLETE mode=$MODE test_evaluated=False"
