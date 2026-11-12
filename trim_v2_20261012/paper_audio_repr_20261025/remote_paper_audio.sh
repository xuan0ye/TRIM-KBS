#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${PAPER_PROJECT_ROOT:-$HERE/../..}"
EVENT="${PAPER_EVENT_DIR:-$HERE/../audio_event_remote_20261024}"
RAW="${PAPER_RAW_ROOT:-$ROOT/MSA Datasets/MOSEI/Raw}"
MODE="${1:-status}"
cd "$HERE"
mkdir -p logs "$HERE/../logs"
if [[ "$MODE" == start ]]; then
  if [[ -f logs/run.pid ]] && kill -0 "$(cat logs/run.pid)" 2>/dev/null; then
    echo "Already running; use: bash remote_paper_audio.sh status"
    exit 0
  fi
  rm -f logs/run.exitcode
  if [[ -f logs/run.log ]]; then
    mv -- "$HERE/logs/run.log" "$HERE/logs/run.$(date +%Y%m%d_%H%M%S).$$.previous.log"
  fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  nohup bash "$HERE/remote_paper_audio.sh" run >> "$HERE/logs/run.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > logs/run.pid
  echo "Started PID=$(cat logs/run.pid). Check: bash remote_paper_audio.sh status"
  exit 0
fi
if [[ "$MODE" == status ]]; then
  if [[ -f logs/run.pid ]]; then ps -p "$(cat logs/run.pid)" -o pid,etime,stat,args || true; fi
  if [[ -f logs/run.exitcode ]]; then echo "EXIT_CODE=$(cat logs/run.exitcode)"; fi
  if [[ -f logs/run.log ]]; then tail -n 30 logs/run.log; else echo "No run log yet."; fi
  exit 0
fi
case "$MODE" in run|smoke|audit|bundle) ;; *) echo "Usage: bash remote_paper_audio.sh start|status|smoke|audit|bundle";exit 2;; esac
exec 9>"$HERE/../logs/suite.lock"
flock -n 9 || { echo "Another TRIM suite holds the GPU experiment lock. Wait for it to finish.";exit 2; }
finish() { code=$?;printf '%s\n' "$code" > "$HERE/logs/${MODE}.exitcode";echo "PAPER_AUDIO_EXIT_CODE=$code mode=$MODE"; }
trap finish EXIT
PY="$EVENT/.venv-events/bin/python"
if [[ ! -x "$PY" ]]; then echo "Existing audio-event environment missing: $PY";exit 2;fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MPLBACKEND=Agg CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$HERE/.cache/matplotlib" NUMBA_CACHE_DIR="$HERE/.cache/numba"
if [[ "$MODE" == bundle ]]; then
  "$PY" -u paper_summarize.py --out "$HERE/runs/pilot"
  exit 0
fi
if [[ "$MODE" == audit ]]; then
  "$PY" -u paper_data.py --root "$ROOT" --raw-root "$RAW" --out "$HERE/runs/pilot" --audit-only
  exit 0
fi
run_stage() {
  stage="$1";shift
  "$PY" -u paper_data.py --root "$ROOT" --raw-root "$RAW" --out "$HERE/runs/$stage" --assets "$EVENT/assets" --device cuda:0 "$@"
  "$PY" -u paper_train.py --root "$ROOT" --out "$HERE/runs/$stage" --device cuda:0 "$@"
  "$PY" -u paper_summarize.py --out "$HERE/runs/$stage"
}
run_stage smoke --smoke
if [[ "$MODE" == run ]]; then run_stage pilot; fi
echo "PAPER_AUDIO_SUITE_COMPLETE mode=$MODE test_evaluated=False"
