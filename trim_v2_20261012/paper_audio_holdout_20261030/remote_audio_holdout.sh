#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-status}"
cd "$HERE"
mkdir -p logs "$HERE/../logs"
if [[ "$MODE" == start ]]; then
  if [[ -f logs/run.pid ]] && kill -0 "$(cat logs/run.pid)" 2>/dev/null; then echo "Already running. Use: bash remote_audio_holdout.sh status";exit 0;fi
  rm -f logs/run.exitcode
  if [[ -f logs/run.log ]]; then mv -- logs/run.log "logs/run.$(date +%Y%m%d_%H%M%S).$$.previous.log";fi
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  nohup bash "$HERE/remote_audio_holdout.sh" run >> logs/run.log 2>&1 < /dev/null &
  printf '%s\n' "$!" > logs/run.pid
  echo "Started PID=$(cat logs/run.pid). Check: bash remote_audio_holdout.sh status"
  exit 0
fi
if [[ "$MODE" == status ]]; then
  if [[ -f logs/run.pid ]]; then ps -p "$(cat logs/run.pid)" -o pid,etime,stat,args || true;fi
  if [[ -f logs/run.exitcode ]]; then echo "EXIT_CODE=$(cat logs/run.exitcode)";fi
  if [[ -f logs/run.log ]]; then tail -n 35 logs/run.log;else echo "No run log yet.";fi
  exit 0
fi
if [[ "$MODE" != run ]]; then echo "Usage: bash remote_audio_holdout.sh start|status";exit 2;fi
finish() { code=$?;printf '%s\n' "$code" > logs/run.exitcode;echo "AUDIO_HOLDOUT_EXIT_CODE=$code"; }
trap finish EXIT
exec 9>"$HERE/../logs/suite.lock"
flock -n 9 || { echo "Another TRIM suite is running.";exit 2; }
EVENT="${PAPER_EVENT_DIR:-$HERE/../audio_event_remote_20261024}"
PY="$EVENT/.venv-events/bin/python"
if [[ ! -x "$PY" ]]; then echo "Existing event environment missing: $PY";exit 2;fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 MPLBACKEND=Agg
"$PY" -u run_holdout.py --device cuda:0
