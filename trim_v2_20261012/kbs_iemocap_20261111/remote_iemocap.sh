#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p "$HERE/logs" "$HERE/runs" "$HERE/../logs"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$(command -v python)"
if [[ -z "$(printenv CUDA_VISIBLE_DEVICES || true)" ]]; then export CUDA_VISIBLE_DEVICES=0; fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
OUT="$HERE/runs/extension"
MODE="${1:-status}"
case "$MODE" in
 check|prepare|smoke|train|evaluate|summarize)
  "$PY" -u run.py "$MODE" --root "$ROOT" --out "$OUT"
  ;;
 start)
  if [[ -f "$HERE/logs/run.pid" ]]; then
   OLD="$(cat "$HERE/logs/run.pid")"
   if [[ "$OLD" =~ ^[0-9]+$ ]] && kill -0 "$OLD" 2>/dev/null && [[ -r "/proc/$OLD/cmdline" ]]; then
    CMDLINE="$(tr '\0' ' ' < "/proc/$OLD/cmdline")"
    if [[ "$CMDLINE" == *"$HERE/remote_iemocap.sh run"* ]]; then echo "Already running PID=$OLD"; exit 0; fi
   fi
  fi
  "$PY" -u run.py check --quick --root "$ROOT" --out "$OUT"
  if [[ -f "$HERE/logs/run.exitcode" ]]; then mv "$HERE/logs/run.exitcode" "$HERE/logs/previous.exitcode"; fi
  nohup bash "$HERE/remote_iemocap.sh" run >> "$HERE/logs/run.log" 2>&1 < /dev/null &
  PID="$!"; echo "$PID" > "$HERE/logs/run.pid"
  echo "Started PID=$PID. Check: bash remote_iemocap.sh status"
  ;;
 run)
  trap 'CODE=$?; echo "$CODE" > "$HERE/logs/run.exitcode"; echo "IEMOCAP_EXIT_CODE=$CODE mode=run"' EXIT
  "$PY" -u run.py run --root "$ROOT" --out "$OUT"
  ;;
 status)
  if [[ -f "$HERE/logs/run.pid" ]]; then ps -p "$(cat "$HERE/logs/run.pid")" -o pid,etime,stat,args || true; fi
  if [[ -f "$HERE/logs/run.exitcode" ]]; then echo "EXIT_CODE=$(cat "$HERE/logs/run.exitcode")"; fi
  if [[ -f "$OUT/suite_progress.json" ]]; then cat "$OUT/suite_progress.json"; echo; fi
  if [[ -f "$HERE/logs/run.log" ]]; then tail -n 30 "$HERE/logs/run.log"; else echo 'No log yet. Run: bash remote_iemocap.sh start'; fi
  ;;
 *) echo 'Usage: bash remote_iemocap.sh {check|prepare|smoke|start|run|train|evaluate|summarize|status}'; exit 2 ;;
esac
