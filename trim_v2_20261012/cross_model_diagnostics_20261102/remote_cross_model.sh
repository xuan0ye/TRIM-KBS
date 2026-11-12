#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p "$HERE/logs" "$HERE/../logs"
ROOT="${CROSS_DIAG_ROOT:-$(cd "$HERE/../.." && pwd)}"
if [[ -n "${CROSS_DIAG_PYTHON:-}" ]]; then PY="$CROSS_DIAG_PYTHON"
elif [[ -x "$HERE/../audio_event_remote_20261024/.venv-events/bin/python" ]]; then PY="$HERE/../audio_event_remote_20261024/.venv-events/bin/python"
else PY="$(command -v python)"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
read -r -a DATASETS <<< "${CROSS_DIAG_DATASETS:-MOSI SIMS}"
OUT="${CROSS_DIAG_OUT:-$HERE/runs/diagnostic}"
MODE="${1:-status}"
case "$MODE" in
 setup)
  "$PY" - <<'PY'
import importlib.util,subprocess,sys
from pathlib import Path
missing=[]
for name,package in [('yaml','PyYAML==6.0.2'),('einops','einops==0.8.1')]:
 if importlib.util.find_spec(name) is None:
  wheel=Path('wheels/einops-0.8.1-py3-none-any.whl')
  missing.append(str(wheel.resolve()) if name=='einops' and wheel.is_file() else package)
if missing:subprocess.run([sys.executable,'-m','pip','install','--no-deps',*missing],check=True)
import torch,transformers,numpy,yaml,einops
print('DIAG_PYTHON',sys.executable)
print('DIAG_VERSIONS',torch.__version__,transformers.__version__,numpy.__version__,yaml.__version__,einops.__version__)
print('DIAG_SETUP_COMPLETE')
PY
  ;;
 check) "$PY" -u preflight.py --root "$ROOT" --datasets "${DATASETS[@]}" ;;
 smoke)
  "$PY" -u preflight.py --root "$ROOT" --smoke
  exec 9>"$HERE/../logs/suite.lock"
  flock -n 9 || { echo 'Another TRIM suite holds the experiment lock.'; exit 2; }
  "$PY" -u run_suite.py --root "$ROOT" --smoke 2>&1 | tee "$HERE/logs/smoke.log"
  ;;
 start)
  if [[ -f "$HERE/logs/run.pid" ]]; then
   OLD_PID="$(cat "$HERE/logs/run.pid")"
   if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null && [[ -r "/proc/$OLD_PID/cmdline" ]]; then
    COMMAND_LINE="$(tr '\0' ' ' < "/proc/$OLD_PID/cmdline")"
    if [[ "$COMMAND_LINE" == *"$HERE/remote_cross_model.sh run"* ]]; then echo "Already running PID=$OLD_PID"; exit 0; fi
   fi
  fi
  "$PY" -u preflight.py --root "$ROOT" --datasets "${DATASETS[@]}"
  rm -f "$HERE/logs/run.exitcode"
  nohup bash "$HERE/remote_cross_model.sh" run >> "$HERE/logs/run.log" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$HERE/logs/run.pid"
  echo "Started PID=$PID. Check: bash remote_cross_model.sh status"
  ;;
 run)
  trap 'CODE=$?; echo "$CODE" > "$HERE/logs/run.exitcode"; echo "CROSS_MODEL_EXIT_CODE=$CODE mode=run"' EXIT
  exec 9>"$HERE/../logs/suite.lock"
  flock -n 9 || { echo 'Another TRIM suite holds the experiment lock.'; exit 2; }
  "$PY" -u run_suite.py --root "$ROOT" --out "$OUT" --datasets "${DATASETS[@]}"
  ;;
 status)
  if [[ -f "$HERE/logs/run.pid" ]]; then ps -p "$(cat "$HERE/logs/run.pid")" -o pid,etime,stat,args || true; fi
  if [[ -f "$HERE/logs/run.exitcode" ]]; then echo "EXIT_CODE=$(cat "$HERE/logs/run.exitcode")"; fi
  if [[ -f "$HERE/logs/run.log" ]]; then tail -n 35 "$HERE/logs/run.log"; else echo 'No run log yet. Run setup, smoke, then start.'; fi
  ;;
 *) echo 'Usage: bash remote_cross_model.sh {setup|check|smoke|start|run|status}'; exit 2 ;;
esac