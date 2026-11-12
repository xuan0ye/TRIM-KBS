#!/usr/bin/env bash
set -euo pipefail

# Full CCF-B supplement run for experiments 2/3/4.
#
# This is the "run everything" launcher for a large-memory GPU server.
# It delegates to remote_run_ccfb_experiments_2_4.sh with full defaults:
#   - all 3 datasets: MOSI, SIMS, MOSEI
#   - all E2E binary ablations
#   - 5 training seeds by default
#   - full fixed-mask combo evaluation, including text-missing and all-missing
#   - random A/V missing and random T/A/V missing for p=0.1,0.3,0.5,0.7
#
# Usage on remote:
#   cd ~/wjh/VWB
#   conda activate wjhenv
#   bash remote_run_ccfb_experiments_2_4_full.sh
#
# Resume behavior:
#   SKIP_EXISTING=1 by default, so interrupted runs can be restarted safely.

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MSA_DATA_ROOT="${MSA_DATA_ROOT:-$PROJECT_DIR/MSA Datasets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Full experiment scope.
export DATASETS="${DATASETS:-MOSI SIMS MOSEI}"
export ABLATIONS="${ABLATIONS:-full no_uni_ctx no_gcm no_mm_ctx no_summary_token no_contrast no_ot no_ama no_missing_repr no_modrop no_aux text_only}"

# Five seeds gives stronger evidence than the 3-seed quick run.
export SEEDS_SPACE="${SEEDS_SPACE:-13 21 42 57 88}"
export MASK_SEEDS="${MASK_SEEDS:-13,21,42,57,73,88,101,123,202,340}"
export MISSING_PS="${MISSING_PS:-0.1 0.3 0.5 0.7}"

# Large-GPU defaults. Adjust down if your job OOMs.
export BATCH="${BATCH:-16}"
export ACC_STEPS="${ACC_STEPS:-2}"
export PHASE1_EP="${PHASE1_EP:-5}"
export PHASE2_EP="${PHASE2_EP:-8}"
export PHASE3_EP_DEFAULT="${PHASE3_EP_DEFAULT:-7}"
export UNFREEZE_N="${UNFREEZE_N:-2}"

export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_MASK_EVAL="${RUN_MASK_EVAL:-1}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"

export RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results_ccfb_2_4_full}"
export LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs_ccfb_2_4_full}"

echo "Launching FULL CCF-B supplement experiments 2/3/4"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "MSA_DATA_ROOT=$MSA_DATA_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "DATASETS=$DATASETS"
echo "ABLATIONS=$ABLATIONS"
echo "SEEDS_SPACE=$SEEDS_SPACE"
echo "BATCH=$BATCH"
echo "ACC_STEPS=$ACC_STEPS"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo

cd "$PROJECT_DIR"
bash remote_run_ccfb_experiments_2_4.sh
