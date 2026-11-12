#!/usr/bin/env bash
set -euo pipefail

# CCF-B supplemental experiments 2/3/4:
# 2) unified binary component ablation
# 3) BERT-base full model vs key mechanism ablations
# 4) text-missing, all-combo, and all-missing missing-modality evaluation
#
# Remote server usage:
#   cd ~/wjh/VWB
#   conda activate wjhenv
#   bash remote_run_ccfb_experiments_2_4.sh
#
# A smaller core run can be launched with:
#   ABLATIONS="full no_gcm no_ot no_missing_repr no_summary_token text_only" bash remote_run_ccfb_experiments_2_4.sh

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
MSA_DATA_ROOT="${MSA_DATA_ROOT:-$PROJECT_DIR/MSA Datasets}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS="${DATASETS:-MOSI SIMS MOSEI}"
SEEDS_SPACE="${SEEDS_SPACE:-42 52 62}"
MASK_SEEDS="${MASK_SEEDS:-13,21,42,57,73,88,101,123,202,340}"
MISSING_PS="${MISSING_PS:-0.1 0.3 0.5 0.7}"

ABLATIONS="${ABLATIONS:-full no_uni_ctx no_gcm no_mm_ctx no_summary_token no_contrast no_ot no_ama no_missing_repr no_modrop no_aux text_only}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_MASK_EVAL="${RUN_MASK_EVAL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

BATCH="${BATCH:-8}"
ACC_STEPS="${ACC_STEPS:-4}"
PHASE1_EP="${PHASE1_EP:-5}"
PHASE2_EP="${PHASE2_EP:-8}"
PHASE3_EP_DEFAULT="${PHASE3_EP_DEFAULT:-7}"
UNFREEZE_N="${UNFREEZE_N:-2}"

RESULT_ROOT="${RESULT_ROOT:-$PROJECT_DIR/results_ccfb_2_4}"
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs_ccfb_2_4}"

cd "$PROJECT_DIR"
export MSA_DATA_ROOT
export OUTPUT_ROOT="$PROJECT_DIR"
export PYTHONIOENCODING=utf-8
mkdir -p "$RESULT_ROOT" "$LOG_DIR"

echo "PROJECT_DIR=$PROJECT_DIR"
echo "MSA_DATA_ROOT=$MSA_DATA_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "DATASETS=$DATASETS"
echo "SEEDS_SPACE=$SEEDS_SPACE"
echo "ABLATIONS=$ABLATIONS"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "LOG_DIR=$LOG_DIR"

run_logged() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
  echo
  echo "======================================================================"
  echo "[START] $name"
  echo "[LOG]   $log_file"
  echo "======================================================================"
  "$@" 2>&1 | tee "$log_file"
  echo "[DONE]  $name"
}

bert_for_dataset() {
  case "$1" in
    SIMS) echo "bert-base-chinese" ;;
    *) echo "bert-base-uncased" ;;
  esac
}

phase3_for_dataset() {
  case "$1" in
    SIMS) echo "0" ;;
    *) echo "$PHASE3_EP_DEFAULT" ;;
  esac
}

ckpt_csv_for_ablation() {
  local dataset="$1"
  local ablation="$2"
  local csv=""
  for seed in $SEEDS_SPACE; do
    local ckpt="$RESULT_ROOT/train/${dataset}/${ablation}/seed_${seed}/${dataset}_e2e_best.pt"
    if [[ -z "$csv" ]]; then
      csv="$ckpt"
    else
      csv="$csv,$ckpt"
    fi
  done
  echo "$csv"
}

all_ckpts_exist_for_ablation() {
  local dataset="$1"
  local ablation="$2"
  for seed in $SEEDS_SPACE; do
    local ckpt="$RESULT_ROOT/train/${dataset}/${ablation}/seed_${seed}/${dataset}_e2e_best.pt"
    if [[ ! -f "$ckpt" ]]; then
      return 1
    fi
  done
  return 0
}

if [[ "$RUN_TRAIN" == "1" ]]; then
  for dataset in $DATASETS; do
    bert_model="$(bert_for_dataset "$dataset")"
    phase3_ep="$(phase3_for_dataset "$dataset")"
    for ablation in $ABLATIONS; do
      for seed in $SEEDS_SPACE; do
        out_dir="$RESULT_ROOT/train/${dataset}/${ablation}/seed_${seed}"
        ckpt="$out_dir/${dataset}_e2e_best.pt"
        if [[ "$SKIP_EXISTING" == "1" && -f "$ckpt" ]]; then
          echo "[SKIP] existing $ckpt"
          continue
        fi
        mkdir -p "$out_dir"
        run_logged "train_${dataset}_${ablation}_seed${seed}" \
          "$PYTHON_BIN" src/run_e2e_msa.py \
            --dataset "$dataset" \
            --bert_model "$bert_model" \
            --ablation "$ablation" \
            --seed "$seed" \
            --data_root "$MSA_DATA_ROOT" \
            --out_dir "$out_dir" \
            --batch "$BATCH" \
            --acc_steps "$ACC_STEPS" \
            --phase1_ep "$PHASE1_EP" \
            --phase2_ep "$PHASE2_EP" \
            --phase3_ep "$phase3_ep" \
            --unfreeze_n "$UNFREEZE_N"
      done
    done
  done
fi

if [[ "$RUN_MASK_EVAL" == "1" ]]; then
  for dataset in $DATASETS; do
    bert_model="$(bert_for_dataset "$dataset")"
    for ablation in $ABLATIONS; do
      if ! all_ckpts_exist_for_ablation "$dataset" "$ablation"; then
        echo "[WARN] skip mask eval for $dataset/$ablation: missing checkpoints"
        continue
      fi
      ckpts="$(ckpt_csv_for_ablation "$dataset" "$ablation")"

      run_logged "mask_combo_all_${dataset}_${ablation}" \
        "$PYTHON_BIN" src/eval_e2e_ckpt_masks.py \
          --dataset "$dataset" \
          --bert_model "$bert_model" \
          --ablation "$ablation" \
          --ckpts "$ckpts" \
          --data_root "$MSA_DATA_ROOT" \
          --eval_combos \
          --combo_set all \
          --include_empty_combo \
          --out_dir "$RESULT_ROOT/mask_eval/${dataset}/${ablation}/combos_all"

      for p in $MISSING_PS; do
        run_logged "mask_random_av_${dataset}_${ablation}_p${p}" \
          "$PYTHON_BIN" src/eval_e2e_ckpt_masks.py \
            --dataset "$dataset" \
            --bert_model "$bert_model" \
            --ablation "$ablation" \
            --ckpts "$ckpts" \
            --data_root "$MSA_DATA_ROOT" \
            --eval_missing \
            --random_modalities av \
            --missing_p "$p" \
            --seeds "$MASK_SEEDS" \
            --out_dir "$RESULT_ROOT/mask_eval/${dataset}/${ablation}/random_av_p${p}"

        run_logged "mask_random_tav_${dataset}_${ablation}_p${p}" \
          "$PYTHON_BIN" src/eval_e2e_ckpt_masks.py \
            --dataset "$dataset" \
            --bert_model "$bert_model" \
            --ablation "$ablation" \
            --ckpts "$ckpts" \
            --data_root "$MSA_DATA_ROOT" \
            --eval_missing \
            --random_modalities tav \
            --missing_p "$p" \
            --seeds "$MASK_SEEDS" \
            --out_dir "$RESULT_ROOT/mask_eval/${dataset}/${ablation}/random_tav_p${p}"
      done
    done
  done
fi

run_logged "summarize_ccfb_2_4" \
  "$PYTHON_BIN" src/summarize_ccfb_experiments.py \
    --root "$RESULT_ROOT"

echo
echo "CCF-B supplement experiments 2/3/4 finished."
echo "Main summaries:"
echo "  $RESULT_ROOT/summary/ccfb_ablation_train_summary.csv"
echo "  $RESULT_ROOT/summary/ccfb_mask_eval_summary_all.csv"
echo "Logs:"
echo "  $LOG_DIR"
