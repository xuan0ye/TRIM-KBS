# KBS TRIM source snapshot

This repository contains implementation and experiment source for the TRIM
manuscript, *Missing-Modality Sentiment Classification with Explicit
Modality-Use Evaluation*. The original VWB entry points are described below.

| Experiment | Source directory and entry point |
| --- | --- |
| End-to-end sentiment model and mask evaluation | `src/run_e2e_msa.py` and `src/eval_e2e_ckpt_masks.py` |
| Fixed-checkpoint removal, cross-video replacement, and frame-order replay | `trim_v2_20261012/cross_model_diagnostics_20261102/run_suite.py` |
| KBS benchmark diagnostics | `trim_v2_20261012/kbs_substantive_20261110/run.py` |
| Six-class IEMOCAP study, including the runtime compatibility fix | `trim_v2_20261012/kbs_iemocap_20261111/run.py` |
| MOSEI audio representations, probes, and fusion controls | `trim_v2_20261012/paper_audio_*` |
| Source-video-disjoint audio routing | `trim_v2_20261012/paper_audio_holdout_20261030/run_holdout.py` |

The experiment directories preserve their historical relative layout under
`trim_v2_20261012/`. The protocol documents and manifests describe the original
settings and required inputs. Inspect the IEMOCAP command-line entry point
from the repository root.

    python trim_v2_20261012/kbs_iemocap_20261111/run.py --help

Obtain benchmark data and media from their providers under the applicable
access terms. Historical replay requires the original checkpoints and
reference assets listed in its manifests, plus the upstream CorMulT and LNLN
sources. The audio experiments require parent-stage features and the pinned
pretrained weights. This snapshot provides code and configuration. The
historical bundle manifests also list supporting files needed to recreate
the original remote-run bundles.

---

# VWB

This repository contains the implementation of a multimodal sentiment analysis
framework for text, audio, and visual inputs. The model combines a pretrained
language backbone with audio/visual projection layers, unimodal context
refinement, gated cross-modal interaction, multimodal summary tokens, missing
modality handling, contrastive alignment, Sinkhorn optimal transport, and
adaptive multi-loss aggregation.

The code is intended for research reproducibility on MOSI, MOSEI, and CH-SIMS
under a unified binary sentiment classification setting.

## Repository Contents

- `src/run_e2e_msa.py`: end-to-end training script for the proposed model.
- `src/eval_e2e_ckpt_masks.py`: missing-modality evaluation for saved checkpoints.
- `src/models/`: reusable model components.
- `remote_run_ccfb_experiments_2_4.sh`: launcher for the supplemental ablation and missing-modality experiments.
- `remote_run_ccfb_experiments_2_4_full.sh`: full multi-seed launcher.

This repository intentionally does not include datasets, generated result files,
training logs, paper drafts, or intermediate checkpoints.

## Installation

```bash
conda create -n vwb python=3.10
conda activate vwb
pip install -r requirements.txt
```

## Data

Prepare the processed MOSI, MOSEI, and CH-SIMS files in the following layout:

```text
MSA Datasets/
  MOSI/Processed/aligned_50.pkl
  MOSEI/Processed/aligned_50.pkl
  SIMS/Processed/unaligned_39.pkl
```

Then set:

```bash
export MSA_DATA_ROOT="/path/to/MSA Datasets"
```

## Training

Run a single end-to-end experiment:

```bash
python src/run_e2e_msa.py \
  --dataset MOSI \
  --bert_model bert-base-uncased \
  --data_root "$MSA_DATA_ROOT" \
  --out_dir results_e2e_mosi
```

Run the full supplemental ablation and missing-modality suite:

```bash
bash remote_run_ccfb_experiments_2_4_full.sh
```

## Checkpoints

Training scripts save model checkpoints to the selected output directory. Large
trained checkpoints are not stored directly in this Git repository because they
can exceed GitHub's recommended file-size limits. If released, checkpoints should
be attached as GitHub Releases or hosted on a model hub and linked from this
README.

## Citation

If you use this code, please cite the corresponding paper.