# Fresh expert training with source-video holdouts, 2026-10-30

## Primary question and scope
Can an audio probability router improve the same newly trained paper TRIM expert when the router learns on videos neither supervised expert has fitted or used for checkpoint selection? Primary condition is gated_correct; gated_shuffled, gated_zero and constant_correct are predeclared controls, not candidates to select after external evaluation.

This is a **single fixed source-video holdout pilot**, not OOF and not a confirmed method improvement. It retains the exact archived method-new.tex/model_legacy.py TRIM backbone and objective. No new test predictions are produced. The original 572-clip validation set has already been reused in earlier pilots and remains exploratory even though this run seals all selections before evaluating it. Media availability remains class biased.

## Committed partition of the original 6065 training clips
Split sorted unique source-video IDs with NumPy RNG seed 20261030, without using labels for assignment or trying alternative splits. Group boundaries at 67.5%,75%,90%,100% of 749 videos give:

| Role | Clips | Videos | Negative / positive | Permitted use |
|---|---:|---:|---:|---|
| expert_train | 4208 | 506 | 2173 / 2035 | Expert gradient updates and audio normalization |
| expert_dev | 460 | 56 | 270 / 190 | Both experts' checkpoint selection only |
| router_train | 909 | 112 | 484 / 425 | Router gradient updates only |
| router_dev | 488 | 75 | 278 / 210 | Router checkpoint selection only |

All video groups and clip IDs are mutually exclusive, and every original training row belongs to exactly one role. Source plan is committed in split_plan.json with index/video sets, class counts, and ID/label SHA-256 values. There are no post-validation exclusions or seed searches. The external validation videos remain disjoint from the full original training set.

The distinction matters: expert_dev labels participate in expert checkpoint selection, so these videos are excluded from router fitting too. Dividing old fitted prediction caches cannot make these roles unseen. Prefit same-data combination has a known overfitting risk; compare the cross-validated prediction approach described in the [official stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html).

## Data and preprocessing
Reuse only verified paper input arrays (token IDs, historical stored-frame mean COVAREP audio, visual features, labels, IDs) and **raw frozen Cnn14 2048D embeddings** from paper_audio_repr_20261025/runs/pilot. No old normalized PANNs representation, trained sentiment classifier or old multimodal checkpoint initializes an expert. Old supervised checkpoints are not read for training.

PANNs mean/std are fitted only on expert_train, using the previous float64 scaling formula and clipping normalized embeddings to [-8,8]. Reuse those statistics for all other roles and final validation. Traditional paper audio remains its exact original stored-frame mean, including padding; do not introduce another backbone variant.

All input/source hashes are checked. The raw local English BERT is reverified against the archived 199-tensor reference with the already-fixed effective config comparison. Its original pretrained tensors initialize the new expert; only new run-owned resume files can resume training. Frozen external AudioSet Cnn14 features may be reused because that sentiment classifier is being fitted anew.

## Fresh TRIM expert
Seed 42, d_model=256, 119,660,936 parameters. Call the unmodified parent paper_train.run_job with **only** expert_train mapped to its train argument and expert_dev mapped to its validation argument. Its original three phases (BERT frozen 5 epochs; last two layers 8; all layers 7), per-phase restoration, global best expert_dev Macro-F1 selection, optimizer rates/scheduler, batch 8 x accumulation 4, CE smoothing .05, class weighting, .02 InfoNCE, .02 OT, .001 legacy regularizer and independent A/V dropout .15 all remain unchanged, including known archived auxiliary quirks.

The paper trainer saves epoch-level optimizer/RNG resumes in the new trim_expert directory. It never receives router videos or original external validation rows. Completed jobs require matching signatures and artifact hashes. Do not initialize from the old all-training-data paper_original/best.pt, including as a warmstart.

## Fresh audio expert
A new random Probe-shaped classifier: Linear(2048,32), LayerNorm, GELU, dropout .1, Linear(32,2), 65,698 parameters. Seed 42. Train on expert_train only, 15 epochs, batch64, AdamW lr1e-3, weight decay1e-3, class-weighted CE smoothing .05, clip norm1; choose best expert_dev Macro-F1. No previous sentiment weights are loaded. Freeze/eval the selected audio classifier before generating router inputs.

The same newly selected experts supply cached four-mask TRIM logits and audio/standardized-zero-input logits to every router control. Experts stay frozen throughout routing. No text latent coordinates are used in the router.

## Four routers
1. gated_correct: correctly aligned full-embedding audio expert probabilities.
2. gated_shuffled: independent fixed label-free cross-video audio donor assignments in router_train/router_dev/external validation; recipient labels retained.
3. gated_zero: expert applied to standardized zero embeddings. This is no-information input, not physical silence or missing audio.
4. constant_correct: aligned audio expert with one learned global mixture weight, to distinguish a sample-specific gate from a constant mixture.

Sample gate has seven bounded features: parent/audio positive probabilities, each normalized binary entropy, absolute probability disagreement, and each confidence |2p-1|. Linear(7,16), GELU, Linear(16,1), sigmoid: **145 trainable parameters**. Last-layer weight starts at zero and bias at -2, giving initial active weight sigmoid(-2). Constant control directly optimizes one weight in [0,1], projected after each step and initialized sigmoid(-2), avoiding a slow extra sigmoid scale on this one-dimensional control. All gated controls share identical initial state, minibatch order and independent A/V mask plans; all controls use the same experts.

    w = enabled * sigmoid(gate(features)) * A_present  # constant control uses its projected weight directly
    p_mix = p_parent + w * (p_audio - p_parent)
    logits_mix = parent_logits + log(p_mix) - log(p_parent)

Zero enable switch is an exact baseline candidate, saved before optimization; active training sets this nontrainable buffer to one. This avoids a zero learned global amplitude blocking sample-gate gradients. Experts' logits are detached. Missing A makes w=0 and exactly preserves T/TV logits. Mixture log arguments are bounded below by 1e-30 for numerical stability.

Train routers for fixed 50 epochs on router_train, batch64, AdamW lr1e-3/wd1e-3, clip norm1, class weights from router_train, CE smoothing .05 plus .001*mean(w^2). Independent A/V dropout .15 chooses the corresponding cached parent mask. Select strict best **router_dev TAV Macro-F1**, including disabled baseline; ties keep the earlier checkpoint. No router hyperparameter search uses external validation. Completed small jobs can be reused; incomplete audio/router jobs restart from their fixed fresh initialization.

## Selection seal and external evaluation
Finish all expert and router fitting/selections first, then write immutable selection_seal.json listing both expert hashes and all four selected router hashes. Only then load original external validation feature arrays, generate its readouts and evaluate. Refuse if any selected file changed. Re-running unchanged completed training may replay evaluation; it cannot silently update selections.

Report original T/TA/TV/TAV expert, audio-only, and each router under four masks, zero audio input and 20 cross-video PANNs assignments. Parent text/legacy acoustic/visual inputs stay fixed in interventions. Correct minus permuted F1 only has true-alignment meaning for correct conditions; the shuffled/zero comparisons are negative controls. Return class predictions, probabilities, actual mixture weights and donors.

**Main paired baseline is the same new TRIM expert fitted on 4208 rows.** Separately show the old 6065-row parent as a historical absolute reference; its training amount and checkpoint selection differ. A gain over a weaker new baseline does not demonstrate beating the old full-training model. This pilot first tests whether expert-unseen routing produces any useful incremental signal before more expensive OOF/full-data extensions.

## Exports and interpretation
Review includes source/input/role hashes, selection seal, expert histories and internal metrics, compact all-role expert logits, small audio classifier and router weights, and audio-replay features (up to64 rows per internal role plus all572 external rows). Full119M TRIM weights, optimizer state and full training embedding arrays stay remote. No raw audio is re-extracted or returned.

Positive results still require stable improvements, correspondence-sensitive controls, calibration and further independent evaluation. Null results must be reported without selecting another validation-winning variant. This does not isolate environmental sounds from speech/prosody in PANNs, and it is not a new universal modality-imbalance discovery.
