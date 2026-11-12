# Full-PANNs supervised residual pilot, 2026-10-28

## Question and scope
Can complete 2048D PANNs embeddings improve the completed paper TRIM checkpoint when audio retains its own sentiment supervision? This is an exploratory validation continuation, not a confirmed improvement or a new test result. PANNs features mix speech, prosody and background sound; a gain would not by itself establish an environmental-sound mechanism.

The frozen backbone is the exact archived method-new.tex/model_legacy.py paper model, paper_original/seed_42 from paper_audio_repr_20261025. None of its 119,660,936 parameters are updated. Previously verified projected text features (256D) and the four T/TA/TV/TAV logits are reused from the causal run. If its cache is absent, the same fixed checkpoint is replayed into this new experiment's cache. Existing experiment files are never rewritten. No raw audio is extracted again.

## Data and normalization
Use precisely the previous available-media, non-neutral MOSEI subset: 6065 train clips and 572 validation clips, from disjoint source videos. This subset has substantial class-dependent missing-media bias. The same IDs/labels, byte hashes, normalization and parent checkpoint are verified before training.

Complete PANNs embeddings are standardized with the previous information probe's training-only mean/std and clipped to [-8,8]. There is no 2048-to-74 random projection. The current code neither reads test split arrays nor evaluates test predictions. All labels used for optimization are training labels. Validation is used for checkpoint selection and diagnostics only.

## Added branch
The audio classifier is the previous Probe('panns_embedding',2048): Linear(2048,32), LayerNorm, GELU, dropout(0.1), Linear(32,2). It is initialized from that seed's best audio-only classifier (seeds 13,21,42); warmstart readout is checked against its saved validation predictions. These weights were already selected on the same validation set, so this continuation is not an independent validation confirmation.

The gate receives frozen projected text (256D), frozen parent class probabilities (2D), and detached audio class probabilities (2D): Linear(260,32), GELU, Linear(32,1), sigmoid. Its final bias starts at -2. Only its 8385 parameters, the 65,698 audio-classifier parameters and a scalar strength are trained: **74,084 added parameters**.

For audio logits u, gate g, trainable scalar s and audio availability a:

    margin = (u_positive - u_negative) / 2
    delta = tanh(s) * g * 2*tanh(margin/2) * a
    fused_logits = frozen_parent_logits + [-delta, +delta]

s starts at zero, making the initial output exactly the parent output. |delta| <= 2. Base logits, projected text, audio input embeddings and gate posterior inputs are detached. The audio classifier remains trainable via its fusion contribution and auxiliary loss. A=0 forces an exact zero correction, including with a trained nonzero head; T and TV must exactly reproduce their respective frozen-parent logits. TAV/TA still keep the original acoustic branch plus this added PANNs correction.

## Four matched conditions
All conditions within a seed use identical warmstart weights, initialization, minibatch order and independent A/V availability masks. Text is always present. A/V are independently dropped with probability 0.15 during branch training; the corresponding exact frozen-parent logits are selected for that mask.

1. full_supervised: aligned full embeddings, continuing audio CE weight 0.2.
2. shuffled_supervised: fixed label-free cross-video bijections, independently in train/validation; recipient labels retained, auxiliary CE weight 0.2. Every donor comes from a different source video. It is a broken-correspondence control, including for warmstart input at fusion training.
3. zero_supervised: all standardized input values zero while audio remains marked available, auxiliary CE weight 0.2. Zero in normalized feature space is a no-information control; it is not physical silence or a missing-audio condition. The text-conditioned gate and learned bias can still change predictions.
4. full_no_aux: aligned full embeddings and the identical supervised warmstart, but continuing audio CE weight 0. This isolates continuing auxiliary supervision, not all historical audio supervision.

Because warmstart is supervised and informative, shuffled/zero controls intentionally remove input correspondence after that same warmstart. No control receives an oracle mask based on validation errors.

## Fixed training and selection
15 epochs, batch size 64, AdamW, audio learning rate 1e-4, gate/strength learning rate 1e-3, weight decay 1e-3, gradient clipping norm 1. FP32, TF32 off. Loss = class-weighted smoothed fusion CE + aux_weight * audio CE on audio-present training rows + 0.001*mean(delta^2). CE weights are sqrt(N/(2*n_class)) normalized to mean 1; label smoothing 0.05. Rows with audio dropped have zero branch contribution.

The global best validation TAV Macro-F1 selects the checkpoint; **epoch zero is eligible**, and later epochs must strictly improve it. A null result can therefore select identity without being forced to change predictions. Audio-only validation F1 is monitored but does not select the fusion checkpoint. There is no hyperparameter search in this bundle.

There are 12 jobs: four conditions x three branch seeds. All use one fixed parent seed 42; branch seed variation does not represent three independent backbone trainings. Completed jobs can be reused only with matching signatures and artifact hashes. An incomplete job restarts from its matched warmstart.

## Diagnostics and returned artifacts
For every selected head, report T/TA/TV/TAV, corrected/broken cases against its matching parent mask, zero-feature inference, and 20 label-free cross-video PANNs-input assignments while parent legacy acoustic/text/vision inputs stay fixed. For full conditions, assigned minus permuted F1 measures correspondence dependence of the new input. For shuffled/zero conditions it is only a negative-control assignment comparison. Export every donor index, class prediction, probability, gate and correction.

The review ZIP includes all small trained heads, signatures, histories, metrics, and **validation_features.npz** (normalized complete validation embeddings, projected text, labels/IDs, four frozen-parent logits), enabling local neural replay. It excludes training embeddings, full backbone weights and raw media. Its manifest hashes every exported member except the manifest itself.

Interpret a validation gain together with correct versus shuffled/zero comparisons, correspondence dependence, per-seed stability and calibration. A single selected validation gain is insufficient for a paper claim. A stronger result should proceed to a separately fixed full-dataset protocol and independent evaluation; do not retroactively call this heavily reused validation set a test set. Do not replace the original paper result table with these subset metrics.
