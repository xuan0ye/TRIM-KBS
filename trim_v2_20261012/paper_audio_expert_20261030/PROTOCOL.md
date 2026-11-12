# Frozen audio expert mechanism pilot, 2026-10-30

## Motivation and fixed scope
The previous supervised residual improved selected validation Macro-F1 by 0.1767 pp, correcting only the same one clip in three branch seeds. On those selected checkpoints, 85 parent errors include 76 with >=90% parent confidence. The learned |delta| cap (0.595-0.693) leaves only two of those errors reachable even with ideal audio signs and gate one; actual warm audio logits leave zero reachable. Replacing the audio expert in the old learned gate restores standalone audio performance but corrects no parent errors. These are read-only posthoc diagnostics of selected checkpoints, not new training results or achievable oracle performance. The residual's learned strength could grow with different training; this is not a proof that its architecture can never correct more errors.

Audio standalone Macro-F1 fell from 67.9249% to 65.9784%, and first-layer weights drifted substantially. This does not by itself prove fusion gradient conflict: continued audio-only training could also overfit. Audio agreement with parent predictions actually decreased; we do not assert that the classifier simply copied text.

Keep the exact paper model_legacy.py TRIM paper_original/seed_42 frozen. Reuse the same 6065 available-media training clips and 572 validation clips, normalization, complete 2048D PANNs embeddings, and previous causal contexts. No BERT retraining, Cnn14 re-extraction, test evaluation or paper table edits. This remains a biased media-availability subset and a heavily reused validation pilot.

## What an expert means here
The TRIM expert is the entire fixed paper checkpoint. The audio expert is both the fixed Cnn14 embedding extractor and the sentiment Probe('panns_embedding',2048): Linear(2048,32), LayerNorm, GELU, dropout(0.1), Linear(32,2), initialized from each previous audio-only best checkpoint (seeds 13/21/42). In frozen conditions, all 65,698 classifier parameters are requires_grad=False and the classifier always stays in eval mode, even while the router trains. Each epoch checks the expert's byte fingerprint and deterministic assigned-input outputs are unchanged.

The router uses the same frozen projected text (256D), parent probabilities (2D), and detached audio probabilities (2D): Linear(260,32), GELU, Linear(32,1), sigmoid. It has 8385 parameters. Original residual conditions additionally train one scalar: **8386 trainable parameters** with a frozen residual expert, 74,084 when audio is trainable. Probability mixtures train **8385 router parameters**; their scalar state is only a fixed 0/1 enable switch. The frozen paper model is never part of the optimizer. Gate conditioning on text does not update the paper text encoder.

## Six matched conditions
1. joint_repeat: reproduce the previous joint supervised residual, continuing audio auxiliary CE weight 0.2. This anchors the comparison to the previous result.
2. stop_fusion_to_audio: same residual, audio warmstart, dropout and auxiliary CE; detach audio logits from both the fusion correction and its residual penalty. The audio classifier receives only its auxiliary supervised gradient. This tests continued audio training against additional fusion gradients.
3. frozen_residual: original residual form, audio expert frozen/eval, router and scalar trainable. This isolates expert freezing while retaining the previous correction form.
4. frozen_mixture: both experts frozen/eval, train a probability mixture router.
5. shuffled_mixture: same mixture and warmstart but fixed independent label-free cross-video audio donor bijections in train/validation.
6. zero_mixture: same mixture with standardized audio embeddings all zero, audio still marked available. This is no-information input, not physical silence or missing audio.

All six share the same initial state values, minibatch order and independent A/V availability masks within each seed. Frozen classifiers intentionally disable audio dropout; joint/stop conditions keep the original dropout. They cannot share identical dropout realizations across frozen versus trainable experts. Three branch seeds share one frozen TRIM seed, not three independent backbone runs.

## Fusion forms
Original residual:

    delta = tanh(s) * sigmoid(router(context)) * 2*tanh((audio_pos-audio_neg)/4) * A_present
    output = parent_logits + [-delta,+delta]

Probability mixture:

    alpha = 1 for active mixture training; 0 for the exact disabled baseline candidate
    w = alpha * sigmoid(router(context)) * A_present
    p_mix = p_parent + w*(p_audio-p_parent)
    output = parent_logits + log(p_mix) - log(p_parent)

All states initially have scalar zero, preserving an exact disabled baseline checkpoint. Before mixture optimization, set its nontrainable scalar switch to one; the sigmoid router then trains directly, starting near gate 0.12. This avoids a zero global trainable amplitude blocking all sample-specific gate gradients. Original residual conditions continue to learn tanh(s) from zero exactly as before. The saved epoch-zero candidate remains disabled; a selected trained mixture checkpoint has its switch enabled. The mixture can approach the audio expert distribution instead of depending on the previous bounded additive correction. The returned output has softmax p_mix. A=0 gives exactly zero shift and reproduces the corresponding T/TV logits. For mixtures, reported delta is half the induced change in the class log-odds; it is not bounded by the old residual cap. Gate values are the sigmoid router output; effective mixture weight is alpha times gate for available audio. Log arguments are bounded below by 1e-30 for numerical stability.

## Fixed optimization and selection
15 epochs, batch 64, FP32, TF32 off, AdamW: audio lr 1e-4 when trainable, router/scalar lr 1e-3, weight decay 1e-3, total gradient norm clip 1. Text always present; independent A/V dropout 0.15 chooses the corresponding cached frozen-parent logits. Loss is class-weighted CE with label smoothing 0.05, plus 0.2 audio CE on audio-present rows for joint/stop conditions, plus 0.001*mean(delta^2). The mixture versus residual comparison changes the correction parameterization including removal of the trainable global amplitude; its router has one fewer trainable scalar, explicitly reported. Frozen conditions have no auxiliary optimization loss; their expert is fixed. Weights are sqrt(N/(2*n_class)), normalized to mean one.

Global validation TAV Macro-F1 selects the checkpoint, with epoch zero eligible; only strict improvements replace it. Do not interpret a selected identity checkpoint as absence of all training-time changes. No temperature or router hyperparameter grid is searched. Eighteen small-head jobs run by default; all substantive training remains remote.

## Training-only diagnosis before optimization
Verify all relevant old input/checkpoint hashes. Export train and validation baseline confidence and error rates, warm versus previously selected joint audio readouts, and correction reachability counts. These counts use labels for diagnosis only; they never create a routing target or a reported achievable F1 upper bound.

On each warm and previously selected joint classifier, compute fusion CE, audio CE and residual penalty gradients over the same eight fixed training minibatches of up to 128 rows. All snapshots use deterministic eval/TAV mode and no optimizer. Export gradient norms and cosine angles, including weighted auxiliary norm. Initial zero strength must yield zero audio fusion gradient. Negative cosine alone is not proof of training-wide causation; historical stochastic/dropout gradients were not saved. Use the joint/stop continuation comparison as a mechanism control.

Return compact full-train readouts and the first 128 sampled training feature rows, so the gradient snapshot can be independently replayed locally. No validation optimization occurs in diagnosis.

## Important remaining limitation
Both existing experts were trained on the same training rows later used to train the router. These are **in-sample expert predictions, not out-of-fold predictions**. The run explicitly reports train-versus-validation confidence/error gaps; it does not claim unbiased stacking or calibrated confidence. If routing rejects audio because the parent is nearly perfect in-sample, a later proper train-only split/OOF protocol must retrain experts excluding router holdout rows. Merely splitting already-fitted cached predictions does not fix this issue. Do not quietly call such a split out-of-fold. Current experts were also selected on this same validation set.

## Review output
Every selected branch exports T/TA/TV/TAV, zero input and 20 cross-video PANNs-input interventions, preserving parent legacy audio/text/vision inputs. Include predictions, probabilities, delta, raw gate, donor indices, weights, histories, source and input hashes. frozen_mixture versus shuffled/zero tests correspondence, frozen_mixture versus frozen_residual tests correction form, frozen_residual versus joint_repeat tests freezing, stop versus joint tests fusion-to-audio gradient involvement. They are exploratory comparisons, not automatically causal proof or confirmation of environmental sound effects. PANNs represents speech/prosody/environment together.

Review ZIP contains complete validation features, compact training readouts, a 128-row training gradient sample and small branch weights. It excludes full training embeddings, full backbone weights and raw audio. No test metrics are produced. Promote a method only after stable gains, useful correspondence, calibration and a separately fixed evaluation protocol are established.
