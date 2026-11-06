# Speaker-assignment audit protocol

The audit evaluates whether the face selected by the frozen assignment pipeline
matches the dataset speaker ID for the target utterance. Emotion labels and model
predictions are not shown during annotation.

## Sampling

- IEMOCAP: 300 target utterances, with 60 sampled from each session.
- MELD: 300 target utterances, with 150/60/90 sampled from the official
  train/development/test splits.
- Sampling seed: 2024.
- Audit unit: the highest-quality accepted target-face assignment for one
  utterance.

Run `drvc-build-audit-sample` after face assignment. The command stores the
stratum population and selected frame so that the design-adjusted estimate can be
recomputed from the released sample list.

## Annotation

Two annotators independently inspect the original clip and the annotated speaker
ID. The assignment output and downstream emotion predictions remain hidden.
Each annotator chooses one label:

- `correct`: the selected face is the annotated current speaker;
- `incorrect`: the selected face is clearly another person;
- `undecidable`: occlusion, blur, multiple plausible faces, or off-screen speech
  prevents a reliable decision.

A 40-example pilot establishes the shared interpretation of these labels. A
third annotator resolves disagreements after the two independent labels have
been saved. The released record keeps `annotator_1`, `annotator_2`, and
`adjudicated` as separate fields.

## Reporting

`drvc-summarize-audit` computes pre-adjudication Cohen's kappa on pairs for which
both annotators made a decidable choice. Accuracy uses adjudicated, decidable,
system-accepted assignments and weights each stratum by its population-to-sample
ratio. The confidence interval uses stratified bootstrap resampling with the same
estimator.

Training-only calibration audits are separate from this assignment audit. They
contain `raw_score`, `target_correct`, and `quality_score`; fit the monotone
speaker-confidence mapping with `drvc-calibrate-speaker-scores`, then select the
quality threshold with `drvc-select-quality-threshold`. IEMOCAP uses candidates
0.20--0.60 and a 95% accepted-match target. MELD uses 0.25--0.65 and a 90%
target. Development and test annotations do not enter calibration.
