# Method mapping and implementation decisions

## Regional observations

Regions are ordered as `global`, `brow`, `eye`, `mouth`. Brow landmarks use FAN
indices 17--26, eyes 36--47, and mouth 48--67 (zero based). A local crop is the
joint landmark box expanded by `0.15 * inter_pupil_distance` on every side. The
face is aligned using the eye centers before local boxes are evaluated.

For frame `t` and region `r`:

`q = valid * clip(det * speaker * visibility * blur * pose, 0, 1)`

where `pose = exp(-(|yaw|/45)^2 - (|pitch|/30)^2)`. The blur factor is the
piecewise-linear empirical-quantile mapping frozen from training data. A regional
utterance is valid only with at least two distinct valid frames. Repeated/padded
sampling positions always have zero masks.

Frozen 512-dimensional ResNet-18 frame features are quality pooled. The cache
stores this pooled raw feature, not a projected vector. Each region has its own
live `512 -> 256` projection and LayerNorm.

## Dual references

Eligibility requires earlier order, same dialogue, same dataset speaker ID,
valid region, quality at least `tau_q`, and the dataset timing boundary. The
latest eligible item is the recent reference. The historical pool is the last
`K=4` eligible items after excluding the recent reference. It is pooled after
each member passes through the same live regional projection as the current
observation.

The branch mask is current-valid AND reference-exists. With exactly one eligible
past observation, only the recent branch is active. Same-speaker turn distance is
computed from all class-retained turns by that speaker, not from wall-clock time.

## Change and gates

For every region and branch, independent weights implement:

`phi([v-b, abs(v-b), v*b]): 768 -> 256 -> 256`, GELU and dropout.

The B6 scalar gate consumes `[v, delta, q_current, q_reference, log1p(count),
log1p(age)]`, uses `516 -> 64 -> 1`, GELU, dropout, sigmoid, and the hard branch
mask. G1 drops the two continuous qualities; G2 uses the fixed product
`mask*q_current*q_reference`; B3/B4/B5 use the hard mask itself.

Region updates are combined with a region-specific attention vector. The
all-invalid case bypasses softmax and returns exact zeros.

## Multimodal context

- Frozen RoBERTa-base: masked mean over at most 128 tokens, raw dim 768.
- Frozen wav2vec2-base-960h: padding-aware temporal mean, raw dim 768.
- Both are projected to 256 with LayerNorm and dropout.
- Anonymous roles are assigned in first-appearance order inside each dialogue,
  embedded in 32 dimensions, with a shared overflow bucket.
- Metadata has 42 entries: text/audio availability and, per region, current
  mask/quality plus recent and historical mask/quality/log-count/log-age.
- The fused 256-dimensional utterance token receives learned order embeddings
  and enters a two-layer, four-head, causal Transformer with FFN dimension 1024.
- A causal window contains the current plus at most 31 prior utterances. Only its
  final real token is classified and supervised.

Three conventional architecture choices are configuration-controlled: classifier
width, Transformer norm order, and positional embedding type. The reproducible
defaults are a 256-hidden classifier, a pre-norm Transformer, and learned
32-position embeddings; every run records these values with the checkpoint.

## Controlled variants

- B0 zeros visual content and visual metadata and uses text/audio.
- B1 uses the global current region only.
- B2 uses all current regions without change branches.
- B3/B4/B5 use recent/historical/both branches with hard gates.
- B3Q/B4Q/B6 use learned quality gates.
- B7 retains B6 at inference and adds the training-only shift objective selected
  with `shift_lambda=0.10`.
- N4 substitutes the recent content for the historical content while retaining
  the historical branch's original eligibility mask and temporal metadata.
- G1 removes continuous quality values from learned gates.
- G2 uses the fixed quality product.
- C1 adds a 1,152-wide residual MLP to each static regional feature.
- C2 maps `[current; reference]` through independent branch MLPs, without an
  explicit relative-difference operator; its 342-wide control layer matches the
  trainable capacity of the change operator.
- C3 uses the current region as a query over the same legal recent and historical
  candidates and applies a 640-wide refinement layer. No candidate outside B6's
  causal history is exposed.

Use `scripts/parameter_report.py` to set the control widths and verify the paper's
within-5% trainable-parameter matching criterion before launching experiments.
