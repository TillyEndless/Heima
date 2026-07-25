# Latent Intervention Qualitative Audit

Checkpoint: `/data/zxl/runs/gpt2_latent_token_ablation_formal/G3/step5000/model_weights.pt`
Split: `/data/zxl/runs/gpt2_latent_token_ablation_formal/data_split.json`
Samples: `100`

## Aggregate Metrics

- correct answer hits: 18/100
- shuffled answer hits: 18/100
- zero answer hits: 18/100
- question-only answer hits: 17/100
- answer flips correct vs shuffle: 0/100
- correct latent success but shuffle failure: 0/100
- correct/shuffle nearly identical: 64/100
- avg edit distance correct/shuffle: 44.30
- avg edit distance correct/zero: 129.41
- avg edit distance correct/q-only: 165.41
- avg similarity correct/shuffle: 0.8428
- avg similarity correct/zero: 0.5460
- avg similarity correct/q-only: 0.3927

## Automatically Found Cases

- correct latent success but shuffle failure sample ids: []
- correct/shuffle nearly identical sample ids: [1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 28, 29, 31, 32, 36, 37, 38, 39, 40, 41]
- low-similarity correct/shuffle sample ids: [0, 9, 22, 27, 33, 42, 48, 57, 58, 79, 80, 81, 89, 91, 98]

## Representative Observations

- Sample 47 asks for the Adderall peak year. Correct and shuffled latent generations are exactly identical and both recover `2012`; question-only also recovers `2012`. This is direct evidence for question leakage / prompt shortcut on at least some samples.
- Sample 1 has exact correct/shuffle identity but neither gives the final option answer. This shows the decoder can ignore which latent vector is supplied while still producing the same template.
- Sample 80 has large surface difference between correct and shuffled generations, but neither condition extracts the gold answer. This kind of difference does not support a useful sample-specific reasoning latent.

## Diagnosis

The near-zero shuffle margin is not an artifact of only looking at scalar NLL. In 100 qualitative samples, correct latent and shuffled latent have identical answer-hit outcomes: 18/100 vs 18/100, with 0 answer flips. There are also no cases where correct latent succeeds and shuffled latent fails after stricter multiple-choice answer matching.

The strongest explanation is B, decoder shortcut, together with D, question leakage. The decoder often emits a question-conditioned reasoning template, and for some samples the question text itself carries enough answer prior for question-only to match correct-latent behavior. A is also plausible: the continuous latent is not carrying usable sample-specific information for this decoder. C is present but secondary: some targets are easy or answer-bearing from the question, but even on examples with surface differences the answer/entity behavior rarely depends on correct vs shuffled latent.

Current ranking: B/D strongest, A likely, C secondary. This audit does not support the claim that G3 final latent has become a necessary sample-specific reasoning state.
