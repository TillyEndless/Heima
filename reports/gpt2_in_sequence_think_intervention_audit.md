# GPT2 In-Sequence THINK Intervention Audit

Run root: `/data/zxl/runs/gpt2_cot_latent_adjacent_matrix_seed42`

This audit loads only the final G3/G4 checkpoints. No training was run. It measures CoT and Answer NLL on the original `Question <THINK> CoT Answer` sequence.

| group | condition | CoT NLL | Answer NLL | CoT delta vs normal | CoT delta CI | Answer delta vs normal | Answer delta CI |
|---|---|---:|---:|---:|---|---:|---|
| G3 | normal | 1.772774 | 1.149548 | - | - | - | - |
| G3 | zero_think_embedding | 1.773284 | 1.148404 | 0.000510 | [0.000295, 0.000720] | -0.001144 | [-0.002717, 0.000127] |
| G3 | zero_think | 1.784021 | 1.152861 | 0.011246 | [0.010499, 0.011920] | 0.003313 | [0.000204, 0.006172] |
| G3 | shuffle_think | 1.833142 | 1.172197 | 0.060368 | [0.044575, 0.077328] | 0.022649 | [0.003682, 0.052020] |
| G3 | remove_think | 1.780481 | 1.150032 | 0.007707 | [0.007321, 0.008086] | 0.000484 | [-0.001382, 0.002097] |
| G4 | normal | 1.782977 | 1.191268 | - | - | - | - |
| G4 | zero_think_embedding | 1.783619 | 1.190358 | 0.000642 | [0.000439, 0.000833] | -0.000910 | [-0.001654, -0.000182] |
| G4 | zero_think | 1.798896 | 1.194606 | 0.015919 | [0.015159, 0.016636] | 0.003339 | [0.000227, 0.006351] |
| G4 | shuffle_think | 1.837996 | 1.215689 | 0.055019 | [0.040230, 0.071123] | 0.024421 | [0.005341, 0.054408] |
| G4 | remove_think | 1.793855 | 1.190863 | 0.010878 | [0.010420, 0.011316] | -0.000404 | [-0.001353, 0.000528] |

## Interpretation

- G3: `PASS_IN_SEQUENCE_CAUSAL_THINK_SIGNAL`. zero THINK CoT delta=0.011246, shuffle THINK CoT delta=0.060368, remove THINK CoT delta=0.007707. zero/shuffle THINK significantly increases CoT NLL in sequence, so the old self-decode evaluator likely understated latent usage.
- G4: `PASS_IN_SEQUENCE_CAUSAL_THINK_SIGNAL`. zero THINK CoT delta=0.015919, shuffle THINK CoT delta=0.055019, remove THINK CoT delta=0.010878. zero/shuffle THINK significantly increases CoT NLL in sequence, so the old self-decode evaluator likely understated latent usage.
