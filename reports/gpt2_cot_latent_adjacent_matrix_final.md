# GPT2 CoT Latent Adjacent Matrix

Run root: `/data/zxl/runs/gpt2_cot_latent_adjacent_matrix_seed42`

Dataset: pure-text CoT adapter over `/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl`, seed 42, train 10000 / eval 1000. No Qwen-VL, Model B, Loss2, projector, role embedding, or cumulative latent was used.

## Final 5000-Step Metrics

| group | status | main NLL | cot NLL | answer acc | latent acc | shuffle margin | latent gain | z-only margin | wrong-Q margin | identical-ish correct/shuffle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0 | VANILLA_COT_BASELINE | 1.762155 | 1.766846 | 0.000000 | - | - | - | - | - | - / 128 |
| G1 | SHUFFLE_SIGNAL_BUT_Q_ONLY_BETTER | 1.468100 | 0.000000 | 0.140625 | 0.000000 | 0.012548 | -0.409688 | 0.008509 | 0.280169 | 49 / 128 |
| G2 | LATENT_TOKEN_FORMATION_WITH_WEAK_CAUSAL_USAGE | 1.436876 | 0.000000 | 0.164062 | 1.000000 | 0.000270 | 0.017842 | 0.000244 | 0.298790 | 106 / 128 |
| G3 | SHUFFLE_SIGNAL_BUT_Q_ONLY_BETTER | 1.756116 | 1.770158 | 0.125000 | 1.000000 | 0.009908 | -0.528969 | 0.007861 | 0.334914 | 89 / 128 |
| G4 | SHUFFLE_SIGNAL_BUT_Q_ONLY_BETTER | 1.766574 | 1.780418 | 0.078125 | 1.000000 | 0.008090 | -0.552963 | 0.010651 | 0.255598 | 93 / 128 |

## Requested G2/G3 Summary

```json
{
  "G2": {
    "shuffle_margin": 0.0002703224420547485,
    "latent_acc": 1.0,
    "generation_acc": 0.1640625,
    "latent_gain": 0.017842129707336427,
    "status": "LATENT_TOKEN_FORMATION_WITH_WEAK_CAUSAL_USAGE"
  },
  "G3": {
    "shuffle_margin": 0.00990807354450226,
    "latent_acc": 1.0,
    "generation_acc": 0.125,
    "latent_gain": -0.5289687694311141,
    "status": "SHUFFLE_SIGNAL_BUT_Q_ONLY_BETTER"
  }
}
```

## Curves

### G1

| step | main NLL | cot NLL | answer acc | latent acc | shuffle margin | latent gain | z-only margin | wrong-Q margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1.498118 | 0.000000 | 0.085938 | 0.000000 | 0.006576 | -0.423721 | 0.013539 | 0.216372 |
| 1000 | 1.462353 | 0.000000 | 0.101562 | 0.000000 | 0.001970 | -0.516849 | 0.012599 | 0.220086 |
| 2500 | 1.434205 | 0.000000 | 0.132812 | 0.000000 | 0.011853 | -0.463949 | 0.017628 | 0.247259 |
| 5000 | 1.468100 | 0.000000 | 0.140625 | 0.000000 | 0.012548 | -0.409688 | 0.008509 | 0.280169 |

### G2

| step | main NLL | cot NLL | answer acc | latent acc | shuffle margin | latent gain | z-only margin | wrong-Q margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 1.509127 | 0.000000 | 0.125000 | 1.000000 | 0.001359 | 0.022546 | 0.000639 | 0.246790 |
| 1000 | 1.466469 | 0.000000 | 0.125000 | 1.000000 | 0.000916 | 0.022214 | 0.000731 | 0.261120 |
| 2500 | 1.428640 | 0.000000 | 0.109375 | 1.000000 | 0.000389 | 0.018331 | 0.000361 | 0.281941 |
| 5000 | 1.436876 | 0.000000 | 0.164062 | 1.000000 | 0.000270 | 0.017842 | 0.000244 | 0.298790 |

### G3

| step | main NLL | cot NLL | answer acc | latent acc | shuffle margin | latent gain | z-only margin | wrong-Q margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2.077221 | 2.084624 | 0.093750 | 1.000000 | -0.001281 | -0.623529 | 0.000952 | 0.242165 |
| 1000 | 1.964261 | 1.982091 | 0.054688 | 1.000000 | 0.000308 | -0.614199 | 0.001438 | 0.270541 |
| 2500 | 1.838601 | 1.854462 | 0.101562 | 1.000000 | 0.007932 | -0.564767 | 0.004916 | 0.299519 |
| 5000 | 1.756116 | 1.770158 | 0.125000 | 1.000000 | 0.009908 | -0.528969 | 0.007861 | 0.334914 |

### G4

| step | main NLL | cot NLL | answer acc | latent acc | shuffle margin | latent gain | z-only margin | wrong-Q margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 2.085916 | 2.090746 | 0.101562 | 1.000000 | -0.000755 | -0.652377 | -0.001668 | 0.207890 |
| 1000 | 1.974993 | 1.991983 | 0.070312 | 1.000000 | 0.001638 | -0.621431 | 0.001713 | 0.219080 |
| 2500 | 1.849374 | 1.864855 | 0.039062 | 1.000000 | 0.005505 | -0.574238 | 0.005850 | 0.242132 |
| 5000 | 1.766574 | 1.780418 | 0.078125 | 1.000000 | 0.008090 | -0.552963 | 0.010651 | 0.255598 |

## Interpretation

G2 is the clearest test of the earlier hypothesis. It reaches latent token accuracy 1.0, but the final shuffle margin is only 0.000270 and 106/128 correct-vs-shuffle generations are nearly identical. This supports only a very weak latent effect, not a reasoning state that the decoder must use.

G3 and G4 produce statistically positive shuffle margins around 0.008-0.010, so their hidden state does carry some sample-specific signal. However, both have strongly negative latent gain: Q-only has lower CoT NLL than Q+correct latent. That means adding the extracted latent hurts reconstruction under this decoder protocol, so the result is better described as `SHUFFLE_SIGNAL_BUT_Q_ONLY_BETTER` rather than proof that latent token supervision solves the shortcut.

The G4 question-dropout variant did not reverse the pattern. Its final shuffle margin is slightly lower than G3 and latent gain remains negative. This weakens the simple explanation that ordinary question leakage alone is responsible; the self-decode evaluation template/objective is still allowing strong shortcut behavior or producing a latent that is not aligned with the decoder input distribution.

Answer generation accuracy is low across all groups on the first 128 eval generations, so the safest conclusion is about relative latent intervention behavior, not task quality.

## Bottom Line

Latent token supervision makes the model reliably emit the special latent token, and adjacent CoT training creates a measurable shuffle-sensitive hidden state. It does not yet make the latent a necessary reasoning bottleneck. The next useful change is likely to tighten the decoder/eval protocol around latent usage rather than simply train longer under the same objective.
