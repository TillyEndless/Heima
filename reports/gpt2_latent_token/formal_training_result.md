# GPT2 Formal G2/G3 Training Result

Run directory: `/data/zxl/runs/gpt2_latent_token_ablation_formal`

Shared setup: seed 42, train 10000, eval 1000, AdamW lr 1e-5, batch size 2, max length 192, generation eval 128 samples. Model weights only were saved at step500/1000/2500/5000; optimizer state was not saved.

## Final Compact Output

```json
{
  "G2": {
    "shuffle_margin": 0.0002561159133911133,
    "latent_acc": 0.0,
    "generation_acc": 0.2578125
  },
  "G3": {
    "shuffle_margin": 0.00015956497192382813,
    "latent_acc": 1.0,
    "generation_acc": 0.265625
  }
}
```

## Eval Curve

|group|step|main loss|latent acc|shuffle margin|latent gain|generation acc|
|---|---:|---:|---:|---:|---:|---:|
|G2|500|1.469737|0.000000|-0.000315|0.003296|0.187500|
|G2|1000|1.416298|0.000000|-0.000790|0.027792|0.171875|
|G2|2500|1.354492|0.000000|-0.000258|0.032794|0.281250|
|G2|5000|1.304356|0.000000|0.000256|0.022932|0.257812|
|G3|500|1.479016|0.000000|0.000181|0.024916|0.171875|
|G3|1000|1.424813|1.000000|-0.000419|0.031454|0.195312|
|G3|2500|1.359370|1.000000|-0.000613|0.028616|0.273438|
|G3|5000|1.306535|1.000000|0.000160|0.022914|0.265625|

## Interpretation

G3 successfully learned the latent token prediction objective: latent token accuracy reached 1.0 from step1000 onward. However, correct-vs-shuffled latent margins remained essentially zero for both G2 and G3 at step5000. In this run, latent token supervision made `<THINK>` predictable but did not make the continuous latent state measurably sample-specific under the current intervention metric.

The smoke margin around 0.03 did not survive the larger 10000/1000 formal run. The current evidence points to a decoder/question shortcut or an insufficiently constraining self-decode objective, not yet to latent token supervision forcing reasoning information into `z`.
