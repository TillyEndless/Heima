# Tiny Acceptance Final Comparison

This is a real-image tiny acceptance run on `/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1`. It validates the scaled A+B Loss1 mechanism; it is not a Heima benchmark reproduction.

## Run Status

- Run root: `/data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1`
- Groups: `tiny_acceptance_detach_baseline` and `tiny_acceptance_no_detach_ours`
- Both logs ended with exit code 0.
- Baseline final checkpoint: `/data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1/baseline/checkpoints/heima_detach_baseline.pt` (9.77 GiB)
- Ours final checkpoint: `/data/zxl/runs/heima_ab_loss1_tiny_acceptance_v1/ours/checkpoints/ours_loss1_no_detach.pt` (9.77 GiB)

## Gradient Audit

| group | detach_encoder_latent | grad_A_from_loss1 | finite |
|---|---:|---:|---|
| Heima detach baseline | True | 0.000000 | True |
| Ours Loss1 no-detach | False | 1.505476 | True |

Loss1 gradient control passed: detach blocks Loss1 to A, no-detach sends finite non-zero Loss1 gradient to A.

## Main Metric

| group | validation main NLL | runtime sec |
|---|---:|---:|
| Heima detach baseline | 0.481689 | 124.208999 |
| Ours Loss1 no-detach | 0.408855 | 132.494063 |

Ours minus baseline main NLL: `-0.072834`. Lower is better, so ours is better on main NLL in this tiny run.

## Summary Latent Intervention NLL

| group | correct NLL | q-only NLL | shuffle NLL | zero NLL | q+z gain | shuffle margin | eff rank | pairwise cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heima detach baseline | 1.560587 | 1.564940 | 1.558741 | 1.555988 | 0.004354 | -0.001846 | 8.142647 | 0.857742 |
| Ours Loss1 no-detach | 1.598668 | 1.581366 | 1.598411 | 1.575441 | -0.017302 | -0.000257 | 11.228210 | 0.843554 |

## Caption Latent Intervention NLL

| group | correct NLL | q-only NLL | shuffle NLL | zero NLL | q+z gain | shuffle margin | eff rank | pairwise cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heima detach baseline | 1.759450 | 1.757403 | 1.758709 | 1.756812 | -0.002047 | -0.000741 | 6.121554 | 0.927575 |
| Ours Loss1 no-detach | 1.772536 | 1.770855 | 1.770774 | 1.771499 | -0.001681 | -0.001761 | 6.975980 | 0.961654 |

## Reasoning Latent Intervention NLL

| group | correct NLL | q-only NLL | shuffle NLL | zero NLL | q+z gain | shuffle margin | eff rank | pairwise cos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Heima detach baseline | 1.698882 | 1.696458 | 1.698964 | 1.698783 | -0.002424 | 0.000082 | 5.227752 | 0.882693 |
| Ours Loss1 no-detach | 1.710302 | 1.691319 | 1.708216 | 1.694817 | -0.018983 | -0.002086 | 4.003932 | 0.830688 |

## Reasoning Generation Diagnostics

These are generation-side exact/coverage diagnostics from JSONL outputs, not teacher-forced token NLL.

| group | condition | exact match | numeric coverage | entity coverage | samples |
|---|---|---:|---:|---:|---:|
| Heima detach baseline | correct | 0.000000 | 0.250000 | 0.760870 | 8 |
| Heima detach baseline | whole_prefix_shuffle | 0.000000 | 0.250000 | 0.739130 | 8 |
| Heima detach baseline | zero | 0.000000 | 0.250000 | 0.739130 | 8 |
| Heima detach baseline | q_only | 0.000000 | 0.250000 | 0.717391 | 8 |
| Ours Loss1 no-detach | correct | 0.000000 | 0.250000 | 0.760870 | 8 |
| Ours Loss1 no-detach | whole_prefix_shuffle | 0.000000 | 0.250000 | 0.695652 | 8 |
| Ours Loss1 no-detach | zero | 0.000000 | 0.250000 | 0.717391 | 8 |
| Ours Loss1 no-detach | q_only | 0.000000 | 0.250000 | 0.826087 | 8 |

## Interpretation

1. Baseline latent interpretability: not established. In the reasoning section, the detach baseline has only a tiny positive normal-over-shuffle margin (`0.000082`), while q-only is still lower NLL than q+z.
2. Ours vs detach: gradient wiring works, and ours improves validation main NLL (`0.408855` vs `0.481689`). However, ours does not improve the reasoning normal-over-shuffle signal; its reasoning margin is negative (`-0.002086`).
3. Loss1 changes the representation path in the computational sense: no-detach gives A a non-zero Loss1 gradient (`1.505476`). But this tiny run does not show that the resulting latent becomes more sample-specific under intervention.
4. Current evidence is enough to say the mechanism runs end-to-end on real images; it is not enough to claim ours is better than Heima-style detach on latent semantics.

## Missing Metrics

- content-token NLL was specified in the acceptance contract but is not computed by the reused strict trainer result JSON; generation-side token coverage is reported separately and not treated as NLL.
- deterministic Model A answer generation accuracy is not computed by this trainer output; validation main_nll is reported instead.
- baseline and ours used the same seed/split/trainer/hyperparameters, but the wrapper reran S0/S1 separately for each group rather than branching both joint runs from one byte-identical S1 checkpoint.

## Recommendation

Before Loss2 or A-only self-decoder claims, first strengthen the A+B Loss1 acceptance evaluator so content-token NLL and Model A answer generation accuracy are computed from checkpoint logits/generation, then rerun a longer warm-B joint comparison from a shared S1 checkpoint.
