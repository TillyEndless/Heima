# Heima Shortcut Root Cause Audit

## Hypothesis 1: latent extraction inconsistent

Conclusion: PASS for the strict official-section implementation. The current audited path uses predictor-hidden extraction (`position-1`) matching the Heima shifted thinking-token mask, plus official-shape projector and embedding replacement. This is unlikely to explain margin≈0.

## Hypothesis 2: prompt shortcut

Official step5000 avg shuffle margin: `-0.00025177001953125`; q_gain: `-0.00021489461263020834`; zero_margin: `4.00543212890625e-05`.
Remove-latent/zero generation exact-match-to-normal: `0.7083`; similarity-to-normal: `0.9233`.
Conclusion: FAIL for latent causality, strong evidence for prompt/question shortcut. Removing or zeroing latent barely changes NLL or generation.

## Hypothesis 3: model scale

Official H0 checkpoint total B/projector params: `1896980352`.
Current H0 checkpoint total B/projector params: `1896980352`.
Official avg NLL_correct: `1.1978848775227864`; current summary/caption/reasoning NLLs are in `/data/zxl/runs/ab_loss1_shortcut_formal/h0_heima_b_probe/eval_step5000/metrics.json`.
Conclusion: scale is not the primary observed difference here. Both official-style and current H0 use Qwen2.5-0.5B decoders per section, and both show near-zero margins.

## Hypothesis 4: teacher-forcing text prefix shortcut

The decoder loss is standard causal teacher forcing over target CoT tokens. Future target tokens are not visible under causal masking, but previous gold target prefix is visible while computing later target-token CE. This is legal teacher forcing history, not future leakage. It can still reduce sensitivity to latent because after the first few target tokens, the gold prefix dominates reconstruction.
Conclusion: PASS as a contributor to reconstruction shortcut, but no evidence of future-token leakage from the audited causal setup.

## Hypothesis 5: Heima reconstruction metric may not imply latent causality

Official correct-vs-shuffle margin≈0: `-0.00025177001953125`.
Conclusion: FAIL for using reconstruction NLL alone as proof of latent causality. Official checkpoint reconstruction can remain strong when latent is shuffled/zeroed/removed.
