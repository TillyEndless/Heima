# Heima Protocol Audit Final

## 1. Current vs Heima Differences

- Latent extraction: strict official-section code is aligned to Heima predictor-hidden extraction (`hidden_states[position-1]`, the state that predicts the thinking token).
- Decoder prompt: current official-section H0 prompt aligns with the audited official-style local latent template. See `reports/prompt_alignment.json`.
- Projector/replacement: strict official-section code uses `HeimaOfficialAbstractProjection` and `inputs_embeds` slot replacement. Older htext reports recorded a prior LayerNorm+Linear mismatch, but this is not the path behind the official-section H0 run.

## 2. Differences Explaining Shuffle Failure

The strongest explanation is not extraction mismatch. The official checkpoint itself has near-zero correct/shuffle/zero/q-only differences, so the failure is mainly protocol-level: prompt/question shortcut plus teacher-forced target prefix makes reconstruction NLL weakly causal with respect to latent.

## 3. Does Official Heima Remove-Latent Fail?

No. In the official H0 step5000 artifacts, zero/remove-latent NLL and generations are almost the same as normal. This means remove-latent does not fail in this scaled official-style probe.

## 4. What To Change Next

- Latent extraction: do not prioritize; audited strict path is aligned.
- Prompt: high priority. Reduce question-only priors and test prompts that force latent-specific details before generic reconstruction.
- Training objective: high priority. Add objectives/interventions that penalize shuffle/zero success or require contrastive sample-specific latent use.
- Model scale: medium priority. Larger B may improve fluency but current evidence shows scale alone does not solve latent causality.
- Architecture: medium/high priority. A decoder architecture that gates or cross-attends to latent, or a bottleneck that makes latent unavoidable, may be needed.

## Bottom Line

The shortcut margin≈0 is most consistent with prompt/question shortcut and teacher-forced reconstruction being non-causal with respect to the latent. Official-style Heima reconstruction quality should not be treated as evidence that the latent is causally used.
