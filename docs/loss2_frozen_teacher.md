# Frozen-Teacher Semantic Loss2

Loss2 keeps the original external Model B design and adds a second, frozen
teacher decoder.

## Roles

- `B_dec` is the trainable student/interpreter. It participates in Loss1 and in
  the latent-side Loss2 feature computation.
- `B_teacher` is a frozen semantic teacher. It is initialized from the same base
  checkpoint and tokenizer family, uses an independent parameter object,
  `requires_grad=False`, stays in `eval()` mode, and never enters the optimizer.

## Losses

The existing original G1 losses remain:

```text
L_main = NTP_A(answer | question, z)
L1 = NTP_Bdec(CoT | latent z)
```

Loss2 adds a representation alignment in Model B space:

```text
h_L = B_dec(Q, ExplainPrompt, injected z, <SEM>, teacher-forced CoT)[<SEM>]
h_T = stopgrad(B_teacher(Q, T_<=i, <SEM>)[<SEM>])
L2 = distance(h_L, h_T)
```

In original G1 there is one whole-CoT stage. Later strict Heima sectioned paths
can map the same mechanism to summary/caption/reasoning.

## Pre-SEM Default

The default `pre_sem` mode reads the student feature at `<SEM>` before the gold
CoT target. Causal masking prevents `<SEM>` from seeing future target tokens.
This avoids the target leakage of taking a feature after the full teacher-forced
CoT.

`post_cot` exists only as an ablation and must be reported as target-leaking.

## Teacher Context

Default teacher context is `cumulative`:

```text
Question + T_<=i + <SEM>
```

For original whole-CoT G1, `T_<=i` is just `record["cot"]`.
`section_only` is also implemented as an ablation.

## Gradient Path

Formal no-detach Loss2:

```text
L2 -> h_L -> B_dec -> injected latent -> z -> producer Model A
```

Teacher features are detached. Teacher parameters never receive gradients.

## Why Loss2 Alone Is Not Enough

A decreasing Loss2 can mean representation matching, shortcut learning, or
feature collapse. Evaluation must also compare normal/shuffle/zero/random
latents, feature variance, pairwise cosine, and retrieval behavior.
