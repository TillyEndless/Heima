# Heima Follow-Up Experiment Plan

Generated: 2026-07-20T09:00:24Z

This plan separates mechanism debugging from official-baseline reproduction.
The small-model experiments are resource-adapted probes. They must not be
reported as the official Heima baseline. The official comparison remains the
11B vision encoder plus 8B language decoder path on `/data`.

## Current Evidence

Completed on `cad218`:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Local weights: `/mnt/nas/share2/home/zxl/small_models/Qwen2.5-0.5B-Instruct`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- GPU: one RTX 4090 24GB
- Data: official micro text subset
  `/mnt/nas/share2/home/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1`
- Run: `/mnt/nas/share2/home/zxl/cad218_runs/heima_small_qwen_l1_200step_intervention/20260720_164836`

The gradient mechanism works:

- `joint_detach`: `grad_A_from_loss1 = 0`
- `ours_l1_no_detach`: `grad_A_from_loss1 = 1.8843`
- CE backend: Torchtune `CEWithChunkedOutputLoss`, `fallback_used = false`

The semantic latent signal is not yet established:

- `joint_detach normal_shuffle_margin = 0.00108`
- `ours_l1_no_detach normal_shuffle_margin = -0.00072`
- `joint_detach qz_gain_over_q = -0.10046`
- `ours_l1_no_detach qz_gain_over_q = -0.18046`

Interpretation: one-shot, single-token Qwen small-model training currently
proves the gradient path but does not prove that B uses sample-specific latent
information.

## Experimental Principles

1. Keep the official comparison on `/data`:
   `Xkev/Llama-3.2V-11B-cot` as Model A and
   `meta-llama/Llama-3.1-8B-Instruct` as Model B, with official Heima adapters
   and Torchtune code.
2. Use `cad218` and `/data` small-model paths for fast mechanism probes only.
3. Baseline and ours must share model, data, batch order, optimizer, scheduler,
   prompt, projector, hidden extraction, label builder, and loss. The only
   algorithmic difference is `detach_encoder_latent`.
4. Every run must write an immutable run directory before training starts:
   `experiment_manifest.json`, full group configs, launcher command, GPU/env
   snapshot, model revision, data paths, and code status.
5. Do not enter Loss2 until Loss1 produces a stable normal-over-shuffle signal.

## Experiment Matrix

### E0: Small-Model Sanity And Wiring

Status: complete.

Purpose:

- Verify model loading, typed thinking token, predictor hidden, official
  projector, embedding replacement, Torchtune CE, and detach/no-detach gradient
  routing.

Model:

- `Qwen/Qwen2.5-0.5B-Instruct`

Data:

- official micro text subset, reasoning-only target.

Groups:

- `S0 main-only`
- `S1 staged detach`
- `joint_detach`
- `ours_l1_no_detach`

Gate:

- `grad_A_from_loss1 == 0` for detach.
- `grad_A_from_loss1 > 0` for no-detach.

Current result: pass for gradient, fail/unclear for semantic latent usage.

### E1: Small-Model Longer One-Shot Probe

Status: next runnable.

Purpose:

- Test whether longer training fixes the missing latent signal in the one-shot
  setting.

Default config:

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Seeds: `42, 43, 44`
- Steps: `500` per group
- Batch size: `2`
- Eval samples: all validation records if memory permits, otherwise `45`
- Target: `reasoning`
- Thinking tokens: `K=1`
- Loss: `Lmain + lambda1 * Loss1`
- `lambda1`: `0.1`, plus optional sweep `{0.03, 0.1, 0.3}`

Required metrics:

- Main NLL
- Q-only NLL
- Q+normal NLL
- Q+shuffle NLL
- Q+zero NLL
- `normal_shuffle_margin`
- `qz_gain_over_q`
- Latent effective rank
- Pairwise cosine
- Grad attribution

Pass condition:

- In at least 2/3 seeds, `normal_shuffle_margin > 0` and `qz_gain_over_q > 0`.

If fail:

- Do not scale this one-shot setting. Move to E2/E3.

### E2: Latent Token Budget Sweep

Status: not implemented.

Purpose:

- Test the teacher suggestion that latent token count should be a fraction of
  original CoT length, e.g. 10% or 20%, rather than always `K=1`.

Required implementation:

- Support `num_thinking_tokens > 1` in HText/Qwen path.
- Extract multiple predictor states: if thinking tokens occupy positions
  `p_1..p_K`, use hidden states `h[p_k - 1]`.
- Replace multiple B typed-thinking-token slots with projected latents.
- Make intervention shuffle the whole latent sequence per sample while keeping
  Q and target fixed.

Configs:

- `K=1`
- `K=ceil(0.10 * cot_token_count)`
- `K=ceil(0.20 * cot_token_count)`
- Optional cap: `K <= 32`

Pass condition:

- Larger K improves normal-over-shuffle and Q+z-over-Q without collapsing
  answer performance.

### E3: Clear CoT / Progressive Distillation Dataset

Status: not implemented.

Purpose:

- Use data with explicit `question / cot1 / cot2 / cot3 / answer`, because the
  current official micro fields are summary/caption/reasoning and do not give
  clean step-by-step reasoning compression.

Candidate data strategies:

- Synthetic arithmetic with verified `cot1`, `cot2`, `cot3`.
- GSM8K-style text reasoning split into setup/calculation/conclusion.
- Official LLaVA-CoT fields mapped as:
  `summary -> cot1`, `caption -> cot2`, `reasoning -> cot3` only for multimodal
  adapter experiments, not as pure reasoning ground truth.

Schedule:

- Explicit SFT: `Q + cot1 + cot2 + cot3 + Answer`
- P1: `Q + THINKING_1 + cot2 + cot3 + Answer`
- P2: `Q + THINKING_1 + THINKING_2 + cot3 + Answer`
- P3/recover: `Q + THINKING_1 + THINKING_2 + THINKING_3 + Answer`

Interpreter:

- `z1 -> cot1`
- `z2 -> cot2`
- `z3 -> cot3`

Pass condition:

- Each stage has positive normal-over-shuffle margin on its own target.

### E4: Context-Conditioned Information Asymmetry

Status: not implemented for Qwen.

Purpose:

- Restore the core Heima asymmetry: Model A sees extra sample information;
  Model B sees only question plus latent. Without this, B may solve from Q or
  language prior and ignore z.

Schema:

- `context`
- `question`
- `cot1/cot2/cot3`
- `answer`
- `pair_group_id`

Visibility:

- A input: `context + question`
- B input: `question + latent`
- B must not see `context`

Evaluation:

- Same-question paired shuffle within `pair_group_id`.
- Context-fact token NLL.
- Context-value retrieval.

Pass condition:

- `Q + normal z` beats `Q + paired-shuffled z` and `Q-only`.

### E5: Official Heima Checkpoint Reproduction

Status: blocked on full official downloads on `/data`.

Purpose:

- Reproduce official Heima with official model types, official checkpoints,
  official data, image input, and three decoders.

Resources:

- Model A: `Xkev/Llama-3.2V-11B-cot`
- Model B: `meta-llama/Llama-3.1-8B-Instruct`
- Checkpoint: `shawnricecake/Heima`
- Data: `Xkev/LLaVA-CoT-100k` plus images
- Code: official Heima repo and bundled Torchtune fork

Outputs:

- checkpoint loading trace
- data pipeline trace
- official metrics on test subset/full split
- latent interventions for summary/caption/reasoning

Pass condition:

- Official checkpoint loads without lightweight code.
- Image enters encoder.
- Decoders do not read image.
- Normal-over-shuffle signal exists in official checkpoint.

### E6: Official Ours-L1

Status: blocked until E5 passes.

Purpose:

- Compare official Heima staged/joint-detach baseline to ours-L1 under the same
  11B+8B path.

Groups:

- `official_staged`
- `joint_detach`
- `ours_l1_no_detach`

Only difference:

- `detach_encoder_latent: true` vs `false`

Pass condition:

- Loss1 returns gradient to A only in `ours_l1_no_detach`.
- Ours improves latent sample specificity and does not hurt answer performance.

### E7: Loss2

Status: do not run yet.

Entry condition:

- E3 or E4 shows stable Loss1 semantic signal.
- E5 official baseline loads and has measurable latent signal.

Purpose:

- Add the next proposed loss only after Loss1 is interpretable and controlled.

## Server Split

### cad218

Use for fast small-model mechanism experiments:

- Qwen 0.5B
- one GPU, 24GB
- official micro text data
- synthetic/progressive/context-conditioned text data
- multi-seed small experiments

Do not use for official 11B+8B full training.

### /data

Use for official work and additional small-model sweeps:

- 8x RTX 4090 class GPUs
- official 11B+8B full path
- official checkpoint reproduction
- official Ours-L1 after reproduction

Also configure Qwen 0.5B there for parallel small-model experiments while
official data/model downloads continue.

## Immediate Next Actions

1. Copy or download `Qwen/Qwen2.5-0.5B-Instruct` to `/data`.
2. Sync the small-model runner to `/data`.
3. Run E1 multi-seed 500-step on one server only if GPU is free.
4. Implement E2 multi-latent-token support.
5. Build E3 clear CoT dataset and progressive schedule.
6. Keep `/data` official download/reproduction path separate from all Qwen runs.
