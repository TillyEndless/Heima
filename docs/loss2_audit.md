# Loss2 Audit

This branch starts from `origin/master` at `73c311b Heima alignment`. It is a
sibling of `feat/model-a-self-decoder`; it does not include the Self-A files and
does not remove the external Model B interpreter.

## Existing Main + Loss1 Path

1. Model A initialization and main forward:
   - `src/g1/trainer.py::load_models`
   - `src/g1/latent_reasoner.py::main_forward`
   - Model A is loaded with `AutoModelForCausalLM`.

2. z extraction:
   - `src/g1/latent_reasoner.py::extract_latent`
   - Current G1 baseline takes the last valid question-token hidden state:
     `last_valid_hidden(out.hidden_states[-1], question_mask)`.
   - This is not strict Heima predictor hidden. The strict predictor path lives
     in later HText scripts, not this original G1 A+B baseline.

3. Model B decoder initialization:
   - `src/g1/trainer.py::load_models`
   - Model B is created only when `config["lambda1"] > 0`.
   - Model A and Model B are checked for parameter independence by
     `assert_parameter_independence`.

4. Model B trainability:
   - In the existing trainer, Model B parameters are appended to the optimizer
     when present and are trainable by default.

5. Projector:
   - Original G1 has no explicit projector. The latent dimension equals the
     decoder embedding dimension because Model A and Model B use the same base
     model.

6. Loss1 prompt, replacement, and labels:
   - `src/g1/whole_cot_decoder.py::DECODER_PROMPT`
   - `src/g1/whole_cot_decoder.py::loss1_forward`
   - The latent placeholder is the existing EOS token.
   - `replace_latent_with_cat` inserts z into Model B token embeddings.
   - `build_loss1_labels` masks prompt and latent slot with `-100`; only CoT
     target tokens participate in Loss1.

7. CoT boundaries:
   - Synthetic records contain a single whole-CoT field `record["cot"]`.
   - There are no summary/caption/reasoning section boundaries in the original
     G1 baseline.

8. Optimizer parameters:
   - Existing Main-only: Model A only.
   - Existing Main+Loss1: Model A and Model B.

9. Existing B-teacher/SEM/cache:
   - No frozen B-teacher, `<SEM>` token, semantic feature loss, or feature cache
     exists in `origin/master`.

10. Existing config entrances:
   - `experiments/g1_gpt2/configs/main_only.yaml`
   - `experiments/g1_gpt2/configs/main_l1.yaml`
   - Script entry: `scripts/train_g1.py`.

Loss2 is therefore added as an independent module and smoke script. It preserves
the existing `src/g1/trainer.py` Main+Loss1 semantics.

## Added Audit Scope

The branch now audits Loss2 without changing the training method:

- intervention metrics for normal/shuffled/zero/random latents before any
  optimizer step and during the 2-step smoke run;
- feature variance, raw cosine, batch-centered cosine, and correct-vs-shuffled
  pair diagnostics;
- exact detach checks showing whether Loss2 reaches the original producer
  latent z;
- separate Main/Loss1/Loss2 gradient attribution for Model A, grouped into
  embedding, early, middle, and late block parameters;
- causal leakage checks verifying that changing only gold CoT tokens after
  `<SEM>` does not change the pre-SEM feature;
- same-question paired intervention reporting when such pairs exist.

This remains the original G1 whole-CoT A+B baseline. It is not the strict
Heima/Qwen-VL official-section Loss2 path.
