# Strict Heima Stage1 + Stage2 Interpreter Supervision

This branch adds an experiment framework for the exact question: after the
official Heima Stage1 interpreter has been trained, what changes if the Stage2
interpreter loss is allowed to update Model A through the thinking latent?

It does not use the previous self-decoder, cumulative, or tiny-acceptance
training logic.

## Stage1

Stage1 delegates to the official Heima trainer:

```bash
scripts/train_stage1.sh \
  --config heima/configs/2_1-llama3_2_vision-11b-main_lora-pure_llm_decoder_lora-split_3_stages-fix_num-progressive.yaml \
  --output-dir /path/to/stage1_B_star
```

The wrapper forces `freeze_base_model=True`, so Model A is frozen while the
official summary/caption/reasoning interpreters and abstract projectors train
with the official dataset, loss, optimizer, schedule, and checkpoint format.

## Stage2

Stage2 loads the Stage1 interpreter checkpoint B* as a frozen teacher. Both
comparison modes keep B frozen and exclude B from the optimizer.

`heima_baseline` matches the paper-style Stage2 control:

```text
L_total = L_NTP
L_interp = CE(B(z.detach()), CoT)  # forward/log/eval only
```

`ours_interp_supervision` changes only the latent detach switch:

```text
L_total = L_NTP + lambda_interp * CE(B(z), CoT)
```

B remains frozen, but autograd is preserved through B input, so the interpreter
loss can update Model A.

## Required Evaluation

The experiment must reuse Heima evaluation for answer accuracy and interpreter
generation quality. Stage2 reports must include NTP loss, interpreter loss,
B(z) generation quality, CoT reconstruction quality, and correct/shuffle/zero
latent interventions.

## Why Heima Does Not Backprop Interpreter Loss

In the original schedule, the interpreter is trained after/alongside the latent
pipeline as an explainer of the existing thinking state, not as a teacher that
reshapes the encoder representation. Blocking interpreter gradients keeps the
interpreter diagnostic from becoming an additional training objective for the
core model.

This branch tests exactly that missing variable: whether frozen-interpreter CE
should become a supervision signal for Model A during Stage2.
