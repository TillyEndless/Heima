# O0/O1/O2 Config Diff

O0/O1/O2 share model revision, debug32 sample ids, base checkpoint, tokenizer/special tokens, LoRA target/modules_to_save, optimizer, LR, scheduler, seed, batch settings, max sequence length, canonical prefix, greedy `use_cache=False` generation, and the same task-aware evaluator.

Only objective differs:

| experiment | objective | source |
|---|---|---|
| O0 | token-weighted mean CE over THINK+answer labels | existing final checkpoint, no retraining |
| O1 | `mean_loss_think + mean_loss_answer` | new run from base |
| O2 | `0.1 * mean_loss_think + mean_loss_answer` | new run from base |

Tensor formula: logits `[B,S-1,V]`, labels `[B,S-1]`; THINK and answer masks are segment masks after causal shift. Question, padding, and `labels=-100` positions are excluded from denominators.
