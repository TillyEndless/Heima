# Adapter Parameter Audit

The large adapter is expected from `modules_to_save=["embed_tokens","lm_head"]`, not from LoRA A/B alone.

| item | value |
|---|---:|
| total model parameters | 8,720,090,624 |
| PEFT trainable parameters | 1,107,326,976 |
| LoRA A/B parameters | 20,185,088 |
| embed_tokens parameters seen in PEFT names | 1,087,141,888 |
| lm_head parameters seen in PEFT names | 1,087,141,888 |
| modules_to_save parameters | 1,087,141,888 |
| embed/lm_head tied at loaded model accessors | False |
| adapter checkpoint size bytes | 6603644184 |

Interpretation: full embedding/output-head saves dominate the parameter count. This matches the teacher concern that output layer must be trainable for THINK-token NTP supervision.
