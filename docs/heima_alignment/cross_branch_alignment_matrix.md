# Cross-Branch Scaled Heima Alignment Matrix

| Field | A+B Loss1 | A+B Loss1+Loss2 | A-only Loss1 | Allowed difference |
|---|---|---|---|---|
| Branch | `feat/heima-aligned-ab-loss1` | `feat/heima-aligned-ab-loss1-loss2` | `feat/heima-aligned-aonly-loss1` | Method branch name |
| Model A | Qwen2.5-VL-3B | Qwen2.5-VL-3B | Qwen2.5-VL-3B | `ALLOWED_MODEL_SCALE_DIFFERENCE` vs official 11B |
| Model B during training | stage-specific Qwen2.5 small LLM | stage-specific Qwen2.5 small LLM plus frozen teacher | none; same A self-decodes | Method variable |
| Dataset default | full Heima-prepared LLaVA-CoT-100k JSON | same | same | none |
| Smoke data | generated micro fixture only under `--smoke/--dry-run` | same | same | none |
| Split/hash recording | launch manifest records split hash | same | same | none |
| A explicit CoT SFT | required stage | required stage | required stage | none |
| Progressive summary/caption/reasoning | cumulative stage sequence | same | same | none |
| Recover | required | required | required | none |
| Marker labels | `main_label_mode=heima_ntp` | same | same | none |
| Hidden extraction | predictor `p-1` | same | same | none |
| Projector | official Linear/ReLU/Linear/Dropout(0) | same for B_dec; teacher features frozen | self path does not train external B; eval interpreters use same projector | Method variable only for training path |
| Prompt | official-style stage prompt with question and typed token | same | same self/eval target prompt | none |
| Target sections | summary/caption/reasoning | same | same | none |
| Training exposure | compute-matched modes exposed | `main_loss1_only` vs `main_loss1_loss2` exposed | compute-matched main-only exposed | none |
| Optimizer config | config driven | config driven | config driven | none |
| Paper evaluator | stage present | stage present | stage present | none |
| Causal evaluator | stage present | stage present | stage present | none |
| Generation profiles | `paper`, `causal_deterministic` | same | same | none |
| Model A accuracy | planned via VLMEvalKit adapter dry-run | same | same | not run full in this task |
| Model B reconstruction | BLEU/METEOR/ROUGE-L/BERTScore planned | same | eval-only interpreters after A-only training | A-only has no training B |
| Causal metrics | correct/shuffle/zero/random/q-only planned | same | same | none |

Permitted method variables only:

- A+B Loss1: external B plus Loss1.
- A+B Loss1+Loss2: external B plus Loss1 plus frozen semantic teacher Loss2.
- A-only Loss1: same Model A self-decodes latent during training, no external B in Loss1 training.
