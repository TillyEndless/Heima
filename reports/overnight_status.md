# Overnight Status

## Branch And Commits

- branch: `feat/latent-cot-overnight-audit-qwen7b`
- HEAD: `0e7c6c2 run: start gpt2 g3 50k in-sequence training`

## Gate Results

- answer_parser_tests: complete
- dataset_audit: complete (1000 valid)
- download_model: complete
- finish: complete
- qwen_baseline_sanity: complete
- stage1_overfit32: complete
- stage1_smoke1k_200: complete
- start: running

## Environment

- server: `a100-1`
- enforced GPU: physical GPU1 via `CUDA_VISIBLE_DEVICES=1`
- GPU peak baseline: 15.3GB during untrained Qwen generation
- GPU after run: free
- large storage: `/data2`, `/weights2`; no old experiments modified

## Model Download

- model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- revision: `916b56a44061fd5cd7d6a8fb632557ed4f724f60`
- local snapshot: `/weights2/zhouxiaoling/hf_cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60`
- status: complete via `HF_ENDPOINT=https://hf-mirror.com`, `HF_HUB_DISABLE_XET=1`

## Dataset Audit

- dataset: `a-m-team/AM-DeepSeek-R1-Distilled-1.4M` config `am_0.9M_sample_1k`
- parsed/valid: 1000/1000
- parse status: `{'jsonl_fallback:assistant_tags': 1000}`
- CoT token count: `{'count': 1000, 'max': 19612, 'mean': 1722.379, 'min': 120}`
- latent count r=0.5: `{'count': 1000, 'max': 9806, 'mean': 861.191, 'min': 60}`
- splits: 32-sample overfit and 1k smoke under ignored `data/processed_large/qwen7b_stage1/`
- note: `datasets` streaming had schema cast error, so audit used direct JSONL fallback as requested

## Token And Trainable Audit

- `<THINK>` single token: True id=151665
- label checks: `{'question_labels_all_ignore': True, 'think_id': 151665, 'think_labels_are_think_id': True}`
- embed_tokens trainable/modules_to_save: True
- lm_head trainable/modules_to_save: True
- trainable params: 1107326976 / 8720090624

## Qwen Baseline Sanity

- cases: 16
- parser answer accuracy: 0.0 (low because this dataset contains broad/open answers, not only numeric GSM8K-style answers)
- tokens/s: 34.24
- cases: `/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b/reports/qwen_baseline_sanity.jsonl`

## Qwen Stage-1 Smoke

- training objective tonight: `L_main = L_think + L_answer`; no Stage2/self-decode, no Model B, no Loss2, no projector
- runtime smoke cap: `max_q=256`, `max_latent=128`, `max_answer=128`; this is not a formal oracle-K result
- overfit32: ok=True, reload_ok=False, step 1: loss_total=9.8326, loss_think=19.0205, think_acc=0.0000; step 30: loss_total=5.5825, loss_think=6.6493, think_acc=0.0000
- smoke1k/200: ok=True, reload_ok=False, step 1: loss_total=9.8326, loss_think=19.0205, think_acc=0.0000; step 200: loss_total=0.1420, loss_think=0.0982, think_acc=0.9922
- overfit checkpoint: `/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b/checkpoints/qwen7b_stage1_overfit32/final_adapter`
- smoke checkpoint: `/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b/checkpoints/qwen7b_stage1_smoke1k_200/final_adapter`
- reload check failed with OOM because it loads an additional base+adapter after training; checkpoint files were still saved

## Skipped

- current-checkpoint semantic/intervention audit: skipped here because CURRENT_CHECKPOINT was not provided on this new server task
- E0/E1/E2 small-model smoke: not started in this run; priority went to Qwen asset/data/stage1 gate
- Qwen Stage2/self-decode: explicitly not run tonight

## Next Decisions

- Decide whether to keep `modules_to_save=[embed_tokens,lm_head]`; it works but creates 6.2GB adapter checkpoints and nearly fills 40GB during training.
- For formal 7B runs, replace oracle-K/capped smoke with a fixed-K or free-K protocol to avoid length leakage.
- If reload audit is required, run it in a fresh process after training or load on CPU/offload to avoid OOM.
