# Official Heima Reproduction First

## Status

Official baseline reproduction was not run. The current machine does not have the official model weights, Heima LoRA/checkpoint files, official generated JSON split, or LLaVA-CoT image files at the paths used by the official configs.

This report intentionally does not substitute GPT-2, `src/htext`, synthetic/context-conditioned data, or any lightweight component.

## Environment

- GPUs detected: 8 CUDA devices; see `environment.json` for per-GPU memory.
- PyTorch: 2.7.1+cu126 / CUDA 12.6
- Transformers: 5.11.0
- HF Hub: 1.18.0

## Official Resources

Requested official resources:

- `Xkev/Llama-3.2V-11B-cot`
- `meta-llama/Llama-3.1-8B-Instruct`
- `Xkev/LLaVA-CoT-100k` plus images
- `shawnricecake/Heima` encoder and three decoder checkpoints
- repository `torchtune_pkg`

The official decode config requires paths under `/mnt/localssd/llava-cot-checkpoints` and `/mnt/localssd/llava-cot-dataset`; these are missing on this server.

## Official Metrics

No BLEU/METEOR/ROUGE-L/BERTScore reproduction was computed. Table A5 reference values are recorded in `official_metric_reproduction.json`.

## Latent Intervention

Not run. It requires the official Encoder and three official Decoder checkpoints loaded with official data/images.

## Feasibility

Current classification: `B_only_official_checkpoint_evaluation_possible_after_downloading_resources`.

The immediate next step is resource acquisition/path mapping, not Ours-L1.
