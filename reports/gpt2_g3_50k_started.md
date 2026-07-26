# GPT2 G3 50k In-Sequence Training Started

- branch: feat/gpt2-inseq-generation-and-50k
- tmux session: gpt2_g3_50k_inseq
- pane PID: 2409587
- python PID: 2409590
- GPU: CUDA_VISIBLE_DEVICES=0
- GPU process: 2328616, [Not Found], 43720 MiB
- run dir: /data/zxl/runs/gpt2_cot_latent_adjacent_50k_seed42/G3_50k
- log path: /data/zxl/runs/gpt2_cot_latent_adjacent_50k_seed42/G3_50k/logs/train.log
- command: CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/coconut/bin/python scripts/run_gpt2_50k_inseq.py --group G3 --output-dir /data/zxl/runs/gpt2_cot_latent_adjacent_50k_seed42 --train-samples 50000 --eval-samples 1000 --seed 42 --max-steps 5000 --batch-size 8 --gradient-accumulation-steps 2 --eval-batch-size 8 --generation-samples 256 --max-length 640 --max-new-tokens 128 --lr 1e-5 --device cuda:0
- expected checkpoints: step500, step1000, step2500, step5000, final
- dataset split hash: 0231130c2d57a6d8567b6b4203d7427554d8216292f9995f0bb59b471a3ef42a
- data: train=50000, eval=1000, overlap_with_eval=0, eval split matches 10k eval prefix
- config: GPT2-small, G3 latent_text_adjacent, lambda_latent=0.05, no external self-decode loss, no Model B, no Loss2, no projector
- dry-run: PASS, 32 train / 8 eval / 1 step
- G4-50k: not started; G4 did not clearly outperform G3 and GPU priority is G3

Current observed progress at start: step50 reached; latent token accuracy still 0.0 at step50, expected early in training.
