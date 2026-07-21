#!/usr/bin/env bash
set -euo pipefail
cat <<'MSG'
This scaled protocol expects official Heima-prepared files:
  data_train-num_thinking_token_summary1_caption1_reasoning1.json
  data_test-num_thinking_token_summary1_caption1_reasoning1.json
and image_files from Xkev/LLaVA-CoT-100k.
This task does not download large data automatically. Approve storage/network first,
then run the official heima/scripts/run-1_1 and run-1_2 preparation flow or an
explicitly reviewed adapter into /data/zxl/official_heima/datasets/LLaVA-CoT-100k.
MSG
