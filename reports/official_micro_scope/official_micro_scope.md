# Official Micro Scope

This scope is for running one or two official-data tasks before the full LLaVA-CoT image archive finishes downloading.

## Selected Tasks

- ChartQA: visual numeric reasoning.
- ScienceQA (`sqa`): science/map/diagram QA.

## Current Subset

- Root: `/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1`
- Train/validation/test turns: 192 / 48 / 48
- Unique image paths required for true MLLM run: 288

## Download Boundary

For text-only Loss1/Loss2 wiring on official section targets, the full image zip is not required.
For a true official multimodal Heima baseline, images are required, but only the image paths in `image_manifest.jsonl` are needed for this micro run.
Because the official HF dataset distributes images as a split monolithic zip, targeted image acquisition requires either an alternate upstream-image source or waiting for the archive extraction.

## Matched Heima Hyperparameters

- Stages 1-3: LoRA rank 16, alpha 32, dropout 0; epochs 1; batch size 6; encoder lr 1e-4; decoder lr 5e-4.
- Recovering/decoder stage: batch size 8; encoder lr 1e-5; decoder lr 5e-4.

## Priority

1. Use the micro subset immediately for official-schema Loss1/Loss2 wiring.
2. Acquire only micro images when possible.
3. Run true MLLM micro baseline and Ours comparisons.
4. Keep full official resource download in the background for paper-level reproduction.
