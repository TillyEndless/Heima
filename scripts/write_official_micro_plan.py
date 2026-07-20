#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


OUT = Path("/data/zxl/Heima/reports/official_micro_scope")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    subset_root = Path("/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1")
    spec = json.loads((subset_root / "dataset_spec.json").read_text())
    image_count = sum(1 for _ in (subset_root / "image_manifest.jsonl").open())
    manifest = {
        "goal": "Run a minimal official-data Heima comparison before full LLaVA-CoT image archive is available.",
        "selected_tasks": ["chartqa", "sqa"],
        "why_these_tasks": {
            "chartqa": "visual numeric reasoning; strong test of caption/reasoning latent content",
            "sqa": "science/map/diagram QA; closer to general visual reasoning examples in Heima",
        },
        "already_available": {
            "official_code": "/data/zxl/Heima/heima",
            "torchtune_fork": "/data/zxl/Heima/torchtune_pkg/torchtune",
            "llava_cot_train_jsonl": "/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl",
            "micro_subset": str(subset_root),
        },
        "micro_subset": spec,
        "needed_for_text_only_loss_framework": {
            "required_now": [
                "official LLaVA-CoT train.jsonl",
                "official section targets: summary/caption/reasoning",
                "official typed thinking token names",
                "official projector/replacement/loss code",
                "decoder/base tokenizer and model weights, or smaller debug adapter explicitly marked non-official",
            ],
            "not_required": [
                "full image.zip",
                "all 98k records",
                "all upstream image datasets",
            ],
        },
        "needed_for_true_official_multimodal_micro_run": {
            "required_now": [
                f"{image_count} image files listed in image_manifest.jsonl",
                "Xkev/Llama-3.2V-11B-cot or official Heima Encoder checkpoint",
                "meta-llama/Llama-3.1-8B-Instruct or official Heima decoder checkpoints",
            ],
            "can_defer": [
                "remaining LLaVA-CoT images not referenced by chartqa_sqa_v1",
                "full 100k training set",
                "full official BLEU/METEOR/ROUGE/BERTScore reproduction on entire test split",
            ],
            "important_caveat": "The Hugging Face dataset publishes images as a monolithic split zip; if images are only obtained through that archive, partial task image extraction may still require the archive central directory or an alternate upstream-image source.",
        },
        "heima_training_params_to_match": {
            "progressive_stages_1_to_3": {
                "encoder_model": "Llama-3.2V-11B-cot LoRA",
                "decoder_model": "Llama-3.1-8B-Instruct LoRA",
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
                "epochs": 1,
                "batch_size": 6,
                "gradient_accumulation_steps": 1,
                "encoder_lr": "1e-4",
                "decoder_lr": "5e-4",
                "thinking_tokens_per_section": 1,
            },
            "recovering_and_decoder_stage": {
                "epochs": 1,
                "batch_size": 8,
                "gradient_accumulation_steps": 1,
                "encoder_lr": "1e-5",
                "decoder_lr": "5e-4",
                "summary_decoder": True,
                "caption_decoder": True,
                "reasoning_decoder": True,
            },
        },
        "priority_execution_order": [
            "Build official-data micro JSON and image manifest. Done.",
            "Run text-only official-schema Loss1/Loss2 smoke on chartqa_sqa_v1 to validate detach/no-detach and loss wiring.",
            "Acquire only manifest images if possible; otherwise continue full image.zip in background.",
            "Run true MLLM micro official baseline with official checkpoints.",
            "Run Ours-L1 and then Loss2 on the same micro split, keeping Heima params/config knobs matched except the method variable.",
            "Scale to full official baseline after full resources are available.",
        ],
    }
    (OUT / "download_scope.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Official Micro Scope",
        "",
        "This scope is for running one or two official-data tasks before the full LLaVA-CoT image archive finishes downloading.",
        "",
        "## Selected Tasks",
        "",
        "- ChartQA: visual numeric reasoning.",
        "- ScienceQA (`sqa`): science/map/diagram QA.",
        "",
        "## Current Subset",
        "",
        f"- Root: `{subset_root}`",
        f"- Train/validation/test turns: {spec['splits']['train']['num_turns']} / {spec['splits']['validation']['num_turns']} / {spec['splits']['test']['num_turns']}",
        f"- Unique image paths required for true MLLM run: {image_count}",
        "",
        "## Download Boundary",
        "",
        "For text-only Loss1/Loss2 wiring on official section targets, the full image zip is not required.",
        "For a true official multimodal Heima baseline, images are required, but only the image paths in `image_manifest.jsonl` are needed for this micro run.",
        "Because the official HF dataset distributes images as a split monolithic zip, targeted image acquisition requires either an alternate upstream-image source or waiting for the archive extraction.",
        "",
        "## Matched Heima Hyperparameters",
        "",
        "- Stages 1-3: LoRA rank 16, alpha 32, dropout 0; epochs 1; batch size 6; encoder lr 1e-4; decoder lr 5e-4.",
        "- Recovering/decoder stage: batch size 8; encoder lr 1e-5; decoder lr 5e-4.",
        "",
        "## Priority",
        "",
        "1. Use the micro subset immediately for official-schema Loss1/Loss2 wiring.",
        "2. Acquire only micro images when possible.",
        "3. Run true MLLM micro baseline and Ours comparisons.",
        "4. Keep full official resource download in the background for paper-level reproduction.",
        "",
    ]
    (OUT / "official_micro_scope.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
