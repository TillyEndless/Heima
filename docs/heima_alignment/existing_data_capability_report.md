# Existing Data Capability Report

This audit is read-only except for writing this report and its JSON sidecar. No training, download, or deletion was performed.

## Micro Subsets

| subset | size | accessible images | JSON/JSONL files | train rows | train image-available rows | required fields in train |
|---|---:|---:|---:|---:|---:|---|
| `/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1` | 17M | 257 | 5 | 167 | 167 | image:100%, question:100%, summary:100%, caption:100%, reasoning:100%, answer:100% |
| `/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1` | 17M | 257 | 7 | 192 | 192 | image:100%, question:100%, summary:100%, caption:100%, reasoning:100%, answer:100% |

## Split Counts and Field Completeness

### `/data/zxl/official_heima/micro_subsets/chartqa_sqa_available_images_v1`

- `dataset_spec.json`: JSON metadata keys=['name', 'source_subset', 'image_root', 'splits', 'total_kept', 'total_dropped', 'strict_note']
- `image_manifest.jsonl`: rows=257, image_available=257, unique_images=253, fields=image:100%, question:0%, summary:0%, caption:0%, reasoning:0%, answer:0%
- `test.jsonl`: rows=45, image_available=45, unique_images=45, fields=image:100%, question:100%, summary:100%, caption:100%, reasoning:100%, answer:100%
- `train.jsonl`: rows=167, image_available=167, unique_images=164, fields=image:100%, question:100%, summary:100%, caption:100%, reasoning:100%, answer:100%
- `validation.jsonl`: rows=45, image_available=45, unique_images=45, fields=image:100%, question:100%, summary:100%, caption:100%, reasoning:100%, answer:100%

### `/data/zxl/official_heima/micro_subsets/chartqa_sqa_v1`

- `dataset_spec.json`: JSON metadata keys=['name', 'source', 'seed', 'tasks', 'splits', 'num_unique_images', 'fields', 'heima_alignment']
- `image_extraction_report.json`: JSON metadata keys=['zip_part', 'manifest', 'out_root', 'needed', 'extracted', 'missing', 'extracted_by_task', 'missing_by_task', 'first_missing', 'unsupported']
- `image_manifest.jsonl`: rows=288, image_available=288, unique_images=253, fields=image:100%, question:0%, summary:0%, caption:0%, reasoning:0%, answer:0%
- `sqa_image_extraction_report.json`: JSON metadata keys=['parquet', 'subset_root', 'needed_sqa', 'extracted', 'missing', 'ambiguous', 'first_missing', 'first_ambiguous']
- `test.jsonl`: rows=48, image_available=48, unique_images=45, fields=image:100%, question:100%, summary:100%, caption:100%, reasoning:100%, answer:100%
- `train.jsonl`: rows=192, image_available=192, unique_images=164, fields=image:100%, question:100%, summary:100%, caption:100%, reasoning:100%, answer:100%
- `validation.jsonl`: rows=48, image_available=48, unique_images=46, fields=image:100%, question:100%, summary:100%, caption:100%, reasoning:100%, answer:100%

## Full LLaVA-CoT train.jsonl Match

- rows: 98582
- schema keys top: [('id', 98582), ('image', 98582), ('conversations', 98582)]
- direct required-field rates: image:100.00%, question:0.00%, summary:0.00%, caption:0.00%, reasoning:0.00%, answer:0.00%
- rows whose image field matches accessible micro images by path/name key: 5796 (5.88%)
- first-100 image matches: 13
- Note: full `train.jsonl` is not already in the acceptance schema with direct `question/summary/caption/reasoning/answer` keys; it needs the official-schema adapter or the micro subset JSONL.

## Runs Search

Found 2 matching files under `/data/zxl/runs`. Top entries:
- `/data/zxl/runs/g1_loss2_audit_smoke_20260720/loss2_smoke/manifest.json` (1421 bytes)
- `/data/zxl/runs/g1_loss2_smoke_20260720/loss2_smoke/manifest.json` (1421 bytes)

## Answers

- Existing local micro data can support up to **192 train samples with accessible images** in a train split, and up to **192 samples** in a single schema-compatible JSONL split with accessible images.
- It is **not enough** for the planned 4096-train / 512-eval mini acceptance. It is enough only for a much smaller real-image acceptance run.
- It can still validate real image loading, schema plumbing, projector/replacement/loss wiring, and correct-vs-shuffle latent evaluation on a small subset.
- Missing for the planned run: more local images and a locked split of sufficient size. Full `train.jsonl` also needs schema adaptation because the direct field names are not the acceptance fields.
- For a tiny acceptance gate, no additional subset image download is needed if we reduce the train/eval size to fit the existing micro subset.
- For the originally requested 4096/512 acceptance, yes: we need to download only the selected subset images or restore enough of `image.zip`; full image.zip is not strictly necessary for this acceptance scale.

Detailed sidecar: `/data/zxl/Heima-ab-loss1-mini-acceptance/docs/heima_alignment/existing_data_capability_audit.json`
