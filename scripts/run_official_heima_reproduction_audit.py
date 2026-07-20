#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "official_heima_reproduction"


def run(cmd: list[str], timeout: int = 20, cwd: Path | None = None) -> dict:
    try:
        p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"returncode": None, "stdout": "", "stderr": repr(exc)}


def write_json(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hash(paths: list[Path]) -> dict:
    entries = []
    for root in paths:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".sh", ".md"}:
                rel = str(p.relative_to(ROOT))
                entries.append(f"{rel}\t{sha256(p)}")
    h = hashlib.sha256("\n".join(entries).encode()).hexdigest()
    return {"tree_sha256": h, "file_count": len(entries), "sample_files": entries[:40]}


def environment() -> dict:
    gpu = run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader"], timeout=10)
    ram = run(["free", "-b"], timeout=10)
    disk = run(["df", "-B1", "/data", "/", "/mnt/nas/share2/home/zxl"], timeout=10)
    py = run([
        "/data/zxl/conda_envs/nlp-final/bin/python",
        "-c",
        "import json, torch, transformers, huggingface_hub, sys; print(json.dumps({'python':sys.version,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'gpu_count':torch.cuda.device_count(),'transformers':transformers.__version__,'huggingface_hub':huggingface_hub.__version__}))",
    ], timeout=20)
    parsed_py = {}
    try:
        parsed_py = json.loads(py["stdout"])
    except Exception:
        parsed_py = {"raw": py}
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "gpu_query": gpu,
        "ram": ram,
        "disk": disk,
        "python_stack": parsed_py,
        "cuda_compiler": run(["bash", "-lc", "nvcc --version 2>/dev/null | tail -n 1"], timeout=10),
    }


def hf_head(repo: str, kind: str) -> dict:
    # This is an access probe only; it does not download weights.
    url = f"https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}/resolve/main/README.md"
    res = run(["curl", "-L", "-I", "--max-time", "15", url], timeout=20)
    status = None
    for line in res["stdout"].splitlines():
        if line.startswith("HTTP/"):
            status = line
    return {"repo": repo, "type": kind, "url": url, "http_status": status, **res}


def official_resources() -> dict:
    local_candidates = {
        "xkev_llama32v_cot": [
            "/mnt/localssd/llava-cot-checkpoints/llava-cot-pretrained/Llama-3.2V-11B-cot",
            "/data/zxl/models/Xkev/Llama-3.2V-11B-cot",
        ],
        "meta_llama31_8b": [
            "/mnt/localssd/llava-cot-checkpoints/llama3_1/Llama-3.1-8B-Instruct",
            "/data/zxl/models/meta-llama/Llama-3.1-8B-Instruct",
            "/data/xinmiao/weights/Llama-3.1-8B",
        ],
        "llava_cot_100k": [
            "/mnt/localssd/llava-cot-dataset/json_files",
            "/mnt/localssd/llava-cot-dataset/image_files",
            "/data/zxl/datasets/LLaVA-CoT-100k",
        ],
        "heima_checkpoints": [
            "/data/zxl/Heima/lora-ckpts",
            "/mnt/localssd/llava-cot-checkpoints/output-checkpoints",
            "/data/zxl/checkpoints/heima",
        ],
    }
    local = {}
    for key, paths in local_candidates.items():
        local[key] = []
        for s in paths:
            p = Path(s)
            if p.exists():
                local[key].append({"path": s, "exists": True, "size_bytes": run(["du", "-sb", s], timeout=30)["stdout"]})
            else:
                local[key].append({"path": s, "exists": False})
    probes = [
        hf_head("Xkev/Llama-3.2V-11B-cot", "model"),
        hf_head("meta-llama/Llama-3.1-8B-Instruct", "model"),
        hf_head("Xkev/LLaVA-CoT-100k", "dataset"),
        hf_head("shawnricecake/Heima", "model"),
    ]
    return {
        "requested_resources": {
            "encoder_base": "Xkev/Llama-3.2V-11B-cot",
            "decoder_base": "meta-llama/Llama-3.1-8B-Instruct",
            "dataset": "Xkev/LLaVA-CoT-100k plus image files",
            "heima_checkpoints": "shawnricecake/Heima encoder and summary/caption/reasoning decoders",
            "torchtune_fork": "repository torchtune_pkg/torchtune",
        },
        "hf_access_probes": probes,
        "local_resource_candidates": local,
        "official_code_hash": tree_hash([ROOT / "heima", ROOT / "torchtune_pkg"]),
        "local_git": {
            "root_head": run(["git", "rev-parse", "HEAD"], cwd=ROOT),
            "heima_head": run(["git", "-C", str(ROOT / "heima"), "rev-parse", "HEAD"]),
            "torchtune_head": run(["git", "-C", str(ROOT / "torchtune_pkg"), "rev-parse", "HEAD"]),
        },
    }


def config_trace() -> dict:
    cfg = ROOT / "heima" / "configs" / "4_1-llama3_2_vision-11b-decode-pure_llm_decoder_lora-split_3_stages.yaml"
    script = ROOT / "heima" / "scripts" / "run-4_1-decode-pure_llm_decoder_lora-split_3_stages.sh"
    text = cfg.read_text(encoding="utf-8") if cfg.exists() else ""
    required_paths = [
        "/mnt/localssd/llava-cot-checkpoints/llava-cot-pretrained/Llama-3.2V-11B-cot",
        "/mnt/localssd/llava-cot-checkpoints/llama3_1/Llama-3.1-8B-Instruct",
        "/mnt/localssd/llava-cot-dataset/json_files",
        "/mnt/localssd/llava-cot-dataset/image_files",
        "lora-ckpts/exp-step_3/main/adapter_0.pt",
        "lora-ckpts/exp-step_4/decoder/summary/adapter_0.pt",
        "lora-ckpts/exp-step_4/decoder/summary/abstract_projector_summary.pth",
        "lora-ckpts/exp-step_4/decoder/caption/adapter_0.pt",
        "lora-ckpts/exp-step_4/decoder/caption/abstract_projector_caption.pth",
        "lora-ckpts/exp-step_4/decoder/reasoning/adapter_0.pt",
        "lora-ckpts/exp-step_4/decoder/reasoning/abstract_projector_reasoning.pth",
    ]
    status = []
    for s in required_paths:
        p = ROOT / s if not s.startswith("/") else Path(s)
        status.append({"path": s, "resolved_path": str(p), "exists": p.exists(), "sha256": sha256(p) if p.is_file() else None})
    return {
        "official_decode_config": str(cfg),
        "official_decode_script": str(script),
        "config_sha256": sha256(cfg),
        "script_sha256": sha256(script),
        "required_path_status": status,
        "image_pipeline_declared": "dataset.image_dir and multimodal llama3_2_vision transform in official config",
        "decoder_image_access": "official decoder config uses pure LLM decoder dataset/prompt and no image_dir in decoder model input; not runtime-verified because checkpoints/data are absent",
        "typed_thinking_tokens": ["<THINKING_OF_SUMMARY>", "<THINKING_OF_CAPTION>", "<THINKING_OF_REASONING>"],
        "official_config_excerpt": "\n".join(text.splitlines()[:140]),
    }


def checkpoint_loading(trace: dict) -> dict:
    missing = [x for x in trace["required_path_status"] if not x["exists"]]
    return {
        "attempted": False,
        "loaded_encoder": False,
        "loaded_decoders": {"summary": False, "caption": False, "reasoning": False},
        "reason": "Official base models, LoRA adapters, projector weights, generated test json, and image files are not present at configured paths.",
        "missing_required_paths": missing,
        "no_lightweight_substitution": True,
    }


def metric_reproduction() -> dict:
    paper_a5 = {
        "Summary": {"BLEU": 15.9, "METEOR": 40.1, "ROUGE-L": 41.6, "BERTScore": 73.4},
        "Caption": {"BLEU": 12.8, "METEOR": 35.5, "ROUGE-L": 37.9, "BERTScore": 71.4},
        "Reasoning": {"BLEU": 11.2, "METEOR": 32.7, "ROUGE-L": 32.7, "BERTScore": 66.6},
    }
    return {
        "official_test_split_evaluated": False,
        "metrics": None,
        "paper_table_a5_reference": paper_a5,
        "absolute_difference": None,
        "reason_not_run": "Official checkpoints/data are absent; running metrics on lightweight or synthetic data is prohibited by this task.",
    }


def latent_interventions() -> dict:
    return {
        "evaluated": False,
        "stages": ["summary", "caption", "reasoning"],
        "planned_interventions": ["normal", "within-stage cyclic shuffle", "same-question shuffle if available", "zero", "mean", "norm-matched random"],
        "metrics": ["full-stage NLL", "first-token NLL", "content-word NLL", "visual-fact NLL", "BLEU", "METEOR", "ROUGE-L", "BERTScore", "free-generation visual-fact match"],
        "reason_not_run": "Requires loaded official Encoder and three official Decoder checkpoints plus official test data/images.",
    }


def feasibility(env: dict, resources: dict) -> dict:
    present = resources["local_resource_candidates"]
    has_all = all(any(x["exists"] for x in vals) for vals in present.values())
    return {
        "classification": "B_only_official_checkpoint_evaluation_possible_after_downloading_resources" if not has_all else "A_possible",
        "can_run_official_steps_1_to_5_now": False,
        "can_use_official_checkpoints_now": False,
        "can_run_resource_adapted_now": False,
        "disk_note": "Only about 388GB free on /data during audit; full official model+dataset+checkpoint set may require additional space.",
        "gpu_note": "8x RTX 4090-class GPUs were detected; memory may be enough for sharded/FSDP inference/training, but official full training feasibility still depends on complete resources and exact torchtune environment.",
        "blocking_items": [
            "Download/obtain Xkev/Llama-3.2V-11B-cot files.",
            "Obtain Meta Llama-3.1-8B-Instruct access and files.",
            "Download Xkev/LLaVA-CoT-100k JSON and image files.",
            "Download shawnricecake/Heima encoder, summary decoder, caption decoder, reasoning decoder checkpoints.",
            "Map paths in heima/configs to local storage with sufficient disk.",
            "Run official data scripts before official decode/eval.",
        ],
    }


def excluded() -> dict:
    return {
        "excluded_components": ["src/htext", "GPT-2", "synthetic data", "context-conditioned data", "self-written projector", "self-written dataset builder", "self-written replacement/loss", "single whole-CoT decoder"],
        "audit_result": "No official reproduction or metrics were produced using lightweight components.",
    }


def report(env, resources, trace, loading, metrics, interventions, feas, excl) -> str:
    return f"""# Official Heima Reproduction First

## Status

Official baseline reproduction was not run. The current machine does not have the official model weights, Heima LoRA/checkpoint files, official generated JSON split, or LLaVA-CoT image files at the paths used by the official configs.

This report intentionally does not substitute GPT-2, `src/htext`, synthetic/context-conditioned data, or any lightweight component.

## Environment

- GPUs detected: {env['python_stack'].get('gpu_count')} CUDA devices; see `environment.json` for per-GPU memory.
- PyTorch: {env['python_stack'].get('torch')} / CUDA {env['python_stack'].get('torch_cuda')}
- Transformers: {env['python_stack'].get('transformers')}
- HF Hub: {env['python_stack'].get('huggingface_hub')}

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

Current classification: `{feas['classification']}`.

The immediate next step is resource acquisition/path mapping, not Ours-L1.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env = environment()
    resources = official_resources()
    trace = config_trace()
    loading = checkpoint_loading(trace)
    metrics = metric_reproduction()
    interventions = latent_interventions()
    feas = feasibility(env, resources)
    excl = excluded()
    write_json("environment.json", env)
    write_json("official_resources.json", resources)
    write_json("data_pipeline_trace.json", trace)
    write_json("checkpoint_loading.json", loading)
    write_json("official_metric_reproduction.json", metrics)
    write_json("official_latent_interventions.json", interventions)
    write_json("training_feasibility.json", feas)
    write_json("lightweight_components_excluded.json", excl)
    (OUT / "official_reproduction_report.md").write_text(report(env, resources, trace, loading, metrics, interventions, feas, excl), encoding="utf-8")
    print(json.dumps({"status": "complete", "out": str(OUT), "official_reproduction_run": False, "classification": feas["classification"]}, indent=2))


if __name__ == "__main__":
    main()
