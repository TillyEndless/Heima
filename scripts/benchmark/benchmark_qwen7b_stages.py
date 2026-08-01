#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, gc, json, math, os, random, statistics, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path('/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b')
HF_HOME = Path('/weights2/zhouxiaoling/hf_cache')
MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
MODEL_REVISION = '916b56a44061fd5cd7d6a8fb632557ed4f724f60'
THINK = '<THINK>'
REPORT_DIR = REPO / 'reports' / 'performance'
JSONL = REPORT_DIR / 'qwen7b_a10040g_benchmark.jsonl'
HIST_JSONL = REPORT_DIR / 'historical_runs.jsonl'
SUMMARY = REPORT_DIR / 'qwen7b_a10040g_summary.json'
REPORT_MD = REPORT_DIR / 'qwen7b_a10040g_report.md'
CSV_PATH = REPORT_DIR / 'qwen7b_stage1_vs_stage2.csv'
HIST_MD = REPORT_DIR / 'historical_timing_audit.md'
HIST_CSV = REPORT_DIR / 'historical_vs_new_benchmarks.csv'
LOCK = REPO / 'qwen7b_gpu1_benchmark.lock'

STAGE1_CONFIGS = {
    'S1-small': dict(stage='stage1', max_q=128, max_latent=64, max_answer=128, warmup=10, measure=30),
    'S1-current': dict(stage='stage1', max_q=256, max_latent=128, max_answer=128, warmup=0, measure=0, skip_if_historical=True),
    'S1-medium': dict(stage='stage1', max_q=256, max_latent=256, max_answer=256, warmup=10, measure=30),
    'S1-large': dict(stage='stage1', max_q=384, max_latent=384, max_answer=256, warmup=6, measure=20),
}
STAGE2_CONFIGS = {
    'S2-small': dict(stage='stage2', max_q=128, max_latent=64, max_answer=64, max_cot=128, warmup=5, measure=20),
    'S2-current': dict(stage='stage2', max_q=256, max_latent=128, max_answer=128, max_cot=256, warmup=5, measure=20),
    'S2-medium': dict(stage='stage2', max_q=256, max_latent=128, max_answer=128, max_cot=512, warmup=3, measure=12),
}
GEN_CONFIGS = {
    'GEN-self-decode-50': dict(stage='semantic_eval', cases=50, max_new_tokens=96, warmup=2, measure=10),
}

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + '\n')

def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + '\n')

def read_json(path: Path) -> Any:
    return json.loads(path.read_text())

def stats(vals: list[float]) -> dict[str, float | int | None]:
    if not vals:
        return dict(count=0, mean=None, median=None, p90=None, p95=None, std=None)
    xs = sorted(vals)
    def pct(p):
        if len(xs) == 1: return xs[0]
        k = (len(xs)-1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return xs[lo] if lo == hi else xs[lo] * (hi-k) + xs[hi] * (k-lo)
    return dict(count=len(xs), mean=sum(xs)/len(xs), median=statistics.median(xs), p90=pct(0.90), p95=pct(0.95), std=statistics.pstdev(xs) if len(xs)>1 else 0.0)

def load_rows(n=128):
    p = REPO / 'data/processed_large/qwen7b_stage1/smoke1k.json'
    rows = read_json(p)
    return rows[:n]

def recover_historical() -> list[dict[str, Any]]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    specs = [
        ('qwen7b_stage1_overfit32', 'reports/qwen7b_stage1_overfit32_stage1_smoke.json', 'checkpoints/qwen7b_stage1_overfit32/final_adapter/adapter_model.safetensors', 32, 30, 256, 128, 128),
        ('qwen7b_stage1_smoke1k_200', 'reports/qwen7b_stage1_smoke1k_200_stage1_smoke.json', 'checkpoints/qwen7b_stage1_smoke1k_200/final_adapter/adapter_model.safetensors', 1000, 200, 256, 128, 128),
    ]
    for run_id, rep, ckpt, train_samples, max_steps, max_q, max_latent, max_answer in specs:
        rp, cp = REPO/rep, REPO/ckpt
        if not rp.exists(): continue
        obj = read_json(rp)
        logs = obj.get('logs', [])
        completed = len(logs)
        ckpt_ts = cp.stat().st_mtime if cp.exists() else None
        first = logs[0] if logs else {}
        last = logs[-1] if logs else {}
        # No per-step wall timestamps were logged in the original smoke; use status/checkpoint mtimes only as low-confidence end-to-end bounds.
        status_name = 'stage1_smoke1k_200' if 'smoke1k' in run_id else 'stage1_overfit32'
        sp = REPO / 'status' / f'{status_name}.json'
        status_ts = read_json(sp).get('time') if sp.exists() else None
        run = dict(
            run_id=run_id, stage='stage1', model=MODEL_ID, model_revision=MODEL_REVISION,
            dataset='a-m-team/AM-DeepSeek-R1-Distilled-1.4M/am_0.9M_sample_1k', ratio=0.5,
            max_q=max_q, max_latent=max_latent, max_answer_or_cot=max_answer,
            micro_batch_size=1, gradient_accumulation=1, effective_batch_size=1,
            max_steps=max_steps, completed_steps=completed, train_samples=train_samples, eval_samples=0,
            checkpoint_path=str(cp) if cp.exists() else None, log_path=str(rp), command='scripts/latent_cot_overnight_qwen7b.py --phase stage1 --stage1', git_sha=sh(['git','rev-parse','HEAD']),
            command_start_time=None, first_step_time=None, last_step_time=None,
            checkpoint_saved_time_iso=iso(ckpt_ts) if ckpt_ts else None, checkpoint_saved_time_unix=ckpt_ts,
            end_to_end_runtime_seconds=None, core_training_runtime_seconds=None,
            timestamp_source='checkpoint/report only; original step wall timestamps missing', timestamp_confidence='low', checkpoint_mtime_unreliable=False,
            seconds_per_step=None, steps_per_hour=None, samples_per_second=None, samples_per_hour=None,
            actual_tokens_per_second=None, padded_tokens_per_second=None, supervised_tokens_per_second=None,
            total_setup_overhead=None, checkpoint_save_overhead=None,
            measured_or_inferred='missing_runtime', whether_setup_time_is_included=None,
            benchmark_status='historical_runtime_missing_needs_profiler' if 'smoke1k' in run_id else 'historical_runtime_missing',
            applicable_configuration=f'Stage1 max_q={max_q}, max_latent={max_latent}, max_answer={max_answer}, LoRA BF16 GC microbatch=1',
            first_log=first, last_log=last,
        )
        runs.append(run)
    HIST_JSONL.write_text('')
    for r in runs: append_jsonl(HIST_JSONL, r)
    lines = ['# Historical Timing Audit', '']
    lines += ['Historical Qwen smoke reports exist, but they did not log per-step wall timestamps. Checkpoint mtime is recorded only as low-confidence endpoint evidence, not pure training throughput.', '']
    for r in runs:
        lines += [f"## {r['run_id']}", f"- completed_steps: {r['completed_steps']}", f"- config: {r['applicable_configuration']}", f"- runtime: missing", f"- timestamp_confidence: {r['timestamp_confidence']}", f"- checkpoint_saved_time: {r['checkpoint_saved_time_iso']}", '']
    HIST_MD.write_text('\n'.join(lines) + '\n')
    return runs

def import_stack():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return torch, AutoTokenizer, AutoModelForCausalLM, LoraConfig, get_peft_model

def load_model(lora=True):
    torch, AutoTokenizer, AutoModelForCausalLM, LoraConfig, get_peft_model = import_stack()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), trust_remote_code=True)
    tok.add_special_tokens({'additional_special_tokens': [THINK]})
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.resize_token_embeddings(len(tok))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if lora:
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], modules_to_save=['embed_tokens','lm_head'])
        model = get_peft_model(model, cfg)
    model.to('cuda:0')
    model.train()
    return torch, tok, model

def make_stage1_batch(torch, tok, row, max_q, max_latent, max_answer):
    think_id = tok.convert_tokens_to_ids(THINK)
    qids = tok('Question:\n' + row['question'] + '\n\n', add_special_tokens=False)['input_ids'][:max_q]
    ans = tok('\nAnswer:\n' + row['answer'] + tok.eos_token, add_special_tokens=False)['input_ids'][:max_answer]
    k = min(int(row['latent_count']), max_latent)
    ids = qids + [think_id] * k + ans
    labels = [-100] * len(qids) + [think_id] * k + ans
    attn = [1] * len(ids)
    meta = dict(actual_tokens=len(ids), padded_tokens=len(ids), supervised_think_tokens=k, supervised_answer_tokens=len(ans), supervised_tokens=k+len(ans), q_tokens=len(qids))
    return {k2: torch.tensor([v], dtype=torch.long, device='cuda:0') for k2,v in dict(input_ids=ids, labels=labels, attention_mask=attn).items()}, meta

def make_decode_batch(torch, tok, row, z, max_q, max_latent, max_cot):
    # Prompt embeds: Question + instruction + K latent slots + target CoT. Labels only on CoT.
    think_id = tok.convert_tokens_to_ids(THINK)
    qids = tok('Question:\n' + row['question'][:2000] + '\n\nExplain the hidden reasoning represented by these latent tokens:\n', add_special_tokens=False)['input_ids'][:max_q]
    cot = tok('\nReasoning:\n' + row['gold_cot'] + tok.eos_token, add_special_tokens=False)['input_ids'][:max_cot]
    k = min(z.shape[1], max_latent)
    ids = qids + [think_id] * k + cot
    labels = [-100] * (len(qids) + k) + cot
    attn = [1] * len(ids)
    input_ids = torch.tensor([ids], dtype=torch.long, device='cuda:0')
    labels_t = torch.tensor([labels], dtype=torch.long, device='cuda:0')
    attn_t = torch.tensor([attn], dtype=torch.long, device='cuda:0')
    return input_ids, labels_t, attn_t, len(qids), k, len(cot)

def bench_one(config_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ['HF_HOME'] = str(HF_HOME)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    result = dict(config_name=config_name, config=cfg, model=MODEL_ID, model_revision=MODEL_REVISION, git_sha=sh(['git','rev-parse','HEAD']), measured_or_inferred='measured', confidence='high', started_unix=time.time(), started_iso=iso(time.time()))
    torch = None
    try:
        torch, tok, model = load_model(lora=True)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        rows = load_rows(256)
        torch.cuda.reset_peak_memory_stats()
        step_records=[]
        total_steps = int(cfg.get('warmup',0)) + int(cfg.get('measure',0))
        stage = cfg['stage']
        if stage in {'stage1','stage2'}:
            for step in range(total_steps):
                row = rows[step % len(rows)]
                t0 = time.perf_counter()
                batch, meta1 = make_stage1_batch(torch, tok, row, cfg['max_q'], cfg['max_latent'], cfg['max_answer'])
                t_dl = time.perf_counter() - t0
                torch.cuda.synchronize(); ev0=torch.cuda.Event(True); ev1=torch.cuda.Event(True); ev2=torch.cuda.Event(True); ev3=torch.cuda.Event(True)
                opt.zero_grad(set_to_none=True)
                wall0=time.perf_counter(); ev0.record()
                if stage == 'stage1':
                    out = model(**batch)
                    loss = out.loss
                    meta = meta1 | dict(cot_target_tokens=0, stage2_decode_tokens=0)
                else:
                    captured = {}
                    norm = model.base_model.model.model.norm if hasattr(model, 'base_model') else model.model.norm
                    def _hook(_module, _inp, out):
                        captured['last_hidden'] = out
                    handle = norm.register_forward_hook(_hook)
                    out1 = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], return_dict=True)
                    handle.remove()
                    seq = batch['input_ids'][0]
                    think_positions = (seq == tok.convert_tokens_to_ids(THINK)).nonzero(as_tuple=False).flatten()
                    z = captured['last_hidden'][:, think_positions, :]
                    dec_ids, dec_labels, dec_attn, dec_q_len, dec_k, cot_len = make_decode_batch(torch, tok, row, z, cfg['max_q'], cfg['max_latent'], cfg['max_cot'])
                    embeds = model.get_input_embeddings()(dec_ids)
                    embeds[:, dec_q_len:dec_q_len+dec_k, :] = z[:, :dec_k, :]
                    out2 = model(inputs_embeds=embeds, attention_mask=dec_attn, labels=dec_labels, return_dict=True)
                    loss = out1.loss + 0.1 * out2.loss
                    meta = meta1 | dict(cot_target_tokens=cot_len, stage2_decode_tokens=int(dec_ids.numel()))
                ev1.record(); loss.backward(); ev2.record(); opt.step(); ev3.record(); torch.cuda.synchronize(); wall1=time.perf_counter()
                if stage == 'stage2' and step == 0:
                    grad_norm = 0.0
                    for n,p in model.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            grad_norm += float(p.grad.detach().float().norm().cpu())
                            break
                    result['stage2_grad_first_trainable_norm'] = grad_norm
                    result['stage2_grad_nonzero_finite'] = math.isfinite(grad_norm) and grad_norm > 0
                rec = dict(step=step+1, warmup=step < cfg.get('warmup',0), wall_step_time=wall1-wall0, dataloader_time=t_dl, gpu_forward_time=ev0.elapsed_time(ev1)/1000.0, gpu_backward_time=ev1.elapsed_time(ev2)/1000.0, gpu_optimizer_time=ev2.elapsed_time(ev3)/1000.0, loss=float(loss.detach().cpu()), **meta)
                step_records.append(rec)
                del loss
        elif stage == 'semantic_eval':
            model.eval()
            for step,row in enumerate(rows[: cfg.get('warmup',0)+cfg.get('measure',0)]):
                prompt = 'Question:\n' + row['question'] + '\n\nReasoning:\n'
                enc = tok(prompt, return_tensors='pt', truncation=True, max_length=512).to('cuda:0')
                torch.cuda.synchronize(); ev0=torch.cuda.Event(True); ev1=torch.cuda.Event(True); wall0=time.perf_counter(); ev0.record()
                with torch.no_grad():
                    gen = model.generate(**enc, max_new_tokens=cfg['max_new_tokens'], do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
                ev1.record(); torch.cuda.synchronize(); wall1=time.perf_counter()
                rec=dict(step=step+1,warmup=step<cfg.get('warmup',0),wall_step_time=wall1-wall0,gpu_forward_time=ev0.elapsed_time(ev1)/1000.0,new_tokens=int(gen.shape[1]-enc.input_ids.shape[1]))
                step_records.append(rec)
        measured = [r for r in step_records if not r.get('warmup')]
        result['status']='complete'
        result['steps_total']=len(step_records); result['steps_measured']=len(measured)
        for key in ['wall_step_time','dataloader_time','gpu_forward_time','gpu_backward_time','gpu_optimizer_time']:
            result[key+'_stats']=stats([r[key] for r in measured if key in r])
        result['step_records']=step_records
        if measured:
            mean_wall = result['wall_step_time_stats']['mean']
            result['seconds_per_step']=mean_wall
            result['steps_per_hour']=3600.0/mean_wall if mean_wall else None
            toks = [r.get('actual_tokens',0)+r.get('stage2_decode_tokens',0) for r in measured]
            sup = [r.get('supervised_tokens',0)+r.get('cot_target_tokens',0) for r in measured]
            result['tokens_per_second']=sum(toks)/sum(r['wall_step_time'] for r in measured) if toks else None
            result['supervised_tokens_per_second']=sum(sup)/sum(r['wall_step_time'] for r in measured) if sup else None
            result['think_tokens_per_second']=sum(r.get('supervised_think_tokens',0) for r in measured)/sum(r['wall_step_time'] for r in measured)
            result['answer_tokens_per_second']=sum(r.get('supervised_answer_tokens',0) for r in measured)/sum(r['wall_step_time'] for r in measured)
            result['samples_per_second']=len(measured)/sum(r['wall_step_time'] for r in measured)
        result['max_memory_allocated']=int(torch.cuda.max_memory_allocated())
        result['max_memory_reserved']=int(torch.cuda.max_memory_reserved())
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        result['memory_allocated_after_cleanup']=int(torch.cuda.memory_allocated())
        result['memory_reserved_after_cleanup']=int(torch.cuda.memory_reserved())
    except Exception as e:
        is_oom = 'OutOfMemoryError' in type(e).__name__ or 'out of memory' in str(e).lower()
        result['status']='oom' if is_oom else 'failed'
        result['error']=repr(e); result['traceback']=traceback.format_exc()[-4000:]
        try:
            if torch is not None:
                torch.cuda.empty_cache(); result['memory_after_cleanup']=int(torch.cuda.memory_allocated())
        except Exception: pass
    result['ended_unix']=time.time(); result['ended_iso']=iso(time.time())
    append_jsonl(JSONL, result)
    return result

def run_suite():
    import fcntl
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK.open('w') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        historical = recover_historical()
        JSONL.write_text('')
        append_jsonl(JSONL, dict(config_name='S1-current', status='skipped_existing_measured_run_missing_runtime', source='historical report exists but no timestamps; not rerun due skip policy', historical_run_id='qwen7b_stage1_smoke1k_200', config=STAGE1_CONFIGS['S1-current']))
        configs = [(n,c) for n,c in STAGE1_CONFIGS.items() if not c.get('skip_if_historical')] + list(STAGE2_CONFIGS.items()) + list(GEN_CONFIGS.items())
        for name, _cfg in configs:
            cmd = [sys.executable, str(Path(__file__)), '--mode', 'one', '--config-name', name]
            env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES']='1'; env['HF_HOME']=str(HF_HOME); env.setdefault('HF_ENDPOINT','https://hf-mirror.com'); env.setdefault('HF_HUB_DISABLE_XET','1')
            p = subprocess.run(cmd, cwd=REPO, env=env, text=True)
            # Child records its own complete/failed/OOM row. Continue regardless.
            time.sleep(2)
        summarize()

def load_jsonl(path: Path) -> list[dict[str,Any]]:
    if not path.exists(): return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def summarize():
    rows = load_jsonl(JSONL)
    hist = load_jsonl(HIST_JSONL)
    measured = [r for r in rows if r.get('status')=='complete']
    s1 = [r for r in measured if r.get('config',{}).get('stage')=='stage1']
    s2 = [r for r in measured if r.get('config',{}).get('stage')=='stage2']
    def best(rs): return min(rs, key=lambda r:r.get('seconds_per_step', 1e9)) if rs else None
    def byname(n): return next((r for r in rows if r.get('config_name')==n), None)
    cur = byname('S1-small') or best(s1)
    s2small = byname('S2-small') or best(s2)
    eta = {}
    basis = byname('S1-medium') or cur
    if basis and basis.get('seconds_per_step'):
        for samples in [1000,10000,100000,500000,900000]:
            sec = samples * basis['seconds_per_step']
            eta[str(samples)] = dict(optimistic_compute_only_seconds=sec, realistic_with_15_percent_overhead_seconds=sec*1.15, basis_config=basis.get('config_name'), note='microbatch=1, grad_accum=1, capped benchmark length; long CoT requires length-bucket estimates')
    ratios = {}
    if cur and s2small and cur.get('seconds_per_step') and s2small.get('seconds_per_step'):
        ratios['stage2_over_stage1_time_ratio'] = s2small['seconds_per_step']/cur['seconds_per_step']
        ratios['stage2_over_stage1_memory_ratio'] = s2small.get('max_memory_allocated',0)/max(cur.get('max_memory_allocated',1),1)
    summary = dict(git_sha=sh(['git','rev-parse','HEAD']), generated_iso=iso(time.time()), rows=len(rows), historical_runs=len(hist), measured_configs=[r.get('config_name') for r in measured], oom_configs=[r.get('config_name') for r in rows if r.get('status')=='oom'], failed_configs=[r.get('config_name') for r in rows if r.get('status')=='failed'], ratios=ratios, eta=eta, raw_jsonl=str(JSONL))
    write_json(SUMMARY, summary)
    with CSV_PATH.open('w', newline='') as f:
        w=csv.writer(f); w.writerow(['config','stage','status','seconds_per_step','steps_per_hour','tokens_per_second','supervised_tokens_per_second','max_memory_allocated_gb','max_memory_reserved_gb'])
        for r in rows:
            w.writerow([r.get('config_name'), r.get('config',{}).get('stage'), r.get('status'), r.get('seconds_per_step'), r.get('steps_per_hour'), r.get('tokens_per_second'), r.get('supervised_tokens_per_second'), (r.get('max_memory_allocated') or 0)/1024**3, (r.get('max_memory_reserved') or 0)/1024**3])
    with HIST_CSV.open('w', newline='') as f:
        w=csv.writer(f); w.writerow(['run_id','stage','status','completed_steps','runtime_source','confidence','seconds_per_step','applicable_configuration'])
        for r in hist: w.writerow([r.get('run_id'), r.get('stage'), r.get('benchmark_status'), r.get('completed_steps'), r.get('timestamp_source'), r.get('timestamp_confidence'), r.get('seconds_per_step'), r.get('applicable_configuration')])
        for r in rows: w.writerow([r.get('config_name'), r.get('config',{}).get('stage'), r.get('status'), r.get('steps_measured'), 'new profiler', r.get('confidence'), r.get('seconds_per_step'), r.get('config')])
    lines=['# Qwen-7B A100 40GB Benchmark Report','',f'Generated: {summary["generated_iso"]}', '', '## Historical Audit', 'Historical Stage1 smoke reports lacked step wall timestamps, so Stage1-current was not rerun but is marked as historical-runtime-missing per skip policy.', '', '## New Measurements']
    for r in rows:
        lines += [f"### {r.get('config_name')}", f"- stage: {r.get('config',{}).get('stage')}", f"- status: {r.get('status')}", f"- seconds_per_step: {r.get('seconds_per_step')}", f"- steps/hour: {r.get('steps_per_hour')}", f"- tokens/s: {r.get('tokens_per_second')}", f"- supervised tokens/s: {r.get('supervised_tokens_per_second')}", f"- peak allocated GB: {(r.get('max_memory_allocated') or 0)/1024**3:.2f}", f"- peak reserved GB: {(r.get('max_memory_reserved') or 0)/1024**3:.2f}", f"- error: {r.get('error')}", '']
    lines += ['## ETA', '']
    for k,v in eta.items():
        lines += [f"- {k} samples: optimistic {v['optimistic_compute_only_seconds']/3600:.2f} h; realistic+15% {v['realistic_with_15_percent_overhead_seconds']/3600:.2f} h; basis {v['basis_config']}"]
    lines += ['', '## Hardware Decision Notes', '- These are single A100 40GB measured values; 8x4090 speedup is not assumed linear.', '- Current bottleneck is primarily memory for Stage2/long sequence and modules_to_save embed/lm_head, not only compute.', '- Single-card is enough for smoke and short bucket profiling; formal 100k+ or long CoT Stage2 likely benefits from multi-GPU or smaller trainable head strategy.']
    REPORT_MD.write_text('\n'.join(lines)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode', choices=['historical','suite','one','summary'], default='suite'); ap.add_argument('--config-name')
    args=ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='1'; os.environ['HF_HOME']=str(HF_HOME); os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com'); os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    if args.mode=='historical': recover_historical(); summarize()
    elif args.mode=='summary': summarize()
    elif args.mode=='one':
        cfg = (STAGE1_CONFIGS | STAGE2_CONFIGS | GEN_CONFIGS)[args.config_name]
        bench_one(args.config_name,cfg); summarize()
    else: run_suite()
if __name__ == '__main__': main()
