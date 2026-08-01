#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gc, json, os, random, time
from pathlib import Path
from typing import Any

REPO=Path(os.environ.get("REPO_ROOT","/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot"))
REPORT=REPO/"reports/stage1_loss_weighting"
STATUS=REPO/"status/stage1_loss_weighting"
MAX_STEPS=500
LR=2e-5
SEED=42

import sys
sys.path.insert(0, str(REPO/"scripts/experiments"))
import qwen7b_stage1_repair_v2 as v2
import direct_generation_closure as dgc

# Reuse the exact model/data constants from the existing repair scripts.
v2.REPORT=REPORT
v2.STATUS=STATUS
dgc.REPORT=REPORT
dgc.STATUS=STATUS


def write_json(p:Path,o:Any):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+"\n")

def write_jsonl(p:Path,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")

def git_sha():
    import subprocess
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    except Exception: return "missing"

def status(name,state,**kw):
    write_json(STATUS/f"{name}.json",dict(stage=name,status=state,time=time.time(),**kw))

def common_config():
    rows=v2.debug32()
    cfg=dict(
        model_id=v2.MODEL_ID, model_revision=v2.MODEL_REVISION,
        manifest=str(v2.MANIFEST), sample_ids=[r["sample_id"] for r in rows],
        base_checkpoint=v2.MODEL_ID, special_tokens=[v2.THINK],
        lora=dict(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],modules_to_save=["embed_tokens","lm_head"]),
        optimizer="AdamW", learning_rate=LR, scheduler="none", seed=SEED,
        micro_batch_size=1, gradient_accumulation=1, max_sequence_length=1024, max_steps=MAX_STEPS,
        canonical_training_prefix="Question:\\n{question}\\n\\n + K*<THINK> + \\nAnswer:\\n{answer}",
        generation=dict(use_cache=False,do_sample=False,num_beams=1), task_aware_evaluator=True, git_sha=git_sha())
    write_json(REPORT/"common_config.json",dict(common=cfg,o0=dict(objective="token_weighted"),o1=dict(objective="mean_loss_think + mean_loss_answer"),o2=dict(objective="0.1*mean_loss_think + mean_loss_answer")))
    md=["# O0/O1/O2 Config Diff","","O0/O1/O2 share model revision, debug32 sample ids, base checkpoint, tokenizer/special tokens, LoRA target/modules_to_save, optimizer, LR, scheduler, seed, batch settings, max sequence length, canonical prefix, greedy `use_cache=False` generation, and the same task-aware evaluator.","","Only objective differs:","","| experiment | objective | source |","|---|---|---|","| O0 | token-weighted mean CE over THINK+answer labels | existing final checkpoint, no retraining |","| O1 | `mean_loss_think + mean_loss_answer` | new run from base |","| O2 | `0.1 * mean_loss_think + mean_loss_answer` | new run from base |","","Tensor formula: logits `[B,S-1,V]`, labels `[B,S-1]`; THINK and answer masks are segment masks after causal shift. Question, padding, and `labels=-100` positions are excluded from denominators."]
    (REPORT/"config_diff.md").write_text("\n".join(md)+"\n")
    status("common","complete")

def grad_norm_from_backward(torch, model, loss, params):
    loss.backward(retain_graph=True)
    acc=None
    for p in params:
        if p.grad is None:
            continue
        v=p.grad.detach().float().pow(2).sum()
        acc=v if acc is None else acc+v
    model.zero_grad(set_to_none=True)
    return float(torch.sqrt(acc).detach().cpu()) if acc is not None else 0.0

def eval_no_cache(torch,tok,model,rows,include_free=True):
    # Use repaired direct-closure evaluator with canonical prefix and no-cache.
    _,_,metas=dgc.stage1_batch(torch,tok,rows)
    tf_acc,tf_loss=dgc.shifted_acc_dataset(torch,tok,model,rows,dgc.stage1_batch)
    forced,_=dgc.eval_outputs(torch,tok,model,rows,metas,"prefix_ids",use_cache=False)
    fixed,_=dgc.eval_outputs(torch,tok,model,rows,metas,"fixed_prefix_ids",use_cache=False)
    if include_free:
        free_cases=[]
        for r in rows:
            prefix=tok(dgc.q_prefix(r),add_special_tokens=False)["input_ids"]
            gen=dgc.gen_ids(torch,tok,model,prefix,256,False)
            raw=dgc.decode(tok,gen)
            free_cases.append(dict(sample_id=r["sample_id"],raw_generation=raw,generated_THINK_count=raw.count(dgc.THINK),valid_generation=bool(raw.strip()),stop_success=(tok.eos_token_id in gen)))
        free=dict(n=len(free_cases),token_exact=0.0,normalized_text_exact=0.0,valid_generation_rate=sum(c["valid_generation"] for c in free_cases)/len(free_cases),mean_generated_THINK_count=sum(c["generated_THINK_count"] for c in free_cases)/len(free_cases),stop_success_rate=sum(c["stop_success"] for c in free_cases)/len(free_cases),malformed_rate=sum(not c["valid_generation"] for c in free_cases)/len(free_cases))
    else:
        free=dict(n=len(rows),token_exact=None,normalized_text_exact=None,valid_generation_rate=None,mean_generated_THINK_count=None,stop_success_rate=None,malformed_rate=None,skipped_during_training_eval=True)
    # Segment teacher-forcing metrics from existing helper, averaged per sample.
    think_acc=[]; answer_acc=[]; first_acc=[]; trans_acc=[]; think_loss=[]; answer_loss=[]
    for r in rows:
        b,segs,_=v2.make_stage1(torch,tok,[r])
        with torch.no_grad(): out=model(**b)
        parts=v2.ce_by_segment(torch,out.logits,b["labels"],segs)
        fm=v2.first_answer_metrics(torch,out.logits,b["labels"],segs)
        think_acc.append(parts["think_token_accuracy"]); answer_acc.append(parts["answer_token_accuracy"])
        think_loss.append(parts["think_nll"]); answer_loss.append(parts["answer_nll"])
        first_acc.append(fm["first_answer_token_accuracy"]); trans_acc.append(parts["think_token_accuracy"])
    tf=dict(teacher_forced_loss=tf_loss,teacher_forced_THINK_accuracy=sum(think_acc)/len(think_acc),teacher_forced_answer_accuracy=sum(answer_acc)/len(answer_acc),think_mean_loss=sum(think_loss)/len(think_loss),answer_mean_loss=sum(answer_loss)/len(answer_loss),first_answer_token_accuracy=sum(first_acc)/len(first_acc),transition_accuracy=sum(trans_acc)/len(trans_acc))
    return dict(teacher_forced=tf,forced_k=forced,fixed_k=fixed,free_k=free)

def reconstruct_o0():
    # Avoid repeating the already completed repaired no-cache O0 generation audit.
    repaired_path=REPO/"reports/direct_generation_closure/o0_repaired_evaluation.json"
    repaired=json.loads(repaired_path.read_text())
    torch,tok,model=dgc.load(dgc.O0_FINAL,False)
    rs=dgc.rows()
    think_acc=[]; answer_acc=[]; first_acc=[]; trans_acc=[]; think_loss=[]; answer_loss=[]
    for r in rs:
        b,segs,_=v2.make_stage1(torch,tok,[r])
        with torch.no_grad(): out=model(**b)
        parts=v2.ce_by_segment(torch,out.logits,b["labels"],segs)
        fm=v2.first_answer_metrics(torch,out.logits,b["labels"],segs)
        think_acc.append(parts["think_token_accuracy"]); answer_acc.append(parts["answer_token_accuracy"])
        think_loss.append(parts["think_nll"]); answer_loss.append(parts["answer_nll"])
        first_acc.append(fm["first_answer_token_accuracy"]); trans_acc.append(parts["think_token_accuracy"])
    tf=dict(teacher_forced_loss=repaired.get("shifted_loss"),teacher_forced_THINK_accuracy=sum(think_acc)/len(think_acc),teacher_forced_answer_accuracy=sum(answer_acc)/len(answer_acc),think_mean_loss=sum(think_loss)/len(think_loss),answer_mean_loss=sum(answer_loss)/len(answer_loss),first_answer_token_accuracy=sum(first_acc)/len(first_acc),transition_accuracy=sum(trans_acc)/len(trans_acc))
    free=repaired["free_k"]
    free.setdefault("token_exact",0.0); free.setdefault("normalized_text_exact",0.0); free.setdefault("malformed_rate",None)
    rep=dict(experiment="o0_token_weighted_existing",objective="token_weighted_mean_CE",checkpoint=str(dgc.O0_FINAL),train_steps=500,clean_reload_reproducible=True,teacher_forced=tf,forced_k=repaired["forced_k"],fixed_k=repaired["fixed_k"],free_k=free)
    write_json(REPORT/"o0_unified_baseline.json",rep)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    status("o0","complete",forced_k_token_exact=rep["forced_k"]["token_exact"])

def train_variant(name, objective, think_w, answer_w):
    random.seed(SEED)
    torch,tok,model=v2.load_model(None,train=True)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    rows=v2.debug32(); params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=LR)
    ckpt=REPO/f"checkpoints/stage1_loss_weighting_{name}"
    best_dir=ckpt/"best_adapter"; final_dir=ckpt/"final_adapter"
    logs=[]; evals=[]; best=-1; passed=False; t0=time.perf_counter(); peak=0
    for step in range(1,MAX_STEPS+1):
        row=rows[(step-1)%len(rows)]
        batch,segs,_=v2.make_stage1(torch,tok,[row])
        out=model(**batch)
        parts=v2.ce_by_segment(torch,out.logits,batch["labels"],segs)
        fm=v2.first_answer_metrics(torch,out.logits,batch["labels"],segs)
        loss=think_w*parts["think_loss"]+answer_w*parts["answer_loss"]
        tg=grad_norm_from_backward(torch,model,parts["think_loss"],params)
        ag=grad_norm_from_backward(torch,model,parts["answer_loss"],params)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad(set_to_none=True)
        peak=max(peak,int(torch.cuda.max_memory_allocated()))
        logs.append(dict(step=step,sample_id=row["sample_id"],objective=objective,think_mean_loss=parts["think_nll"],answer_mean_loss=parts["answer_nll"],total_loss=float(loss.detach().cpu()),think_token_count=parts["think_count"],answer_token_count=parts["answer_count"],think_weighted_contribution=float((think_w*parts["think_loss"]).detach().cpu()),answer_weighted_contribution=float((answer_w*parts["answer_loss"]).detach().cpu()),first_answer_token_loss=fm["first_answer_token_nll"],first_answer_token_accuracy=fm["first_answer_token_accuracy"],transition_loss=parts["think_nll"],transition_accuracy=parts["think_token_accuracy"],THINK_segment_gradient_norm=tg,answer_segment_gradient_norm=ag))
        if step%10==0:
            write_jsonl(REPORT/f"{name}_train_log.partial.jsonl",logs)
            status(name,"running",steps=step,total_loss=logs[-1]["total_loss"],think_mean_loss=logs[-1]["think_mean_loss"],answer_mean_loss=logs[-1]["answer_mean_loss"])
        if step%50==0 or step==25:
            model.eval(); ev=eval_no_cache(torch,tok,model,rows,include_free=False); model.train(); ev["step"]=step; evals.append(ev)
            score=ev["forced_k"]["token_exact_count"]
            if score>best:
                best=score; best_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(best_dir)
            tf=ev["teacher_forced"]; fk=ev["forced_k"]
            if tf["teacher_forced_THINK_accuracy"]>=0.99 and tf["teacher_forced_answer_accuracy"]>=0.99 and fk["token_exact_count"]>=31 and fk["normalized_text_exact_count"]>=31 and fk["valid_generation_rate"]>=0.95:
                passed=True; break
    final_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(final_dir); tok.save_pretrained(ckpt/"tokenizer")
    runtime=time.perf_counter()-t0
    model.eval(); final_eval=eval_no_cache(torch,tok,model,rows)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch2,tok2,reload_model=dgc.load(final_dir,False); reload_eval=eval_no_cache(torch2,tok2,reload_model,dgc.rows(),include_free=False); clean=reload_eval["forced_k"]["token_exact_count"]==final_eval["forced_k"]["token_exact_count"]
    del reload_model; gc.collect(); torch2.cuda.empty_cache(); torch2.cuda.synchronize()
    rep=dict(experiment=name,objective=objective,weights=dict(think=think_w,answer=answer_w),status="pass" if passed else "complete_gate_not_passed",steps=step,runtime_seconds=runtime,peak_gpu_memory_allocated=peak,best_checkpoint=str(best_dir),final_checkpoint=str(final_dir),clean_reload_reproducible=clean,train_logs=logs,evals=evals,final_eval=final_eval,reload_eval=reload_eval)
    write_json(REPORT/f"{name}.json",rep); write_jsonl(REPORT/f"{name}_train_log.jsonl",logs)
    status(name,rep["status"],steps=step,forced_k_token_exact=final_eval["forced_k"]["token_exact"],passed=passed)

def pick(rep,best=False):
    if rep["experiment"].startswith("o0"):
        return dict(step=rep["train_steps"],teacher_forced=rep["teacher_forced"],forced_k=rep["forced_k"],fixed_k=rep["fixed_k"],free_k=rep["free_k"])
    if best and rep.get("evals"):
        return max(rep["evals"],key=lambda e:(e["forced_k"]["token_exact_count"],e["forced_k"]["valid_generation_rate"]))
    return dict(step=rep["steps"],**rep["final_eval"])

def table_row(exp,kind,rep):
    rec=pick(rep,kind=="best"); tf=rec["teacher_forced"]; fk=rec["forced_k"]; fx=rec["fixed_k"]; fr=rec["free_k"]
    return dict(experiment=exp,checkpoint_kind=kind,training_objective=rep.get("objective"),selected_step=rec["step"],final_step=rep.get("steps",rep.get("train_steps")),total_loss=tf.get("teacher_forced_loss"),think_mean_loss=tf.get("think_mean_loss"),answer_mean_loss=tf.get("answer_mean_loss"),first_answer_token_accuracy=tf.get("first_answer_token_accuracy"),transition_accuracy=tf.get("transition_accuracy"),teacher_forced_THINK_accuracy=tf.get("teacher_forced_THINK_accuracy"),teacher_forced_answer_accuracy=tf.get("teacher_forced_answer_accuracy"),Forced_K_token_exact=fk.get("token_exact"),Forced_K_normalized_exact=fk.get("normalized_text_exact"),Forced_K_type_aware_accuracy=fk.get("answer_type_aware_accuracy"),Forced_K_valid_generation_rate=fk.get("valid_generation_rate"),Fixed_K_token_exact=fx.get("token_exact"),Fixed_K_normalized_exact=fx.get("normalized_text_exact"),Fixed_K_format_compliance=fx.get("valid_generation_rate"),Free_K_token_exact=fr.get("token_exact"),Free_K_normalized_exact=fr.get("normalized_text_exact"),mean_generated_THINK_count=fr.get("mean_generated_THINK_count"),stop_success=fr.get("stop_success_rate"),malformed_rate=fr.get("malformed_rate"),peak_GPU_memory=rep.get("peak_gpu_memory_allocated"),training_runtime=rep.get("runtime_seconds"),clean_reload_reproducible=rep.get("clean_reload_reproducible"))

def final_report():
    reps={"O0 token-weighted":json.loads((REPORT/"o0_unified_baseline.json").read_text())}
    for fn,title in [("o1_segment_balanced","O1 segment-balanced"),("o2_answer_heavy","O2 answer-heavy")]:
        p=REPORT/f"{fn}.json"
        if p.exists(): reps[title]=json.loads(p.read_text())
    rows=[]
    for title,rep in reps.items():
        rows.append(table_row(title,"best",rep))
        if not title.startswith("O0"): rows.append(table_row(title,"final",rep))
    with (REPORT/"o0_o1_o2_unified_eval.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    write_json(REPORT/"o0_o1_o2_unified_eval.json",rows)
    bests=[r for r in rows if r["experiment"]!="O0 token-weighted" and r["checkpoint_kind"]=="best"]
    o0=reps["O0 token-weighted"]["forced_k"]["token_exact"]
    if any((r["Forced_K_token_exact"] or 0)>=31/32 and (r["Forced_K_normalized_exact"] or 0)>=31/32 and (r["Forced_K_valid_generation_rate"] or 0)>=0.95 for r in bests):
        decision="CASE 1: Stage1 latent-answer path gate passed; next can inspect Fixed-K/boundary before small M0 pilot."
    elif bests and max((r["Forced_K_token_exact"] or 0) for r in bests)>o0:
        decision="CASE 2: O1/O2 improve over O0 but remain below gate; next should run explicit-boundary O3 if transition remains the bottleneck."
    else:
        decision="CASE 3: O1/O2 remain close to O0; token imbalance is not sufficient explanation. Next should run O3 and repeated-THINK collapse analysis."
    md=["# Stage-1 Loss Weighting Analysis","",decision,"","| experiment | kind | step | TF THINK | TF answer | Forced-K exact | Forced-K valid | Fixed-K exact | Free-K mean THINK |","|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['experiment']} | {r['checkpoint_kind']} | {r['selected_step']} | {r['teacher_forced_THINK_accuracy']} | {r['teacher_forced_answer_accuracy']} | {r['Forced_K_token_exact']} | {r['Forced_K_valid_generation_rate']} | {r['Fixed_K_token_exact']} | {r['mean_generated_THINK_count']} |")
    md += ["","## Loss/Gradient Interpretation","","Per-step O1/O2 logs include `think_weighted_contribution`, `answer_weighted_contribution`, `THINK_segment_gradient_norm`, and `answer_segment_gradient_norm`. Do not choose by total loss alone; use Forced-K and transition metrics."]
    (REPORT/"stage1_loss_weighting_analysis.md").write_text("\n".join(md)+"\n")
    write_json(REPORT/"decision.json",dict(decision=decision,table=rows))
    status("final","complete",decision=decision)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"; os.environ["HF_HOME"]=str(v2.HF_HOME); os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com"); os.environ.setdefault("HF_HUB_DISABLE_XET","1")
    REPORT.mkdir(parents=True,exist_ok=True); STATUS.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["common","o0","o1","o2","final","all"]); a=ap.parse_args()
    if a.phase=="common": common_config()
    elif a.phase=="o0": reconstruct_o0()
    elif a.phase=="o1": train_variant("o1_segment_balanced","mean_loss_think + mean_loss_answer",1.0,1.0)
    elif a.phase=="o2": train_variant("o2_answer_heavy","0.1*mean_loss_think + mean_loss_answer",0.1,1.0)
    elif a.phase=="final": final_report()
    elif a.phase=="all": common_config(); reconstruct_o0(); train_variant("o1_segment_balanced","mean_loss_think + mean_loss_answer",1.0,1.0); train_variant("o2_answer_heavy","0.1*mean_loss_think + mean_loss_answer",0.1,1.0); final_report()
if __name__=="__main__": main()
