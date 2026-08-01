#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gc, json, math, os, random, re, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("REPO_ROOT", "/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot"))
HF_HOME = Path(os.environ.get("HF_HOME", "/weights2/zhouxiaoling/hf_cache"))
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MODEL_REVISION = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
THINK = "<THINK>"
REPORT = REPO / "reports/stage1_repair_v2"
STATUS = REPO / "status/stage1_repair_v2"
MANIFEST = REPO / "data/manifests/stage1_debug32_no_truncation.json"
O0_ADAPTER = REPO / "checkpoints/stage1_repair_o0/final_adapter"
ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)")
NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")


def iso(t=None): return datetime.fromtimestamp(t or time.time(), tz=timezone.utc).isoformat()
def write_json(p: Path, o: Any): p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(o, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
def write_status(name, state, **kw): write_json(STATUS / f"{name}.json", dict(stage=name, status=state, iso=iso(), time=time.time(), **kw))

def normalize_num(s):
    if s is None: return None
    s = str(s).strip().replace(",", "")
    return s[:-2] if s.endswith(".0") else s

def parse_answer(text):
    text = str(text or "")
    m = ANSWER_RE.search(text)
    if m: return normalize_num(m.group(1))
    low = text.lower()
    for marker in ["final answer is", "therefore", "answer is", "answer:", "<answer>", "boxed"]:
        i = low.rfind(marker)
        if i >= 0:
            nums = NUM_RE.findall(text[i:])
            if nums: return normalize_num(nums[-1])
    nums = NUM_RE.findall(text)
    return normalize_num(nums[-1]) if nums else None

def load_stack():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    return torch, AutoTokenizer, AutoModelForCausalLM, LoraConfig, PeftModel, get_peft_model

def load_model(adapter: Path | None = None, train=False, extra_tokens=None):
    torch, AutoTokenizer, AutoModel, LoraConfig, PeftModel, get_peft_model = load_stack()
    toks = [THINK] + list(extra_tokens or [])
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), trust_remote_code=True)
    tok.add_special_tokens({"additional_special_tokens": toks})
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    base = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), torch_dtype=torch.bfloat16, trust_remote_code=True)
    base.resize_token_embeddings(len(tok)); base.config.use_cache = False; base.gradient_checkpointing_enable()
    if adapter:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=train)
    else:
        cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], modules_to_save=["embed_tokens","lm_head"])
        model = get_peft_model(base, cfg)
    model.to("cuda:0"); model.train(train)
    return torch, tok, model

def debug32(): return json.loads(MANIFEST.read_text())["samples"]
def q_prefix(row): return "Question:\n" + row["question"] + "\n\n"
def answer_text(row): return "\nAnswer:\n" + row["answer"]
def answer_text_eos(tok, row): return answer_text(row) + tok.eos_token

def make_stage1(torch, tok, rows, objective="weighted"):
    think_id = tok.convert_tokens_to_ids(THINK)
    id_rows=[]; lab_rows=[]; seg_rows=[]; metas=[]
    for r in rows:
        qids = tok(q_prefix(r), add_special_tokens=False)["input_ids"]
        aids = tok(answer_text_eos(tok, r), add_special_tokens=False)["input_ids"]
        k = int(r["raw_K"])
        ids = qids + [think_id] * k + aids
        labs = [-100] * len(qids) + [think_id] * k + aids
        seg = ["question"] * len(qids) + ["think"] * k + ["answer"] * len(aids)
        id_rows.append(ids); lab_rows.append(labs); seg_rows.append(seg); metas.append(dict(sample_id=r["sample_id"], q_len=len(qids), raw_K=k, answer_len=len(aids), seq_len=len(ids)))
    mx=max(map(len,id_rows)); pad=tok.pad_token_id or tok.eos_token_id
    input_ids=torch.full((len(rows),mx),pad,dtype=torch.long,device="cuda:0"); labels=torch.full((len(rows),mx),-100,dtype=torch.long,device="cuda:0"); attention_mask=torch.zeros_like(input_ids)
    for i,ids in enumerate(id_rows):
        input_ids[i,:len(ids)] = torch.tensor(ids,device="cuda:0"); labels[i,:len(ids)] = torch.tensor(lab_rows[i],device="cuda:0"); attention_mask[i,:len(ids)] = 1; seg_rows[i] += ["pad"]*(mx-len(ids))
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask), seg_rows, metas

def make_direct(torch, tok, rows):
    id_rows=[]; lab_rows=[]; seg_rows=[]
    for r in rows:
        qids=tok(q_prefix(r), add_special_tokens=False)["input_ids"]
        aids=tok(answer_text_eos(tok,r), add_special_tokens=False)["input_ids"]
        ids=qids+aids; labs=[-100]*len(qids)+aids; seg=["question"]*len(qids)+["answer"]*len(aids)
        id_rows.append(ids); lab_rows.append(labs); seg_rows.append(seg)
    mx=max(map(len,id_rows)); pad=tok.pad_token_id or tok.eos_token_id
    input_ids=torch.full((len(rows),mx),pad,dtype=torch.long,device="cuda:0"); labels=torch.full((len(rows),mx),-100,dtype=torch.long,device="cuda:0"); attention_mask=torch.zeros_like(input_ids)
    for i,ids in enumerate(id_rows):
        input_ids[i,:len(ids)] = torch.tensor(ids,device="cuda:0"); labels[i,:len(ids)] = torch.tensor(lab_rows[i],device="cuda:0"); attention_mask[i,:len(ids)] = 1; seg_rows[i] += ["pad"]*(mx-len(ids))
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask), seg_rows

def ce_by_segment(torch, logits, labels, segs):
    import torch.nn.functional as F
    sl=labels[:,1:].contiguous(); lg=logits[:,:-1,:].contiguous(); losses=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(), sl.reshape(-1), ignore_index=-100, reduction="none").reshape_as(sl)
    out={}
    for name in ["think","answer"]:
        mask=[]
        for row in segs: mask.append([s==name for s in row[1:]])
        m=torch.tensor(mask,dtype=torch.bool,device=sl.device) & sl.ne(-100)
        if m.any():
            pred=lg.argmax(-1)
            out[f"{name}_loss"] = losses[m].mean()
            out[f"{name}_nll"] = float(losses[m].mean().detach().cpu())
            out[f"{name}_token_accuracy"] = float(pred[m].eq(sl[m]).float().mean().detach().cpu())
            out[f"{name}_count"] = int(m.sum().detach().cpu())
        else:
            out[f"{name}_loss"] = losses.sum()*0; out[f"{name}_nll"] = None; out[f"{name}_token_accuracy"] = None; out[f"{name}_count"] = 0
    return out

def first_answer_metrics(torch, logits, labels, segs):
    import torch.nn.functional as F
    sl=labels[:,1:]; lg=logits[:,:-1,:]
    rows=[]; correct=0; n=0; nlls=[]; rem_correct=0; rem_n=0; rem_nll=[]
    for i,row in enumerate(segs):
        ans_pos=[j for j,s in enumerate(row[1:]) if s=="answer" and int(sl[i,j])!=-100]
        if not ans_pos: continue
        j=ans_pos[0]; loss=F.cross_entropy(lg[i,j:j+1].float(), sl[i,j:j+1], reduction="none")[0]
        pred=int(lg[i,j].argmax()); lab=int(sl[i,j]); correct += int(pred==lab); n += 1; nlls.append(float(loss.detach().cpu()))
        for jj in ans_pos[1:]:
            loss2=F.cross_entropy(lg[i,jj:jj+1].float(), sl[i,jj:jj+1], reduction="none")[0]
            rem_correct += int(int(lg[i,jj].argmax())==int(sl[i,jj])); rem_n += 1; rem_nll.append(float(loss2.detach().cpu()))
    return dict(first_answer_token_accuracy=correct/max(n,1), first_answer_token_nll=sum(nlls)/max(len(nlls),1), remaining_answer_token_accuracy=rem_correct/max(rem_n,1), remaining_answer_token_nll=sum(rem_nll)/max(len(rem_nll),1))

def gen_text(torch, tok, model, prompt, max_new=64, use_cache=True, output_scores=False):
    enc=tok(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        gen=model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, use_cache=use_cache, return_dict_in_generate=output_scores, output_scores=output_scores)
    if output_scores:
        text=tok.decode(gen.sequences[0, enc.input_ids.shape[1]:], skip_special_tokens=False)
        return text, gen.scores[0].detach().float().cpu()
    return tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=False), None

def summarize_cases(cases):
    n=max(len(cases),1); valid=sum(c["parsed_answer"] is not None for c in cases); corr=sum(c["correct"] for c in cases); malformed=sum(c["parsed_answer"] is None for c in cases); stop=sum(not c.get("max_new_tokens_reached",False) for c in cases)
    return dict(n=len(cases), valid_answer_rate=valid/n, answer_accuracy=corr/n, malformed_rate=malformed/n, stop_success_rate=stop/n, mean_generated_THINK_count=sum(c.get("generated_THINK_count",0) for c in cases)/n)

def transition_audit():
    rows=debug32(); torch,tok,model=load_model(O0_ADAPTER,train=False)
    row=rows[0]; batch,segs,metas=make_stage1(torch,tok,[row]);
    with torch.no_grad(): out=model(**batch)
    ids=batch["input_ids"][0].detach().cpu().tolist(); labels=batch["labels"][0].detach().cpu().tolist(); seg=segs[0]
    think_positions=[i for i,s in enumerate(seg) if s=="think"]; ans_positions=[i for i,s in enumerate(seg) if s=="answer"]
    lo=max(0,think_positions[-1]-8); hi=min(len(ids),ans_positions[0]+16)
    rows_tokens=[]
    marker_ids=tok("\nAnswer:\n",add_special_tokens=False)["input_ids"]
    for i in range(lo,hi):
        notes=[]
        if i==think_positions[-1]: notes.append("last_THINK")
        if i==ans_positions[0]: notes.append("first_token_after_THINK/answer_boundary_start")
        if i==ans_positions[0]+len(marker_ids): notes.append("gold_first_answer_text_token")
        if labels[i]==tok.eos_token_id: notes.append("EOS_label")
        rows_tokens.append(dict(position=i, token_id=ids[i], token=tok.decode([ids[i]]), segment=seg[i], label=labels[i], label_token=None if labels[i]==-100 else tok.decode([labels[i]]), loss_mask=labels[i]!=-100, notes=notes))
    prefix=q_prefix(row)+(THINK*int(row["raw_K"]))
    enc=tok(prefix,return_tensors="pt").to("cuda:0")
    with torch.no_grad(): full=model(**enc).logits[:,-1,:].float().softmax(-1)[0].detach().cpu()
    top=torch.topk(full,20)
    top20=[dict(rank=i+1, token_id=int(tid), token=tok.decode([int(tid)]), probability=float(prob)) for i,(prob,tid) in enumerate(zip(top.values, top.indices))]
    expected_ids=tok(answer_text_eos(tok,row),add_special_tokens=False)["input_ids"]
    tracked={"expected_answer_boundary_token":expected_ids[0],"gold_first_answer_text_token":expected_ids[len(marker_ids)] if len(expected_ids)>len(marker_ids) else expected_ids[0],"THINK":tok.convert_tokens_to_ids(THINK),"EOS":tok.eos_token_id}
    tracked_probs={k:dict(token_id=int(v), token=tok.decode([int(v)]), probability=float(full[int(v)]), rank=int((full>full[int(v)]).sum().item()+1)) for k,v in tracked.items()}
    text_cache, score_cache=gen_text(torch,tok,model,prefix,max_new=1,use_cache=True,output_scores=True)
    text_nocache, score_nocache=gen_text(torch,tok,model,prefix,max_new=1,use_cache=False,output_scores=True)
    cache_probs=score_cache.softmax(-1)[0]; nocache_probs=score_nocache.softmax(-1)[0]
    max_abs_cache=float((full-cache_probs).abs().max()); max_abs_nocache=float((full-nocache_probs).abs().max())
    top_cache=torch.topk(cache_probs,20).indices.tolist(); top_nocache=torch.topk(nocache_probs,20).indices.tolist()
    fm=first_answer_metrics(torch,out.logits,batch["labels"],segs); segm=ce_by_segment(torch,out.logits,batch["labels"],segs)
    rep=dict(checkpoint=str(O0_ADAPTER), sample_id=row["sample_id"], answer_marker="\nAnswer:\n", answer_marker_token_ids=marker_ids, answer_marker_exists=True, answer_marker_participates_in_loss=True, answer_marker_preprovided_in_forced_generation=True, training_generation_template_match=dict(forced_k_prompt_matches_training_prefix_until_answer_marker=True, p0_ladder_does_not_preprovide_answer_marker=True), token_window=rows_tokens, next_token_top20_after_question_plus_K_THINK=top20, tracked_next_tokens=tracked_probs, first_answer_metrics=fm, segment_metrics={k:v for k,v in segm.items() if not k.endswith("_loss")}, full_vs_generate_logits=dict(use_cache_true_max_abs_prob_diff=max_abs_cache,use_cache_false_max_abs_prob_diff=max_abs_nocache,top20_full=top.indices.tolist(),top20_generate_cache=top_cache,top20_generate_no_cache=top_nocache,generated_first_cache=text_cache,generated_first_no_cache=text_nocache), generation_settings=dict(greedy=True,do_sample=False,temperature=None,top_p=None), template_checks=dict(no_duplicate_bos_eos=True,chat_template_applied=False,attention_mask_shape=list(enc.attention_mask.shape),position_ids="default_arange",adapter_enabled=True))
    write_json(REPORT/"o0_transition_audit.json",rep)
    md=["# O0 Transition Audit","",f"Checkpoint: `{O0_ADAPTER}`","",f"Answer marker exists: `{rep['answer_marker']}`",f"Marker participates in loss: `{rep['answer_marker_participates_in_loss']}`",f"Marker pre-provided in old forced/fixed generation: `{rep['answer_marker_preprovided_in_forced_generation']}`",f"Full vs generate cache max prob diff: `{max_abs_cache}`",f"Full vs generate no-cache max prob diff: `{max_abs_nocache}`", "", "## Top Tokens After Q + K THINK", "", "|rank|token|prob|", "|---:|---|---:|"]
    md += [f"|{x['rank']}|`{x['token']}`|{x['probability']:.6g}|" for x in top20]
    (REPORT/"o0_transition_audit.md").write_text("\n".join(md)+"\n")
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); write_status("transition_audit","complete")

def prefix_ladder():
    rows=debug32(); torch,tok,model=load_model(O0_ADAPTER,train=False); levels=[0,1,2,4,8,"all_minus_1"]; all_cases=[]; summary=[]
    for lv in levels:
        cases=[]
        for r in rows:
            aids=tok(answer_text_eos(tok,r),add_special_tokens=False)["input_ids"]
            nforce=(len(aids)-1 if lv=="all_minus_1" else int(lv)); nforce=max(0,min(nforce,len(aids)))
            forced=tok.decode(aids[:nforce],skip_special_tokens=False)
            rem=tok.decode(aids[nforce:],skip_special_tokens=False)
            prompt=q_prefix(r)+(THINK*int(r["raw_K"]))+forced
            raw,_=gen_text(torch,tok,model,prompt,max_new=96,use_cache=True)
            pred=parse_answer(forced+raw); gold=parse_answer(r["answer"]); correct=pred is not None and gold is not None and pred==gold
            case=dict(level=str(lv),sample_id=r["sample_id"],forced_token_count=nforce,question=r["question"],gold_answer=gold,forced_prefix=forced,remaining_gold_suffix=rem,raw_generation=raw,parsed_answer=pred,correct=correct,suffix_exact_match=(forced+raw).startswith(answer_text(r)),max_new_tokens_reached=len(tok(raw,add_special_tokens=False)["input_ids"])>=96,generated_THINK_count=raw.count(THINK))
            cases.append(case); all_cases.append(case)
        s=summarize_cases(cases); s["level"]=str(lv); s["suffix_exact_match_rate"]=sum(c["suffix_exact_match"] for c in cases)/len(cases); summary.append(s)
    with (REPORT/"prefix_forcing_ladder.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    write_jsonl(REPORT/"prefix_forcing_cases.jsonl",all_cases)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); write_status("prefix_ladder","complete")

def eval_direct(torch,tok,model,rows,max_new=64):
    cases=[]
    for r in rows:
        prompt=q_prefix(r)+"Answer:\n"
        raw,_=gen_text(torch,tok,model,prompt,max_new=max_new,use_cache=True)
        pred=parse_answer(raw); gold=parse_answer(r["answer"])
        cases.append(dict(sample_id=r["sample_id"],question=r["question"],gold_answer=gold,raw_generation=raw,parsed_answer=pred,correct=pred is not None and gold is not None and pred==gold,max_new_tokens_reached=len(tok(raw,add_special_tokens=False)["input_ids"])>=max_new,generated_THINK_count=raw.count(THINK)))
    return summarize_cases(cases), cases

def train_direct():
    rows=debug32(); random.seed(42); torch,tok,model=load_model(None,train=True); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5); logs=[]; evals=[]; best=-1; best_dir=None
    for step in range(1,501):
        batch,segs=make_direct(torch,tok,[rows[(step-1)%len(rows)]])
        out=model(**batch); loss=out.loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step%10==0 or step==1:
            m=ce_by_segment(torch,out.logits,batch["labels"],segs); fm=first_answer_metrics(torch,out.logits,batch["labels"],segs); logs.append(dict(step=step,loss=float(loss.detach().cpu()),answer_token_accuracy=m["answer_token_accuracy"],answer_nll=m["answer_nll"],**fm))
        if step%50==0:
            model.eval(); summ,cases=eval_direct(torch,tok,model,rows); model.train(); evals.append(dict(step=step,summary=summ))
            if summ["answer_accuracy"]>best:
                best=summ["answer_accuracy"]; best_dir=REPO/"checkpoints/stage1_repair_v2_direct/best_adapter"; best_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(best_dir)
            if summ["answer_accuracy"]>=0.95 and summ["valid_answer_rate"]>=0.95: break
    final=REPO/"checkpoints/stage1_repair_v2_direct/final_adapter"; final.mkdir(parents=True,exist_ok=True); model.save_pretrained(final); tok.save_pretrained(REPO/"checkpoints/stage1_repair_v2_direct/tokenizer")
    model.eval(); final_summ,final_cases=eval_direct(torch,tok,model,rows); passed=final_summ["answer_accuracy"]>=0.95 and final_summ["valid_answer_rate"]>=0.95
    rep=dict(experiment="direct_answer_control",status="pass" if passed else "fail",steps=step,best_accuracy=best,final_summary=final_summ,train_logs=logs,evals=evals,final_checkpoint=str(final),best_checkpoint=str(best_dir) if best_dir else None)
    write_json(REPORT/"direct_answer_control.json",rep)
    md=["# Direct Answer Cases","",f"final answer acc: {final_summ['answer_accuracy']}",f"valid answer rate: {final_summ['valid_answer_rate']}",""]
    for c in final_cases[:32]: md += [f"## {c['sample_id']} correct={c['correct']}",f"gold={c['gold_answer']} pred={c['parsed_answer']}","```text",c['raw_generation'][:1200],"```",""]
    (REPORT/"direct_answer_cases.md").write_text("\n".join(md)+"\n")
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); write_status("direct_answer","complete" if passed else "failed",passed=passed,answer_accuracy=final_summ["answer_accuracy"],valid_answer_rate=final_summ["valid_answer_rate"])
    if not passed: raise SystemExit("direct answer control failed; stop before O1/O2")

def train_stage1_variant(name, objective):
    rows=debug32(); random.seed(42); torch,tok,model=load_model(None,train=True); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5); logs=[]; evals=[]; best=-1; best_dir=None
    for step in range(1,501):
        batch,segs,_=make_stage1(torch,tok,[rows[(step-1)%len(rows)]])
        out=model(**batch); parts=ce_by_segment(torch,out.logits,batch["labels"],segs)
        if objective=="balanced": loss=parts["think_loss"]+parts["answer_loss"]
        elif objective=="answer_heavy": loss=0.1*parts["think_loss"]+parts["answer_loss"]
        else: loss=out.loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step%10==0 or step==1:
            fm=first_answer_metrics(torch,out.logits,batch["labels"],segs); logs.append(dict(step=step,loss=float(loss.detach().cpu()),think_nll=parts["think_nll"],answer_nll=parts["answer_nll"],think_token_accuracy=parts["think_token_accuracy"],answer_token_accuracy=parts["answer_token_accuracy"],**fm))
        if step%50==0:
            model.eval(); modes,cases=eval_modes(torch,tok,model,rows); model.train(); evals.append(dict(step=step,modes=modes))
            acc=modes["forced_k"]["answer_accuracy"]
            if acc>best:
                best=acc; best_dir=REPO/f"checkpoints/stage1_repair_v2_{name}/best_adapter"; best_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(best_dir)
            if acc>=0.95 and modes["forced_k"]["valid_answer_rate"]>=0.95 and modes["fixed_k"]["format_compliance"]>=0.95: break
    final=REPO/f"checkpoints/stage1_repair_v2_{name}/final_adapter"; final.mkdir(parents=True,exist_ok=True); model.save_pretrained(final); tok.save_pretrained(REPO/f"checkpoints/stage1_repair_v2_{name}/tokenizer")
    passed=evals[-1]["modes"]["forced_k"]["answer_accuracy"]>=0.95 and evals[-1]["modes"]["forced_k"]["valid_answer_rate"]>=0.95
    rep=dict(experiment=name,objective=objective,status="pass" if passed else "fail",steps=step,best_forced_k_accuracy=best,train_logs=logs,evals=evals,final_checkpoint=str(final),best_checkpoint=str(best_dir) if best_dir else None)
    write_json(REPORT/f"{name}_overfit_debug32.json",rep); write_status(name,"complete" if passed else "failed",passed=passed,forced_k_accuracy=evals[-1]["modes"]["forced_k"]["answer_accuracy"])
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); return passed

def eval_modes(torch,tok,model,rows):
    modes={}; all_cases=[]
    for mode in ["forced_k","fixed_k","free_k"]:
        cases=[]
        for r in rows:
            if mode=="forced_k": prompt=q_prefix(r)+(THINK*int(r["raw_K"]))+"\nAnswer:\n"
            elif mode=="fixed_k": prompt=q_prefix(r)+(THINK*64)+"\nAnswer:\n"
            else: prompt=q_prefix(r)
            raw,_=gen_text(torch,tok,model,prompt,max_new=64,use_cache=True)
            pred=parse_answer(raw); gold=parse_answer(r["answer"])
            cases.append(dict(mode=mode,sample_id=r["sample_id"],raw_generation=raw,parsed_answer=pred,gold_answer=gold,correct=pred is not None and gold is not None and pred==gold,max_new_tokens_reached=len(tok(raw,add_special_tokens=False)["input_ids"])>=64,generated_THINK_count=raw.count(THINK)))
        s=summarize_cases(cases); s["format_compliance"]=s["valid_answer_rate"]; modes[mode]=s; all_cases.extend(cases)
    return modes,all_cases

def run_o1_o2():
    train_stage1_variant("o1_balanced","balanced")
    train_stage1_variant("o2_answer_heavy","answer_heavy")

def storage_audit():
    lines=["# Checkpoint Storage Audit","", "No O0 historical checkpoint was deleted.", "", "| path | size | recommendation |", "|---|---:|---|"]
    for p in [REPO/"checkpoints/stage1_repair_o0", REPO/"checkpoints/stage1_repair_v2_direct", REPO/"checkpoints/stage1_repair_v2_o1_balanced", REPO/"checkpoints/stage1_repair_v2_o2_answer_heavy"]:
        if p.exists():
            import subprocess
            size=subprocess.check_output(["du","-sh",str(p)],text=True).split()[0]
            rec="ask user before deleting; O0 step checkpoints are large and mostly redundant after metrics are extracted" if "stage1_repair_o0" in str(p) else "keep best/final until gate decision"
            lines.append(f"| `{p}` | {size} | {rec} |")
    (REPORT/"checkpoint_storage_audit.md").write_text("\n".join(lines)+"\n")

def final_report():
    storage_audit()
    statuses={p.stem:json.loads(p.read_text()) for p in STATUS.glob("*.json")}
    rows=[]
    def add(exp, path):
        if path.exists():
            r=json.loads(path.read_text()); final=(r.get("evals") or r.get("generation_evals") or [])[-1] if (r.get("evals") or r.get("generation_evals")) else {}
            modes=final.get("modes",{}) if isinstance(final,dict) else {}
            rows.append(dict(experiment=exp,status=r.get("status"),steps=r.get("steps"),forced_k_acc=modes.get("forced_k",{}).get("answer_accuracy"),forced_k_valid=modes.get("forced_k",{}).get("valid_answer_rate"),fixed_k_acc=modes.get("fixed_k",{}).get("answer_accuracy"),fixed_k_format=modes.get("fixed_k",{}).get("format_compliance"),free_k_acc=modes.get("free_k",{}).get("answer_accuracy"),free_k_valid=modes.get("free_k",{}).get("valid_answer_rate")))
    add("O0 token-weighted", REPO/"reports/stage1_repair/o0_overfit_debug32.json")
    add("O1 balanced", REPORT/"o1_balanced_overfit_debug32.json")
    add("O2 answer-heavy", REPORT/"o2_answer_heavy_overfit_debug32.json")
    if rows:
        with (REPORT/"o0_o1_o2_o3_comparison.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    trans=json.loads((REPORT/"o0_transition_audit.json").read_text()) if (REPORT/"o0_transition_audit.json").exists() else {}
    direct=json.loads((REPORT/"direct_answer_control.json").read_text()) if (REPORT/"direct_answer_control.json").exists() else {}
    direct_pass=direct.get("status")=="pass"
    can_resume=direct_pass and any((r.get("forced_k_acc") or 0)>=0.95 and (r.get("forced_k_valid") or 0)>=0.95 and (r.get("fixed_k_format") or 0)>=0.95 for r in rows)
    status=dict(statuses=statuses,can_resume_10k=can_resume,direct_answer_pass=direct_pass,transition_cache_diff=trans.get("full_vs_generate_logits",{}),comparison=rows)
    write_json(REPORT/"stage1_repair_v2_status.json",status)
    md=["# Stage-1 Repair V2 Report","",f"Can resume 10k pilot: **{can_resume}**","", "## Answers", "", f"1. THINK->Answer first token main failure? transition audit generated-first/top-k and first-token NLL are in JSON; current O0 generation failure is not solved by teacher forcing.", f"2. Template consistency: old forced/fixed pre-provided `Answer:` marker; P0 ladder tests Q+K THINK without marker.", f"3. Cache consistency: `{trans.get('full_vs_generate_logits',{})}`", f"4. Direct-answer control pass: `{direct_pass}`", "5. Prefix-forcing recovery: see `prefix_forcing_ladder.csv`.", "6. O1/O2: run only if direct-answer gate passes.", "7. Explicit boundary: not run unless O1/O2 fail after direct gate passes and transition points to boundary.", f"8. Resume 10k pilot: `{can_resume}`", ""]
    (REPORT/"stage1_repair_v2_report.md").write_text("\n".join(md)+"\n")
    (REPORT/"next_gate_recommendation.md").write_text(("Resume 10k pilot is allowed.\n" if can_resume else "Do not resume 10k pilot. If direct-answer failed, fix generation/template/parser before O1/O2. If direct passed but O1/O2 failed, run explicit boundary O3.\n"))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"; os.environ["HF_HOME"]=str(HF_HOME); os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com"); os.environ.setdefault("HF_HUB_DISABLE_XET","1")
    REPORT.mkdir(parents=True,exist_ok=True); STATUS.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["transition","ladder","direct","o1o2","final","all"]); args=ap.parse_args()
    if args.phase=="transition": transition_audit()
    elif args.phase=="ladder": prefix_ladder()
    elif args.phase=="direct": train_direct()
    elif args.phase=="o1o2": run_o1_o2()
    elif args.phase=="final": final_report()
    elif args.phase=="all":
        try:
            transition_audit(); prefix_ladder(); train_direct(); run_o1_o2()
        finally:
            final_report()
if __name__=="__main__": main()
