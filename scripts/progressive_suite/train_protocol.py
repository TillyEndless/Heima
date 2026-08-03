#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gc, hashlib, json, os, random, re, subprocess, time, traceback
from pathlib import Path
REPO=Path(os.environ.get("REPO_ROOT", Path.cwd()))
ASSET_ROOT=Path(os.environ.get("ASSET_ROOT","/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b"))
HF_HOME=Path(os.environ.get("HF_HOME","/weights2/zhouxiaoling/hf_cache"))
MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"; MODEL_REVISION="916b56a44061fd5cd7d6a8fb632557ed4f724f60"
TOKENS=["<THINK>","<THINK_1>","<THINK_2>","<THINK_3>","<THINK_4>","<THINK_END>","<ANSWER>"]; THINK,THINK_END,ANSWER=TOKENS[0],TOKENS[5],TOKENS[6]
OUT=REPO/"reports/progressive_suite"; RUNS=REPO/"runs/progressive_suite"; CKPT=REPO/"checkpoints/progressive_suite"; STATUS=REPO/"status/progressive_suite"
ANSWER_RE=re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)"); NUM_RE=re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")
def wj(p,o):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    tmp.replace(p)
def wjl(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+f".{os.getpid()}.tmp")
    tmp.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in rows))
    tmp.replace(p)
def stat(name,state,**kw): wj(STATUS/f"{name}.json",dict(time=time.time(),stage=name,status=state,**kw))
def git_sha():
    try: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    except Exception: return "missing"
def parse_answer(text):
    text=str(text or ""); m=ANSWER_RE.search(text)
    if m: return m.group(1).replace(",","").removesuffix(".0")
    nums=NUM_RE.findall(text); return nums[-1].replace(",","").removesuffix(".0") if nums else None
def correct(gold,gen):
    g,p=parse_answer(gold),parse_answer(gen); return bool(g is not None and p is not None and g==p)
def q_prefix(r): return "Question:\n"+r["question"].strip()+"\n\n"
def answer_text(tok,r): return "\nAnswer:\n"+r["answer"].strip()+tok.eos_token
def load_raw_rows():
    d32=REPO/"data/manifests/stage1_debug32_no_truncation.json"; smoke=ASSET_ROOT/"data/processed_large/qwen7b_stage1/smoke1k.json"
    rows=json.loads(smoke.read_text()) if smoke.exists() else (json.loads(d32.read_text()).get("samples",[]) if d32.exists() else [])
    out=[]
    for i,r in enumerate(rows):
        q=r.get("question",""); cot=r.get("gold_cot") or r.get("cot") or r.get("think_content",""); ans=r.get("answer") or r.get("reference_answer","")
        if q and cot and ans: out.append(dict(r,sample_id=str(r.get("sample_id",r.get("id",i))),question=str(q),gold_cot=str(cot),answer=str(ans)))
    return out
def tok_only():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),trust_remote_code=True); tok.add_special_tokens({"additional_special_tokens":TOKENS})
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    return tok
def split_chunks(tok,cot):
    ids=tok(cot,add_special_tokens=False)["input_ids"]
    if len(ids)<4: return [cot," "," "," "]
    cuts=[round(len(ids)*i/4) for i in range(5)]
    return [tok.decode(ids[cuts[i]:cuts[i+1]],skip_special_tokens=False).strip() or " " for i in range(4)]
def prepare_data():
    tok=tok_only(); rows=load_raw_rows(); random.Random(42).shuffle(rows)
    for r in rows:
        r["cot_chunks"]=split_chunks(tok,r["gold_cot"]); r["q_token_count"]=len(tok(q_prefix(r),add_special_tokens=False)["input_ids"]); r["cot_token_count"]=len(tok(r["gold_cot"],add_special_tokens=False)["input_ids"]); r["answer_token_count"]=len(tok(answer_text(tok,r),add_special_tokens=False)["input_ids"]); r["raw_K"]=max(1,round(0.5*r["cot_token_count"])) if r["cot_token_count"] else 0
    max_train_tokens=int(os.environ.get("PROGRESSIVE_MAX_SEQUENCE_TOKENS","1024"))
    filtered=[]; dropped_long=0
    for r in rows:
        if r["q_token_count"] + r["cot_token_count"] + r["answer_token_count"] + 8 <= max_train_tokens:
            filtered.append(r)
        else:
            dropped_long += 1
    rows=filtered
    scales={"D32":(32,32),"S2K":(2048,512),"S10K":(10000,1000),"S50K":(50000,2000)}; manifest={"source_rows":len(load_raw_rows()),"filtered_rows":len(rows),"dropped_too_long":dropped_long,"max_sequence_tokens":max_train_tokens,"model_id":MODEL_ID,"model_revision":MODEL_REVISION,"git_sha":git_sha(),"tokens":TOKENS,"scales":{}}
    for s,(tr,ev) in scales.items():
        ok=len(rows)>=tr+ev; train=rows[:tr] if ok else (rows[:32] if s=="D32" and len(rows)>=32 else []); eval_=rows[tr:tr+ev] if ok else (rows[:32] if s=="D32" and len(rows)>=32 else [])
        path=RUNS/f"{s}/data_split.json"; wj(path,{"train":train,"eval":eval_})
        manifest["scales"][s]={"requested_train":tr,"requested_eval":ev,"actual_train":len(train),"actual_eval":len(eval_),"usable":bool(train and eval_ and (ok or s=="D32")),"status":"ok" if ok or s=="D32" else "DATA_INSUFFICIENT","split_path":str(path),"split_hash":hashlib.sha256(path.read_bytes()).hexdigest()}
    wj(OUT/"data_manifest.json",manifest); return manifest
def env_report():
    import torch, transformers, peft, accelerate
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip()
    wj(OUT/"environment.json",dict(git_sha=git_sha(),python=os.sys.version,torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,accelerate=accelerate.__version__,cuda=torch.version.cuda,gpu=gpu,model_id=MODEL_ID,model_revision=MODEL_REVISION))
def load_model(adapter=None, train=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),trust_remote_code=True); tok.add_special_tokens({"additional_special_tokens":TOKENS})
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),torch_dtype=torch.bfloat16,trust_remote_code=True); base.resize_token_embeddings(len(tok)); base.config.use_cache=False; base.gradient_checkpointing_enable()
    if adapter: model=PeftModel.from_pretrained(base,adapter,is_trainable=train)
    else:
        cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],modules_to_save=["embed_tokens","lm_head"]); model=get_peft_model(base,cfg)
    model.to("cuda:0"); model.train(train); return torch,tok,model
def tid_list(tok,repeated=False): return [tok.convert_tokens_to_ids(THINK)]*4 if repeated else [tok.convert_tokens_to_ids(f"<THINK_{i}>") for i in range(1,5)]
def make_sequence(tok,r,protocol,stage):
    repeated="REPEATED" in protocol or protocol.startswith("FIXED"); no_ntp="NO_LATENT_NTP" in protocol; tids=tid_list(tok,repeated); end=tok.convert_tokens_to_ids(THINK_END); ansm=tok.convert_tokens_to_ids(ANSWER)
    q=tok(q_prefix(r),add_special_tokens=False)["input_ids"]; ans=tok(answer_text(tok,r),add_special_tokens=False)["input_ids"]; chunks=[tok(c,add_special_tokens=False)["input_ids"] for c in r["cot_chunks"]]
    if protocol.startswith("FIXED"):
        k=int(protocol.split("K")[-1]); ids=q+[tok.convert_tokens_to_ids(THINK)]*k+[end,ansm]+ans; seg=["question"]*len(q)+["think"]*k+["boundary"]*2+["answer"]*len(ans)
    elif "ONESHOT" in protocol:
        ids=q+tids+[end,ansm]+ans; seg=["question"]*len(q)+["think"]*4+["boundary"]*2+["answer"]*len(ans)
    else:
        s=int(stage)
        if s==0: ids=q+sum(chunks,[])+ans; seg=["question"]*len(q)+sum(([f"text{i+1}"]*len(chunks[i]) for i in range(4)),[])+["answer"]*len(ans)
        elif s in [1,2,3]: ids=q+tids[:s]+sum(chunks[s:],[])+ans; seg=["question"]*len(q)+["think"]*s+sum(([f"text{i+1}"]*len(chunks[i]) for i in range(s,4)),[])+["answer"]*len(ans)
        else: ids=q+tids+[end,ansm]+ans; seg=["question"]*len(q)+["think"]*4+["boundary"]*2+["answer"]*len(ans)
    labs=[-100 if sg=="question" or (sg=="think" and no_ntp) else tid for tid,sg in zip(ids,seg)]
    return ids,labs,seg
def make_batch(torch,tok,rows,protocol,stage):
    seq=[make_sequence(tok,r,protocol,stage) for r in rows]; mx=max(len(x[0]) for x in seq); pad=tok.pad_token_id or tok.eos_token_id
    inp=torch.full((len(rows),mx),pad,dtype=torch.long,device="cuda:0"); lab=torch.full_like(inp,-100); att=torch.zeros_like(inp); segs=[]
    for i,(ids,labs,seg) in enumerate(seq): inp[i,:len(ids)]=torch.tensor(ids,device="cuda:0"); lab[i,:len(ids)]=torch.tensor(labs,device="cuda:0"); att[i,:len(ids)]=1; segs.append(seg+["pad"]*(mx-len(ids)))
    return dict(input_ids=inp,labels=lab,attention_mask=att),segs
def metrics(torch,logits,labels,segs):
    import torch.nn.functional as F
    sl=labels[:,1:]; lg=logits[:,:-1,:].contiguous(); pred=lg.argmax(-1); losses=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),sl.reshape(-1),ignore_index=-100,reduction="none").reshape_as(sl); out={}
    for name in ["think","text","boundary","answer"]:
        mr=[]
        for row in segs: mr.append([(s=="think" if name=="think" else s.startswith("text") if name=="text" else s==name) for s in row[1:]])
        m=torch.tensor(mr,dtype=torch.bool,device=sl.device)&sl.ne(-100); out[name+"_count"]=int(m.sum().cpu()); out[name+"_nll"]=float(losses[m].mean().detach().cpu()) if m.any() else None; out[name+"_acc"]=float(pred[m].eq(sl[m]).float().mean().detach().cpu()) if m.any() else None
    return out,losses,sl
def eval_model(torch,tok,model,rows,protocol,stage):
    model.eval(); batch,segs=make_batch(torch,tok,rows,protocol,stage)
    with torch.no_grad(): out=model(**batch)
    met,_,_=metrics(torch,out.logits,batch["labels"],segs); acc=valid=boundary=term=0; gens=[]
    for r in rows[:min(32,len(rows))]:
        ids,labs,seg=make_sequence(tok,r,protocol,stage); start=seg.index("answer") if "answer" in seg else len(ids); prefix=ids[:start]; gold=tok.decode(ids[start:],skip_special_tokens=False)
        inp=torch.tensor([prefix],device="cuda:0"); att=torch.ones_like(inp)
        with torch.no_grad(): gen=model.generate(input_ids=inp,attention_mask=att,max_new_tokens=min(128,len(ids[start:])+16),do_sample=False,num_beams=1,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=False)
        txt=tok.decode(gen[0,inp.shape[1]:].detach().cpu().tolist(),skip_special_tokens=False); c=correct(gold,txt); acc+=c; valid+=bool(txt.strip()); boundary+=(THINK_END in txt or ANSWER in txt); term+=(tok.eos_token in txt); gens.append({"sample_id":r["sample_id"],"gold":gold,"generation":txt,"correct":c})
    n=max(1,len(gens)); met.update(type_aware_answer_accuracy=acc/n,valid_rate=valid/n,boundary_success=boundary/n,termination_success=term/n); return met,gens
def train_job(protocol,scale,steps=None):
    manifest=prepare_data(); sc=manifest["scales"].get(scale,{}); job=f"{protocol}_{scale}"; stat(job,"starting")
    if not sc.get("usable"):
        reason=sc.get("status","DATA_MISSING"); wj(OUT/f"{job}_failure.json",{"status":"FAILED","reason":reason,"scale":sc}); stat(job,"failed",reason=reason); return 2
    data=json.loads(Path(sc["split_path"]).read_text()); train=data["train"]; ev=data["eval"]; steps=steps or (600 if scale=="D32" else min(2000,len(train)))
    torch,tok,model=load_model(None,True); random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42); params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=2e-5)
    plan=[0,1,2,3,4,4] if (protocol.startswith("P1") or protocol.startswith("P3") or protocol.startswith("P5")) else [4]; per=max(1,steps//len(plan)); logs=[]; evals=[]; best=(-1,-1); best_dir=CKPT/job/"best_adapter"; final_dir=CKPT/job/"final_adapter"; step=0; t0=time.perf_counter()
    try:
      for stg in plan:
        for _ in range(per):
          step+=1; row=train[(step-1)%len(train)]; batch,segs=make_batch(torch,tok,[row],protocol,stg); out=model(**batch); met,losses,sl=metrics(torch,out.logits,batch["labels"],segs); loss=0
          for name,w in [("think",0.1),("text",1.0),("boundary",1.0),("answer",1.0)]:
            mr=[]
            for rowseg in segs: mr.append([(s=="think" if name=="think" else s.startswith("text") if name=="text" else s==name) for s in rowseg[1:]])
            m=torch.tensor(mr,dtype=torch.bool,device=sl.device)&sl.ne(-100)
            if m.any(): loss=loss+w*losses[m].mean()
          loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad(set_to_none=True); rec={"step":step,"stage":stg,"total_loss":float(loss.detach().cpu()),**met}; logs.append(rec)
          if step%10==0: wjl(OUT/f"{job}_train.partial.jsonl",logs); stat(job,"running",step=step,stage_id=stg,total_loss=rec["total_loss"])
          if step%25==0:
            e,g=eval_model(torch,tok,model,ev,protocol,stg); e.update(step=step,stage=stg); evals.append(e); score=(e.get("type_aware_answer_accuracy") or 0,e.get("valid_rate") or 0)
            if score>best: best=score; best_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(best_dir); tok.save_pretrained(CKPT/job/"tokenizer"); wjl(OUT/f"{job}_best_generations.jsonl",g)
            model.train(True)
      final_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(final_dir); tok.save_pretrained(CKPT/job/"tokenizer"); e,g=eval_model(torch,tok,model,ev,protocol,plan[-1]); wjl(OUT/f"{job}_generations.jsonl",g)
      rep={"job":job,"protocol":protocol,"scale":scale,"status":"complete","steps":step,"runtime_seconds":time.perf_counter()-t0,"best_score":best,"final_eval":e,"evals":evals,"best_checkpoint":str(best_dir),"final_checkpoint":str(final_dir),"peak_gpu_memory_allocated":int(torch.cuda.max_memory_allocated())}; wj(OUT/f"{job}.json",rep); wjl(OUT/f"{job}_train.jsonl",logs); stat(job,"complete",step=step,best_score=best); return 0
    except Exception as e:
      wj(OUT/f"{job}_failure.json",{"job":job,"status":"FAILED","reason":repr(e),"traceback":traceback.format_exc()}); stat(job,"failed",reason=repr(e)); return 1
    finally:
      try: del model
      except Exception: pass
      gc.collect(); torch.cuda.empty_cache()
def aggregate():
    rows=[]
    for p in sorted(OUT.glob("P*.json"))+sorted(OUT.glob("FIXED*.json")):
      if p.name.endswith("failure.json"): continue
      o=json.loads(p.read_text()); f=o.get("final_eval",{}); rows.append({"job":o.get("job"),"protocol":o.get("protocol"),"scale":o.get("scale"),"steps":o.get("steps"),"answer_acc":f.get("type_aware_answer_accuracy"),"valid":f.get("valid_rate"),"boundary":f.get("boundary_success"),"runtime":o.get("runtime_seconds")})
    wj(OUT/"all_metrics.json",rows)
    if rows:
      with (OUT/"all_metrics.csv").open("w",newline="") as f: cw=csv.DictWriter(f,fieldnames=list(rows[0].keys())); cw.writeheader(); cw.writerows(rows)
    (OUT/"final_recommendation.md").write_text("# Progressive Suite Summary\n\n"+json.dumps(rows,indent=2)+"\n")
def audit(): env_report(); prepare_data(); wj(OUT/"existing_results_audit.json",{"o012":str(REPO/"reports/stage1_loss_weighting/o0_o1_o2_unified_eval.csv"),"o3":str(REPO/"reports/stage1_boundary/o3_best_final_comparison.csv")}); (OUT/"existing_results_audit.md").write_text("# Existing Results Audit\n\nO0/O1/O2/O3 found in existing reports; this suite does not rerun them verbatim.\n")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("cmd",choices=["audit","train","aggregate","smoke"]); ap.add_argument("--protocol",default="P1_PROGRESSIVE_4_TYPED"); ap.add_argument("--scale",default="D32"); ap.add_argument("--steps",type=int); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True); STATUS.mkdir(parents=True,exist_ok=True)
    if a.cmd=="audit": audit()
    elif a.cmd=="train": raise SystemExit(train_job(a.protocol,a.scale,a.steps))
    elif a.cmd=="smoke": audit(); raise SystemExit(train_job(a.protocol,"D32",10))
    elif a.cmd=="aggregate": aggregate()
if __name__=="__main__": main()
