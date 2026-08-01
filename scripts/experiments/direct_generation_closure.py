#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gc, json, os, re, subprocess, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get('REPO_ROOT','/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot'))
HF_HOME = Path(os.environ.get('HF_HOME','/weights2/zhouxiaoling/hf_cache'))
MODEL_ID='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
MODEL_REVISION='916b56a44061fd5cd7d6a8fb632557ed4f724f60'
THINK='<THINK>'
MANIFEST=REPO/'data/manifests/stage1_debug32_no_truncation.json'
DIRECT_FINAL=REPO/'checkpoints/stage1_repair_v2_direct/final_adapter'
O0_FINAL=REPO/'checkpoints/stage1_repair_o0/final_adapter'
REPORT=REPO/'reports/direct_generation_closure'
STATUS=REPO/'status/direct_generation_closure'
ANSWER_RE=re.compile(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)')
NUM_RE=re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?')
CHOICE_RE=re.compile(r'\b([A-E])\b', re.I)

def iso(t=None): return datetime.fromtimestamp(t or time.time(), tz=timezone.utc).isoformat()
def write_json(p:Path,o:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def write_jsonl(p:Path,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
def status(name,state,**kw): write_json(STATUS/f'{name}.json',dict(stage=name,status=state,iso=iso(),time=time.time(),**kw))
def norm_ws(s): return re.sub(r'\s+', ' ', str(s or '').replace('<｜end▁of▁sentence｜>','').strip())
def norm_num(s):
    if s is None: return None
    s=str(s).strip().replace(',','')
    if s.endswith('.0'): s=s[:-2]
    return s
def parse_num(text):
    text=str(text or '')
    m=ANSWER_RE.search(text)
    if m: return norm_num(m.group(1))
    nums=NUM_RE.findall(text)
    return norm_num(nums[-1]) if nums else None

def rows(): return json.loads(MANIFEST.read_text())['samples']
def q_prefix(r): return 'Question:\n'+r['question']+'\n\n'
def answer_target(tok,r): return '\nAnswer:\n'+r['answer']+tok.eos_token

def infer_answer_type(r):
    ans=str(r.get('answer','')).strip(); q=str(r.get('question',''))
    if '```' in ans or re.search(r'\bdef\s+\w+\s*\(', ans) or re.search(r'\breturn\b', ans): return 'code'
    if re.fullmatch(r'\s*[A-Ea-e]\s*', ans) or re.search(r'\b(A|B|C|D|E)[\).]', q): return 'multiple_choice'
    if parse_num(ans) is not None and len(NUM_RE.findall(ans)) <= 2 and len(ans) < 80: return 'numeric'
    return 'natural_language'

def task_correct(gold_text, gen_text, typ):
    if typ=='numeric': return parse_num(gold_text)==parse_num(gen_text) and parse_num(gold_text) is not None
    if typ=='multiple_choice':
        g=CHOICE_RE.findall(gold_text); p=CHOICE_RE.findall(gen_text)
        return bool(g and p and g[-1].upper()==p[-1].upper())
    return norm_ws(gold_text)==norm_ws(gen_text)

def load(adapter:Path,train=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),trust_remote_code=True)
    tok.add_special_tokens({'additional_special_tokens':[THINK]})
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),torch_dtype=torch.bfloat16,trust_remote_code=True)
    base.resize_token_embeddings(len(tok)); base.config.use_cache=False
    model=PeftModel.from_pretrained(base,adapter,is_trainable=train)
    model.to('cuda:0'); model.train(train)
    return torch,tok,model

def direct_batch(torch,tok,rs):
    ids_rows=[]; lab_rows=[]; seg_rows=[]; meta=[]
    for r in rs:
        qids=tok(q_prefix(r),add_special_tokens=False)['input_ids']
        aids=tok(answer_target(tok,r),add_special_tokens=False)['input_ids']
        ids=qids+aids; labs=[-100]*len(qids)+aids; seg=['question']*len(qids)+['answer']*len(aids)
        ids_rows.append(ids); lab_rows.append(labs); seg_rows.append(seg)
        meta.append(dict(sample_id=r['sample_id'],prefix_ids=qids,answer_ids=aids,answer_text=answer_target(tok,r)))
    mx=max(map(len,ids_rows)); pad=tok.pad_token_id or tok.eos_token_id
    input_ids=torch.full((len(rs),mx),pad,dtype=torch.long,device='cuda:0'); labels=torch.full((len(rs),mx),-100,dtype=torch.long,device='cuda:0'); attention_mask=torch.zeros_like(input_ids)
    for i,ids in enumerate(ids_rows):
        input_ids[i,:len(ids)]=torch.tensor(ids,device='cuda:0'); labels[i,:len(ids)]=torch.tensor(lab_rows[i],device='cuda:0'); attention_mask[i,:len(ids)]=1; seg_rows[i]+=['pad']*(mx-len(ids))
    return dict(input_ids=input_ids,labels=labels,attention_mask=attention_mask),seg_rows,meta

def stage1_batch(torch,tok,rs):
    tid=tok.convert_tokens_to_ids(THINK); ids_rows=[]; lab_rows=[]; seg_rows=[]; meta=[]
    for r in rs:
        qids=tok(q_prefix(r),add_special_tokens=False)['input_ids']; k=int(r['raw_K']); aids=tok(answer_target(tok,r),add_special_tokens=False)['input_ids']
        ids=qids+[tid]*k+aids; labs=[-100]*len(qids)+[tid]*k+aids; seg=['question']*len(qids)+['think']*k+['answer']*len(aids)
        ids_rows.append(ids); lab_rows.append(labs); seg_rows.append(seg); meta.append(dict(sample_id=r['sample_id'],prefix_ids=qids+[tid]*k,fixed_prefix_ids=qids+[tid]*64,answer_ids=aids,raw_K=k))
    mx=max(map(len,ids_rows)); pad=tok.pad_token_id or tok.eos_token_id
    input_ids=torch.full((len(rs),mx),pad,dtype=torch.long,device='cuda:0'); labels=torch.full((len(rs),mx),-100,dtype=torch.long,device='cuda:0'); attention_mask=torch.zeros_like(input_ids)
    for i,ids in enumerate(ids_rows):
        input_ids[i,:len(ids)]=torch.tensor(ids,device='cuda:0'); labels[i,:len(ids)]=torch.tensor(lab_rows[i],device='cuda:0'); attention_mask[i,:len(ids)]=1; seg_rows[i]+=['pad']*(mx-len(ids))
    return dict(input_ids=input_ids,labels=labels,attention_mask=attention_mask),seg_rows,meta

def decode(tok,ids): return tok.decode(ids,skip_special_tokens=False)
def first_div(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return i
    return None if len(a)==len(b) else min(len(a),len(b))

def gen_ids(torch,tok,model,prefix_ids,max_new,use_cache=False):
    inp=torch.tensor([prefix_ids],device='cuda:0'); attn=torch.ones_like(inp)
    with torch.no_grad(): out=model.generate(input_ids=inp,attention_mask=attn,max_new_tokens=max_new,do_sample=False,num_beams=1,temperature=None,top_p=None,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=use_cache)
    return out[0,inp.shape[1]:].detach().cpu().tolist()

def one_step_info(torch,tok,model,prefix_ids,gold_id):
    inp=torch.tensor([prefix_ids],device='cuda:0'); attn=torch.ones_like(inp)
    with torch.no_grad(): logits=model(input_ids=inp,attention_mask=attn).logits[:,-1,:].float()[0]
    probs=logits.softmax(-1); pred=int(logits.argmax()); rank=int((probs>probs[gold_id]).sum().item()+1); top1=float(probs[pred].detach().cpu()); gp=float(probs[gold_id].detach().cpu())
    return dict(pred_id=pred,pred_token=decode(tok,[pred]),gold_rank=rank,gold_probability=gp,top1_probability=top1,logit_margin=float((logits[pred]-logits[gold_id]).detach().cpu()))

def eval_outputs(torch,tok,model,rs,metas,prefix_key='prefix_ids',use_cache=False):
    cases=[]
    for r,m in zip(rs,metas):
        gold=m['answer_ids']; max_new=len(gold)+8; gen=gen_ids(torch,tok,model,m[prefix_key],max_new,use_cache=use_cache)
        gtrim=gen[:len(gold)]; div=first_div(gtrim,gold); raw=decode(tok,gen); gold_text=decode(tok,gold); typ=infer_answer_type(r)
        token_exact=(gtrim==gold); ntx=(norm_ws(raw)==norm_ws(gold_text)); aware=task_correct(gold_text,raw,typ)
        valid=bool(raw.strip()) and not (len(gen)>=max_new and tok.eos_token_id not in gen)
        cls='correct' if token_exact else 'other'
        if not token_exact:
            if div is not None and div < len(gold):
                gi=gold[div]; pi=gtrim[div] if div<len(gtrim) else None
                if pi==tok.eos_token_id: cls='early_eos'
                elif gi==tok.eos_token_id: cls='missing_eos'
                elif decode(tok,[gi]).strip()==decode(tok,[pi]).strip() if pi is not None else False: cls='whitespace_or_newline_only'
                elif typ=='code' and norm_ws(raw)==norm_ws(gold_text): cls='code_format_only'
                elif aware: cls='semantically_correct_non_exact'
                elif div==0: cls='first_token_error'
                else: cls='interior_token_error'
            elif len(gen)>=max_new and tok.eos_token_id not in gen: cls='missing_eos'
        info=one_step_info(torch,tok,model,m[prefix_key]+gold[:div],gold[div]) if div is not None and div < len(gold) else {}
        cases.append(dict(sample_id=r['sample_id'],source=r.get('source'),answer_type=typ,question=r['question'],gold_answer_text=gold_text,generated_answer_text=raw,gold_token_ids=gold,generated_token_ids=gen,answer_length=len(gold),first_divergence_position=div,gold_divergence_token=None if div is None or div>=len(gold) else decode(tok,[gold[div]]),generated_divergence_token=None if div is None or div>=len(gtrim) else decode(tok,[gtrim[div]]),eos_position=(gen.index(tok.eos_token_id) if tok.eos_token_id in gen else None),stop_reason='eos' if tok.eos_token_id in gen else 'max_new_or_other',max_new_tokens_reached=len(gen)>=max_new and tok.eos_token_id not in gen,token_exact_match=token_exact,normalized_text_exact_match=ntx,answer_type_aware_correctness=aware,valid_generation=valid,failure_class=cls,**info))
    summ=dict(n=len(cases),token_exact=sum(c['token_exact_match'] for c in cases)/len(cases),token_exact_count=sum(c['token_exact_match'] for c in cases),normalized_text_exact=sum(c['normalized_text_exact_match'] for c in cases)/len(cases),normalized_text_exact_count=sum(c['normalized_text_exact_match'] for c in cases),answer_type_aware_accuracy=sum(c['answer_type_aware_correctness'] for c in cases)/len(cases),answer_type_aware_count=sum(c['answer_type_aware_correctness'] for c in cases),valid_generation_rate=sum(c['valid_generation'] for c in cases)/len(cases),failure_counts=dict(Counter(c['failure_class'] for c in cases)),type_counts=dict(Counter(c['answer_type'] for c in cases)))
    return summ,cases

def shifted_acc(torch,model,batch):
    import torch.nn.functional as F
    with torch.no_grad(): out=model(**batch)
    logits=out.logits[:,:-1,:]; labels=batch['labels'][:,1:]; mask=labels.ne(-100); pred=logits.argmax(-1)
    loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),labels.reshape(-1),ignore_index=-100,reduction='mean')
    return float(pred[mask].eq(labels[mask]).float().mean().cpu()), float(loss.detach().cpu())

def shifted_acc_dataset(torch,tok,model,rs,builder):
    vals=[]; losses=[]
    for r in rs:
        batch,_,_=builder(torch,tok,[r])
        acc,loss=shifted_acc(torch,model,batch)
        vals.append(acc); losses.append(loss)
    return sum(vals)/max(len(vals),1), sum(losses)/max(len(losses),1)


def analyze_existing_direct():
    torch,tok,model=load(DIRECT_FINAL,False); rs=rows(); batch,segs,meta=direct_batch(torch,tok,rs); acc,loss=shifted_acc_dataset(torch,tok,model,rs,direct_batch); summ,cases=eval_outputs(torch,tok,model,rs,meta,use_cache=False)
    failures=[c for c in cases if not c['token_exact_match']]
    write_jsonl(REPORT/'direct_failure_cases.jsonl',failures)
    md=['# Direct Failure Analysis','',f'Existing direct final token exact: {summ["token_exact_count"]}/32',f'Shifted teacher-forced acc: {acc}', '', '| sample | class | type | divergence | gold | generated | gold rank | gold prob |', '|---|---|---|---:|---|---|---:|---:|']
    for c in failures:
        md.append(f"| {c['sample_id']} | {c['failure_class']} | {c['answer_type']} | {c['first_divergence_position']} | `{str(c['gold_divergence_token']).replace('|','/')}` | `{str(c['generated_divergence_token']).replace('|','/')}` | {c.get('gold_rank')} | {c.get('gold_probability')} |")
    (REPORT/'direct_failure_analysis.md').write_text('\n'.join(md)+'\n')
    write_json(REPORT/'direct_existing_eval.json',dict(shifted_teacher_forced_accuracy=acc,shifted_loss=loss,summary=summ,cases=cases))
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('existing_direct','complete',token_exact=summ['token_exact'])

def cache_divergence():
    torch,tok,model=load(DIRECT_FINAL,False); rs=rows(); _,_,meta=direct_batch(torch,tok,rs); rows_out=[]
    for r,m in zip(rs,meta):
        gold=m['answer_ids']; max_new=len(gold)+8; nc=gen_ids(torch,tok,model,m['prefix_ids'],max_new,False); ca=gen_ids(torch,tok,model,m['prefix_ids'],max_new,True)
        if nc!=ca:
            div=first_div(nc,ca); rows_out.append(dict(sample_id=r['sample_id'],padding_side=tok.padding_side,input_length=len(m['prefix_ids']),no_cache_text=decode(tok,nc),cache_text=decode(tok,ca),first_divergence_step=div,no_cache_token=None if div is None or div>=len(nc) else decode(tok,[nc[div]]),cache_token=None if div is None or div>=len(ca) else decode(tok,[ca[div]]),single_sample_cache_matches_no_cache=False,batched_not_tested=True,conclusion='single-sample cache/no-cache divergence; validity pipeline uses no-cache'))
    write_json(REPORT/'cache_divergence.json',dict(n_divergent=len(rows_out),divergent_cases=rows_out,validity_pipeline='use_cache=False'))
    md=['# Cache Divergence Audit','',f'Divergent single-sample cases: {len(rows_out)}','', 'Validity pipeline uses `use_cache=False`.']
    (REPORT/'cache_divergence.md').write_text('\n'.join(md)+'\n')
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('cache_divergence','complete',n_divergent=len(rows_out))

def continue_direct_if_needed():
    prior=json.loads((REPORT/'direct_existing_eval.json').read_text()); summ=prior['summary']
    if summ['token_exact_count']>=31 or summ['normalized_text_exact_count']>=31:
        write_json(REPORT/'direct_continuation.json',dict(status='skipped_existing_pass_like',summary=summ,checkpoint=str(DIRECT_FINAL)))
        status('direct_continuation','complete',ran=False,passed=True); return
    torch,tok,model=load(DIRECT_FINAL,True); rs=rows(); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-6)
    best_count=summ['token_exact_count']; best_dir=REPO/'checkpoints/direct_generation_closure_continued/best_adapter'; logs=[]; evals=[]; passed=False
    for step in range(1,201):
        batch,segs,meta=direct_batch(torch,tok,[rs[(step-1)%len(rs)]])
        out=model(**batch); loss=out.loss; loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step%20==0:
            model.eval(); _,_,full_meta=direct_batch(torch,tok,rs); tf_acc,tf_loss=shifted_acc_dataset(torch,tok,model,rs,direct_batch); ev,cases=eval_outputs(torch,tok,model,rs,full_meta,use_cache=False); model.train()
            rec=dict(step=step,shifted_teacher_forced_accuracy=tf_acc,shifted_loss=tf_loss,summary=ev); evals.append(rec)
            if ev['token_exact_count']>best_count:
                best_count=ev['token_exact_count']; best_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(best_dir)
            if tf_acc>=0.99 and ev['token_exact_count']>=31 and ev['normalized_text_exact_count']>=31 and ev['valid_generation_rate']>=0.95:
                passed=True; break
    final_dir=REPO/'checkpoints/direct_generation_closure_continued/final_adapter'; final_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(final_dir); tok.save_pretrained(REPO/'checkpoints/direct_generation_closure_continued/tokenizer')
    model.eval(); _,_,fm=direct_batch(torch,tok,rs); tf_acc,tf_loss=shifted_acc_dataset(torch,tok,model,rs,direct_batch); final_summ,final_cases=eval_outputs(torch,tok,model,rs,fm,use_cache=False)
    clean_ok=False
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    torch2,tok2,model2=load(final_dir,False); _,_,fm2=direct_batch(torch2,tok2,rs); reload_summ,_=eval_outputs(torch2,tok2,model2,rs,fm2,use_cache=False); clean_ok=reload_summ['token_exact_count']==final_summ['token_exact_count']
    del model2; gc.collect(); torch2.cuda.empty_cache(); torch2.cuda.synchronize()
    passed=passed or (tf_acc>=0.99 and final_summ['token_exact_count']>=31 and final_summ['normalized_text_exact_count']>=31 and final_summ['valid_generation_rate']>=0.95 and clean_ok)
    write_json(REPORT/'direct_continuation.json',dict(status='pass' if passed else 'fail',steps=step,lr=5e-6,best_token_exact_count=best_count,final_checkpoint=str(final_dir),best_checkpoint=str(best_dir) if best_dir.exists() else None,final_shifted_teacher_forced_accuracy=tf_acc,final_summary=final_summ,reload_summary=reload_summ,clean_reload_reproducible=clean_ok,evals=evals))
    status('direct_continuation','complete' if passed else 'failed',ran=True,passed=passed,token_exact_count=final_summ['token_exact_count'])

def eval_o0_if_direct_passed():
    direct=json.loads((REPORT/'direct_continuation.json').read_text())
    if direct.get('status') not in ['pass','skipped_existing_pass_like']:
        write_json(REPORT/'o0_repaired_evaluation.json',dict(status='skipped',reason='direct gate failed'))
        with (REPORT/'o0_old_vs_repaired.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['metric','old','repaired','note']); w.writeheader(); w.writerow(dict(metric='status',old='O0 forced token exact 0.125 parsed-ish old',repaired='skipped',note='direct gate failed'))
        return
    torch,tok,model=load(O0_FINAL,False); rs=rows(); batch,segs,meta=stage1_batch(torch,tok,rs); tf_acc,tf_loss=shifted_acc_dataset(torch,tok,model,rs,stage1_batch)
    forced_s,forced_c=eval_outputs(torch,tok,model,rs,meta,'prefix_ids',False)
    fixed_s,fixed_c=eval_outputs(torch,tok,model,rs,meta,'fixed_prefix_ids',False)
    free_cases=[]
    for r in rs:
        prefix=tok(q_prefix(r),add_special_tokens=False)['input_ids']; gen=gen_ids(torch,tok,model,prefix,256,False); raw=decode(tok,gen); free_cases.append(dict(sample_id=r['sample_id'],raw_generation=raw,generated_THINK_count=raw.count(THINK),valid_generation=bool(raw.strip()),stop_success=('<｜end▁of▁sentence｜>' in raw)))
    free_s=dict(valid_generation_rate=sum(c['valid_generation'] for c in free_cases)/len(free_cases),mean_generated_THINK_count=sum(c['generated_THINK_count'] for c in free_cases)/len(free_cases),stop_success_rate=sum(c['stop_success'] for c in free_cases)/len(free_cases))
    rep=dict(status='complete',checkpoint=str(O0_FINAL),shifted_teacher_forced_accuracy=tf_acc,shifted_loss=tf_loss,forced_k=forced_s,fixed_k=fixed_s,free_k=free_s,free_cases=free_cases)
    write_json(REPORT/'o0_repaired_evaluation.json',rep)
    with (REPORT/'o0_old_vs_repaired.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['metric','old','repaired','note']); w.writeheader();
        w.writerow(dict(metric='forced_k_token_exact',old='0.125 parsed answer acc old',repaired=forced_s['token_exact'],note='old used noncanonical/parser-style metric'))
        w.writerow(dict(metric='fixed_k_token_exact',old='0.09375 parsed answer acc old',repaired=fixed_s['token_exact'],note='canonical prefix + task-aware metric'))
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('o0_reeval','complete')

def storage_plan():
    lines=['# Checkpoint Retention Plan','','No checkpoints were deleted.','','| path | size | role | recommendation |','|---|---:|---|---|']
    paths=[REPO/'checkpoints/stage1_repair_o0/final_adapter',REPO/'checkpoints/stage1_repair_o0/step50',REPO/'checkpoints/stage1_repair_o0/step250',REPO/'checkpoints/stage1_repair_o0/step100',REPO/'checkpoints/stage1_repair_o0/step150',REPO/'checkpoints/stage1_repair_o0/step200',REPO/'checkpoints/stage1_repair_o0/step300',REPO/'checkpoints/stage1_repair_o0/step350',REPO/'checkpoints/stage1_repair_o0/step400',REPO/'checkpoints/stage1_repair_o0/step450',REPO/'checkpoints/stage1_repair_o0/step500',REPO/'checkpoints/stage1_repair_v2_direct/best_adapter',REPO/'checkpoints/stage1_repair_v2_direct/final_adapter',REPO/'checkpoints/direct_generation_closure_continued/best_adapter',REPO/'checkpoints/direct_generation_closure_continued/final_adapter']
    reclaim=0
    for p in paths:
        if not p.exists(): continue
        size=subprocess.check_output(['du','-sh',str(p)],text=True).split()[0]
        keep=('final_adapter' in str(p) or 'step50' in str(p) or 'step250' in str(p) or 'best_adapter' in str(p))
        rec='keep' if keep else 'candidate delete after user approval'
        lines.append(f'| `{p}` | {size} | checkpoint | {rec} |')
    (REPORT/'checkpoint_retention_plan.md').write_text('\n'.join(lines)+'\n')

def final_gate():
    existing=json.loads((REPORT/'direct_existing_eval.json').read_text())
    cont=json.loads((REPORT/'direct_continuation.json').read_text())
    cache=json.loads((REPORT/'cache_divergence.json').read_text())
    direct_pass=cont.get('status') in ['pass','skipped_existing_pass_like']
    o0=json.loads((REPORT/'o0_repaired_evaluation.json').read_text())
    if not direct_pass: decision='CASE 1: Direct gate failed; keep O1/O2, 10k pilot, and Stage2 forbidden.'
    elif o0.get('status')=='complete' and o0['forced_k']['token_exact']>=0.95: decision='CASE 2: Direct gate and O0 Forced-K pass; O1/O2 not needed, next can be small M0 pilot.'
    elif o0.get('status')=='complete': decision='CASE 3: Direct gate passed but O0 Forced-K failed; next may run O1/O2.'
    else: decision='CASE 1: Direct gate status blocks O0.'
    gate=dict(direct_existing=existing['summary'],direct_continuation=cont,o0_repaired_status=o0,cache_divergence=cache,direct_gate_passed=direct_pass,may_run_o1_o2=direct_pass and o0.get('status')=='complete' and o0['forced_k']['token_exact']<0.95,may_resume_small_m0=direct_pass and o0.get('status')=='complete' and o0['forced_k']['token_exact']>=0.95,ten_k_and_stage2_forbidden=True,decision=decision)
    write_json(REPORT/'direct_gate_status.json',gate); storage_plan()
    md=['# Direct Generation Closure Report','',f'Decision: {decision}','', '## Direct Failures', '']
    failures=[json.loads(x) for x in (REPORT/'direct_failure_cases.jsonl').read_text().splitlines() if x.strip()]
    md.append(f'Existing direct final failed {len(failures)} token-exact cases; see `direct_failure_cases.jsonl`.')
    md += ['', '## Answers', '', f'1. direct 剩余 3 条为什么失败？ See failure classes: `{dict(Counter(f["failure_class"] for f in failures))}`.', '2. 它们是真错误还是格式等价？ See token/normalized/type-aware fields per case; continuation decision uses token+normalized exact.', f'3. cache divergence 根因是什么？ `{cache.get("n_divergent")}` single-sample cache/no-cache divergent cases; validity uses no-cache.', f'4. direct gate 是否通过？ `{direct_pass}`.', f'5. 修复后的 O0 Forced-K 是否通过？ `{o0.get("forced_k",{}).get("token_exact") if o0.get("status")=="complete" else "skipped"}`.', f'6. 下一步？ {decision}', '7. 10k pilot 和 Stage2 仍禁止。']
    (REPORT/'direct_gate_report.md').write_text('\n'.join(md)+'\n')
    (REPORT/'next_experiment_recommendation.md').write_text(decision+'\n')
    status('final','complete')

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='1'; os.environ['HF_HOME']=str(HF_HOME); os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com'); os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    REPORT.mkdir(parents=True,exist_ok=True); STATUS.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser(); ap.add_argument('phase',choices=['existing','cache','continue_direct','o0','final','all']); args=ap.parse_args()
    if args.phase=='existing': analyze_existing_direct()
    elif args.phase=='cache': cache_divergence()
    elif args.phase=='continue_direct': continue_direct_if_needed()
    elif args.phase=='o0': eval_o0_if_direct_passed()
    elif args.phase=='final': final_gate()
    elif args.phase=='all':
        analyze_existing_direct(); cache_divergence(); continue_direct_if_needed(); eval_o0_if_direct_passed(); final_gate()
if __name__=='__main__': main()
