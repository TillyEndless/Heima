#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gc, json, os, random, re, subprocess, time
from collections import Counter
from pathlib import Path
from typing import Any

REPO=Path(os.environ.get('REPO_ROOT','/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot'))
HF_HOME=Path(os.environ.get('HF_HOME','/weights2/zhouxiaoling/hf_cache'))
MODEL_ID='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
MODEL_REVISION='916b56a44061fd5cd7d6a8fb632557ed4f724f60'
THINK='<THINK>'; THINK_END='<THINK_END>'; ANSWER='<ANSWER>'
MANIFEST=REPO/'data/manifests/stage1_debug32_no_truncation.json'
REPORT=REPO/'reports/stage1_boundary'; STATUS=REPO/'status/stage1_boundary'
MAX_STEPS=500; LR=2e-5; SEED=42; EVAL_EVERY=25
ANSWER_RE=re.compile(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)')
NUM_RE=re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?')
CHOICE_RE=re.compile(r'\b([A-E])\b', re.I)

def write_json(p:Path,o:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def write_jsonl(p:Path,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
def status(name,state,**kw): write_json(STATUS/f'{name}.json',dict(stage=name,status=state,time=time.time(),**kw))
def git_sha():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    except Exception: return 'missing'
def rows(): return json.loads(MANIFEST.read_text())['samples']
def q_prefix(r): return 'Question:\n'+r['question']+'\n\n'
def answer_text(tok,r): return r['answer']+tok.eos_token
def norm_ws(s): return re.sub(r'\s+',' ',str(s or '').replace('<｜end▁of▁sentence｜>','').strip())
def norm_num(s):
    if s is None: return None
    s=str(s).strip().replace(',','')
    return s[:-2] if s.endswith('.0') else s
def parse_num(text):
    text=str(text or '')
    m=ANSWER_RE.search(text)
    if m: return norm_num(m.group(1))
    nums=NUM_RE.findall(text)
    return norm_num(nums[-1]) if nums else None
def infer_type(r):
    ans=str(r.get('answer','')).strip(); q=str(r.get('question',''))
    if '```' in ans or re.search(r'\bdef\s+\w+\s*\(',ans) or re.search(r'\breturn\b',ans): return 'code'
    if re.fullmatch(r'\s*[A-Ea-e]\s*',ans) or re.search(r'\b(A|B|C|D|E)[\).]',q): return 'multiple_choice'
    if parse_num(ans) is not None and len(NUM_RE.findall(ans))<=2 and len(ans)<80: return 'numeric'
    return 'natural_language'
def task_correct(gold,gen,typ):
    if typ=='numeric':
        g,p=parse_num(gold),parse_num(gen); return g is not None and p is not None and g==p
    if typ=='multiple_choice':
        g,p=CHOICE_RE.findall(gold),CHOICE_RE.findall(gen); return bool(g and p and g[-1].upper()==p[-1].upper())
    return norm_ws(gold)==norm_ws(gen)

def load_model(adapter:Path|None=None, train=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),trust_remote_code=True)
    tok.add_special_tokens({'additional_special_tokens':[THINK,THINK_END,ANSWER]})
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),torch_dtype=torch.bfloat16,trust_remote_code=True)
    base.resize_token_embeddings(len(tok)); base.config.use_cache=False; base.gradient_checkpointing_enable()
    if adapter:
        model=PeftModel.from_pretrained(base,adapter,is_trainable=train)
    else:
        cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],modules_to_save=['embed_tokens','lm_head'])
        model=get_peft_model(base,cfg)
    model.to('cuda:0'); model.train(train)
    return torch,tok,model

def ids_for(tok,r):
    tids=[tok.convert_tokens_to_ids(x) for x in [THINK,THINK_END,ANSWER]]
    q=tok(q_prefix(r),add_special_tokens=False)['input_ids']; k=int(r['raw_K']); ans=tok(answer_text(tok,r),add_special_tokens=False)['input_ids']
    return q,k,ans,tids

def make_batch(torch,tok,rs):
    rows_ids=[]; rows_lab=[]; rows_seg=[]; metas=[]
    think_id,end_id,answer_id=[tok.convert_tokens_to_ids(x) for x in [THINK,THINK_END,ANSWER]]
    for r in rs:
        q,k,ans,_=ids_for(tok,r)
        ids=q+[think_id]*k+[end_id,answer_id]+ans
        labs=[-100]*len(q)+[think_id]*k+[end_id,answer_id]+ans
        seg=['question']*len(q)+['think']*k+['boundary_end','boundary_answer']+['answer']*len(ans)
        rows_ids.append(ids); rows_lab.append(labs); rows_seg.append(seg)
        metas.append(dict(sample_id=r['sample_id'],q_len=len(q),raw_K=k,answer_len=len(ans),prefix_d1=q+[think_id]*k+[end_id,answer_id],prefix_d2=q+[think_id]*k,prefix_free=q,boundary_ids=[end_id,answer_id],answer_ids=ans,full_target_ids=[end_id,answer_id]+ans))
    mx=max(map(len,rows_ids)); pad=tok.pad_token_id or tok.eos_token_id
    input_ids=torch.full((len(rs),mx),pad,dtype=torch.long,device='cuda:0'); labels=torch.full((len(rs),mx),-100,dtype=torch.long,device='cuda:0'); attention_mask=torch.zeros_like(input_ids)
    for i,ids in enumerate(rows_ids):
        input_ids[i,:len(ids)]=torch.tensor(ids,device='cuda:0'); labels[i,:len(ids)]=torch.tensor(rows_lab[i],device='cuda:0'); attention_mask[i,:len(ids)]=1; rows_seg[i]+=['pad']*(mx-len(ids))
    return dict(input_ids=input_ids,labels=labels,attention_mask=attention_mask), rows_seg, metas

def shifted_parts(torch,logits,labels,segs):
    import torch.nn.functional as F
    sl=labels[:,1:].contiguous(); lg=logits[:,:-1,:].contiguous(); loss=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),sl.reshape(-1),ignore_index=-100,reduction='none').reshape_as(sl); pred=lg.argmax(-1); out={}
    for name,match in [('think',lambda s:s=='think'),('boundary',lambda s:s in ['boundary_end','boundary_answer']),('answer',lambda s:s=='answer'),('boundary_end',lambda s:s=='boundary_end'),('answer_marker',lambda s:s=='boundary_answer')]:
        mask=torch.tensor([[match(s) for s in row[1:]] for row in segs],dtype=torch.bool,device=labels.device)&sl.ne(-100)
        if mask.any():
            out[name+'_loss']=loss[mask].mean(); out[name+'_mean_loss']=float(loss[mask].mean().detach().cpu()); out[name+'_accuracy']=float(pred[mask].eq(sl[mask]).float().mean().detach().cpu()); out[name+'_count']=int(mask.sum().detach().cpu())
        else:
            out[name+'_loss']=loss.sum()*0; out[name+'_mean_loss']=None; out[name+'_accuracy']=None; out[name+'_count']=0
    return out

def token_position_metrics(torch,tok,logits,labels,segs):
    import torch.nn.functional as F
    sl=labels[:,1:]; lg=logits[:,:-1,:]
    out={}
    names={'first_boundary_token':lambda row:[i for i,s in enumerate(row[1:]) if s=='boundary_end'][:1], 'answer_marker':lambda row:[i for i,s in enumerate(row[1:]) if s=='boundary_answer'][:1], 'first_answer_token':lambda row:[i for i,s in enumerate(row[1:]) if s=='answer'][:1], 'eos':lambda row:[i for i,s in enumerate(row[1:]) if s=='answer'][-1:]}
    for name,fn in names.items():
        losses=[]; ok=0; n=0
        for bi,row in enumerate(segs):
            for j in fn(row):
                if int(sl[bi,j])==-100: continue
                losses.append(float(F.cross_entropy(lg[bi,j:j+1].float(),sl[bi,j:j+1],reduction='none')[0].detach().cpu()))
                ok+=int(int(lg[bi,j].argmax())==int(sl[bi,j])); n+=1
        out[name+'_loss']=sum(losses)/max(len(losses),1); out[name+'_accuracy']=ok/max(n,1)
    return out

def grad_norm_from_backward(torch,model,loss,params):
    loss.backward(retain_graph=True)
    acc=None
    for p in params:
        if p.grad is None: continue
        v=p.grad.detach().float().pow(2).sum(); acc=v if acc is None else acc+v
    model.zero_grad(set_to_none=True)
    return float(torch.sqrt(acc).detach().cpu()) if acc is not None else 0.0

def decode(tok,ids): return tok.decode(ids,skip_special_tokens=False)
def first_div(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return i
    return None if len(a)==len(b) else min(len(a),len(b))
def gen(torch,tok,model,prefix,max_new):
    inp=torch.tensor([prefix],device='cuda:0'); attn=torch.ones_like(inp)
    with torch.no_grad(): out=model.generate(input_ids=inp,attention_mask=attn,max_new_tokens=max_new,do_sample=False,num_beams=1,temperature=None,top_p=None,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=False)
    return out[0,inp.shape[1]:].detach().cpu().tolist()

def summarize_answer(tok,r,gold_ans,generated_ans):
    raw=decode(tok,generated_ans); gold=decode(tok,gold_ans); typ=infer_type(r); trim=generated_ans[:len(gold_ans)]
    return dict(token_exact=trim==gold_ans,normalized_exact=norm_ws(raw)==norm_ws(gold),type_aware=task_correct(gold,raw,typ),valid=bool(raw.strip()) and not (len(generated_ans)>=len(gold_ans)+8 and tok.eos_token_id not in generated_ans),eos_success=(tok.eos_token_id in generated_ans),stop_success=(tok.eos_token_id in generated_ans),raw_generation=raw)

def eval_o3(torch,tok,model,rs,include_free=True):
    _,_,metas=make_batch(torch,tok,rs); d1=[]; d2=[]; d3=[]
    for r,m in zip(rs,metas):
        g1=gen(torch,tok,model,m['prefix_d1'],len(m['answer_ids'])+8); s1=summarize_answer(tok,r,m['answer_ids'],g1); d1.append(s1)
        g2=gen(torch,tok,model,m['prefix_d2'],len(m['full_target_ids'])+8); boundary_ok=g2[:2]==m['boundary_ids']; ans_part=g2[2:] if boundary_ok else g2[2:]
        s2=summarize_answer(tok,r,m['answer_ids'],ans_part); s2.update(boundary_sequence_exact=boundary_ok,think_end_first_token_accuracy=(len(g2)>0 and g2[0]==m['boundary_ids'][0]),answer_marker_accuracy=(len(g2)>1 and g2[1]==m['boundary_ids'][1])); d2.append(s2)
        if include_free:
            gf=gen(torch,tok,model,m['prefix_free'],256); raw=decode(tok,gf); think_id,end_id,answer_id=[tok.convert_tokens_to_ids(x) for x in [THINK,THINK_END,ANSWER]]
            term=end_id in gf; marker=answer_id in gf; boundary_seq=any(gf[i:i+2]==[end_id,answer_id] for i in range(max(0,len(gf)-1)))
            d3.append(dict(raw_generation=raw,think_started=(len(gf)>0 and gf[0]==think_id),generated_THINK_count=sum(x==think_id for x in gf),think_termination=term,boundary_sequence_accuracy=boundary_seq,valid_generation=bool(raw.strip()),malformed=not marker,stop_success=(tok.eos_token_id in gf)))
    def agg_answer(cases):
        n=max(len(cases),1)
        return dict(n=len(cases),token_exact=sum(c['token_exact'] for c in cases)/n,token_exact_count=sum(c['token_exact'] for c in cases),normalized_exact=sum(c['normalized_exact'] for c in cases)/n,normalized_exact_count=sum(c['normalized_exact'] for c in cases),type_aware_accuracy=sum(c['type_aware'] for c in cases)/n,type_aware_count=sum(c['type_aware'] for c in cases),valid_generation_rate=sum(c['valid'] for c in cases)/n,eos_success_rate=sum(c['eos_success'] for c in cases)/n,stop_success_rate=sum(c['stop_success'] for c in cases)/n)
    out={'D1_program_inserted_boundary':agg_answer(d1),'D2_model_generated_boundary':agg_answer(d2)}
    n=max(len(d2),1); out['D2_model_generated_boundary'].update(think_end_first_token_accuracy=sum(c['think_end_first_token_accuracy'] for c in d2)/n,answer_marker_accuracy=sum(c['answer_marker_accuracy'] for c in d2)/n,boundary_sequence_accuracy=sum(c['boundary_sequence_exact'] for c in d2)/n)
    if include_free:
        n=max(len(d3),1); out['D3_free_k']=dict(n=len(d3),THINK_started_rate=sum(c['think_started'] for c in d3)/n,mean_generated_THINK_count=sum(c['generated_THINK_count'] for c in d3)/n,THINK_termination_rate=sum(c['think_termination'] for c in d3)/n,boundary_sequence_accuracy=sum(c['boundary_sequence_accuracy'] for c in d3)/n,valid_generation_rate=sum(c['valid_generation'] for c in d3)/n,malformed_rate=sum(c['malformed'] for c in d3)/n,stop_success_rate=sum(c['stop_success'] for c in d3)/n)
    else:
        out['D3_free_k']=dict(skipped_during_training_eval=True)
    return out

def transition_probs(torch,tok,model,rs,step,label):
    records=[]; think_id,end_id,answer_id=[tok.convert_tokens_to_ids(x) for x in [THINK,THINK_END,ANSWER]]
    for r in rs[:8]:
        q,k,ans,_=ids_for(tok,r); prefixes=[('after_last_THINK',q+[think_id]*k),('after_THINK_END',q+[think_id]*k+[end_id]),('after_ANSWER',q+[think_id]*k+[end_id,answer_id])]
        for pos,prefix in prefixes:
            inp=torch.tensor([prefix],device='cuda:0'); attn=torch.ones_like(inp)
            with torch.no_grad(): logits=model(input_ids=inp,attention_mask=attn).logits[:,-1,:].float()[0]
            probs=logits.softmax(-1); top=torch.topk(probs,10)
            rec=dict(step=step,label=label,sample_id=r['sample_id'],position=pos,p_THINK=float(probs[think_id].cpu()),p_THINK_END=float(probs[end_id].cpu()),p_ANSWER=float(probs[answer_id].cpu()),boundary_margin=float((logits[end_id]-logits[think_id]).cpu()),top10=[dict(token_id=int(i),token=decode(tok,[int(i)]),prob=float(v)) for v,i in zip(top.values.cpu(),top.indices.cpu())])
            if pos=='after_ANSWER' and ans:
                gold=ans[0]; rec.update(gold_first_answer_token=decode(tok,[gold]),gold_first_answer_probability=float(probs[gold].cpu()),gold_first_answer_rank=int((probs>probs[gold]).sum().cpu()+1),first_answer_token_accuracy=int(int(logits.argmax())==gold))
            records.append(rec)
    return records

def audits(torch,tok):
    rs=rows(); batch,segs,metas=make_batch(torch,tok,rs)
    vocab=len(tok); emb=model_emb=None
    token_ids={x:tok.convert_tokens_to_ids(x) for x in [THINK,THINK_END,ANSWER]}
    single={k:(len(tok(k,add_special_tokens=False)['input_ids'])==1) for k in token_ids}
    rows_a=[]
    for r,seg,m in zip(rs[:3],segs[:3],metas[:3]):
        ids=batch['input_ids'][0].detach().cpu().tolist() if not rows_a else None
    r=rs[0]; b,s,m=make_batch(torch,tok,[r]); ids=b['input_ids'][0].detach().cpu().tolist(); labels=b['labels'][0].detach().cpu().tolist()
    for i,(tid,lab,seg) in enumerate(zip(ids,labels,s[0])):
        if seg!='pad' and (seg!='question' or i>max(0,m[0]['q_len']-5)):
            rows_a.append(dict(position=i,token_id=tid,token=decode(tok,[tid]),label=lab,label_token=None if lab==-100 else decode(tok,[lab]),segment=seg,loss_mask=lab!=-100))
    write_json(REPORT/'o3_boundary_token_audit.json',dict(token_ids=token_ids,single_token=single,sample_id=r['sample_id'],raw_K=m[0]['raw_K'],sequence_fits=len(ids)<=1024,latent_cap_hit=False,question_truncated=False,answer_truncated=False,boundary_truncated=False,token_window=rows_a[-80:]))

def train():
    random.seed(SEED); torch,tok,model=load_model(None,True); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    audits(torch,tok); rs=rows(); params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=LR)
    ckpt=REPO/'checkpoints/stage1_boundary_o3_dynamic'; best_dir=ckpt/'best_adapter'; final_dir=ckpt/'final_adapter'
    logs=[]; evals=[]; trans=[]; best_tuple=(-1,-1,-1,-1,-1); best_step=0; passed=False; degradation=False; d2_hist=[]; t0=time.perf_counter(); peak=0
    trans.extend(transition_probs(torch,tok,model,rs,0,'initialization'))
    for step in range(1,MAX_STEPS+1):
        row=rs[(step-1)%len(rs)]; batch,segs,_=make_batch(torch,tok,[row]); out=model(**batch); parts=shifted_parts(torch,out.logits,batch['labels'],segs); pos=token_position_metrics(torch,tok,out.logits,batch['labels'],segs)
        loss=0.1*parts['think_loss']+parts['boundary_loss']+parts['answer_loss']
        tg=grad_norm_from_backward(torch,model,parts['think_loss'],params); bg=grad_norm_from_backward(torch,model,parts['boundary_loss'],params); ag=grad_norm_from_backward(torch,model,parts['answer_loss'],params)
        loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad(set_to_none=True); peak=max(peak,int(torch.cuda.max_memory_allocated()))
        rec=dict(step=step,sample_id=row['sample_id'],think_mean_loss=parts['think_mean_loss'],boundary_mean_loss=parts['boundary_mean_loss'],answer_mean_loss=parts['answer_mean_loss'],total_loss=float(loss.detach().cpu()),first_boundary_token_loss=pos['first_boundary_token_loss'],answer_marker_loss=pos['answer_marker_loss'],first_answer_token_loss=pos['first_answer_token_loss'],EOS_loss=pos['eos_loss'],think_token_count=parts['think_count'],boundary_token_count=parts['boundary_count'],answer_token_count=parts['answer_count'],think_weighted_contribution=float((0.1*parts['think_loss']).detach().cpu()),boundary_weighted_contribution=float(parts['boundary_loss'].detach().cpu()),answer_weighted_contribution=float(parts['answer_loss'].detach().cpu()),think_gradient_norm=tg,boundary_gradient_norm=bg,answer_gradient_norm=ag,boundary_token_accuracy=parts['boundary_accuracy'],answer_token_accuracy=parts['answer_accuracy'],EOS_accuracy=pos['eos_accuracy'])
        logs.append(rec)
        if step%10==0:
            write_jsonl(REPORT/'o3_train_log.partial.jsonl',logs); status('o3','running',steps=step,total_loss=rec['total_loss'])
        if step%EVAL_EVERY==0:
            model.eval(); ev=eval_o3(torch,tok,model,rs,include_free=True); model.train(); ev['step']=step; evals.append(ev); trans.extend(transition_probs(torch,tok,model,rs,step,f'step{step}'))
            d1=ev['D1_program_inserted_boundary']; d2=ev['D2_model_generated_boundary']; d3=ev['D3_free_k']
            key=(d2['token_exact_count'],d1['token_exact_count'],d2['valid_generation_rate'],d2['boundary_sequence_accuracy'],d3['THINK_termination_rate'])
            if key>best_tuple:
                best_tuple=key; best_step=step; best_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(best_dir)
            d2_hist.append(d2['token_exact_count'])
            if len(d2_hist)>=4 and d2_hist[-1]<d2_hist[-2]<d2_hist[-3]<d2_hist[-4]: degradation=True
            tf_ok=rec['boundary_token_accuracy']>=0.99 and rec['answer_token_accuracy']>=0.99
            d1_ok=(d1['token_exact_count']>=31 or d1['normalized_exact_count']>=31) and d1['valid_generation_rate']>=0.95
            d2_ok=d2['boundary_sequence_accuracy']>=0.95 and (d2['token_exact_count']>=31 or d2['normalized_exact_count']>=31) and d2['valid_generation_rate']>=0.95
            if tf_ok and d1_ok and d2_ok:
                passed=True; break
    final_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(final_dir); tok.save_pretrained(ckpt/'tokenizer')
    model.eval(); final_eval=eval_o3(torch,tok,model,rs,include_free=True); trans.extend(transition_probs(torch,tok,model,rs,step,'final'))
    runtime=time.perf_counter()-t0; write_jsonl(REPORT/'o3_transition_probabilities.jsonl',trans); write_jsonl(REPORT/'o3_train_log.jsonl',logs)
    rep=dict(status='pass' if passed else 'complete_gate_not_passed',steps=step,best_step=best_step,best_checkpoint=str(best_dir),final_checkpoint=str(final_dir),runtime_seconds=runtime,peak_gpu_memory_allocated=peak,best_tuple=best_tuple,degradation=degradation,evals=evals,final_eval=final_eval,train_tail=logs[-10:])
    write_json(REPORT/'o3_status.json',rep)
    write_curves(evals); final_reports(rep)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('o3',rep['status'],steps=step,best_step=best_step,passed=passed)

def write_curves(evals):
    rows_out=[]
    for ev in evals:
        d1=ev['D1_program_inserted_boundary']; d2=ev['D2_model_generated_boundary']; d3=ev['D3_free_k']
        rows_out.append(dict(step=ev['step'],D1_token_exact=d1['token_exact'],D1_normalized_exact=d1['normalized_exact'],D1_type_aware=d1['type_aware_accuracy'],D1_valid=d1['valid_generation_rate'],D1_eos=d1['eos_success_rate'],D2_token_exact=d2['token_exact'],D2_normalized_exact=d2['normalized_exact'],D2_type_aware=d2['type_aware_accuracy'],D2_valid=d2['valid_generation_rate'],D2_boundary_sequence=d2['boundary_sequence_accuracy'],D2_THINK_END_first=d2['think_end_first_token_accuracy'],D2_ANSWER_marker=d2['answer_marker_accuracy'],D3_THINK_started=d3['THINK_started_rate'],D3_mean_THINK=d3['mean_generated_THINK_count'],D3_THINK_termination=d3['THINK_termination_rate'],D3_boundary_sequence=d3['boundary_sequence_accuracy'],D3_valid=d3['valid_generation_rate'],D3_malformed=d3['malformed_rate'],D3_stop=d3['stop_success_rate']))
    with (REPORT/'o3_metrics_curve.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)

def final_reports(rep):
    evals=rep['evals']; best=max(evals,key=lambda ev:(ev['D2_model_generated_boundary']['token_exact_count'],ev['D1_program_inserted_boundary']['token_exact_count'],ev['D2_model_generated_boundary']['valid_generation_rate'],ev['D2_model_generated_boundary']['boundary_sequence_accuracy'],ev['D3_free_k']['THINK_termination_rate'])) if evals else None
    final={'step':rep['steps'],**rep['final_eval']}; rows_out=[]
    for kind,ev in [('best',best),('final',final)]:
        if not ev: continue
        d1=ev['D1_program_inserted_boundary']; d2=ev['D2_model_generated_boundary']; d3=ev['D3_free_k']
        rows_out.append(dict(kind=kind,step=ev['step'],D1_token_exact=d1['token_exact'],D1_normalized_exact=d1['normalized_exact'],D1_valid=d1['valid_generation_rate'],D2_token_exact=d2['token_exact'],D2_normalized_exact=d2['normalized_exact'],D2_boundary_sequence=d2['boundary_sequence_accuracy'],D2_valid=d2['valid_generation_rate'],D3_THINK_termination=d3['THINK_termination_rate'],D3_mean_THINK=d3['mean_generated_THINK_count'],D3_boundary_sequence=d3['boundary_sequence_accuracy']))
    with (REPORT/'o3_best_final_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    b=rows_out[0]; d1_pass=b['D1_token_exact']>=31/32 or b['D1_normalized_exact']>=31/32; d1_pass=d1_pass and b['D1_valid']>=0.95
    d2_pass=b['D2_boundary_sequence']>=0.95 and (b['D2_token_exact']>=31/32 or b['D2_normalized_exact']>=31/32) and b['D2_valid']>=0.95
    if d1_pass and d2_pass: decision='CASE A: D1 and D2 passed; next may run small Stage1-only pilot, but not 10k/Stage2 in this round.'
    elif d1_pass and not d2_pass: decision='CASE B: D1 passed, D2 failed; latent-answer content path works, boundary generation remains the issue.'
    elif not d1_pass: decision='CASE C: D1 still failed; explicit boundary did not solve latent-answer content path, do not enter Stage2.'
    else: decision='CASE D: D1/D2 pass but Free-K fails; oracle/fixed-budget diagnostic only.'
    md=['# O3 Explicit Boundary Report','',decision,'','## Best vs Final','','| kind | step | D1 exact | D1 valid | D2 exact | D2 boundary | D2 valid | D3 termination | D3 mean THINK |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows_out: md.append(f"| {r['kind']} | {r['step']} | {r['D1_token_exact']} | {r['D1_valid']} | {r['D2_token_exact']} | {r['D2_boundary_sequence']} | {r['D2_valid']} | {r['D3_THINK_termination']} | {r['D3_mean_THINK']} |")
    md += ['','## Required Answers','',f"1. D1 pass: {d1_pass}",f"2. D2 pass: {d2_pass}",f"3. Explicit boundary solved Forced-K: {d2_pass}",f"4. Best-to-final degradation: {rep['degradation']}",f"5. Free-K termination at best: {b['D3_THINK_termination']}",'6. Old Fixed-K should not be treated as hard failure for dynamic-K O3; it is out-of-training-length-protocol diagnostic.','7. Small Stage1 pilot can start only in CASE A; otherwise no.', '8. 10k pilot and Stage2 remain forbidden in this round.']
    (REPORT/'o3_report.md').write_text('\n'.join(md)+'\n'); (REPORT/'next_experiment_recommendation.md').write_text(decision+'\n')
    (REPORT/'o3_fixed_proposal.md').write_text('# O3-Fixed Proposal\n\nDo not train automatically. Proposed fixed protocol: choose K=64 or debug32 P50 K, train and evaluate all samples with the same K, keep explicit `<THINK_END><ANSWER>` boundary, and compare against dynamic-K O3 using the same base checkpoint, seed, LoRA config, optimizer, max steps, and no-cache evaluator. Fixed-K metrics should only be formal for a separately trained O3-Fixed model.\n')
    write_json(REPORT/'decision.json',dict(decision=decision,d1_pass=d1_pass,d2_pass=d2_pass,best=b,final=rows_out[-1]))

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='1'; os.environ['HF_HOME']=str(HF_HOME); os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com'); os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    REPORT.mkdir(parents=True,exist_ok=True); STATUS.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser(); ap.add_argument('phase',choices=['train','report']); a=ap.parse_args()
    if a.phase=='train': train()
    else:
        rep=json.loads((REPORT/'o3_status.json').read_text()); final_reports(rep)
if __name__=='__main__': main()
