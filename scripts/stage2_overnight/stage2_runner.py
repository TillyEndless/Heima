#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, gc, hashlib, json, os, random, re, subprocess, time, traceback
from pathlib import Path
from typing import Any

REPO=Path(os.environ.get('REPO_ROOT', Path.cwd()))
HF_HOME=Path(os.environ.get('HF_HOME','/weights2/zhouxiaoling/hf_cache'))
MODEL_ID='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
MODEL_REVISION='916b56a44061fd5cd7d6a8fb632557ed4f724f60'
THINK='<THINK>'; THINK_END='<THINK_END>'; ANSWER='<ANSWER>'
TOKENS=[THINK,THINK_END,ANSWER]
MANIFEST=Path(os.environ.get('STAGE2_MANIFEST', str(REPO/'data/manifests/stage1_debug32_no_truncation.json')))
OUT=REPO/'reports/stage2_overnight'; LOGS=REPO/'logs/stage2_overnight'; RUNS=REPO/'runs/stage2_overnight'; STATUS=REPO/'status/stage2_overnight'; CKPT=REPO/'checkpoints/stage2_overnight'
O3_CKPT=Path(os.environ.get('O3_CKPT_ROOT','/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot/checkpoints/stage1_boundary_o3_dynamic'))
ANSWER_RE=re.compile(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)'); NUM_RE=re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?'); CHOICE_RE=re.compile(r'\b([A-E])\b', re.I)
SEED=42; LR=float(os.environ.get('STAGE2_LR','2e-5')); EVAL_EVERY=int(os.environ.get('STAGE2_EVAL_EVERY','100')); STAGE2_MAX_K=int(os.environ.get('STAGE2_MAX_K','64')); STAGE2_EVAL_N=int(os.environ.get('STAGE2_EVAL_N','64')); STAGE2_SAVE_BEST=os.environ.get('STAGE2_SAVE_BEST','0')=='1'; STAGE2_SAVE_FINAL=os.environ.get('STAGE2_SAVE_FINAL','1')=='1'

def atomic_json(p:Path,o:Any):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp'); tmp.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n'); tmp.replace(p)
def atomic_jsonl(p:Path,rows):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+f'.{os.getpid()}.tmp'); tmp.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows)); tmp.replace(p)
def status(name,state,**kw): atomic_json(STATUS/f'{name}.json',dict(stage=name,status=state,time=time.time(),**kw))
def git_sha():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    except Exception: return 'missing'
def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def rows():
    obj=json.loads(MANIFEST.read_text())
    return obj['samples'] if isinstance(obj,dict) and 'samples' in obj else obj
def q_prefix(r): return 'Question:\n'+r['question']+'\n\n'
def answer_text(tok,r): return r['answer']+tok.eos_token
def cot_text(tok,r): return (r.get('gold_cot') or r.get('cot') or '').strip()+tok.eos_token
def explain_prefix(r): return 'Explain the hidden reasoning for the question.\nQuestion:\n'+r['question'].strip()+'\n\nReasoning:\n'
def norm_ws(s): return re.sub(r'\s+',' ',str(s or '').replace('<｜end▁of▁sentence｜>','').strip())
def norm_num(s):
    if s is None: return None
    s=str(s).strip().replace(',',''); return s[:-2] if s.endswith('.0') else s
def parse_num(text):
    text=str(text or ''); m=ANSWER_RE.search(text)
    if m: return norm_num(m.group(1))
    nums=NUM_RE.findall(text); return norm_num(nums[-1]) if nums else None
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

def load_model(adapter:Path|None=None, train=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),trust_remote_code=True)
    tok.add_special_tokens({'additional_special_tokens':TOKENS})
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),torch_dtype=torch.bfloat16,trust_remote_code=True)
    base.resize_token_embeddings(len(tok)); base.config.use_cache=False; base.gradient_checkpointing_enable()
    if adapter: model=PeftModel.from_pretrained(base,adapter,is_trainable=train)
    else:
        cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],modules_to_save=['embed_tokens','lm_head'])
        model=get_peft_model(base,cfg)
    model.to('cuda:0'); model.train(train)
    return torch,tok,model

def ids_for(tok,r,k_cap:int|None=None):
    q=tok(q_prefix(r),add_special_tokens=False)['input_ids']; raw_k=int(r.get('raw_K') or r.get('latent_count') or max(1, round(float(r.get('cot_token_count') or 2)*float(r.get('latent_ratio') or 0.5)))); k=min(raw_k,k_cap) if k_cap else raw_k; ans=tok(answer_text(tok,r),add_special_tokens=False)['input_ids']
    return q,k,ans,raw_k

def make_main_batch(torch,tok,rs,k_cap:int|None=None):
    think_id,end_id,answer_id=[tok.convert_tokens_to_ids(x) for x in TOKENS]
    rows_ids=[]; rows_lab=[]; segs=[]; metas=[]
    for r in rs:
        q,k,ans,raw_k=ids_for(tok,r,k_cap); ids=q+[think_id]*k+[end_id,answer_id]+ans; labs=[-100]*len(q)+[think_id]*k+[end_id,answer_id]+ans; seg=['question']*len(q)+['think']*k+['boundary_end','boundary_answer']+['answer']*len(ans)
        rows_ids.append(ids); rows_lab.append(labs); segs.append(seg); metas.append(dict(sample_id=r.get('sample_id'),q_len=len(q),k=k,raw_K=raw_k,answer_ids=ans,prefix_d1=q+[think_id]*k+[end_id,answer_id],prefix_d2=q+[think_id]*k,boundary_ids=[end_id,answer_id],full_target_ids=[end_id,answer_id]+ans))
    mx=max(map(len,rows_ids)); pad=tok.pad_token_id or tok.eos_token_id
    input_ids=torch.full((len(rs),mx),pad,dtype=torch.long,device='cuda:0'); labels=torch.full_like(input_ids,-100); attn=torch.zeros_like(input_ids)
    for i,ids in enumerate(rows_ids): input_ids[i,:len(ids)]=torch.tensor(ids,device='cuda:0'); labels[i,:len(ids)]=torch.tensor(rows_lab[i],device='cuda:0'); attn[i,:len(ids)]=1; segs[i]+=['pad']*(mx-len(ids))
    return dict(input_ids=input_ids,labels=labels,attention_mask=attn),segs,metas

def parts_loss(torch,logits,labels,segs):
    import torch.nn.functional as F
    sl=labels[:,1:].contiguous(); lg=logits[:,:-1,:].contiguous(); losses=F.cross_entropy(lg.reshape(-1,lg.size(-1)).float(),sl.reshape(-1),ignore_index=-100,reduction='none').reshape_as(sl); pred=lg.argmax(-1); out={}
    for name,fn in [('think',lambda s:s=='think'),('boundary',lambda s:s in ['boundary_end','boundary_answer']),('answer',lambda s:s=='answer')]:
        mask=torch.tensor([[fn(s) for s in row[1:]] for row in segs],dtype=torch.bool,device=labels.device)&sl.ne(-100)
        out[name+'_loss']=losses[mask].mean() if mask.any() else losses.sum()*0
        out[name+'_nll']=float(losses[mask].mean().detach().cpu()) if mask.any() else None
        out[name+'_acc']=float(pred[mask].eq(sl[mask]).float().mean().detach().cpu()) if mask.any() else None
        out[name+'_count']=int(mask.sum().detach().cpu())
    out['main_loss']=0.1*out['think_loss']+out['boundary_loss']+out['answer_loss']
    return out

def make_decode_inputs(torch,tok,model,r,z):
    # z shape: [K, hidden], inserted after explain prompt as continuous latent slots.
    think_id=tok.convert_tokens_to_ids(THINK); prefix=tok(explain_prefix(r),add_special_tokens=False)['input_ids']; cot=tok(cot_text(tok,r),add_special_tokens=False)['input_ids']
    slot=[think_id]*int(z.shape[0]); ids=prefix+slot+cot; labels=[-100]*(len(prefix)+len(slot))+cot
    input_ids=torch.tensor([ids],device='cuda:0'); labels_t=torch.tensor([labels],device='cuda:0'); attn=torch.ones_like(input_ids)
    embeds=model.get_input_embeddings()(input_ids)
    embeds[:,len(prefix):len(prefix)+len(slot),:]=z.unsqueeze(0).to(embeds.dtype)
    return dict(inputs_embeds=embeds,labels=labels_t,attention_mask=attn),dict(prefix_len=len(prefix),latent_len=len(slot),cot_len=len(cot))

def forward_losses(torch,tok,model,r,mode,lambda_decode=0.1,detach_decode=False):
    need_decode=mode in ['M1','M2']
    batch,segs,metas=make_main_batch(torch,tok,[r],STAGE2_MAX_K if need_decode else None)
    out=model(**batch,output_hidden_states=need_decode,use_cache=False)
    mp=parts_loss(torch,out.logits,batch['labels'],segs); total=mp['main_loss']; rec={k:v for k,v in mp.items() if not k.endswith('_loss')}; rec['k']=metas[0]['k']; rec['raw_K']=metas[0]['raw_K']
    if need_decode:
        hs=out.hidden_states[-1][0]; q_len=metas[0]['q_len']; k=metas[0]['k']; z=hs[q_len:q_len+k]
        if detach_decode: z=z.detach()
        dec,dm=make_decode_inputs(torch,tok,model,r,z); dout=model(**dec,use_cache=False); dp=parts_loss(torch,dout.logits,dec['labels'],[['decode_prefix']*(dm['prefix_len']+dm['latent_len'])+['answer']*dm['cot_len']])
        total=total+lambda_decode*dp['answer_loss']; rec.update(decode_nll=dp['answer_nll'],decode_token_acc=dp['answer_acc'],decode_tokens=dp['answer_count'])
    rec['total_loss_tensor']=total; rec['main_loss_value']=float(mp['main_loss'].detach().cpu()); rec['total_loss_value']=float(total.detach().cpu())
    return rec

def decode(tok,ids): return tok.decode(ids,skip_special_tokens=False)
def gen(torch,tok,model,prefix,max_new):
    inp=torch.tensor([prefix],device='cuda:0'); attn=torch.ones_like(inp)
    with torch.no_grad(): out=model.generate(input_ids=inp,attention_mask=attn,max_new_tokens=max_new,do_sample=False,num_beams=1,temperature=None,top_p=None,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=False)
    return out[0,inp.shape[1]:].detach().cpu().tolist()
def eval_model(torch,tok,model,rs,job,k_cap=None):
    model.eval(); d1=[]; d2=[]; tf=[]
    for r in rs:
        batch,segs,metas=make_main_batch(torch,tok,[r],k_cap); m=metas[0]
        with torch.no_grad(): out=model(**batch,use_cache=False)
        pp=parts_loss(torch,out.logits,batch['labels'],segs); tf.append({k:v for k,v in pp.items() if not k.endswith('_loss')})
        g1=gen(torch,tok,model,m['prefix_d1'],len(m['answer_ids'])+8); raw1=decode(tok,g1); gold=decode(tok,m['answer_ids']); typ=infer_type(r)
        d1.append(task_correct(gold,raw1,typ))
        g2=gen(torch,tok,model,m['prefix_d2'],len(m['full_target_ids'])+8); ok=g2[:2]==m['boundary_ids']; raw2=decode(tok,g2[2:] if len(g2)>=2 else g2); d2.append(ok and task_correct(gold,raw2,typ))
    n=max(len(rs),1)
    model.train(True)
    return dict(job=job,n=len(rs),D1_type_aware=sum(d1)/n,D2_type_aware=sum(d2)/n,main_nll=sum(x['answer_nll'] or 0 for x in tf)/n,think_nll=sum(x['think_nll'] or 0 for x in tf)/n,boundary_nll=sum(x['boundary_nll'] or 0 for x in tf)/n)

def asset_manifest():
    OUT.mkdir(parents=True,exist_ok=True); rs=rows(); best=O3_CKPT/'best_adapter'; final=O3_CKPT/'final_adapter'; tok=O3_CKPT/'tokenizer'
    obj=dict(git_sha=git_sha(),model_id=MODEL_ID,model_revision=MODEL_REVISION,tokenizer_path=str(tok),o3_best_step475=str(best),o3_final_step500=str(final),best_adapter_sha256=sha256_file(best/'adapter_model.safetensors'),final_adapter_sha256=sha256_file(final/'adapter_model.safetensors'),adapter_config=json.loads((best/'adapter_config.json').read_text()),debug32_sample_ids=[r['sample_id'] for r in rs],seed=SEED,stage2_max_k=STAGE2_MAX_K,exploratory_note='exploratory Stage2 on imperfect Stage1; decode path uses capped K for memory.')
    atomic_json(OUT/'asset_manifest.json',obj)
    (OUT/'existing_o3_audit.md').write_text(f"# Existing O3 Audit\n\n- step475: `{best}`\n- step500: `{final}`\n- model revision: `{MODEL_REVISION}`\n- debug32 samples: {len(rs)}\n- stage2 max K cap: {STAGE2_MAX_K}\n\nThis Stage2 run is exploratory on imperfect Stage1.\n")
    return obj

def gradient_smoke():
    name='gradient_smoke'; status(name,'running'); torch,tok,model=load_model(O3_CKPT/'best_adapter',True); r=rows()[0]; params=[p for p in model.parameters() if p.requires_grad]
    rec=forward_losses(torch,tok,model,r,'M2',0.1,False); loss=rec['total_loss_tensor']; loss.backward(); grad=sum(float(p.grad.detach().float().pow(2).sum().cpu()) for p in params if p.grad is not None) ** 0.5; model.zero_grad(set_to_none=True)
    rec_det=forward_losses(torch,tok,model,r,'M2',0.1,True); rec_det['total_loss_tensor'].backward(); grad_det=sum(float(p.grad.detach().float().pow(2).sum().cpu()) for p in params if p.grad is not None) ** 0.5
    out=dict(status='pass' if grad>0 and grad_det>0 else 'fail',grad_A_total_no_detach=grad,grad_A_total_detach_control=grad_det,note='detach control here removes decode gradient to first-forward z but main loss still gives A gradient; full split-gradient audit is deferred.',sample_id=r['sample_id'],k=rec['k'],decode_nll=rec.get('decode_nll'))
    atomic_json(OUT/'stage2_gradient_smoke.json',out); status(name,out['status'],result=out); del model; gc.collect(); torch.cuda.empty_cache(); return 0 if out['status']=='pass' else 1

def save_with_lock(model,tok,out_dir:Path,tokenizer_dir:Path):
    lock=CKPT/'checkpoint_save.lock'
    while True:
        try:
            lock.mkdir(parents=True,exist_ok=False)
            break
        except FileExistsError:
            time.sleep(30)
    try:
        out_dir.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(out_dir)
        tok.save_pretrained(tokenizer_dir)
    finally:
        try: lock.rmdir()
        except OSError: pass

def train_job(job,mode,adapter_kind='none',steps=300,lambda_decode=0.1):
    status(job,'running',step=0); adapter=None
    if adapter_kind=='o3_475': adapter=O3_CKPT/'best_adapter'
    if adapter_kind=='o3_500': adapter=O3_CKPT/'final_adapter'
    torch,tok,model=load_model(adapter,True); random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); params=[p for p in model.parameters() if p.requires_grad]; opt=torch.optim.AdamW(params,lr=LR)
    rs=rows(); logs=[]; evals=[]; best=(-1,-1); best_dir=CKPT/job/'best_adapter'; final_dir=CKPT/job/'final_adapter'; t0=time.perf_counter()
    try:
        for step in range(1,steps+1):
            r=rs[(step-1)%len(rs)]; rec=forward_losses(torch,tok,model,r,mode,lambda_decode,False); loss=rec.pop('total_loss_tensor'); loss.backward(); torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); opt.zero_grad(set_to_none=True); rec.update(step=step,sample_id=r['sample_id']); logs.append(rec)
            if step%10==0: atomic_jsonl(OUT/f'{job}_train.partial.jsonl',logs); status(job,'running',step=step,total_loss=rec['total_loss_value'])
            if step%EVAL_EVERY==0:
                ev=eval_model(torch,tok,model,rs[:STAGE2_EVAL_N],job,STAGE2_MAX_K if mode in ['M1','M2'] else None); ev['step']=step; evals.append(ev); score=(ev['D2_type_aware'],ev['D1_type_aware'])
                if score>best:
                    best=score
                    if STAGE2_SAVE_BEST:
                        save_with_lock(model,tok,best_dir,CKPT/job/'tokenizer')
        if STAGE2_SAVE_FINAL:
            save_with_lock(model,tok,final_dir,CKPT/job/'tokenizer')
        fev=eval_model(torch,tok,model,rs[:STAGE2_EVAL_N],job,STAGE2_MAX_K if mode in ['M1','M2'] else None)
        rep=dict(job=job,mode=mode,adapter_kind=adapter_kind,lambda_decode=lambda_decode,steps=steps,status='complete',runtime_seconds=time.perf_counter()-t0,best_score=best,evals=evals,final_eval=fev,best_checkpoint=str(best_dir),final_checkpoint=str(final_dir),stage2_max_k=STAGE2_MAX_K,eval_n=STAGE2_EVAL_N,save_best=STAGE2_SAVE_BEST,save_final=STAGE2_SAVE_FINAL,manifest=str(MANIFEST),exploratory=True,peak_gpu_memory=int(torch.cuda.max_memory_allocated()))
        atomic_json(OUT/f'{job}.json',rep); atomic_jsonl(OUT/f'{job}_train.jsonl',logs); status(job,'complete',step=steps,best_score=best); rc=0
    except Exception as e:
        atomic_json(OUT/f'{job}_failure.json',dict(job=job,status='failed',reason=repr(e),traceback=traceback.format_exc())); status(job,'failed',reason=repr(e)); rc=1
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()
    return rc

def aggregate():
    rows_out=[]
    for p in sorted(OUT.glob('*.json')):
        if p.name.endswith('_failure.json') or p.name in ['asset_manifest.json','stage2_gradient_smoke.json','task_registry.json','live_status.json','launch_manifest.json']: continue
        try: o=json.loads(p.read_text())
        except Exception: continue
        if 'job' in o and 'final_eval' in o:
            f=o['final_eval']; rows_out.append(dict(job=o['job'],mode=o['mode'],adapter=o['adapter_kind'],lambda_decode=o['lambda_decode'],steps=o['steps'],D1=f.get('D1_type_aware'),D2=f.get('D2_type_aware'),main_nll=f.get('main_nll'),runtime=o.get('runtime_seconds'),best=str(o.get('best_score'))))
    atomic_json(OUT/'all_metrics.json',rows_out)
    if rows_out:
        with (OUT/'all_metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    (OUT/'final_recommendation.md').write_text('# Stage2 Overnight Summary\n\n'+json.dumps(rows_out,indent=2)+'\n')

def gpt2_audit():
    files=[str(p) for p in REPO.rglob('*') if p.is_file() and ('gpt2' in str(p).lower() or 'gsm8k' in str(p).lower())]
    atomic_json(OUT/'gpt2_legacy_inventory.json',dict(count=len(files),files=files[:500]))
    (OUT/'gpt2_static_audit.md').write_text('# GPT2 Legacy Static Audit\n\nInventory created. Deep replay not run in this minimal overnight bootstrap. Missing items are not guessed.\n')
    (OUT/'gpt2_root_cause_attribution.md').write_text('# GPT2 Root Cause Attribution\n\nPending checkpoint replay. This CPU audit did not block H200 Stage2 launch.\n')

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('inventory'); sub.add_parser('smoke'); sub.add_parser('aggregate'); sub.add_parser('gpt2-audit')
    tr=sub.add_parser('train'); tr.add_argument('--job',required=True); tr.add_argument('--mode',choices=['M0','M1','M2'],required=True); tr.add_argument('--adapter-kind',default='none'); tr.add_argument('--steps',type=int,default=300); tr.add_argument('--lambda-decode',type=float,default=0.1)
    a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True); RUNS.mkdir(parents=True,exist_ok=True); STATUS.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('HF_HOME',str(HF_HOME)); os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com'); os.environ.setdefault('HF_HUB_DISABLE_XET','1'); os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
    if a.cmd=='inventory': asset_manifest()
    elif a.cmd=='smoke': raise SystemExit(gradient_smoke())
    elif a.cmd=='train': raise SystemExit(train_job(a.job,a.mode,a.adapter_kind,a.steps,a.lambda_decode))
    elif a.cmd=='aggregate': aggregate()
    elif a.cmd=='gpt2-audit': gpt2_audit()
if __name__=='__main__': main()
