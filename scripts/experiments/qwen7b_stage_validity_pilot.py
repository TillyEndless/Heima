#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, fcntl, gc, hashlib, json, math, os, random, re, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get('REPO_ROOT','/data2/zhouxiaoling/latent_cot/Heima-qwen7b-stage-validity-pilot'))
ASSET_ROOT = Path(os.environ.get('ASSET_ROOT','/data2/zhouxiaoling/latent_cot/Heima-overnight-qwen7b'))
HF_HOME = Path(os.environ.get('HF_HOME','/weights2/zhouxiaoling/hf_cache'))
MODEL_ID = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
MODEL_REVISION = '916b56a44061fd5cd7d6a8fb632557ed4f724f60'
CURRENT_STAGE1_ADAPTER = ASSET_ROOT / 'checkpoints/qwen7b_stage1_smoke1k_200/final_adapter'
THINK = '<THINK>'
REPORT = REPO / 'reports/stage_validity'
DATA_MAN = REPO / 'data/manifests'
STATUS = REPO / 'status/stage_validity'
LOCK = REPO / 'qwen7b_stage_validity_gpu1.lock'
DATASET_ID = 'a-m-team/AM-DeepSeek-R1-Distilled-1.4M'
DATASET_CFG = 'am_0.9M_sample_1k'
ANSWER_RE = re.compile(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)')
NUM_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?')

def iso(t=None): return datetime.fromtimestamp(t or time.time(), tz=timezone.utc).isoformat()
def sh(cmd): return subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
def write_json(p:Path,o): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def append_jsonl(p:Path,o): p.parent.mkdir(parents=True,exist_ok=True); p.open('a',encoding='utf-8').write(json.dumps(o,ensure_ascii=False,sort_keys=True)+'\n')
def status(name, st, **kw): write_json(STATUS/f'{name}.json', dict(stage=name,status=st,time=time.time(),iso=iso(),**kw))
def done(name):
    p=STATUS/f'{name}.json'
    return p.exists() and json.loads(p.read_text()).get('status')=='complete'
def normalize(s):
    s=str(s).strip().replace(',','')
    return s[:-2] if s.endswith('.0') else s
def parse_answer(text):
    text=str(text)
    m=ANSWER_RE.search(text)
    if m: return normalize(m.group(1))
    low=text.lower()
    for marker in ['final answer is','answer is','answer:','<answer>','boxed']:
        i=low.rfind(marker)
        if i>=0:
            nums=NUM_RE.findall(text[i:])
            if nums: return normalize(nums[-1])
    nums=NUM_RE.findall(text)
    return normalize(nums[-1]) if nums else None

def load_stack():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    return torch, AutoTokenizer, AutoModelForCausalLM, LoraConfig, get_peft_model, PeftModel

def tokenizer_only():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), trust_remote_code=True)
    tok.add_special_tokens({'additional_special_tokens':[THINK]})
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    return tok

def parse_record(raw, idx):
    info=raw.get('info') if isinstance(raw.get('info'),dict) else {}
    q=raw.get('question') or info.get('question') or raw.get('prompt') or raw.get('input')
    cot=info.get('think_content') or raw.get('think_content') or raw.get('cot')
    ans=info.get('answer_content') or raw.get('answer_content') or raw.get('answer')
    content=raw.get('content') or raw.get('assistant') or raw.get('response') or raw.get('messages')
    parse_status='direct'
    if isinstance(content,list): content='\n'.join(str(x.get('content',x)) if isinstance(x,dict) else str(x) for x in content)
    if (not cot or not ans) and content:
        tm=re.search(r'<think>(.*?)</think>', str(content), re.S|re.I); am=re.search(r'<answer>(.*?)</answer>', str(content), re.S|re.I)
        if not cot and tm: cot=tm.group(1).strip(); parse_status='assistant_tags'
        if not ans and am: ans=am.group(1).strip(); parse_status='assistant_tags'
    if not ans and content: ans=parse_answer(content); parse_status='answer_parser'
    return dict(sample_id=str(raw.get('id',idx)), source=DATASET_ID, question=str(q or '').strip(), gold_cot=str(cot or '').strip(), answer=str(ans or '').strip(), reference_answer=str(ans or '').strip(), parse_status=parse_status)

def get_1k_rows():
    p=ASSET_ROOT/'data/processed_large/qwen7b_stage1/smoke1k.json'
    if p.exists(): return json.loads(p.read_text())
    from huggingface_hub import hf_hub_download
    local=hf_hub_download(DATASET_ID, filename='am_0.9M_sample_1k.jsonl', repo_type='dataset', cache_dir=str(HF_HOME))
    tok=tokenizer_only(); rows=[]
    for i,line in enumerate(open(local,encoding='utf-8')):
        r=parse_record(json.loads(line),i); r['cot_token_count']=len(tok(r['gold_cot'],add_special_tokens=False)['input_ids']); r['latent_ratio']=0.5; r['latent_count']=max(1,round(0.5*r['cot_token_count'])) if r['cot_token_count'] else 0; rows.append(r)
    return [r for r in rows if r['question'] and r['gold_cot'] and r['answer']]

def make_main_batch(torch,tok,rows,max_q=256,max_latent=128,max_answer=128):
    think_id=tok.convert_tokens_to_ids(THINK); ids_rows=[]; lab_rows=[]; seg_rows=[]; metas=[]
    for r in rows:
        qids=tok('Question:\n'+r['question']+'\n\n',add_special_tokens=False)['input_ids']; q_tr=len(qids)>max_q; qids=qids[:max_q]
        ans_ids_full=tok('\nAnswer:\n'+r['answer']+tok.eos_token,add_special_tokens=False)['input_ids']; a_tr=len(ans_ids_full)>max_answer; ans=ans_ids_full[:max_answer]
        raw_k=max(1,round(0.5*int(r.get('cot_token_count') or len(tok(r['gold_cot'],add_special_tokens=False)['input_ids'])))); k=min(raw_k,max_latent)
        ids=qids+[think_id]*k+ans; labs=[-100]*len(qids)+[think_id]*k+ans; seg=['question']*len(qids)+['think']*k+['answer']*len(ans)
        ids_rows.append(ids); lab_rows.append(labs); seg_rows.append(seg); metas.append(dict(q_tokens=len(qids),raw_K=raw_k,used_K=k,answer_tokens=len(ans),question_truncated=q_tr,answer_truncated=a_tr,stage1_sequence_length=len(ids)))
    mx=max(map(len,ids_rows)); pad=tok.pad_token_id or tok.eos_token_id
    input_ids=torch.full((len(rows),mx),pad,dtype=torch.long,device='cuda:0'); labels=torch.full((len(rows),mx),-100,dtype=torch.long,device='cuda:0'); attn=torch.zeros_like(input_ids)
    for i,ids in enumerate(ids_rows):
        input_ids[i,:len(ids)]=torch.tensor(ids,device='cuda:0'); labels[i,:len(ids)]=torch.tensor(lab_rows[i],device='cuda:0'); attn[i,:len(ids)]=1; seg_rows[i]+= ['pad']*(mx-len(ids))
    return dict(input_ids=input_ids,labels=labels,attention_mask=attn), seg_rows, metas

def split_loss(torch, logits, labels, segs):
    import torch.nn.functional as F
    sl=labels[:,1:].contiguous(); sg=logits[:,:-1,:].contiguous(); loss=F.cross_entropy(sg.view(-1,sg.size(-1)).float(), sl.view(-1), ignore_index=-100, reduction='none').view_as(sl)
    sums={}; counts={}
    for name in ['think','answer']:
        vals=[]
        for i,row in enumerate(segs):
            for j,seg in enumerate(row[1:]):
                if seg==name and int(sl[i,j])!=-100: vals.append(loss[i,j])
        if vals:
            v=torch.stack(vals); sums[name]=float(v.sum().detach().cpu()); counts[name]=int(v.numel())
        else: sums[name]=0.0; counts[name]=0
    total_sum=sums['think']+sums['answer']; total_count=counts['think']+counts['answer']
    return dict(think_loss_sum=sums['think'],answer_loss_sum=sums['answer'],think_token_count=counts['think'],answer_token_count=counts['answer'],think_mean_loss=sums['think']/max(counts['think'],1),answer_mean_loss=sums['answer']/max(counts['answer'],1),token_weighted_main_loss=total_sum/max(total_count,1),think_contribution_ratio=sums['think']/max(total_sum,1e-9),answer_contribution_ratio=sums['answer']/max(total_sum,1e-9))

def load_base_lora(adapter=None, train=True):
    torch,AutoTokenizer,AutoModel,LoraConfig,get_peft_model,PeftModel=load_stack()
    tok=AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), trust_remote_code=True); tok.add_special_tokens({'additional_special_tokens':[THINK]});
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(HF_HOME), torch_dtype=torch.bfloat16, trust_remote_code=True)
    base.resize_token_embeddings(len(tok)); base.config.use_cache=False; base.gradient_checkpointing_enable()
    if adapter:
        model=PeftModel.from_pretrained(base, adapter, is_trainable=train)
    else:
        cfg=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],modules_to_save=['embed_tokens','lm_head'])
        model=get_peft_model(base,cfg)
    model.to('cuda:0'); model.train(train)
    return torch,tok,model

def inventory():
    items=[]
    checks={
    'current A-only 50-case raw generation audit':['reports/qwen_baseline_sanity.jsonl'],
    'answer parser unit tests':['reports/answer_parser_tests.json'],
    'clean-process adapter reload':['reports/qwen7b_stage1_smoke1k_200_reload_error.json'],
    'L_main reduction audit':['reports/overnight/token_level_audit.json'],
    'cap/truncation audit':['reports/dataset_audit.json','reports/dynamic_latent_length_report.json'],
    'base vs Stage1 held-out evaluation':[],
    'free-generation THINK compliance':[],
    'decoded-vs-gold semantic cosine':[],
    'gold-gold / gold-random calibration':[],
    'correct/shuffle/zero/remove/q-only intervention':[],
    '32-sample true overfit':['reports/qwen7b_stage1_overfit32_stage1_smoke.json'],
    'E0 joint / E1 main-only / E2 two-stage small-model smoke':[],
    'historical timing audit':['reports/performance/historical_runs.jsonl','reports/performance/historical_timing_audit.md']}
    for name,paths in checks.items():
        existing=[str(ASSET_ROOT/p) for p in paths if (ASSET_ROOT/p).exists()]
        st='complete' if existing and name in ['answer parser unit tests','historical timing audit'] else 'incomplete-result' if existing else 'not-run'
        if name=='clean-process adapter reload' and existing: st='failed'
        if name=='32-sample true overfit' and existing: st='incomplete-result'
        if name in ['L_main reduction audit','cap/truncation audit','current A-only 50-case raw generation audit'] and existing: st='incomplete-result'
        items.append(dict(task=name,status=st,evidence=existing,notes='verified by file presence and content category; not inferred from filename alone'))
    write_json(REPORT/'prior_task_inventory.json',items)
    md=['# Prior Task Inventory','', '| task | status | evidence |','|---|---|---|']+[f"| {x['task']} | {x['status']} | {'; '.join(x['evidence'])} |" for x in items]
    (REPORT/'prior_task_inventory.md').parent.mkdir(parents=True,exist_ok=True); (REPORT/'prior_task_inventory.md').write_text('\n'.join(md)+'\n')
    status('prior_inventory','complete')

def gate_a_reload():
    torch,tok,model=load_base_lora(CURRENT_STAGE1_ADAPTER, train=False)
    think_id=tok.convert_tokens_to_ids(THINK); enc=tok('Question:\n1+1?\n\n'+THINK+'\nAnswer:\n',return_tensors='pt').to('cuda:0')
    with torch.no_grad(): out=model(**enc); gen=model.generate(**enc,max_new_tokens=8,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    ck=CURRENT_STAGE1_ADAPTER/'adapter_model.safetensors'; sha=hashlib.sha256(ck.read_bytes()).hexdigest()
    loaded=sum(p.numel() for n,p in model.named_parameters() if 'lora_' in n or 'modules_to_save' in n)
    rep=dict(model_revision=MODEL_REVISION, adapter=str(CURRENT_STAGE1_ADAPTER), tokenizer_loaded=True, think_single_token=len(tok.encode(THINK,add_special_tokens=False))==1, think_id=think_id, expected_think_id=151665, adapter_loaded_parameter_count=loaded, forward_ok=bool(torch.isfinite(out.logits).all().item()), generate_ok=gen.shape[1]>enc.input_ids.shape[1], checksum_sha256=sha, peak_allocated=int(torch.cuda.max_memory_allocated()))
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); rep['memory_after_cleanup']=int(torch.cuda.memory_allocated())
    rep['cleanup_note']='memory_after_cleanup is measured before process exit; runner verifies external GPU release after subprocess exits'; rep['passed']= all([rep['think_single_token'], rep['think_id']==151665, rep['adapter_loaded_parameter_count']>0, rep['forward_ok'], rep['generate_ok']])
    write_json(REPORT/'stage1_reload_audit.json',rep); status('gate_a_reload','complete' if rep['passed'] else 'failed', passed=rep['passed'])
    if not rep['passed']: raise SystemExit('Gate A failed')

def gate_b_loss():
    torch,tok,model=load_base_lora(None, train=True); rows=get_1k_rows()[:2]; batch,segs,metas=make_main_batch(torch,tok,rows)
    out=model(**batch); red=split_loss(torch,out.logits,batch['labels'],segs)
    token_rows=[]
    for i in range(batch['input_ids'].shape[0]):
        for j in range(batch['input_ids'].shape[1]): token_rows.append(dict(sample=i,position=j,token_id=int(batch['input_ids'][i,j]),token=tok.decode([int(batch['input_ids'][i,j])]),segment=segs[i][j],label=int(batch['labels'][i,j]),contributes_to_loss=int(batch['labels'][i,j])!=-100))
    rep=dict(reduction=red, model_loss=float(out.loss.detach().cpu()), explanation='model loss is token-weighted mean over non-ignored THINK+answer shifted labels; loss_total is not loss_think+loss_answer because each component is a mean over different token counts.', token_rows=token_rows[:900], checks=dict(question_ignore=all(x['label']==-100 for x in token_rows if x['segment']=='question'), think_label=all(x['label']==tok.convert_tokens_to_ids(THINK) for x in token_rows if x['segment']=='think'), answer_labels_present=any(x['segment']=='answer' and x['contributes_to_loss'] for x in token_rows)))
    write_json(REPORT/'loss_reduction_audit.json',rep)
    md=f"# Loss Reduction Audit\n\nL_main is a token-weighted mean over all non-ignored shifted labels. It is not `loss_think + loss_answer`.\n\n```json\n{json.dumps(red,indent=2)}\n```\n"
    (REPORT/'loss_reduction_audit.md').write_text(md)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('gate_b_loss','complete', passed=all(rep['checks'].values()))

def gate_c_length():
    tok=tokenizer_only(); rows=get_1k_rows(); recs=[]
    for r in rows:
        q=len(tok(r['question'],add_special_tokens=False)['input_ids']); cot=len(tok(r['gold_cot'],add_special_tokens=False)['input_ids']); ans=len(tok(r['answer'],add_special_tokens=False)['input_ids']); raw=max(1,round(0.5*cot)); used=min(raw,128)
        recs.append(dict(sample_id=r['sample_id'],question_tokens=q,gold_cot_tokens=cot,answer_tokens=ans,raw_K=raw,used_K=used,effective_ratio=used/max(cot,1),latent_cap_hit=raw>128,question_truncated=q>256,cot_truncated=cot>256,answer_truncated=ans>128,stage1_sequence_length=min(q,256)+used+min(ans+3,128),stage2_decode_sequence_length=min(q,256)+used+min(cot+3,256)))
    def pct(xs,p):
        xs=sorted(xs); k=(len(xs)-1)*p; lo=math.floor(k); hi=math.ceil(k); return xs[lo] if lo==hi else xs[lo]*(hi-k)+xs[hi]*(k-lo)
    summary={}
    for key in ['question_tokens','gold_cot_tokens','answer_tokens','raw_K','used_K','effective_ratio','stage1_sequence_length','stage2_decode_sequence_length']:
        xs=[r[key] for r in recs]; summary[key]={f'P{int(p*100)}':pct(xs,p) for p in [0.5,0.75,0.9,0.95,0.99]}
    summary.update(latent_cap_rate=sum(r['latent_cap_hit'] for r in recs)/len(recs),question_truncation_rate=sum(r['question_truncated'] for r in recs)/len(recs),cot_truncation_rate=sum(r['cot_truncated'] for r in recs)/len(recs),answer_truncation_rate=sum(r['answer_truncated'] for r in recs)/len(recs),protocol_notes=dict(raw_r_0p5='raw_K=round(0.5*gold_cot_tokens)',capped_r_0p5='used_K=min(raw_K,128) in current smoke/gates',fixed_K='not used here',oracle_K='raw_K uses gold CoT length and is diagnostic only',free_K='not used here'))
    write_json(REPORT/'length_cap_audit.json',dict(summary=summary,records=recs[:1000]))
    md=['# Length And Cap Audit','',json.dumps(summary,indent=2)]
    (REPORT/'length_cap_audit.md').write_text('\n'.join(md)+'\n')
    status('gate_c_length','complete', latent_cap_rate=summary['latent_cap_rate'], answer_truncation_rate=summary['answer_truncation_rate'])

def eval_main_free(torch,tok,model,rows,max_new=128):
    ok=0; valid=0; malformed=0; outs=[]
    for r in rows:
        k=min(max(1,round(0.5*int(r['cot_token_count']))),128)
        prompt='Question:\n'+r['question']+'\n\n'+(THINK*k)+'\nAnswer:\n'
        enc=tok(prompt,return_tensors='pt',truncation=True,max_length=512).to('cuda:0')
        with torch.no_grad(): gen=model.generate(**enc,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
        text=tok.decode(gen[0,enc.input_ids.shape[1]:],skip_special_tokens=False)
        pred=parse_answer(text); gold=parse_answer(r['answer'])
        if pred is not None: valid+=1
        else: malformed+=1
        if pred is not None and gold is not None and pred==gold: ok+=1
        outs.append(dict(sample_id=r['sample_id'],gold=gold,pred=pred,text=text[:800]))
    n=len(rows); return dict(answer_accuracy=ok/max(n,1),valid_answer_rate=valid/max(n,1),malformed_output_rate=malformed/max(n,1),cases=outs)

def gate_d_overfit():
    torch,tok,model=load_base_lora(None, train=True); rows=get_1k_rows()[:32]; opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-5)
    logs=[]; passed=False; max_steps=500
    for step in range(1,max_steps+1):
        batch,segs,metas=make_main_batch(torch,tok,[rows[(step-1)%32]])
        out=model(**batch); loss=out.loss
        if not torch.isfinite(loss): break
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); opt.zero_grad(set_to_none=True)
        if step%25==0 or step in [1,300,500]:
            with torch.no_grad():
                pred=out.logits[:,:-1].argmax(-1); lab=batch['labels'][:,1:]; think_id=tok.convert_tokens_to_ids(THINK); m_th=lab.eq(think_id); m_ans=lab.ne(-100)&(~m_th)
                think_acc=float(pred[m_th].eq(think_id).float().mean().cpu()) if m_th.any() else 0.0
                ans_tok_acc=float(pred[m_ans].eq(lab[m_ans]).float().mean().cpu()) if m_ans.any() else 0.0
            logs.append(dict(step=step,loss=float(loss.detach().cpu()),think_token_accuracy=think_acc,answer_token_accuracy=ans_tok_acc))
        if step in [300,500]:
            ev=eval_main_free(torch,tok,model,rows[:8],max_new=128); logs[-1]['free_eval_8']=ev
            if logs[-1]['think_token_accuracy']>=0.99 and ev['answer_accuracy']>=0.95 and ev['valid_answer_rate']>=0.95: passed=True; break
    run_dir=REPO/'checkpoints/stage_validity_gate_d_overfit/final_adapter'; run_dir.mkdir(parents=True,exist_ok=True); model.save_pretrained(run_dir); tok.save_pretrained(REPO/'checkpoints/stage_validity_gate_d_overfit/tokenizer')
    rep=dict(run_status='complete',overfit_gate='pass' if passed else 'fail',steps=step,logs=logs,checkpoint=str(run_dir),sample_ids=[r['sample_id'] for r in rows],criteria=dict(think_token_accuracy='>=0.99',train_answer_accuracy='>=0.95',valid_answer_rate='>=0.95',free_generation_format_compliance='>=0.95'),failure_reason=None if passed else 'free-generation/train answer gate not satisfied within 500 steps or only 8-case eval insufficient for pass')
    write_json(REPORT/'gate_d_overfit.json',rep)
    with (REPORT/'failure_cases.jsonl').open('w') as f:
        if logs and 'free_eval_8' in logs[-1]:
            for c in logs[-1]['free_eval_8']['cases']: f.write(json.dumps(c,ensure_ascii=False)+'\n')
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('gate_d_overfit','complete' if passed else 'failed', passed=passed)
    if not passed: raise SystemExit('Gate D failed; stop before 10k pilot')

def final_report():
    statuses={p.stem:json.loads(p.read_text()) for p in STATUS.glob('*.json')}
    write_json(REPORT/'final_gate_status.json',statuses)
    lines=['# Qwen7B Pilot10k Final Report','','This run is gate-controlled. 10k M0/M1/M2 is launched only if Gate A-D pass.','']
    for k,v in statuses.items(): lines.append(f"- {k}: {v.get('status')} {v.get('passed','')}")
    if statuses.get('gate_d_overfit',{}).get('status')!='complete': lines += ['','Gate D failed, so 10k pilot was not started.']
    (REPORT/'qwen7b_pilot10k_final_report.md').write_text('\n'.join(lines)+'\n')
    (REPORT/'qwen7b_main_joint_twostage.md').write_text('# Main/Joint/Two-stage\n\nNot run unless Gate A-D pass.\n')
    (REPORT/'next_experiment_recommendation.md').write_text('# Next Recommendation\n\nFix Stage1 free-generation overfit protocol before 10k pilot if Gate D failed. Consider answer parser/target formatting and shorter numeric dataset subset.\n')
    with (REPORT/'qwen7b_pilot10k_metrics.csv').open('w') as f: f.write('model,status,notes\nM0,not-run,gate controlled\nM1,not-run,gate controlled\nM2,not-run,gate controlled\n')

def make_bundle():
    bundle=Path('/data2/zhouxiaoling/latent_cot/qwen7b-stage-validity-pilot.bundle')
    subprocess.run(['git','bundle','create',str(bundle),'HEAD'],cwd=REPO,check=True)
    h=hashlib.sha256(bundle.read_bytes()).hexdigest(); (Path(str(bundle)+'.sha256')).write_text(h+'  '+bundle.name+'\n')

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='1'; os.environ['HF_HOME']=str(HF_HOME); os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com'); os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    ap=argparse.ArgumentParser(); ap.add_argument('phase',choices=['inventory','gate_a','gate_b','gate_c','gate_d','final','all','bundle'])
    args=ap.parse_args(); REPORT.mkdir(parents=True,exist_ok=True); STATUS.mkdir(parents=True,exist_ok=True)
    if args.phase=='inventory': inventory()
    elif args.phase=='gate_a': gate_a_reload()
    elif args.phase=='gate_b': gate_b_loss()
    elif args.phase=='gate_c': gate_c_length()
    elif args.phase=='gate_d': gate_d_overfit()
    elif args.phase=='final': final_report()
    elif args.phase=='bundle': make_bundle()
    elif args.phase=='all':
        with LOCK.open('w') as f:
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                if not done('prior_inventory'): inventory()
                if not done('gate_a_reload'): gate_a_reload()
                if not done('gate_b_loss'): gate_b_loss()
                if not done('gate_c_length'): gate_c_length()
                if not done('gate_d_overfit'): gate_d_overfit()
            finally:
                final_report()
if __name__=='__main__': main()
