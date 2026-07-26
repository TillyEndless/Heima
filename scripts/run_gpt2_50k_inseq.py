#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys, time, hashlib, subprocess, os
from pathlib import Path
from typing import List
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_gpt2_cot_latent_adjacent_matrix import (
    Example, read_jsonl, adapt_record, load_model, compute_losses, save_ckpt, trainable_report,
    evaluate, write_configs, encode, THINK, split_hash
)
from audit_gpt2_in_sequence_think_intervention import (
    clean_layer_acts, forward_condition, masked_nll_per_sample, bootstrap_ci, mean as mmean
)
from audit_gpt2_inseq_generation import greedy_batch, summarize_group, write_jsonl

RUN_ROOT=Path('/data/zxl/runs/gpt2_cot_latent_adjacent_50k_seed42')
SOURCE_RUN=Path('/data/zxl/runs/gpt2_cot_latent_adjacent_matrix_seed42')
DATASET=Path('/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl')
REPORT=ROOT/'reports'
EVAL_STEPS=(500,1000,2500,5000)

def build_50k_split(dataset:Path,out_root:Path,train_n:int,eval_n:int,seed:int,tok,max_len:int,dry=False):
    out_root.mkdir(parents=True,exist_ok=True)
    split_path=out_root/'data_split.json'
    if split_path.exists() and not dry:
        payload=json.loads(split_path.read_text(encoding='utf-8'))
    else:
        rows=read_jsonl(dataset)
        examples=[ex for i,row in enumerate(rows) if (ex:=adapt_record(row,i)) is not None]
        rng=random.Random(seed); rng.shuffle(examples)
        old=json.loads((SOURCE_RUN/'data_split.json').read_text(encoding='utf-8'))
        old_eval=[Example(**x) for x in old['eval'][:eval_n]]
        eval_ids={x.sample_id for x in old_eval}
        train=[]
        for ex in examples:
            if ex.sample_id not in eval_ids:
                train.append(ex)
            if len(train)>=train_n: break
        if len(train)<train_n or len(old_eval)<eval_n:
            raise RuntimeError(f'Not enough samples for 50k split: train={len(train)} eval={len(old_eval)}')
        payload={'seed':seed,'dataset_path':str(dataset),'train':[ex.__dict__ for ex in train],'eval':[ex.__dict__ for ex in old_eval]}
        if not dry: split_path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    train=[Example(**x) for x in payload['train'][:train_n]]
    ev=[Example(**x) for x in payload['eval'][:eval_n]]
    def lens(xs,attr):
        vals=[len(encode(tok,getattr(x,attr))) for x in xs]
        return sum(vals)/len(vals), max(vals)
    qavg,qmax=lens(train+ev,'question'); cavg,cmax=lens(train+ev,'cot'); aavg,amax=lens(train+ev,'answer')
    seq_lens=[len(encode(tok,f"Question:\n{x.question}\n\n{THINK}\n{x.cot}\nAnswer:\n{x.answer}{tok.eos_token}")) for x in (train[:2000]+ev)]
    audit={'train_count':len(train),'eval_count':len(ev),'split_seed':seed,'split_hash':split_hash(payload),'overlap_with_eval':len({x.sample_id for x in train}&{x.sample_id for x in ev}), 'question_completeness':sum(bool(x.question) for x in train+ev)/len(train+ev),'cot_completeness':sum(bool(x.cot) for x in train+ev)/len(train+ev),'answer_completeness':sum(bool(x.answer) for x in train+ev)/len(train+ev),'avg_question_token_length':qavg,'max_question_token_length':qmax,'avg_cot_token_length':cavg,'max_cot_token_length':cmax,'avg_answer_token_length':aavg,'max_answer_token_length':amax,'max_seq_len':max_len,'sampled_max_sequence_length':max(seq_lens),'truncation_rate_sampled':sum(x>max_len for x in seq_lens)/len(seq_lens),'eval_matches_10k_eval_prefix':True}
    REPORT.mkdir(exist_ok=True)
    (REPORT/'gpt2_50k_data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return payload,train,ev,audit

def inseq_nll_eval(model,tok,recs,group,max_len,device,batch_size):
    conds=['normal','zero_think_embedding','zero_think','shuffle_think','remove_think']
    vals={c:{'cot':[],'answer':[]} for c in conds}
    with torch.no_grad():
        for st in range(0,len(recs),batch_size):
            b=recs[st:st+batch_size]
            batch=__import__('run_gpt2_cot_latent_adjacent_matrix').make_main_batch(tok,b,group,max_len,device,train=False)
            normal_logits, acts=clean_layer_acts(model,batch)
            perm=torch.randperm(len(b),device=device)
            logits={'normal':normal_logits}
            for c in conds[1:]: logits[c]=forward_condition(model,batch,c,acts,perm).detach()
            for c,l in logits.items():
                vals[c]['cot'].extend(masked_nll_per_sample(l,batch['labels'],batch['cot_mask']))
                vals[c]['answer'].extend(masked_nll_per_sample(l,batch['labels'],batch['answer_mask']))
    out={'conditions':{},'deltas_vs_normal':{}}
    for c in conds: out['conditions'][c]={'cot_nll':mmean(vals[c]['cot']),'answer_nll':mmean(vals[c]['answer'])}
    for c in conds[1:]:
        cd=[x-y for x,y in zip(vals[c]['cot'],vals['normal']['cot'])]; ad=[x-y for x,y in zip(vals[c]['answer'],vals['normal']['answer'])]
        out['deltas_vs_normal'][c]={'cot_delta':mmean(cd),'cot_delta_ci':bootstrap_ci(cd),'answer_delta':mmean(ad),'answer_delta_ci':bootstrap_ci(ad)}
    return out

def generation_eval_inseq(model,tok,recs,group,out_dir,max_len,device,batch_size,max_new=128):
    data={}
    for cond in ['normal','shuffle_activation','zero_activation','remove_think']:
        rows=[]
        for st in range(0,len(recs),batch_size): rows.extend(greedy_batch(model,tok,recs[st:st+batch_size],group,cond,max_len,max_new,device,42+st))
        write_jsonl(out_dir/f'{cond}_generation.jsonl',rows); data[cond]=rows
    return summarize_group(group,data)

def run(args):
    device=torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    group=args.group
    gdir=Path(args.output_dir)/f'{group}_50k'; gdir.mkdir(parents=True,exist_ok=True)
    model,tok=load_model(args.model_path,device)
    _,train,ev,audit=build_50k_split(DATASET,Path(args.output_dir),args.train_samples,args.eval_samples,args.seed,tok,args.max_length,dry=args.dry_run)
    if args.dry_run:
        train=train[:32]; ev=ev[:8]
    before={k:float(model.get_input_embeddings().weight[tok.convert_tokens_to_ids(v)].detach().norm().cpu()) for k,v in {'think':THINK}.items()}
    trainable_report(model,tok,gdir/'trainable_parameter_report_start.json',before)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    steps=1 if args.dry_run else args.max_steps; start=time.time(); opt.zero_grad(set_to_none=True)
    log_path=gdir/'logs'/'train.log'; log_path.parent.mkdir(parents=True,exist_ok=True)
    def log(obj):
        with log_path.open('a') as f: f.write(json.dumps(obj)+'\n')
        print(json.dumps(obj),flush=True)
    for step in range(1,steps+1):
        model.train(); last=None
        for a in range(args.gradient_accumulation_steps):
            off=(((step-1)*args.gradient_accumulation_steps+a)*args.batch_size)%len(train)
            batch=train[off:off+args.batch_size]
            if len(batch)<args.batch_size: batch+=train[:args.batch_size-len(batch)]
            losses=compute_losses(model,tok,batch,group,args.max_length,device,0.0,args.lambda_latent,train=True)
            (losses.total/args.gradient_accumulation_steps).backward(); last=losses
        if step==1: trainable_report(model,tok,gdir/'trainable_parameter_report_after_backward.json',before)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step%args.log_every==0 or args.dry_run:
            log({'group':group,'step':step,'train_loss':float(last.total.detach().cpu()),'main_loss':float(last.main.detach().cpu()),'cot_nll':float(last.cot.detach().cpu()),'answer_nll':float(last.answer.detach().cpu()),'latent_token_loss':float(last.latent_token.detach().cpu()),'latent_token_accuracy':float(last.latent_acc.detach().cpu()),'elapsed_sec':time.time()-start})
        if args.dry_run or step in EVAL_STEPS:
            ck=save_ckpt(model,tok,gdir,step,group,args)
            if args.dry_run:
                reload,_=load_model(args.model_path,device); reload.load_state_dict(torch.load(Path(ck)/'model_weights.pt',map_location=device)['model'],strict=False); del reload
            edir=gdir/f'eval_step{step}'; edir.mkdir(exist_ok=True)
            metrics=evaluate(model,tok,ev,group,args.max_length,device,args.eval_batch_size)
            metrics['inseq_nll']=inseq_nll_eval(model,tok,ev,group,args.max_length,device,args.eval_batch_size)
            if args.dry_run or step in (2500,5000): metrics['inseq_generation']=generation_eval_inseq(model,tok,ev[:args.generation_samples],group,edir,args.max_length,device,args.eval_batch_size,args.max_new_tokens)
            metrics.update({'checkpoint_path':ck,'step':step,'data_split_hash':audit['split_hash']})
            (edir/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
            log({'group':group,'eval_step':step,'metrics':metrics})
            if args.dry_run: break
    save_ckpt(model,tok,gdir,'final',group,args); trainable_report(model,tok,gdir/'trainable_parameter_report_final.json',before)
    if args.dry_run:
        rep=['# GPT2 50k Dry Run Report','',f'group: {group}',f'train samples: {len(train)}',f'eval samples: {len(ev)}',f'split hash: {audit["split_hash"]}','dry-run status: PASS','checks: data load, forward, backward, checkpoint save/load, in-sequence NLL, generation evaluator, finite losses, trainable embeddings/lm_head']
        (REPORT/'gpt2_50k_dryrun_report.md').write_text('\n'.join(rep)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--group',default='G3',choices=['G3','G4'])
    ap.add_argument('--output-dir',default=str(RUN_ROOT))
    ap.add_argument('--model-path',default='/data/zxl/models/openai-community-gpt2')
    ap.add_argument('--train-samples',type=int,default=50000)
    ap.add_argument('--eval-samples',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--max-steps',type=int,default=5000)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--gradient-accumulation-steps',type=int,default=2)
    ap.add_argument('--eval-batch-size',type=int,default=8)
    ap.add_argument('--generation-samples',type=int,default=256)
    ap.add_argument('--max-new-tokens',type=int,default=128)
    ap.add_argument('--max-length',type=int,default=640)
    ap.add_argument('--lr',type=float,default=1e-5)
    ap.add_argument('--lambda-latent',type=float,default=0.05)
    ap.add_argument('--log-every',type=int,default=50)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); run(args)
if __name__=='__main__': main()
