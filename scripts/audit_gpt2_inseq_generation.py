#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re, sys
from pathlib import Path
from typing import Dict, List, Sequence
import torch
from difflib import SequenceMatcher

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from run_gpt2_cot_latent_adjacent_matrix import Example, THINK, encode, load_model, make_main_batch
from audit_gpt2_in_sequence_think_intervention import patch_block_outputs

DEFAULT_RUN_ROOT=Path('/data/zxl/runs/gpt2_cot_latent_adjacent_matrix_seed42')
DEFAULT_MODEL='/data/zxl/models/openai-community-gpt2'
OUT_DIR=ROOT/'reports'/'gpt2_inseq_generation_audit'
STOPWORDS=set('the a an and or to of in on for with is are was were be as by at from this that it there then therefore because image picture shows show likely maybe question answer reasoning conclusion'.split())


def load_split(run_root:Path,n:int)->List[Example]:
    payload=json.loads((run_root/'data_split.json').read_text(encoding='utf-8'))
    return [Example(**x) for x in payload['eval'][:n]]

def load_group(group,run_root,model_path,device):
    model,tok=load_model(model_path,device)
    state=torch.load(run_root/group/'step5000'/'model_weights.pt',map_location=device)
    model.load_state_dict(state['model'],strict=True); model.eval()
    return model,tok

def mean(xs):
    xs=list(xs); return sum(xs)/max(1,len(xs))

def token_sim(a,b):
    ta=str(a).split(); tb=str(b).split()
    if not ta and not tb: return 1.0
    return len(set(ta)&set(tb))/max(1,len(set(ta)|set(tb)))

def char_sim(a,b): return SequenceMatcher(None,str(a),str(b)).ratio()

def entities(text):
    toks=re.findall(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?", str(text))
    out=set()
    for t in toks:
        low=t.lower()
        if len(low)>2 and low not in STOPWORDS:
            if t[:1].isupper() or low not in STOPWORDS:
                out.add(low)
    return out

def overlap(a,b):
    ea,eb=entities(a),entities(b)
    if not ea and not eb: return 1.0
    return len(ea&eb)/max(1,len(ea|eb))

def extract_answer(text):
    m=re.search(r"Answer:\s*(.*)", text, flags=re.I|re.S)
    if m:
        ans=m.group(1).strip().split('\n')[0]
    else:
        ans=str(text).strip().split('\n')[-1]
    return ans[:160]

def split_generated(text):
    if 'Answer:' in text:
        cot,ans=text.split('Answer:',1)
        return cot.strip(), ans.strip()
    return text.strip(), ''

def make_prompt(tok, ex:Example, condition:str):
    if condition=='remove_think':
        prompt=f"Question:\n{ex.question}\n\n"
    else:
        prompt=f"Question:\n{ex.question}\n\n{THINK}\n"
    return encode(tok,prompt)

def clean_prompt_acts(model,tok,recs,group,max_len,device):
    # Use prompt-only sequence so shuffled source is the actual generation-time THINK context.
    ids=[]; pos=[]; pad=tok.pad_token_id or tok.eos_token_id
    think_ids=encode(tok,THINK)
    for ex in recs:
        seq=make_prompt(tok,ex,'normal')[-max_len:]
        p=-1
        for i in range(len(seq)-len(think_ids)+1):
            if seq[i:i+len(think_ids)]==think_ids: p=i; break
        ids.append(seq); pos.append(p)
    mx=max(len(x) for x in ids)
    input_ids=torch.tensor([x+[pad]*(mx-len(x)) for x in ids],dtype=torch.long,device=device)
    mask=torch.tensor([[1]*len(x)+[0]*(mx-len(x)) for x in ids],dtype=torch.long,device=device)
    positions=torch.tensor(pos,dtype=torch.long,device=device)
    with torch.no_grad():
        out=model(input_ids=input_ids,attention_mask=mask,output_hidden_states=True)
    return [h.detach() for h in out.hidden_states[1:]], positions

def greedy_batch(model,tok,recs,group,condition,max_len,max_new,device,seed):
    pad=tok.pad_token_id or tok.eos_token_id
    eos=tok.eos_token_id
    seqs=[]; attn=[]
    for ex in recs:
        ids=make_prompt(tok,ex,condition)[-max_len:]
        seqs.append(ids)
    source_acts=None; positions=None; perm=None
    if condition in ('shuffle_activation','zero_activation'):
        source_acts, positions=clean_prompt_acts(model,tok,recs,group,max_len,device)
        rng=random.Random(seed); order=list(range(len(recs))); rng.shuffle(order)
        if len(order)>1 and all(i==p for i,p in enumerate(order)): order=order[1:]+order[:1]
        perm=torch.tensor(order,dtype=torch.long,device=device)
    done=[False]*len(recs)
    for _ in range(max_new):
        mx=max(len(x) for x in seqs)
        input_ids=torch.tensor([x+[pad]*(mx-len(x)) for x in seqs],dtype=torch.long,device=device)
        mask=torch.tensor([[1]*len(x)+[0]*(mx-len(x)) for x in seqs],dtype=torch.long,device=device)
        handles=[]
        if condition=='zero_activation':
            handles=patch_block_outputs(model,source_acts,positions,'zero_think',perm)
        elif condition=='shuffle_activation':
            handles=patch_block_outputs(model,source_acts,positions,'shuffle_think',perm)
        try:
            with torch.no_grad(): out=model(input_ids=input_ids,attention_mask=mask)
        finally:
            for h in handles: h.remove()
        next_ids=[]
        for i,s in enumerate(seqs):
            if done[i]:
                nxt=eos
            else:
                nxt=int(out.logits[i,len(s)-1].argmax(-1).detach().cpu())
                if nxt==eos: done[i]=True
            next_ids.append(nxt)
        for i,nxt in enumerate(next_ids):
            if not done[i] or nxt==eos:
                seqs[i].append(nxt)
        if all(done): break
    rows=[]
    for ex,full in zip(recs,seqs):
        prompt_len=len(make_prompt(tok,ex,condition))
        gen_ids=full[prompt_len:]
        text=tok.decode(gen_ids,skip_special_tokens=True)
        cot,ans=split_generated(text)
        ext=extract_answer('Answer:'+ans if ans else text)
        hit=bool(ex.answer.strip() and ex.answer.strip().lower() in (ans or text).lower())
        rows.append({'sample_id':ex.sample_id,'question':ex.question,'gold_cot':ex.cot,'gold_answer':ex.answer,'condition':condition,'generated_text':text,'generated_cot':cot,'generated_answer':ans,'extracted_answer':ext,'exact_answer_hit':hit,'output_length':len(text.split()),'intervention_type':'in_sequence_activation_patch' if 'activation' in condition else ('remove_think_token' if condition=='remove_think' else 'normal')})
    return rows

def write_jsonl(path,rows):
    with open(path,'w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def diagnose(normal, other):
    bits=[]
    if normal.get('extracted_answer')!=other.get('extracted_answer'): bits.append('answer changes')
    if other.get('output_length',0)<5: bits.append('output degenerates')
    if token_sim(normal.get('generated_text',''),other.get('generated_text',''))>0.95: bits.append('no visible change')
    if overlap(normal.get('generated_text',''),other.get('generated_text',''))<0.5: bits.append('entity changes')
    if len(set(str(other.get('generated_text','')).split()))<8 and other.get('output_length',0)>20: bits.append('reasoning becomes templated')
    return '; '.join(bits) or 'surface form changes'

def summarize_group(group,data):
    normal=data['normal']; out={}
    for cond,rows in data.items(): out[f'{cond}_answer_accuracy']=mean(float(r['exact_answer_hit']) for r in rows)
    for cond in ['shuffle_activation','zero_activation','remove_think']:
        rows=data[cond]
        sims=[token_sim(n['generated_text'],r['generated_text']) for n,r in zip(normal,rows)]
        entity_gold_n=[overlap(n['generated_text'],n['gold_cot']+' '+n['gold_answer']) for n in normal]
        entity_gold_r=[overlap(r['generated_text'],r['gold_cot']+' '+r['gold_answer']) for r in rows]
        out[f'normal_{cond}_avg_similarity']=mean(sims)
        out[f'answer_flip_normal_vs_{cond}']=sum(n['extracted_answer']!=r['extracted_answer'] for n,r in zip(normal,rows))
        out[f'normal_success_but_{cond}_failure_count']=sum(n['exact_answer_hit'] and not r['exact_answer_hit'] for n,r in zip(normal,rows))
        out[f'normal_{cond}_nearly_identical_count']=sum(s>0.98 for s in sims)
        out[f'avg_length_change_{cond}']=mean(r['output_length']-n['output_length'] for n,r in zip(normal,rows))
        out[f'entity_overlap_drop_normal_to_{cond}']=mean(a-b for a,b in zip(entity_gold_n,entity_gold_r))
        for r,n,s in zip(rows,normal,sims):
            r['token_similarity_to_normal']=s
            r['entity_overlap_with_gold']=overlap(r['generated_text'],r['gold_cot']+' '+r['gold_answer'])
            r['entity_overlap_with_normal']=overlap(r['generated_text'],n['generated_text'])
    for r in normal:
        r['token_similarity_to_normal']=1.0
        r['entity_overlap_with_gold']=overlap(r['generated_text'],r['gold_cot']+' '+r['gold_answer'])
        r['entity_overlap_with_normal']=1.0
    gate=[]
    if out['normal_shuffle_activation_avg_similarity']<=0.90: gate.append('similarity<=0.90')
    if out['answer_flip_normal_vs_shuffle_activation']>=5: gate.append('answer_flip>=5')
    if out['normal_success_but_shuffle_activation_failure_count']>=3: gate.append('normal_success_shuffle_failure>=3')
    if out['entity_overlap_drop_normal_to_shuffle_activation']>=0.05: gate.append('entity_drop>=0.05')
    # Qualitative differences proxy: low similarity or diagnostic not no visible change.
    visible=sum(token_sim(n['generated_text'],r['generated_text'])<0.90 for n,r in zip(normal,data['shuffle_activation']))
    if visible>=10: gate.append('visible_differences>=10')
    gate.append('positive_inseq_nll_delta_prior')
    out['visible_shuffle_difference_count']=visible
    out['gate_reasons']=gate
    out['gate_pass']=len(gate)>=2
    return out

def case_blocks(group,data):
    normal=data['normal']; sh=data['shuffle_activation']; ze=data['zero_activation']; rm=data['remove_think']
    cases=[]
    def add(title, idxs):
        cases.append(f'### {group} {title}\n')
        for idx in idxs[:10]:
            n,s,z,r=normal[idx],sh[idx],ze[idx],rm[idx]
            cases.append(f"#### sample {n['sample_id']}\nQuestion: {n['question']}\n\nGold answer: {n['gold_answer']}\n\nNormal output: {n['generated_text'][:700]}\n\nShuffle output: {s['generated_text'][:700]}\n\nZero output: {z['generated_text'][:700]}\n\nRemove output: {r['generated_text'][:700]}\n\nDiagnosis: shuffle: {diagnose(n,s)}; zero: {diagnose(n,z)}; remove: {diagnose(n,r)}\n")
    sim_sh=[token_sim(n['generated_text'],s['generated_text']) for n,s in zip(normal,sh)]
    sim_ze=[token_sim(n['generated_text'],z['generated_text']) for n,z in zip(normal,ze)]
    sim_rm=[token_sim(n['generated_text'],r['generated_text']) for n,r in zip(normal,rm)]
    add('normal vs shuffle clearly different', sorted(range(len(normal)), key=lambda i: sim_sh[i]))
    add('normal vs zero clearly different', sorted(range(len(normal)), key=lambda i: sim_ze[i]))
    add('normal vs remove clearly different', sorted(range(len(normal)), key=lambda i: sim_rm[i]))
    add('nearly unchanged', sorted(range(len(normal)), key=lambda i: -(sim_sh[i]+sim_ze[i]+sim_rm[i])))
    return '\n'.join(cases)

def run_audit(args):
    device=torch.device(args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu')
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    examples=load_split(args.run_root,args.eval_samples)
    summary={'eval_samples':len(examples),'seed':args.seed,'max_new_tokens':args.max_new_tokens,'groups':{}}
    analyses=['# GPT2 In-Sequence Generation Audit','',f'Run root: `{args.run_root}`','']
    for group in [g.strip() for g in args.groups.split(',') if g.strip()]:
        model,tok=load_group(group,args.run_root,args.model_path,device)
        data={}
        for cond,fncond in [('normal','normal'),('shuffle_activation','shuffle_activation'),('zero_activation','zero_activation'),('remove_think','remove_think')]:
            rows=[]
            for st in range(0,len(examples),args.batch_size):
                rows.extend(greedy_batch(model,tok,examples[st:st+args.batch_size],group,cond,args.max_length,args.max_new_tokens,device,args.seed+st))
            data[cond]=rows
            out_name={'normal':'normal','shuffle_activation':'shuffle_activation','zero_activation':'zero_activation','remove_think':'remove_think'}[cond]
            write_jsonl(OUT_DIR/f'{group}_{out_name}_generation.jsonl',rows)
        metrics=summarize_group(group,data)
        summary['groups'][group]=metrics
        analyses.append(f'## {group} Summary\n')
        analyses.append('| metric | value |\n|---|---:|')
        for k,v in metrics.items():
            if isinstance(v,(int,float,bool)): analyses.append(f'| {k} | {v} |')
        analyses.append(f"\nGate pass: `{metrics['gate_pass']}`; reasons: {', '.join(metrics['gate_reasons'])}\n")
        analyses.append(case_blocks(group,data))
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    (OUT_DIR/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT_DIR/'analysis.md').write_text('\n'.join(analyses)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-root',type=Path,default=DEFAULT_RUN_ROOT)
    ap.add_argument('--model-path',default=DEFAULT_MODEL)
    ap.add_argument('--groups',default='G3,G4')
    ap.add_argument('--eval-samples',type=int,default=256)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--max-length',type=int,default=640)
    ap.add_argument('--max-new-tokens',type=int,default=128)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args(); run_audit(args)
if __name__=='__main__': main()
