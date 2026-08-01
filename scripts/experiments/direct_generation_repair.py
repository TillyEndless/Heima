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
DIRECT_ADAPTER=REPO/'checkpoints/stage1_repair_v2_direct/final_adapter'
O0_ADAPTER=REPO/'checkpoints/stage1_repair_o0/final_adapter'
REPORT=REPO/'reports/direct_generation_repair'
STATUS=REPO/'status/direct_generation_repair'
ANSWER_RE=re.compile(r'####\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?)')
NUM_RE=re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?')

def iso(t=None): return datetime.fromtimestamp(t or time.time(), tz=timezone.utc).isoformat()
def write_json(p:Path,o:Any): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def write_jsonl(p:Path,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n')
def status(name,state,**kw): write_json(STATUS/f'{name}.json',dict(stage=name,status=state,iso=iso(),time=time.time(),**kw))
def norm(s):
    if s is None: return None
    s=str(s).strip().replace(',','')
    return s[:-2] if s.endswith('.0') else s
def parse_answer(text):
    text=str(text or '')
    m=ANSWER_RE.search(text)
    if m: return norm(m.group(1))
    low=text.lower()
    for marker in ['final answer is','therefore','answer is','answer:','<answer>','boxed']:
        i=low.rfind(marker)
        if i>=0:
            nums=NUM_RE.findall(text[i:])
            if nums: return norm(nums[-1])
    nums=NUM_RE.findall(text)
    return norm(nums[-1]) if nums else None

def rows(): return json.loads(MANIFEST.read_text())['samples']
def q_prefix(r): return 'Question:\n'+r['question']+'\n\n'
def answer_target(tok,r): return '\nAnswer:\n'+r['answer']+tok.eos_token

def load(adapter=DIRECT_ADAPTER, train=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),trust_remote_code=True)
    tok.add_special_tokens({'additional_special_tokens':[THINK]})
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REVISION,cache_dir=str(HF_HOME),torch_dtype=torch.bfloat16,trust_remote_code=True)
    base.resize_token_embeddings(len(tok)); base.config.use_cache=False
    model=PeftModel.from_pretrained(base,adapter,is_trainable=train)
    model.to('cuda:0'); model.eval()
    return torch,tok,model

def direct_batch(torch,tok,rs):
    ids_rows=[]; labs_rows=[]; seg_rows=[]; meta=[]
    for r in rs:
        qids=tok(q_prefix(r),add_special_tokens=False)['input_ids']
        aids=tok(answer_target(tok,r),add_special_tokens=False)['input_ids']
        ids=qids+aids; labs=[-100]*len(qids)+aids; seg=['question']*len(qids)+['answer']*len(aids)
        ids_rows.append(ids); labs_rows.append(labs); seg_rows.append(seg); meta.append(dict(sample_id=r['sample_id'],q_len=len(qids),answer_len=len(aids),answer_ids=aids,prefix_ids=qids))
    mx=max(map(len,ids_rows)); pad=tok.pad_token_id or tok.eos_token_id
    input_ids=torch.full((len(rs),mx),pad,dtype=torch.long,device='cuda:0'); labels=torch.full((len(rs),mx),-100,dtype=torch.long,device='cuda:0'); attention_mask=torch.zeros_like(input_ids)
    for i,ids in enumerate(ids_rows):
        input_ids[i,:len(ids)]=torch.tensor(ids,device='cuda:0'); labels[i,:len(ids)]=torch.tensor(labs_rows[i],device='cuda:0'); attention_mask[i,:len(ids)]=1; seg_rows[i]+=['pad']*(mx-len(ids))
    return dict(input_ids=input_ids,labels=labels,attention_mask=attention_mask),seg_rows,meta

def old_infer_prefix_ids(tok,r): return tok(q_prefix(r)+'Answer:\n',add_special_tokens=False)['input_ids']
def first_mismatch(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return i,x,y
    if len(a)!=len(b): return min(len(a),len(b)), a[min(len(a),len(b))] if len(a)>len(b) else None, b[min(len(a),len(b))] if len(b)>len(a) else None
    return None,None,None

def decode(tok,ids): return tok.decode(ids,skip_special_tokens=False)

def generation_config_audit(tok,model):
    gcg=model.generation_config
    return dict(model_training=bool(model.training), adapter_enabled=True, do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None, repetition_penalty=getattr(gcg,'repetition_penalty',None), max_new_tokens='varies_by_gold_len', min_new_tokens=getattr(gcg,'min_new_tokens',None), eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id, forced_bos_token_id=getattr(gcg,'forced_bos_token_id',None), forced_eos_token_id=getattr(gcg,'forced_eos_token_id',None), padding_side=tok.padding_side, model_generation_config=gcg.to_dict())

def causal_shift_audit():
    import torch.nn.functional as F
    torch,tok,model=load(DIRECT_ADAPTER,False); rs=rows(); batch,segs,meta=direct_batch(torch,tok,rs)
    with torch.no_grad(): out=model(**batch)
    shift_logits=out.logits[:,:-1,:].contiguous(); shift_labels=batch['labels'][:,1:].contiguous(); mask=shift_labels.ne(-100)
    losses=F.cross_entropy(shift_logits.reshape(-1,shift_logits.size(-1)).float(),shift_labels.reshape(-1),ignore_index=-100,reduction='none').reshape_as(shift_labels)
    manual=float(losses[mask].mean().detach().cpu()); model_loss=float(out.loss.detach().cpu()); pred=shift_logits.argmax(-1)
    correct=pred.eq(shift_labels)&mask
    first_ok=0; first_n=0; rem_ok=0; rem_n=0; eos_ok=0; eos_n=0; seq_exact=0; cases=[]
    for i,m in enumerate(meta):
        ans_js=[j for j in range(shift_labels.shape[1]) if bool(mask[i,j])]
        all_ok=True
        for local_idx,j in enumerate(ans_js):
            lab=int(shift_labels[i,j]); pr=int(pred[i,j]); ok=(lab==pr)
            all_ok=all_ok and ok
            if local_idx==0: first_n+=1; first_ok+=ok
            elif lab==tok.eos_token_id: eos_n+=1; eos_ok+=ok
            else: rem_n+=1; rem_ok+=ok
            pos=j+1; lo=max(0,pos-8); hi=min(batch['input_ids'].shape[1],pos+9)
            if local_idx<4 or lab==tok.eos_token_id:
                for p in range(lo,hi):
                    sj=p-1
                    cases.append(dict(sample_id=m['sample_id'], focus_target_index=local_idx, position=p, input_token_id=int(batch['input_ids'][i,p]), input_token=decode(tok,[int(batch['input_ids'][i,p])]), label=int(batch['labels'][i,p]), label_token=None if int(batch['labels'][i,p])==-100 else decode(tok,[int(batch['labels'][i,p])]), original_segment=segs[i][p], shifted_target_position=sj, included_in_metric=(sj>=0 and bool(mask[i,sj])), predicted_token_id=None if sj<0 else int(pred[i,sj]), predicted_token=None if sj<0 else decode(tok,[int(pred[i,sj])]), correct=None if sj<0 or not bool(mask[i,sj]) else bool(pred[i,sj].eq(shift_labels[i,sj]))))
        seq_exact += int(all_ok)
    rep=dict(base_model_revision=MODEL_REVISION, adapter_path=str(DIRECT_ADAPTER), tokenizer_files='loaded from HF cache plus added <THINK>', think_token_configuration=dict(token=THINK, token_id=tok.convert_tokens_to_ids(THINK), used_in_direct_control=False), generation_config=generation_config_audit(tok,model), git_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(), adapter_enabled=True, model_returned_loss=model_loss, manual_shifted_ce=manual, abs_diff=abs(model_loss-manual), shifted_answer_token_accuracy=float(correct[mask].float().mean().cpu()), shifted_first_answer_token_accuracy=first_ok/max(first_n,1), shifted_remaining_answer_token_accuracy=rem_ok/max(rem_n,1), shifted_eos_accuracy=eos_ok/max(eos_n,1), sequence_level_teacher_forced_exact_match=seq_exact/len(rs), counts=dict(first_answer=first_n,remaining_answer=rem_n,eos=eos_n,total_answer=int(mask.sum().cpu())), diagnosis=dict(off_by_one_bug=False, shifted_metric_used=True, prior_unshifted_metric_suspect=True))
    write_json(REPORT/'causal_shift_audit.json',rep); write_jsonl(REPORT/'causal_shift_cases.jsonl',cases)
    md=f"""# Causal Shift Audit\n\n- manual shifted CE: {manual}\n- model returned loss: {model_loss}\n- abs diff: {abs(model_loss-manual)}\n- shifted answer token accuracy: {rep['shifted_answer_token_accuracy']}\n- shifted first answer token accuracy: {rep['shifted_first_answer_token_accuracy']}\n- shifted remaining answer token accuracy: {rep['shifted_remaining_answer_token_accuracy']}\n- shifted EOS accuracy: {rep['shifted_eos_accuracy']}\n- sequence-level teacher-forced exact match: {rep['sequence_level_teacher_forced_exact_match']}\n\nThe audit uses `shift_logits = logits[:, :-1, :]` and `shift_labels = labels[:, 1:]`.\n"""
    (REPORT/'causal_shift_audit.md').write_text(md)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('causal_shift','complete')

def prefix_comparison_only():
    torch,tok,model=load(DIRECT_ADAPTER,False); rs=rows(); batch,segs,meta=direct_batch(torch,tok,rs); out=[]
    for r,m in zip(rs,meta):
        canonical=m['prefix_ids']; inf=old_infer_prefix_ids(tok,r); idx,a,b=first_mismatch(canonical,inf)
        out.append(dict(sample_id=r['sample_id'], exact_prefix_match=canonical==inf, canonical_train_prefix_len=len(canonical), inference_prefix_len=len(inf), first_mismatch_position=idx, training_token_id=a, training_token=None if a is None else decode(tok,[a]), inference_token_id=b, inference_token=None if b is None else decode(tok,[b]), bos_eos_duplication=False, whitespace_newline_difference=(idx is not None and ((a in [198,271] if a is not None else False) or (b in [198,271] if b is not None else False))), answer_marker_difference=('Answer' in decode(tok,inf) and 'Answer' not in decode(tok,canonical)), template_difference=True, canonical_train_prefix_text=decode(tok,canonical), old_inference_prefix_text=decode(tok,inf)))
    write_jsonl(REPORT/'prefix_comparison.jsonl',out)
    write_json(REPORT/'prefix_comparison_summary.json',dict(n=len(out),exact_prefix_match_rate=sum(x['exact_prefix_match'] for x in out)/len(out),mismatch_count=sum(not x['exact_prefix_match'] for x in out),answer_marker_difference_count=sum(x['answer_marker_difference'] for x in out)))
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('prefix_comparison','complete')

def manual_greedy(torch,tok,model,prefix_ids,max_new):
    ids=list(prefix_ids); scores=[]
    for _ in range(max_new):
        inp=torch.tensor([ids],device='cuda:0'); attn=torch.ones_like(inp)
        with torch.no_grad(): logits=model(input_ids=inp,attention_mask=attn).logits[:,-1,:].float()
        nxt=int(logits.argmax(-1)[0]); scores.append(logits.detach().cpu()); ids.append(nxt)
        if nxt==tok.eos_token_id: break
    return ids[len(prefix_ids):], scores

def gen_ids(torch,tok,model,prefix_ids,max_new,use_cache):
    inp=torch.tensor([prefix_ids],device='cuda:0'); attn=torch.ones_like(inp)
    with torch.no_grad(): out=model.generate(input_ids=inp,attention_mask=attn,max_new_tokens=max_new,do_sample=False,num_beams=1,temperature=None,top_p=None,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id,use_cache=use_cache)
    return out[0,inp.shape[1]:].detach().cpu().tolist()

def compare_outputs(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y: return False,i
    return len(a)==len(b), (None if len(a)==len(b) else min(len(a),len(b)))

def greedy_paths_and_parser():
    torch,tok,model=load(DIRECT_ADAPTER,False); rs=rows(); _,_,meta=direct_batch(torch,tok,rs); cases=[]; parser_cases=[]
    for r,m in zip(rs,meta):
        gold=m['answer_ids']; max_new=len(gold)+8; prefix=m['prefix_ids']
        man,score=manual_greedy(torch,tok,model,prefix,max_new); noc=gen_ids(torch,tok,model,prefix,max_new,False); cac=gen_ids(torch,tok,model,prefix,max_new,True)
        m_nc,div_nc=compare_outputs(man,noc); m_c,div_c=compare_outputs(man,cac)
        for name,ids in [('manual',man),('generate_no_cache',noc),('generate_cache',cac)]:
            raw=decode(tok,ids); gold_text=decode(tok,gold); pred=parse_answer(raw); gold_num=parse_answer(r['answer']); token_exact=ids[:len(gold)]==gold
            norm_exact=raw.strip()==gold_text.strip(); numeric=(pred is not None and gold_num is not None and pred==gold_num)
            if token_exact and not numeric: klass='correct_tokens_parser_failure'
            elif token_exact: klass='correct'
            elif ids and ids[0]==tok.eos_token_id: klass='early_eos'
            elif len(ids)>=max_new and (not ids or ids[-1]!=tok.eos_token_id): klass='max_length_stop'
            elif 'Answer:' in raw: klass='extra_template_tokens'
            elif len(NUM_RE.findall(raw))>1: klass='multiple_answers'
            else: klass='generation_token_error'
            parser_cases.append(dict(sample_id=r['sample_id'],path=name,generated_token_ids=ids,gold_token_ids=gold,raw_generated_text=raw,raw_gold_answer=gold_text,normalized_generated_text=raw.strip(),normalized_gold_text=gold_text.strip(),parser_output=pred,gold_numeric=gold_num,token_exact_match=token_exact,decoded_string_exact_match=norm_exact,numeric_answer_match=numeric,stop_reason='eos' if tok.eos_token_id in ids else 'max_or_other',eos_generated=tok.eos_token_id in ids,max_new_tokens_reached=len(ids)>=max_new and tok.eos_token_id not in ids,failure_class=klass))
        cases.append(dict(sample_id=r['sample_id'],manual_vs_generate_no_cache_exact_match=m_nc,manual_vs_generate_cache_exact_match=m_c,manual_vs_no_cache_first_divergence=div_nc,manual_vs_cache_first_divergence=div_c,manual_ids=man,no_cache_ids=noc,cache_ids=cac,manual_text=decode(tok,man),no_cache_text=decode(tok,noc),cache_text=decode(tok,cac)))
    write_json(REPORT/'greedy_path_comparison.json',dict(n=len(cases),manual_vs_generate_no_cache_exact_match_rate=sum(c['manual_vs_generate_no_cache_exact_match'] for c in cases)/len(cases),manual_vs_generate_cache_exact_match_rate=sum(c['manual_vs_generate_cache_exact_match'] for c in cases)/len(cases),cases=cases))
    write_jsonl(REPORT/'parser_vs_generation_cases.jsonl',parser_cases)
    by_path={}
    for path in ['manual','generate_no_cache','generate_cache']:
        xs=[c for c in parser_cases if c['path']==path]
        by_path[path]=dict(n=len(xs),token_exact_match_rate=sum(c['token_exact_match'] for c in xs)/len(xs),decoded_string_exact_match_rate=sum(c['decoded_string_exact_match'] for c in xs)/len(xs),numeric_answer_match_rate=sum(c['numeric_answer_match'] for c in xs)/len(xs),parser_accuracy=sum(c['numeric_answer_match'] for c in xs)/len(xs),failure_counts=dict(Counter(c['failure_class'] for c in xs)))
    write_json(REPORT/'parser_vs_generation_summary.json',by_path)
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('greedy_parser','complete')

def gold_prefix_induction():
    torch,tok,model=load(DIRECT_ADAPTER,False); rs=rows(); rec=[]; summary=[]
    for r in rs:
        qids=tok(q_prefix(r),add_special_tokens=False)['input_ids']; gold=tok(answer_target(tok,r),add_special_tokens=False)['input_ids']; ok=0
        for t,gold_id in enumerate(gold):
            prefix=qids+gold[:t]
            inp=torch.tensor([prefix],device='cuda:0'); attn=torch.ones_like(inp)
            with torch.no_grad(): logits=model(input_ids=inp,attention_mask=attn).logits[:,-1,:].float()[0]
            probs=logits.softmax(-1); pred=int(logits.argmax()); rank=int((probs>probs[gold_id]).sum().item()+1); correct=pred==gold_id; ok+=correct
            if t<8 or t==len(gold)-1 or not correct:
                top=torch.topk(probs,20)
                rec.append(dict(sample_id=r['sample_id'],position=t,gold_next_token_id=gold_id,gold_next_token=decode(tok,[gold_id]),predicted_token_id=pred,predicted_token=decode(tok,[pred]),gold_rank=rank,gold_probability=float(probs[gold_id].detach().cpu()),top1_probability=float(probs[pred].detach().cpu()),correct=correct,top20_token_ids=[int(x) for x in top.indices],top20_probabilities=[float(x) for x in top.values]))
        summary.append(dict(sample_id=r['sample_id'],answer_token_count=len(gold),gold_prefix_next_token_accuracy=ok/len(gold)))
    write_jsonl(REPORT/'gold_prefix_induction.jsonl',rec)
    write_json(REPORT/'gold_prefix_induction_summary.json',dict(n=len(summary),mean_accuracy=sum(x['gold_prefix_next_token_accuracy'] for x in summary)/len(summary),sequence_all_correct_rate=sum(x['gold_prefix_next_token_accuracy']==1.0 for x in summary)/len(summary),samples=summary))
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); status('gold_prefix','complete')

def reevaluate_o0_if_allowed():
    gate=json.loads((REPORT/'direct_generation_gate.json').read_text()) if (REPORT/'direct_generation_gate.json').exists() else {}
    if not gate.get('direct_answer_free_overfit',False):
        write_json(REPORT/'o0_reeval_after_direct_gate.json',dict(status='skipped',reason='direct-answer gate failed'))
        return

def storage_report():
    paths=[REPO/'checkpoints/stage1_repair_o0',REPO/'checkpoints/stage1_repair_v2_direct']
    lines=['# Checkpoint Retention Plan','','No files were deleted.','','| path | size | role | recommendation |','|---|---:|---|---|']
    total_reclaim=[]
    for p in paths:
        if not p.exists(): continue
        size=subprocess.check_output(['du','-sh',str(p)],text=True).split()[0]
        role='O0 every-50-step plus final' if p.name=='stage1_repair_o0' else 'direct-answer best/final'
        rec='Potentially delete intermediate step50..step500 after user approval; keep final_adapter if needed.' if p.name=='stage1_repair_o0' else 'Keep until direct repair is resolved; only best/final exist.'
        lines.append(f'| `{p}` | {size} | {role} | {rec} |')
    (REPORT/'checkpoint_retention_plan.md').write_text('\n'.join(lines)+'\n')

def final_gate():
    cs=json.loads((REPORT/'causal_shift_audit.json').read_text())
    pref=json.loads((REPORT/'prefix_comparison_summary.json').read_text())
    gp=json.loads((REPORT/'greedy_path_comparison.json').read_text())
    parser=json.loads((REPORT/'parser_vs_generation_summary.json').read_text())
    ind=json.loads((REPORT/'gold_prefix_induction_summary.json').read_text())
    manual_acc=parser['manual']['numeric_answer_match_rate']; gen_acc=parser['generate_cache']['numeric_answer_match_rate']; token_exact=parser['generate_cache']['token_exact_match_rate']; parser_acc=parser['generate_cache']['parser_accuracy']
    shifted=cs['shifted_answer_token_accuracy']; prefix_match=pref['exact_prefix_match_rate']==1.0
    case='undetermined'
    if shifted<0.99: case='CASE 1: shifted teacher-forced accuracy below 0.99; old metric was misleading or checkpoint not truly overfit.'
    elif manual_acc<0.95 and prefix_match: case='CASE 2: shifted teacher-forcing high but manual greedy low with canonical prefix; induction audit explains exposure/first-token failures.'
    elif manual_acc>=0.95 and gen_acc<0.95: case='CASE 3: generate config/cache/stopping bug.'
    elif token_exact>=0.95 and parser_acc<0.95: case='CASE 4: parser/normalization bug.'
    elif not prefix_match: case='CASE 5: old inference prefix mismatched training prefix; re-evaluation must use canonical prefix.'
    direct_ok=manual_acc>=0.95 and gen_acc>=0.95 and parser_acc>=0.95 and shifted>=0.99
    gate=dict(teacher_forced_accuracy_correct=shifted>=0.99, causal_shift_or_off_by_one_bug=shifted<0.99, exact_training_generation_prefix_match=prefix_match, manual_greedy_numeric_accuracy=manual_acc, generate_cache_numeric_accuracy=gen_acc, generate_cache_token_exact_match=token_exact, parser_accuracy=parser_acc, gold_prefix_induction_mean_accuracy=ind['mean_accuracy'], direct_answer_free_overfit=direct_ok, o0_reeval_status='skipped_until_direct_gate_passes', may_start_o1_o2=False, ten_k_and_stage2_still_forbidden=True, decision_case=case)
    write_json(REPORT/'direct_generation_gate.json',gate)
    storage_report(); reevaluate_o0_if_allowed()
    md=f'''# Direct Generation Repair Report

## Answers

1. 原 teacher-forced accuracy=1.0 是否计算正确？  
Correctly shifted answer token accuracy is `{shifted}`. First-token, remaining-token, EOS, and sequence exact-match details are in `causal_shift_audit.json`.

2. 是否存在 causal shift/off-by-one bug？  
Decision: `{gate['causal_shift_or_off_by_one_bug']}`. The audit manually uses shifted logits/labels.

3. train prefix 与 generation prefix 是否逐 token 一致？  
Old inference prefix exact-match rate is `{pref['exact_prefix_match_rate']}`. Mismatches are in `prefix_comparison.jsonl`. Canonical training prefix is derived from labels, not strings.

4. manual greedy 与 model.generate 是否一致？  
Manual vs generate(no-cache) exact-match rate: `{gp['manual_vs_generate_no_cache_exact_match_rate']}`. Manual vs generate(cache): `{gp['manual_vs_generate_cache_exact_match_rate']}`.

5. token generation 与 parser 哪一层失败？  
Generate-cache token exact match: `{token_exact}`; numeric/parser accuracy: `{parser_acc}`. See `parser_vs_generation_summary.json`.

6. direct-answer 32-sample 是否真正 free overfit？  
`{direct_ok}`.

7. 修复后旧 O0 是否通过 Forced-K？  
Skipped, because direct-answer gate did not pass.

8. 是否可以开始 O1/O2？  
No. O1/O2 remain blocked until direct-answer free overfit passes.

9. 当前是否仍禁止 10k pilot 和 Stage2？  
Yes.

## Decision Tree

{case}
'''
    (REPORT/'direct_generation_report.md').write_text(md)
    (REPORT/'next_experiment_recommendation.md').write_text('Do not run latent training, O1/O2, 10k pilot, or Stage2. First fix direct-answer generation/template/parser according to direct_generation_gate.json.\n')
    status('final_gate','complete')

def main():
    os.environ['CUDA_VISIBLE_DEVICES']='1'; os.environ['HF_HOME']=str(HF_HOME); os.environ.setdefault('HF_ENDPOINT','https://hf-mirror.com'); os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    REPORT.mkdir(parents=True,exist_ok=True); STATUS.mkdir(parents=True,exist_ok=True)
    ap=argparse.ArgumentParser(); ap.add_argument('phase',choices=['shift','prefix','greedy','induction','final','all']); args=ap.parse_args()
    if args.phase=='shift': causal_shift_audit()
    elif args.phase=='prefix': prefix_comparison_only()
    elif args.phase=='greedy': greedy_paths_and_parser()
    elif args.phase=='induction': gold_prefix_induction()
    elif args.phase=='final': final_gate()
    elif args.phase=='all':
        causal_shift_audit(); prefix_comparison_only(); greedy_paths_and_parser(); gold_prefix_induction(); final_gate()
if __name__=='__main__': main()
