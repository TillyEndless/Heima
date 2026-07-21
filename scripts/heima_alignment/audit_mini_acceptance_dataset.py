#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from transformers import AutoProcessor
except Exception:
    AutoProcessor = None

SECTIONS=("summary","caption","reasoning")

def read_any(path: Path):
    if not path.exists(): return []
    txt=path.read_text(encoding='utf-8')
    if path.suffix=='.jsonl': return [json.loads(x) for x in txt.splitlines() if x.strip()]
    obj=json.loads(txt)
    if isinstance(obj,list): return obj
    for k in ['data','train','validation','test','samples']:
        if isinstance(obj.get(k),list): return obj[k]
    return []

def find_files(root: Path):
    candidates=[]
    for p in root.rglob('*'):
        if p.is_file() and p.suffix in {'.json','.jsonl'}:
            name=p.name.lower()
            if any(x in name for x in ['train','test','val','validation']) or 'data_' in name:
                candidates.append(p)
    return sorted(candidates)

def get(row: dict[str,Any], *keys, default=''):
    for k in keys:
        v=row.get(k)
        if v not in (None,''):
            return str(v)
    return default

def normalize(row, idx):
    conv=row.get('conversations')
    if isinstance(conv,list) and len(conv)>=2:
        q=str(conv[0].get('value','')).replace('<image>','').strip()
        ans=str(conv[1].get('value',''))
    else:
        q=get(row,'question','prompt')
        ans=get(row,'answer','final_answer')
    def section(name):
        v=get(row,name)
        if v: return v
        up=name.upper()
        if f'<{up}>' in ans and f'</{up}>' in ans:
            return ans.split(f'<{up}>',1)[1].split(f'</{up}>',1)[0].strip()
        return ''
    return {'id':get(row,'id','sample_id',default=str(idx)),'image':get(row,'image','image_path'),'question':q,'summary':section('summary'),'caption':section('caption'),'reasoning':section('reasoning'),'answer':get(row,'answer','final_answer',default=ans)}

def pct(vals,p):
    if not vals: return None
    vals=sorted(vals); k=(len(vals)-1)*p/100; lo=math.floor(k); hi=math.ceil(k)
    if lo==hi: return vals[lo]
    return vals[lo]*(hi-k)+vals[hi]*(k-lo)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset-root',type=Path,default=Path('/data/zxl/official_heima/datasets/LLaVA-CoT-100k')); ap.add_argument('--model-a-path',default='/data/zxl/small_models/Qwen2.5-VL-3B-Instruct'); ap.add_argument('--out',type=Path,default=Path('docs/heima_alignment/mini_acceptance_dataset_audit.json')); ap.add_argument('--max-tokenize',type=int,default=20000); ns=ap.parse_args()
    root=ns.dataset_root
    files=find_files(root)
    splits={}
    for p in files:
        rows=read_any(p)
        if not rows: continue
        lname=p.name.lower()
        if 'train' in lname: key='train'
        elif 'validation' in lname or 'val' in lname: key='validation'
        elif 'test' in lname: key='test'
        else: key=p.stem
        # prefer prepared num_thinking files over raw metadata for train/test
        if key in splits and 'num_thinking' not in p.name:
            continue
        splits[key]={'path':str(p),'raw_count':len(rows),'records':[normalize(r,i) for i,r in enumerate(rows)]}
    image_dirs=[root/'image_files', root/'images', root/'image']
    image_dirs += [p for p in root.rglob('*') if p.is_dir() and 'image' in p.name.lower()][:10]
    tokenizer=None
    if AutoProcessor is not None and Path(ns.model_a_path).exists():
        try: tokenizer=AutoProcessor.from_pretrained(ns.model_a_path,local_files_only=True,trust_remote_code=True).tokenizer
        except Exception: tokenizer=None
    report={'dataset_root':str(root),'exists':root.exists(),'files_considered':[str(p) for p in files[:50]],'splits':{},'overall_status':'pass'}
    all_keys=[]
    for key,obj in splits.items():
        recs=obj['records']; n=len(recs)
        complete={field:sum(bool(r[field]) for r in recs)/max(n,1) for field in ['image','question','summary','caption','reasoning','answer']}
        accessible=0
        checked=0
        for r in recs[:min(n,20000)]:
            checked+=1
            img=Path(r['image'])
            paths=[img] if img.is_absolute() else [d/img for d in image_dirs]
            if any(p.exists() for p in paths): accessible+=1
        lens=[]
        trunc=0
        sample=recs[:min(n,ns.max_tokenize)]
        if tokenizer is not None:
            for r in sample:
                text=f"<image>\nQuestion: {r['question']}\n<SUMMARY> {r['summary']} </SUMMARY>\n<CAPTION> {r['caption']} </CAPTION>\n<REASONING> {r['reasoning']} </REASONING>\n<CONCLUSION> {r['answer']} </CONCLUSION>"
                ids=tokenizer(text,add_special_tokens=False).get('input_ids',[])
                lens.append(len(ids)); trunc += int(len(ids)>2048)
        pair=[(r['image'],r['question']) for r in recs]
        dup=sum(c-1 for c in Counter(pair).values() if c>1)/max(n,1)
        report['splits'][key]={'path':obj['path'],'sample_count':n,'field_complete_rate':complete,'image_accessible_checked':checked,'image_accessible_ratio':accessible/max(checked,1),'token_length_stats':{'p50':pct(lens,50),'p90':pct(lens,90),'p95':pct(lens,95),'p99':pct(lens,99),'max':max(lens) if lens else None,'tokenized_count':len(lens)},'truncation_ratio_over_2048':trunc/max(len(lens),1) if lens else None,'duplicate_image_question_ratio':dup}
        if key in ['train','validation','test']:
            all_keys.append(key)
    # Determine pass: root exists, train >=4096, validation/test or test >=512, required fields complete enough, images >95% accessible.
    train=report['splits'].get('train',{})
    eval_split=report['splits'].get('validation') or report['splits'].get('test') or {}
    failures=[]
    if not root.exists(): failures.append('dataset_root_missing')
    if train.get('sample_count',0)<4096: failures.append('train_lt_4096')
    if eval_split.get('sample_count',0)<512: failures.append('eval_lt_512')
    for name,sp in [('train',train),('eval',eval_split)]:
        if sp:
            if sp.get('image_accessible_ratio',0)<0.95: failures.append(f'{name}_image_access_lt_95pct')
            for f,v in sp.get('field_complete_rate',{}).items():
                if v<0.95: failures.append(f'{name}_{f}_complete_lt_95pct')
    report['failures']=failures; report['overall_status']='pass' if not failures else 'fail'
    ns.out.parent.mkdir(parents=True,exist_ok=True); ns.out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'out':str(ns.out),'status':report['overall_status'],'failures':failures},indent=2))
    if failures: raise SystemExit(2)
if __name__=='__main__': main()
