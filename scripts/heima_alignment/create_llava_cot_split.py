#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.heima_alignment.data.image_resolver import ImageResolver

def h(text: str) -> str: return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
def read_jsonl(path: Path):
    with path.open(encoding='utf-8') as f:
        for i,line in enumerate(f):
            if line.strip(): yield i,json.loads(line)
def question(row):
    conv=row.get('conversations') or []
    if conv: return str(conv[0].get('value','')).replace('<image>','').strip()
    return str(row.get('question',''))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl')); ap.add_argument('--dataset-root',type=Path,default=Path('/data/zxl/official_heima/datasets/LLaVA-CoT-100k')); ap.add_argument('--out',type=Path,default=Path('/data/zxl/runs/heima_ab_loss1_acceptance_v1/data_split.json')); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--train-ratio',type=float,default=0.95); ap.add_argument('--max-train',type=int,default=4096); ap.add_argument('--max-eval',type=int,default=512); ns=ap.parse_args()
    rows=list(read_jsonl(ns.input)); rng=random.Random(ns.seed); order=list(range(len(rows))); rng.shuffle(order)
    n_eval=max(ns.max_eval, int(len(rows)*(1-ns.train_ratio)))
    eval_idx=order[:n_eval][:ns.max_eval]; train_idx=order[n_eval:][:ns.max_train]
    resolver=ImageResolver(ns.dataset_root)
    def item(idx):
        orig,row=rows[idx]
        image=row.get('image','')
        resolved=resolver.resolve(image)
        return {'index':orig,'id':str(row.get('id',orig)),'image_field':image,'resolved_image_path':resolved.resolved_path,'image_path_hash':resolved.sha256_16,'question_hash':h(question(row))}
    out={'seed':ns.seed,'source':str(ns.input),'dataset_root':str(ns.dataset_root),'train_ratio':ns.train_ratio,'train':[item(i) for i in train_idx],'eval':[item(i) for i in eval_idx]}
    train_ids={x['index'] for x in out['train']}; eval_ids={x['index'] for x in out['eval']}
    if train_ids & eval_ids: raise RuntimeError('train/eval overlap')
    ns.out.parent.mkdir(parents=True,exist_ok=True); ns.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'out':str(ns.out),'train':len(out['train']),'eval':len(out['eval'])},indent=2))
if __name__=='__main__': main()
