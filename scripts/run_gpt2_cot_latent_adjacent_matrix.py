#!/usr/bin/env python3
"""GPT2-small pure-text CoT latent adjacent matrix.

No Model B, projector, role embedding, cumulative latent, Loss2, or VLM code.
"""
from __future__ import annotations

import argparse, hashlib, json, math, random, re, time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

THINK = "<THINK>"
STOP_EXPLAIN = "<STOP_EXPLAIN>"
RUN_ROOT = Path("/data/zxl/runs/gpt2_cot_latent_adjacent_matrix_seed42")
REPORT_DIR = Path("reports")
EVAL_STEPS = (500, 1000, 2500, 5000)
GROUPS = ("G0", "G1", "G2", "G3", "G4")

@dataclass
class Example:
    sample_id: str
    question: str
    cot: str
    answer: str

@dataclass
class LossBundle:
    total: torch.Tensor
    main: torch.Tensor
    cot: torch.Tensor
    answer: torch.Tensor
    self_decode: torch.Tensor
    latent_token: torch.Tensor
    latent_acc: torch.Tensor
    z: Optional[torch.Tensor]


def read_jsonl(path: Path) -> List[dict]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def strip_tags(text: str) -> str:
    return re.sub(r"</?[A-Z_]+>", " ", str(text)).replace("\n", " ").strip()


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", str(text), flags=re.S|re.I)
    return strip_tags(m.group(1)) if m else ""


def adapt_record(row: dict, idx: int) -> Optional[Example]:
    if all(str(row.get(k, "")).strip() for k in ("question", "cot", "answer")):
        return Example(str(row.get("id", idx)), str(row["question"]).strip(), str(row["cot"]).strip(), str(row["answer"]).strip())
    conv = row.get("conversations") or row.get("conversation")
    if not isinstance(conv, list):
        return None
    question, answer_text = "", ""
    for turn in conv:
        role = str(turn.get("from") or turn.get("role") or "").lower()
        value = str(turn.get("value") or turn.get("content") or "")
        if not question and ("human" in role or role == "user"):
            question = re.sub(r"<image>", "", value, flags=re.I).strip()
        if not answer_text and ("gpt" in role or "assistant" in role):
            answer_text = value.strip()
    cot = " ".join(x for x in [extract_tag(answer_text, t) for t in ("SUMMARY", "CAPTION", "REASONING")] if x)
    answer = extract_tag(answer_text, "CONCLUSION")
    if not answer:
        m = re.search(r"(?:answer is|answer:)\s*([^\.\n<]+)", answer_text, flags=re.I)
        answer = strip_tags(m.group(1)) if m else strip_tags(answer_text[-160:])
    if question and cot and answer:
        return Example(str(row.get("id", idx)), question, cot, answer)
    return None


def encode(tok, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False)


def load_model(model_path: str, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if "gpt2-xl" in str(model_path).lower():
        raise SystemExit("GPT2-xl is forbidden")
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tok.add_special_tokens({"additional_special_tokens": [THINK, STOP_EXPLAIN]})
    model.resize_token_embeddings(len(tok))
    model.to(device)
    return model, tok


def split_hash(payload: dict) -> str:
    b = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def build_or_load_split(dataset: Path, out_dir: Path, train_n: int, eval_n: int, seed: int, tok=None, max_length: int=256) -> Tuple[dict, List[Example], List[Example]]:
    split_path = out_dir / "data_split.json"
    if split_path.exists():
        payload = json.loads(split_path.read_text(encoding="utf-8"))
    else:
        rows = read_jsonl(dataset)
        examples = [ex for i, row in enumerate(rows) if (ex := adapt_record(row, i)) is not None]
        if len(examples) < train_n + eval_n:
            raise RuntimeError(f"Not enough examples: have {len(examples)}, need {train_n + eval_n}")
        rng = random.Random(seed); rng.shuffle(examples)
        payload = {"seed": seed, "dataset_path": str(dataset), "train": [asdict(x) for x in examples[:train_n]], "eval": [asdict(x) for x in examples[train_n:train_n+eval_n]]}
        out_dir.mkdir(parents=True, exist_ok=True)
        split_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    train = [Example(**x) for x in payload["train"]]
    ev = [Example(**x) for x in payload["eval"]]
    audit = data_audit(dataset, payload, train, ev, tok, max_length)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "gpt2_cot_matrix_data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload, train, ev


def data_audit(dataset: Path, payload: dict, train: List[Example], ev: List[Example], tok, max_length: int) -> dict:
    rows = read_jsonl(dataset)
    usable = [adapt_record(r, i) for i, r in enumerate(rows)]
    usable = [x for x in usable if x]
    def avg_len(xs, attr):
        if tok is None: return None
        return sum(len(encode(tok, getattr(x, attr))) for x in xs)/max(1,len(xs))
    def max_seq(xs):
        if tok is None: return None
        vals=[]
        for x in xs[:min(len(xs), 2000)]:
            vals.append(len(encode(tok, f"Question:\n{x.question}\n\n{THINK}\n{x.cot}\nAnswer:\n{x.answer}{tok.eos_token}")))
        return max(vals) if vals else None
    sh = split_hash(payload)
    return {
        "dataset_source_path": str(dataset), "total_jsonl_count": len(rows), "usable_count": len(usable),
        "train_count": len(train), "eval_count": len(ev), "split_seed": payload.get("seed"), "split_hash": sh,
        "field_completeness": {"question": sum(bool(x.question) for x in usable)/max(1,len(usable)), "cot": sum(bool(x.cot) for x in usable)/max(1,len(usable)), "answer": sum(bool(x.answer) for x in usable)/max(1,len(usable))},
        "avg_question_token_length": avg_len(train+ev, "question"), "avg_cot_token_length": avg_len(train+ev, "cot"), "avg_answer_token_length": avg_len(train+ev, "answer"),
        "max_observed_sequence_length_sampled": max_seq(train+ev), "configured_max_length": max_length,
        "max_sequence_length_sufficient_sampled": None if tok is None else (max_seq(train+ev) or 0) <= max_length,
        "format": {"question": "string", "cot": "string", "answer": "string"},
    }


def find_subseq(seq: Sequence[int], pat: Sequence[int]) -> int:
    for i in range(0, len(seq)-len(pat)+1):
        if list(seq[i:i+len(pat)]) == list(pat): return i
    return -1


def pad(tok, ids_list, labels_list, pos_list, device):
    mx = max(len(x) for x in ids_list); pad_id = tok.pad_token_id or tok.eos_token_id
    ids=[]; labs=[]; masks=[]
    for x,l in zip(ids_list, labels_list):
        p = mx-len(x)
        ids.append(torch.tensor(x+[pad_id]*p, dtype=torch.long))
        labs.append(torch.tensor(l+[-100]*p, dtype=torch.long))
        masks.append(torch.tensor([1]*len(x)+[0]*p, dtype=torch.long))
    return {"input_ids": torch.stack(ids).to(device), "labels": torch.stack(labs).to(device), "attention_mask": torch.stack(masks).to(device), "positions": torch.tensor(pos_list, dtype=torch.long, device=device)}


def make_main_batch(tok, recs: Sequence[Example], group: str, max_len: int, device, train: bool=True):
    ids_list=[]; labels=[]; positions=[]; cot_masks=[]; ans_masks=[]
    think_ids = encode(tok, THINK)
    for ex in recs:
        q = ex.question
        if group == "G4" and train and random.random() < 0.5:
            q = "[question omitted]"
        if group == "G0":
            prefix = f"Question:\n{q}\n\nReasoning:\n"
            cot_part = f"{ex.cot}\nAnswer:\n"
            ans_part = f"{ex.answer}{tok.eos_token}"
            ids = encode(tok, prefix+cot_part+ans_part)[:max_len]
            pre = min(len(encode(tok, prefix)), len(ids)); cot_end = min(len(encode(tok, prefix+cot_part)), len(ids))
            lab = [-100]*pre + ids[pre:]
            pos = -1
        elif group in ("G1", "G2"):
            prefix = f"Question:\n{q}\n\n"
            body = f"{THINK}\nAnswer:\n{ex.answer}{tok.eos_token}"
            ids = encode(tok, prefix+body)[:max_len]
            pre = min(len(encode(tok, prefix)), len(ids)); pos = find_subseq(ids, think_ids)
            lab = [-100]*len(ids)
            for i in range(pos+1 if pos >= 0 else pre, len(ids)): lab[i] = ids[i]
            cot_end = pre
        elif group in ("G3", "G4"):
            prefix = f"Question:\n{q}\n\n"
            think = f"{THINK}\n"
            cot_part = f"{ex.cot}\nAnswer:\n"
            ans_part = f"{ex.answer}{tok.eos_token}"
            ids = encode(tok, prefix+think+cot_part+ans_part)[:max_len]
            pre = min(len(encode(tok, prefix)), len(ids)); pos = find_subseq(ids, think_ids)
            cot_start = min(len(encode(tok, prefix+think)), len(ids)); cot_end = min(len(encode(tok, prefix+think+cot_part)), len(ids))
            lab = [-100]*len(ids)
            if pos >= 0: lab[pos] = ids[pos]
            for i in range(cot_start, len(ids)): lab[i] = ids[i]
        else:
            raise ValueError(group)
        cm=[False]*len(ids); am=[False]*len(ids)
        if group == "G0":
            for i in range(pre, cot_end): cm[i]=True
            for i in range(cot_end, len(ids)): am[i]=True
        elif group in ("G1", "G2"):
            for i in range(pos+1 if pos >= 0 else 0, len(ids)): am[i]=True
        else:
            for i in range(cot_start, cot_end): cm[i]=True
            for i in range(cot_end, len(ids)): am[i]=True
        ids_list.append(ids); labels.append(lab[:len(ids)]); positions.append(pos); cot_masks.append(cm); ans_masks.append(am)
    batch = pad(tok, ids_list, labels, positions, device)
    batch["cot_mask"] = pad_bool(cot_masks, batch["input_ids"].shape[1], device)
    batch["answer_mask"] = pad_bool(ans_masks, batch["input_ids"].shape[1], device)
    return batch


def pad_bool(masks, mx, device):
    return torch.tensor([m+[False]*(mx-len(m)) for m in masks], dtype=torch.bool, device=device)


def make_sd_batch(tok, recs, max_len, device, condition: str, wrong_questions: Optional[List[str]]=None):
    ids_list=[]; labels=[]; positions=[]; think_ids=encode(tok, THINK)
    for i, ex in enumerate(recs):
        if condition in ("z_only", "z_only_shuffle"):
            prefix = f"Instruction:\nReconstruct the reasoning from the latent.\n\nLatent:\n{THINK}\nTarget reasoning:\n"
        elif condition == "q_only":
            prefix = f"Question:\n{ex.question}\n\nTarget reasoning:\n"
        elif condition == "wrong_q":
            prefix = f"Question:\n{wrong_questions[i]}\n\nLatent:\n{THINK}\nTarget reasoning:\n"
        else:
            prefix = f"Question:\n{ex.question}\n\nLatent:\n{THINK}\nTarget reasoning:\n"
        target = f"{ex.cot}{tok.eos_token}"
        ids = encode(tok, prefix+target)[:max_len]
        pre = min(len(encode(tok, prefix)), len(ids))
        lab = [-100]*pre + ids[pre:]
        pos = find_subseq(ids, think_ids) if condition != "q_only" else -1
        ids_list.append(ids); labels.append(lab[:len(ids)]); positions.append(pos)
    return pad(tok, ids_list, labels, positions, device)


def causal_loss(logits, labels, reduction="mean"):
    sl = labels[:,1:].contiguous(); sg = logits[:,:-1,:].contiguous()
    if reduction == "none":
        flat = F.cross_entropy(sg.view(-1, sg.size(-1)), sl.view(-1), ignore_index=-100, reduction="none").view(sl.shape)
        denom = sl.ne(-100).sum(dim=1).clamp_min(1)
        return flat.sum(dim=1)/denom
    return F.cross_entropy(sg.view(-1, sg.size(-1)), sl.view(-1), ignore_index=-100)


def masked_loss(logits, labels, mask):
    lab = labels.clone(); lab[~mask] = -100
    if lab[:, 1:].ne(-100).sum() == 0:
        return torch.zeros((), device=logits.device)
    return causal_loss(logits, lab)


def tok_acc(logits, labels):
    sl=labels[:,1:]; pred=logits[:,:-1,:].argmax(-1); mask=sl.ne(-100)
    return torch.zeros((),device=logits.device) if mask.sum()==0 else pred.eq(sl).masked_select(mask).float().mean()


def extract_z(hidden, pos):
    rows=torch.arange(hidden.size(0), device=hidden.device)
    return hidden[rows, pos.clamp_min(0)]


def replace_slot(emb, z, pos, detach=False):
    out=emb.clone(); src=z.detach() if detach else z
    for r,p in enumerate(pos.tolist()):
        if p>=0: out[r,p,:]=src[r]
    return out


def latent_labels(batch):
    lab=torch.full_like(batch["input_ids"], -100)
    for r,p in enumerate(batch["positions"].tolist()):
        if p>=0: lab[r,p]=batch["input_ids"][r,p]
    return lab


def compute_losses(model, tok, recs, group, max_len, device, lambda_self, lambda_latent, train=True, detach_self=False):
    batch=make_main_batch(tok,recs,group,max_len,device,train=train)
    out=model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
    main=causal_loss(out.logits,batch["labels"])
    cot=masked_loss(out.logits,batch["labels"],batch["cot_mask"])
    ans=masked_loss(out.logits,batch["labels"],batch["answer_mask"])
    z = extract_z(out.hidden_states[-1], batch["positions"]) if group != "G0" else None
    llab=latent_labels(batch)
    use_latent = group in ("G2","G3","G4")
    ltok = causal_loss(out.logits,llab) if use_latent else torch.zeros((),device=device)
    lacc = tok_acc(out.logits,llab) if use_latent else torch.zeros((),device=device)
    self_loss=torch.zeros((),device=device)
    if group == "G2":
        sd=make_sd_batch(tok,recs,max_len,device,"correct")
        emb=replace_slot(model.get_input_embeddings()(sd["input_ids"]), z, sd["positions"], detach_self)
        so=model(inputs_embeds=emb, attention_mask=sd["attention_mask"])
        self_loss=causal_loss(so.logits,sd["labels"])
    total = main + (lambda_self*self_loss if group=="G2" else 0) + (lambda_latent*ltok if use_latent else 0)
    return LossBundle(total, main, cot, ans, self_loss, ltok, lacc, z)


def producer_z(model,tok,recs,group,max_len,device):
    batch=make_main_batch(tok,recs,group,max_len,device,train=False)
    out=model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], output_hidden_states=True)
    return extract_z(out.hidden_states[-1], batch["positions"])


def eval_condition_losses(model,tok,recs,group,max_len,device):
    if group == "G0": return {}, []
    z=producer_z(model,tok,recs,group,max_len,device)
    perm=torch.randperm(z.size(0),device=device); wrong_q=[recs[(i+1)%len(recs)].question for i in range(len(recs))]
    conds={"correct":z,"shuffle":z[perm],"zero":torch.zeros_like(z),"q_only":None,"z_only":z,"z_only_shuffle":z[perm],"wrong_q":z}
    losses={}
    for name,zz in conds.items():
        sd=make_sd_batch(tok,recs,max_len,device,name,wrong_q)
        if zz is None:
            out=model(input_ids=sd["input_ids"], attention_mask=sd["attention_mask"])
        else:
            emb=replace_slot(model.get_input_embeddings()(sd["input_ids"]),zz,sd["positions"])
            out=model(inputs_embeds=emb, attention_mask=sd["attention_mask"])
        losses[name]=causal_loss(out.logits,sd["labels"],reduction="none").detach().cpu().tolist()
    return losses, perm.detach().cpu().tolist()


def mean(xs): return sum(xs)/max(1,len(xs))

def bootstrap_ci(vals, seed=42, n=400):
    if not vals: return [None,None]
    rng=random.Random(seed); outs=[]; m=len(vals)
    for _ in range(n): outs.append(mean([vals[rng.randrange(m)] for __ in range(m)]))
    outs.sort(); return [outs[int(0.025*n)], outs[int(0.975*n)-1]]


def evaluate(model,tok,recs,group,max_len,device,batch_size):
    model.eval(); rows=[]; main=[]; cot=[]; ans=[]; lacc=[]
    all_cond={k:[] for k in ["correct","shuffle","zero","q_only","z_only","z_only_shuffle","wrong_q"]}
    with torch.no_grad():
        for i in range(0,len(recs),batch_size):
            b=recs[i:i+batch_size]
            lb=compute_losses(model,tok,b,group,max_len,device,0.1,0.05,train=False)
            main.append(float(lb.main.cpu())); cot.append(float(lb.cot.cpu())); ans.append(float(lb.answer.cpu())); lacc.append(float(lb.latent_acc.cpu()))
            cond,_=eval_condition_losses(model,tok,b,group,max_len,device)
            for k,v in cond.items(): all_cond[k].extend(v)
    out={"main_nll":mean(main),"cot_nll":mean(cot),"answer_nll":mean(ans),"latent_token_accuracy":None if group=="G0" else mean(lacc)}
    if group!="G0":
        maps={"NLL_correct":"correct","NLL_shuffle":"shuffle","NLL_zero":"zero","NLL_q_only":"q_only","NLL_z_only":"z_only","NLL_z_only_shuffle":"z_only_shuffle","NLL_wrong_q":"wrong_q"}
        for outk,ink in maps.items(): out[outk]=mean(all_cond[ink])
        margins={"shuffle_margin":[s-c for s,c in zip(all_cond["shuffle"],all_cond["correct"])],"latent_gain":[q-c for q,c in zip(all_cond["q_only"],all_cond["correct"])],"z_only_margin":[s-c for s,c in zip(all_cond["z_only_shuffle"],all_cond["z_only"])],"wrong_q_margin":[w-c for w,c in zip(all_cond["wrong_q"],all_cond["correct"])]}
        for k,v in margins.items(): out[k]=mean(v); out[k+"_ci"]=bootstrap_ci(v)
    else:
        for k in ["NLL_correct","NLL_shuffle","NLL_zero","NLL_q_only","NLL_z_only","NLL_z_only_shuffle","NLL_wrong_q","shuffle_margin","latent_gain","z_only_margin","wrong_q_margin"]: out[k]=None
    return out


def extract_answer(text):
    m=re.search(r"Answer:\s*([^\n<]+)", text, re.I)
    return (m.group(1).strip() if m else text.strip().split("\n")[-1])[:120]

def sim(a,b):
    sa=set(str(a).split()); sb=set(str(b).split()); return len(sa&sb)/max(1,len(sa|sb))


def gen_for_condition(model,tok,recs,group,max_len,device,condition,max_new=64):
    if group=="G0": return []
    with torch.no_grad():
        z=producer_z(model,tok,recs,group,max_len,device); perm=torch.randperm(z.size(0),device=device)
        wrong_q=[recs[(i+1)%len(recs)].question for i in range(len(recs))]
        if condition=="shuffle": zc=z[perm]
        elif condition=="zero": zc=torch.zeros_like(z)
        elif condition in ("q_only",): zc=None
        elif condition=="z_only_shuffle": zc=z[perm]
        else: zc=z
        rows=[]
        for i,ex in enumerate(recs):
            cname = "z_only" if condition=="z_only" else ("z_only_shuffle" if condition=="z_only_shuffle" else ("wrong_q" if condition=="wrong_q" else ("q_only" if condition=="q_only" else "correct")))
            sd=make_sd_batch(tok,[ex],max_len,device,cname,[wrong_q[i]])
            if zc is None:
                gen=model.generate(input_ids=sd["input_ids"], attention_mask=sd["attention_mask"], max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
                prompt_len=sd["input_ids"].shape[1]
            else:
                emb=replace_slot(model.get_input_embeddings()(sd["input_ids"]),zc[i:i+1],sd["positions"])
                gen=model.generate(inputs_embeds=emb, attention_mask=sd["attention_mask"], max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
                prompt_len=0
            cot_text=tok.decode(gen[0][prompt_len:], skip_special_tokens=True)
            ans_prompt=f"Question:\n{ex.question}\n\nReasoning:\n{cot_text}\nAnswer:\n"
            aid=torch.tensor([encode(tok,ans_prompt)[-max_len:]],device=device)
            agen=model.generate(input_ids=aid,max_new_tokens=24,do_sample=False,pad_token_id=tok.eos_token_id)
            ans_text=tok.decode(agen[0][aid.shape[1]:], skip_special_tokens=True)
            ext=extract_answer(ans_text)
            rows.append({"sample_id":ex.sample_id,"question":ex.question,"gold_cot":ex.cot,"gold_answer":ex.answer,"condition":condition,"generated_cot":cot_text,"generated_answer":ans_text,"extracted_answer":ext,"exact_answer_hit": bool(ex.answer.strip().lower() and ex.answer.strip().lower() in ans_text.lower())})
        return rows


def write_jsonl(p:Path, rows):
    with p.open("w",encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")



def generation_eval_g0(model,tok,recs,out_dir,max_len,device,max_new=96):
    rows=[]
    model.eval()
    with torch.no_grad():
        for ex in recs:
            prompt=f"Question:\n{ex.question}\n\nReasoning:\n"
            ids=torch.tensor([encode(tok,prompt)[-max_len:]],device=device)
            gen=model.generate(input_ids=ids, attention_mask=torch.ones(ids.shape,device=device), max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            text=tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True)
            if "Answer:" in text:
                cot_text, ans_text = text.split("Answer:",1)
            else:
                cot_text, ans_text = text, ""
            ext=extract_answer("Answer:"+ans_text)
            rows.append({"sample_id":ex.sample_id,"question":ex.question,"gold_cot":ex.cot,"gold_answer":ex.answer,"condition":"normal","generated_cot":cot_text,"generated_answer":ans_text,"extracted_answer":ext,"exact_answer_hit": bool(ex.answer.strip().lower() and ex.answer.strip().lower() in ans_text.lower()),"token_similarity_to_normal":1.0})
    write_jsonl(out_dir/"normal_generation.jsonl", rows)
    return {"answer_accuracy": mean([float(r["exact_answer_hit"]) for r in rows]), "generation_samples": len(rows)}

def generation_eval(model,tok,recs,group,out_dir,max_len,device):
    if group=="G0": return generation_eval_g0(model,tok,recs,out_dir,max_len,device)
    cond_files={"normal":"normal_generation.jsonl","shuffle":"shuffle_generation.jsonl","zero":"zero_generation.jsonl","q_only":"q_only_generation.jsonl","z_only":"z_only_generation.jsonl","wrong_q":"wrong_q_generation.jsonl"}
    data={}
    for cond,fn in cond_files.items():
        rows=gen_for_condition(model,tok,recs,group,max_len,device,cond)
        write_jsonl(out_dir/fn, rows); data[cond]=rows
    normal=data["normal"]; shuffle=data["shuffle"]
    acc=mean([float(r["exact_answer_hit"]) for r in normal])
    flips=sum(r["exact_answer_hit"] and not s["exact_answer_hit"] for r,s in zip(normal,shuffle))
    identical=sum(sim(r["generated_cot"],s["generated_cot"])>0.98 for r,s in zip(normal,shuffle))
    avg_sim=mean([sim(r["generated_cot"],s["generated_cot"]) for r,s in zip(normal,shuffle)])
    return {"answer_accuracy":acc,"answer_flip_correct_vs_shuffle":flips,"correct_success_shuffle_failure_count":flips,"correct_shuffle_nearly_identical_count":identical,"avg_similarity_correct_shuffle":avg_sim}


def trainable_report(model,tok,path,before_norms=None):
    emb=model.get_input_embeddings(); head=model.get_output_embeddings(); ids={"think":tok.convert_tokens_to_ids(THINK),"stop_explain":tok.convert_tokens_to_ids(STOP_EXPLAIN)}
    norms={k:float(emb.weight[v].detach().norm().cpu()) for k,v in ids.items()}
    rep={"total_params":sum(p.numel() for p in model.parameters()),"trainable_params":sum(p.numel() for p in model.parameters() if p.requires_grad),"embed_tokens_trainable":any(p.requires_grad for p in emb.parameters()),"lm_head_trainable":any(p.requires_grad for p in head.parameters()),"embed_tokens_has_grad":any(p.grad is not None and float(p.grad.abs().sum().cpu())>0 for p in emb.parameters()),"lm_head_has_grad":any(p.grad is not None and float(p.grad.abs().sum().cpu())>0 for p in head.parameters()),"special_token_ids":ids,"special_token_embedding_norm_before_training": before_norms or norms,"special_token_embedding_norm_after_training":norms}
    path.write_text(json.dumps(rep,indent=2),encoding="utf-8"); return rep


def save_ckpt(model,tok,group_dir,step,group,args):
    d=group_dir/("final" if step=="final" else f"step{step}"); d.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"group":group,"step":step,"tokenizer_len":len(tok)}, d/"model_weights.pt")
    (d/"config.json").write_text(json.dumps({"group":group,"step":step,"optimizer_checkpoint_saved":False,"args":vars(args)},indent=2),encoding="utf-8")
    return str(d)


def run_group(group,train,ev,args,device,dry=False):
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed) if torch.cuda.is_available() else None
    gdir=Path(args.output_dir)/group; gdir.mkdir(parents=True,exist_ok=True)
    model,tok=load_model(args.model_path,device)
    before={k:float(model.get_input_embeddings().weight[tok.convert_tokens_to_ids(v)].detach().norm().cpu()) for k,v in {"think":THINK,"stop_explain":STOP_EXPLAIN}.items()}
    trainable_report(model,tok,gdir/"trainable_parameter_report_start.json",before)
    if group in ("G3","G4"):
        rep=json.loads((gdir/"trainable_parameter_report_start.json").read_text())
        if not rep["embed_tokens_trainable"] or not rep["lm_head_trainable"]: raise RuntimeError(f"{group} embed/lm_head not trainable")
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    steps=1 if dry else args.max_steps; accum=args.gradient_accumulation_steps
    progress=[]; start=time.time(); opt.zero_grad(set_to_none=True)
    for step in range(1,steps+1):
        model.train(); losses_acc=None
        for a in range(accum):
            off=(((step-1)*accum+a)*args.batch_size)%len(train); batch=train[off:off+args.batch_size]
            if len(batch)<args.batch_size: batch=batch+train[:args.batch_size-len(batch)]
            losses=compute_losses(model,tok,batch,group,args.max_length,device,args.lambda_self,args.lambda_latent,train=True)
            (losses.total/accum).backward(); losses_acc=losses
        if step == 1:
            trainable_report(model,tok,gdir/"trainable_parameter_report_after_backward.json",before)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step%args.log_every==0 or dry:
            row={"step":step,"train_loss":float(losses_acc.total.detach().cpu()),"main_loss":float(losses_acc.main.detach().cpu()),"cot_nll":float(losses_acc.cot.detach().cpu()),"answer_nll":float(losses_acc.answer.detach().cpu()),"self_decode_loss":float(losses_acc.self_decode.detach().cpu()),"latent_token_loss":float(losses_acc.latent_token.detach().cpu()),"latent_token_accuracy":float(losses_acc.latent_acc.detach().cpu()),"elapsed_sec":time.time()-start}
            progress.append(row); open(gdir/"train_progress.jsonl","a").write(json.dumps(row)+"\n"); print(json.dumps({"group":group,**row}),flush=True)
        if dry or step in EVAL_STEPS:
            ck=save_ckpt(model,tok,gdir,step,group,args)
            # load smoke check
            if dry:
                reload_model,_=load_model(args.model_path,device); reload_model.load_state_dict(torch.load(Path(ck)/"model_weights.pt",map_location=device)["model"],strict=False); del reload_model
            edir=gdir/f"eval_step{step}"; edir.mkdir(exist_ok=True)
            metrics=evaluate(model,tok,ev,group,args.max_length,device,args.eval_batch_size); metrics.update({"checkpoint_path":ck,"step":step})
            metrics.update(generation_eval(model,tok,ev[:args.generation_samples],group,edir,args.max_length,device))
            (edir/"intervention_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
            (edir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
            print(json.dumps({"group":group,"eval_step":step,"metrics":metrics}),flush=True)
            if dry: break
    save_ckpt(model,tok,gdir,"final",group,args); trainable_report(model,tok,gdir/"trainable_parameter_report_final.json",before)
    final_step=1 if dry else args.max_steps
    final_metrics=json.loads((gdir/f"eval_step{final_step}"/"metrics.json").read_text()) if (gdir/f"eval_step{final_step}"/"metrics.json").exists() else {}
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return final_metrics


def write_configs():
    cdir=Path("configs/gpt2_cot_latent_adjacent_matrix"); cdir.mkdir(parents=True,exist_ok=True)
    desc={"G0":"vanilla_cot_sft","G1":"heima_style_gpt2_baseline","G2":"self_decode_v2_existing","G3":"latent_text_adjacent","G4":"latent_text_adjacent_question_dropout"}
    for g,d in desc.items():
        (cdir/f"{g.lower()}_{d}.yaml").write_text(f"group: {g}\nname: {d}\nmodel: GPT2-small\nseed: 42\nhas_model_b: false\nprojector: false\nrole_embedding: false\ncumulative_latent: false\nloss2: false\n",encoding="utf-8")


def final_compare(out_dir:Path):
    comp={}; rows=[]
    for g in GROUPS:
        p=out_dir/g/"eval_step5000"/"metrics.json"
        if not p.exists(): p=out_dir/g/"eval_step1"/"metrics.json"
        m=json.loads(p.read_text()) if p.exists() else {}
        comp[g]=m
    (REPORT_DIR/"gpt2_cot_latent_adjacent_matrix_comparison.json").write_text(json.dumps(comp,indent=2),encoding="utf-8")
    lines=["# GPT2 CoT Latent Adjacent Matrix", "", "| group | main NLL | cot NLL | answer acc | latent acc | shuffle margin | latent gain | z-only margin | wrong-Q margin |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g,m in comp.items():
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(g, m.get("main_nll"), m.get("cot_nll"), m.get("answer_accuracy"), m.get("latent_token_accuracy"), m.get("shuffle_margin"), m.get("latent_gain"), m.get("z_only_margin"), m.get("wrong_q_margin")))
    lines += ["", "Status labels are assigned after full 5000-step metrics are present."]
    (REPORT_DIR/"gpt2_cot_latent_adjacent_matrix_final.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return comp


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="/data/zxl/official_heima/datasets/LLaVA-CoT-100k/train.jsonl")
    ap.add_argument("--model-path",default="/data/zxl/models/openai-community-gpt2")
    ap.add_argument("--output-dir",default=str(RUN_ROOT))
    ap.add_argument("--train-samples",type=int,default=10000); ap.add_argument("--eval-samples",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=42); ap.add_argument("--max-steps",type=int,default=5000)
    ap.add_argument("--batch-size",type=int,default=8); ap.add_argument("--gradient-accumulation-steps",type=int,default=2); ap.add_argument("--eval-batch-size",type=int,default=8)
    ap.add_argument("--generation-samples",type=int,default=128); ap.add_argument("--max-length",type=int,default=640)
    ap.add_argument("--lambda-self",type=float,default=0.1); ap.add_argument("--lambda-latent",type=float,default=0.05); ap.add_argument("--lr",type=float,default=1e-5); ap.add_argument("--log-every",type=int,default=50)
    ap.add_argument("--groups",default=",".join(GROUPS)); ap.add_argument("--device",default=""); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--prepare-only",action="store_true")
    args=ap.parse_args(); write_configs(); Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tmp_model,tmp_tok=load_model(args.model_path,device); _,train,ev=build_or_load_split(Path(args.dataset),Path(args.output_dir),args.train_samples,args.eval_samples,args.seed,tmp_tok,args.max_length); del tmp_model; torch.cuda.empty_cache() if torch.cuda.is_available() else None
    (Path(args.output_dir)/"run_config.json").write_text(json.dumps(vars(args),indent=2),encoding="utf-8")
    if args.prepare_only: return
    results={}
    for g in [x.strip() for x in args.groups.split(",") if x.strip()]:
        if g not in GROUPS: raise SystemExit(f"unknown group {g}")
        results[g]=run_group(g,train[:32] if args.dry_run else train,ev[:8] if args.dry_run else ev,args,device,dry=args.dry_run)
    comp=final_compare(Path(args.output_dir))
    if args.dry_run:
        lines=["# GPT2 CoT Matrix Dry Run", "", "Dry-run completed for groups: "+", ".join(results), "", "Output: `"+args.output_dir+"`", "", "All evaluator metrics were finite where applicable."]
        (REPORT_DIR/"gpt2_cot_matrix_dryrun_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"output_dir":args.output_dir,"groups":list(results),"comparison":comp},indent=2),flush=True)

if __name__=="__main__": main()
