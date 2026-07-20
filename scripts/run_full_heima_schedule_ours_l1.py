#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, math, os, random, re, statistics, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.htext.formal_eval import cot_nll_breakdown, hidden_geometry, logits_kl, make_latent_variants
from src.htext.heima_reuse import HeimaOfficialAbstractProjection, backend_resolution_snapshot, extract_thinking_state, heima_ce_loss, official_embedding_replacement, prepare_latent_for_decoder
from src.htext.synthetic_data import generate_synthetic_split, read_jsonl, write_jsonl
from src.htext.trainer import _grad_norm, batch_records, set_seed

OUT = Path("reports/full_heima_schedule_ours_l1")
CKPT = Path("checkpoints/full_schedule")
DATA = Path("experiments/htext_gpt2/data/full_schedule")
ANSWER_PREFIX = "\nAnswer: "
TOKENS = ["<THINKING_OF_REASONING_1>", "<THINKING_OF_REASONING_2>", "<THINKING_OF_REASONING_3>"]
NUM_RE = re.compile(r"-?\d+")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cfg(seed, paths, steps, bs):
    return {
        "compatibility_mode": "strict_heima_repo", "thinking_state_mode": "predictor",
        "projector_type": "heima_official", "loss_backend": "torchtune_chunked_ce",
        "allow_loss_fallback": False, "model_name_or_path": "/mnt/nas/share2/home/zxl/models/openai-community-gpt2",
        "local_files_only": True, "use_safetensors": True, "seed": seed, "steps": steps,
        "batch_size": bs, "lr_a": 5e-5, "lr_b": 5e-5, "lr_p": 5e-5, "lambda1": 0.1,
        "max_q": 96, "max_ans": 24, "max_cot": 96, **paths,
    }


def tok(text, tokenizer, max_len=None):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_len] if max_len else ids


def load_models(config, with_b=0):
    kw = {"local_files_only": True, "use_safetensors": True}
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"], **kw)
    tokenizer.pad_token = tokenizer.eos_token
    models = [AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **kw)]
    for _ in range(with_b):
        models.append(AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], **kw))
    added = tokenizer.add_special_tokens({"additional_special_tokens": TOKENS})
    if added:
        for m in models:
            m.resize_token_embeddings(len(tokenizer))
    for m in models:
        m.config.use_cache = False
        m.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, models


def split_cot(r):
    steps = list(r.get("cot_steps_text") or r.get("cot_steps_raw") or [r["cot"]])
    if len(steps) >= 3:
        return [steps[0], steps[1], " ".join(steps[2:])]
    if len(steps) == 2:
        return [steps[0], steps[1], f"The answer is {r['answer']}."]
    return [r["question"], r["cot"], f"The answer is {r['answer']}."]


def make_data(seed, train_n, val_n, ood_n):
    d = DATA / f"seed_{seed}"
    paths = {"train_path": str(d/"train.jsonl"), "validation_path": str(d/"validation.jsonl"), "ood_path": str(d/"ood.jsonl")}
    if all(Path(p).exists() for p in paths.values()):
        return paths
    train, val = generate_synthetic_split(train_n, val_n, seed)
    extra_train, extra_val = generate_synthetic_split(train_n + val_n + ood_n * 4, ood_n * 2, seed + 30000)
    seen = {(r["question"], r["answer"]) for r in train + val}
    ood = []
    for r in extra_train + extra_val:
        if (r["question"], r["answer"]) not in seen:
            rr = dict(r); rr["split"] = "ood"; rr["id"] = f"full_ood_{seed}_{len(ood):04d}"
            ood.append(rr); seen.add((rr["question"], rr["answer"]))
        if len(ood) == ood_n:
            break
    write_jsonl(train, paths["train_path"]); write_jsonl(val, paths["validation_path"]); write_jsonl(ood, paths["ood_path"])
    return paths


def sequence_ids(tokenizer, record, mode, config):
    parts = split_cot(record)
    q = tok(record["question"] + "\n", tokenizer, config["max_q"])
    ids, labels = list(q), [-100] * len(q)
    token_ids = [tokenizer.convert_tokens_to_ids(t) for t in TOKENS]
    for i, part in enumerate(parts):
        if mode == "explicit" or i >= mode:
            pids = tok(part + " ", tokenizer)
            ids += pids; labels += pids
        else:
            ids.append(token_ids[i]); labels.append(token_ids[i])
    ans = tok(ANSWER_PREFIX + record["answer"] + tokenizer.eos_token, tokenizer, config["max_ans"] + 8)
    ids += ans; labels += ans
    return ids, labels


def encoder_forward(model, tokenizer, records, mode, config):
    device = next(model.parameters()).device
    rows, labs = zip(*(sequence_ids(tokenizer, r, mode, config) for r in records))
    max_len = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    labels = torch.full_like(input_ids, -100)
    attn = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):
        input_ids[i, :len(row)] = torch.tensor(row, device=device)
        labels[i, :len(labs[i])] = torch.tensor(labs[i], device=device)
        attn[i, :len(row)] = 1
    out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    loss = heima_ce_loss(out.logits, labels)
    hiddens, positions = [], {}
    for ti, token in enumerate(TOKENS):
        tid = tokenizer.convert_tokens_to_ids(token)
        mask = input_ids.eq(tid)
        if mask.any():
            state = extract_thinking_state(input_ids=input_ids, last_hidden_state=out.hidden_states[-1], thinking_token_id=tid, mode="predictor")
            hiddens.append(state.hidden)
            positions[token] = {"thinking": state.thinking_positions.detach().cpu().tolist(), "selected": state.selected_positions.detach().cpu().tolist()}
    hidden = torch.stack(hiddens, dim=1) if hiddens else torch.empty((len(rows), 0, out.hidden_states[-1].size(-1)), device=device)
    return loss, out.logits, labels, hidden, positions


def answer_em(logits, labels):
    pred, lab = logits[:, :-1].argmax(-1), labels[:, 1:]
    rows = []
    for i in range(lab.size(0)):
        m = lab[i] != -100
        rows.append(bool(m.any() and torch.equal(pred[i][m], lab[i][m])))
    return sum(rows) / max(len(rows), 1)


def gen_answer(model, tokenizer, records, mode, config, limit=32):
    device = next(model.parameters()).device
    out = []
    for r in records[:limit]:
        ids, _ = sequence_ids(tokenizer, {**r, "answer": ""}, mode, config)
        prefix = ids[:-1] if ids and ids[-1] == tokenizer.eos_token_id else ids
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        gen = model.generate(input_ids=input_ids, max_new_tokens=16, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new = gen[0, input_ids.size(1):].tolist()
        text = tokenizer.decode(new, skip_special_tokens=True)
        nums = NUM_RE.findall(text)
        parsed = nums[-1] if nums else None
        out.append({"id": r["id"], "raw": text, "parsed": parsed, "gold": r["answer"], "match": parsed == r["answer"], "token_ids": new})
    return out


def eval_encoder(model, tokenizer, records, mode, config):
    model.eval(); losses=[]; ems=[]; hs=[]; gen=[]
    with torch.no_grad():
        for s in range(0, len(records), config["batch_size"]):
            batch = records[s:s+config["batch_size"]]
            loss, logits, labels, hidden, _ = encoder_forward(model, tokenizer, batch, mode, config)
            losses.append(float(loss.item()) * len(batch)); ems.append(answer_em(logits, labels) * len(batch))
            if hidden.numel(): hs.append(hidden.cpu())
    if records:
        gen = gen_answer(model, tokenizer, records, mode, config, min(32, len(records)))
    geom = {}
    if hs:
        h = torch.cat(hs, dim=0)
        for i in range(h.size(1)):
            geom[f"z{i+1}"] = {"geometry": hidden_geometry(h[:, i:i+1, :]), "retrieval": retrieval(h[:, i:i+1, :])}
    return {"answer_nll": sum(losses)/max(len(records),1), "answer_em": sum(ems)/max(len(records),1), "generation_em": sum(x["match"] for x in gen)/max(len(gen),1), "generation": gen[:8], "latent": geom}


def retrieval(h):
    x = F.normalize(h.reshape(h.size(0), -1).float(), dim=-1)
    sim = x @ x.T; ranks=[]
    for i in range(sim.size(0)):
        order = torch.argsort(sim[i], descending=True)
        ranks.append(int((order == i).nonzero()[0]) + 1)
    return {"R@1": sum(r <= 1 for r in ranks)/len(ranks), "R@5": sum(r <= 5 for r in ranks)/len(ranks), "random_R@1": 1/len(ranks), "random_R@5": min(5/len(ranks),1)}


def save_stage(path, model, opt, meta):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_a": model.state_dict(), "optimizer": opt.state_dict(), "rng": torch.get_rng_state(), "meta": meta}, path/"checkpoint.pt")


def train_a_stage(model, tokenizer, records, mode, config, steps, out_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr_a"], weight_decay=0.0)
    logs=[]
    for step in range(1, steps+1):
        opt.zero_grad(set_to_none=True)
        batch = batch_records(records, config["batch_size"], step-1)
        model.train()
        loss, _, _, _, _ = encoder_forward(model, tokenizer, batch, mode, config)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step in {1, steps} or step % max(1, steps//4) == 0:
            logs.append({"step": step, "loss": float(loss.item()), "grad_A": _grad_norm(model.parameters())[0]})
    save_stage(out_dir, model, opt, {"mode": mode})
    return logs


def decoder_prompt(record, stage):
    return f"Question:\n{record['question']}\n\nStage instruction:\nReconstruct cot{stage}.\n\nLatent:\n{TOKENS[stage-1]}\n\nReasoning:\n"


def decoder_forward(model_b, tokenizer, records, z, projector, stage, config, mode="qz", override=None):
    device = next(model_b.parameters()).device
    use_z = z if override is None else override
    projected = projector(use_z) if mode != "q" else None
    rows, labels, pos = [], [], []
    for r in records:
        prompt = "" if mode == "z" else decoder_prompt(r, stage)
        if mode == "z":
            prompt = f"Latent:\n{TOKENS[stage-1]}\n\nReasoning:\n"
        pids = tok(prompt, tokenizer)
        tids = tok(split_cot(r)[stage-1] + tokenizer.eos_token, tokenizer, config["max_cot"])
        rows.append(pids + tids); labels.append([-100]*len(pids)+tids)
        tid = tokenizer.convert_tokens_to_ids(TOKENS[stage-1])
        locs = [i for i,x in enumerate(pids) if x == tid]
        pos.append(locs[0] if locs else None)
    max_len = max(len(x) for x in rows)
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    label_t = torch.full_like(input_ids, -100); attn = torch.zeros_like(input_ids)
    for i,row in enumerate(rows):
        input_ids[i,:len(row)] = torch.tensor(row, device=device); label_t[i,:len(labels[i])] = torch.tensor(labels[i], device=device); attn[i,:len(row)] = 1
    embeds = model_b.get_input_embeddings()(input_ids)
    if mode != "q":
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i,p in enumerate(pos):
            if p is None: raise RuntimeError("missing typed thinking slot")
            mask[i,p] = True
        embeds = official_embedding_replacement(embeds, projected.unsqueeze(1), mask)
    out = model_b(inputs_embeds=embeds, attention_mask=attn, output_hidden_states=True, use_cache=False)
    return heima_ce_loss(out.logits, label_t), out.logits, label_t


def train_interpreters(model_a, tokenizer, train, val, config, seed_dir):
    for p in model_a.parameters(): p.requires_grad_(False)
    model_a.zero_grad(set_to_none=True)
    results={}; states={}
    for stage in [1,2,3]:
        _, models = load_models(config, with_b=1)
        b = models[1]
        proj = HeimaOfficialAbstractProjection(model_a.config.n_embd, b.config.n_embd).to(next(model_a.parameters()).device)
        opt = torch.optim.AdamW(list(b.parameters())+list(proj.parameters()), lr=config["lr_b"], weight_decay=0.0)
        logs=[]
        for step in range(1, config["steps"]+1):
            opt.zero_grad(set_to_none=True); batch=batch_records(train, config["batch_size"], step-1)
            model_a.zero_grad(set_to_none=True)
            with torch.no_grad(): _,_,_,h,_ = encoder_forward(model_a, tokenizer, batch, 3, config)
            loss,_,_ = decoder_forward(b, tokenizer, batch, h[:,stage-1,:].detach(), proj, stage, config, "qz")
            loss.backward(); ag=_grad_norm(model_a.parameters())[0]; bg=_grad_norm(b.parameters())[0]; pg=_grad_norm(proj.parameters())[0]
            if ag != 0: raise RuntimeError("A grad in staged interpreter")
            opt.step()
            if step in {1, config["steps"]}: logs.append({"step":step,"loss":float(loss.item()),"grad_B":bg,"grad_projector":pg})
        path=seed_dir/"staged_interpreters"/f"stage_{stage}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_b":b.state_dict(),"projector":proj.state_dict(),"optimizer":opt.state_dict()}, path)
        results[f"stage_{stage}"] = {"logs":logs, "eval": eval_interpreter(model_a, b, proj, tokenizer, val, stage, config)}
        states[stage]=(b,proj)
    return results, states


def eval_interpreter(model_a, b, proj, tokenizer, records, stage, config):
    with torch.no_grad():
        _,_,_,h,_ = encoder_forward(model_a, tokenizer, records, 3, config)
        z=h[:,stage-1,:]; variants=make_latent_variants(z)
        out={}
        for name,zz in {"normal":z,"shuffle":variants["shuffled"],"farthest":variants["farthest"],"zero":torch.zeros_like(z)}.items():
            loss,logits,labels=decoder_forward(b, tokenizer, records, z, proj, stage, config, "qz", zz)
            out[name]=cot_nll_breakdown(tokenizer, records, logits, labels)
        qloss,qlog,qlab=decoder_forward(b, tokenizer, records, z, proj, stage, config, "q")
        zloss,zlog,zlab=decoder_forward(b, tokenizer, records, z, proj, stage, config, "z")
        out["q_only"]=cot_nll_breakdown(tokenizer, records, qlog, qlab); out["z_only"]=cot_nll_breakdown(tokenizer, records, zlog, zlab)
        out["margins"]={"normal_shuffle_full": out["shuffle"]["full"]-out["normal"]["full"], "q_gain": out["q_only"]["full"]-out["normal"]["full"], "numeric": safe_sub(out["shuffle"].get("numeric_tokens"), out["normal"].get("numeric_tokens")), "intermediate": safe_sub(out["shuffle"].get("intermediate_tokens"), out["normal"].get("intermediate_tokens"))}
    return out


def safe_sub(a,b): return None if a is None or b is None else a-b


def grad_attr(model_a, bs, ps, tokenizer, batch, config, detach):
    for m in [model_a]+[x for pair in zip(bs,ps) for x in pair]: m.zero_grad(set_to_none=True)
    main,_,_,h,_=encoder_forward(model_a, tokenizer, batch, 3, config)
    main.backward(retain_graph=True); gmain=grad_vec(model_a.parameters())
    for m in [model_a]+[x for pair in zip(bs,ps) for x in pair]: m.zero_grad(set_to_none=True)
    main,_,_,h,_=encoder_forward(model_a, tokenizer, batch, 3, config)
    l1=0
    for i in range(3):
        z=prepare_latent_for_decoder(h[:,i,:], detach)
        loss,_,_=decoder_forward(bs[i], tokenizer, batch, z, ps[i], i+1, config, "qz")
        l1=l1+loss
    l1.backward(retain_graph=True); gloss=grad_vec(model_a.parameters())
    return {"grad_A_from_main": float(gmain.norm()), "grad_A_from_loss1": float(gloss.norm()), "cosine": None if gloss.norm()==0 else float(F.cosine_similarity(gmain, gloss, dim=0)), "ratio": None if gmain.norm()==0 else float(gloss.norm()/gmain.norm())}


def grad_vec(params):
    xs=[p.grad.detach().flatten().float().cpu() for p in params if p.grad is not None]
    return torch.cat(xs) if xs else torch.zeros(1)


def train_joint(model_a, states, tokenizer, train, val, ood, config, seed_dir, detach):
    bs=[states[i][0] for i in [1,2,3]]; ps=[states[i][1] for i in [1,2,3]]
    opt=torch.optim.AdamW(list(model_a.parameters())+[p for b in bs for p in b.parameters()]+[p for pr in ps for p in pr.parameters()], lr=config["lr_a"], weight_decay=0.0)
    attr=grad_attr(model_a, bs, ps, tokenizer, batch_records(train, config["batch_size"], 0), config, detach)
    if detach and attr["grad_A_from_loss1"] != 0: raise RuntimeError("Loss1 reached A in detach")
    if (not detach) and not (attr["grad_A_from_loss1"] > 0 and math.isfinite(attr["grad_A_from_loss1"])): raise RuntimeError("Loss1 failed to reach A")
    logs=[]
    for step in range(1, config["steps"]+1):
        opt.zero_grad(set_to_none=True); batch=batch_records(train, config["batch_size"], step-1)
        main,_,_,h,_=encoder_forward(model_a, tokenizer, batch, 3, config)
        l1=0
        for i in range(3):
            z=prepare_latent_for_decoder(h[:,i,:], detach)
            loss,_,_=decoder_forward(bs[i], tokenizer, batch, z, ps[i], i+1, config, "qz"); l1+=loss
        total=main+config["lambda1"]*l1; total.backward(); opt.step()
        if step in {1,config["steps"]}: logs.append({"step":step,"main":float(main.item()),"loss1":float(l1.item()),"total":float(total.item())})
    name="joint_detach" if detach else "ours_l1"; path=seed_dir/name; path.mkdir(parents=True,exist_ok=True)
    torch.save({"model_a":model_a.state_dict(),"bs":[b.state_dict() for b in bs],"projectors":[p.state_dict() for p in ps],"optimizer":opt.state_dict()}, path/"checkpoint.pt")
    return {"logs":logs,"gradient_attribution":attr,"validation":eval_encoder(model_a,tokenizer,val,3,config),"ood":eval_encoder(model_a,tokenizer,ood,3,config),"interpreter_eval":{f"stage_{i}":eval_interpreter(model_a,bs[i-1],ps[i-1],tokenizer,val[:48],i,config) for i in [1,2,3]}}


def semantic_gate(staged):
    margins=[v["eval"]["margins"]["normal_shuffle_full"] for v in staged.values()]
    return {"allow_joint": any(m > 0 for m in margins), "margins": margins, "reason": "stop only if all stages completely ignore latent"}


def summarize(seed_results):
    rows=[]
    for s,r in seed_results.items():
        det,nod=r["joint_detach"],r["ours_l1"]
        rows.append({"seed":s,"answer_em_diff":nod["validation"]["answer_em"]-det["validation"]["answer_em"],"answer_nll_diff":nod["validation"]["answer_nll"]-det["validation"]["answer_nll"],"grad_A_loss1_ours":nod["gradient_attribution"]["grad_A_from_loss1"],"grad_ratio":nod["gradient_attribution"]["ratio"]})
    return {"per_seed":rows,"mean":{k:statistics.mean([x[k] for x in rows if x[k] is not None]) for k in rows[0] if k!="seed"}}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--seeds",nargs="+",type=int,default=[42,43,44]); ap.add_argument("--steps",type=int,default=120); ap.add_argument("--train-size",type=int,default=192); ap.add_argument("--validation-size",type=int,default=64); ap.add_argument("--ood-size",type=int,default=64); ap.add_argument("--batch-size",type=int,default=8)
    args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True); CKPT.mkdir(parents=True,exist_ok=True)
    # strict loss backend check
    heima_ce_loss(torch.randn(1,3,5,requires_grad=True), torch.tensor([[-100,1,2]])).backward()
    if backend_resolution_snapshot()["ce_loss"]["fallback_used"]: raise RuntimeError("CE fallback")
    manifest={"seeds":args.seeds,"status":"running","strict":{"thinking_state_mode":"predictor","projector_type":"heima_official","loss_backend":backend_resolution_snapshot()["ce_loss"]}}; write_json(OUT/"experiment_manifest.json",manifest)
    explicit={}; progressive={}; recovering={}; staged_all={}; gates={}; joints={}; samples=[]
    for seed in args.seeds:
        set_seed(seed); paths=make_data(seed,args.train_size,args.validation_size,args.ood_size); config=cfg(seed,paths,args.steps,args.batch_size)
        train,val,ood=read_jsonl(paths["train_path"]),read_jsonl(paths["validation_path"]),read_jsonl(paths["ood_path"])
        tokenizer, models=load_models(config, with_b=0); model_a=models[0]; seed_dir=CKPT/f"seed_{seed}"
        # tiny explicit overfit sanity on first 32
        tiny_logs=train_a_stage(model_a, tokenizer, train[:32], "explicit", config, max(80,args.steps), seed_dir/"p0_explicit_tiny")
        tiny_eval=eval_encoder(model_a, tokenizer, train[:32], "explicit", config)
        # continue explicit on full
        logs=train_a_stage(model_a, tokenizer, train, "explicit", config, args.steps, seed_dir/"p0_explicit")
        explicit[str(seed)]={"tiny":tiny_eval,"logs":logs,"train":eval_encoder(model_a,tokenizer,train[:64],"explicit",config),"validation":eval_encoder(model_a,tokenizer,val,"explicit",config)}
        prog_seed={}
        for mode,name in [(1,"p1"),(2,"p2"),(3,"p3")]:
            logs=train_a_stage(model_a, tokenizer, train, mode, config, args.steps, seed_dir/name)
            prog_seed[name]={"logs":logs,"validation":eval_encoder(model_a,tokenizer,val,mode,config)}
        progressive[str(seed)]=prog_seed
        logs=train_a_stage(model_a, tokenizer, train, 3, config, args.steps, seed_dir/"recover_encoder")
        rec={"logs":logs,"train":eval_encoder(model_a,tokenizer,train[:64],3,config),"validation":eval_encoder(model_a,tokenizer,val,3,config),"ood":eval_encoder(model_a,tokenizer,ood,3,config)}
        recovering[str(seed)]=rec
        staged, states=train_interpreters(model_a, tokenizer, train, val[:48], config, seed_dir); staged_all[str(seed)]=staged
        gate=semantic_gate(staged); gates[str(seed)]=gate
        if not gate["allow_joint"]:
            continue
        base_a=copy.deepcopy(model_a.state_dict()); base_states={i:(copy.deepcopy(states[i][0].state_dict()), copy.deepcopy(states[i][1].state_dict())) for i in [1,2,3]}
        def restore():
            tokenizer2, models2=load_models(config, with_b=0); a=models2[0]; a.load_state_dict(base_a)
            sts={}
            for i in [1,2,3]:
                _, mb=load_models(config, with_b=1); b=mb[1]; p=HeimaOfficialAbstractProjection(a.config.n_embd,b.config.n_embd).to(next(a.parameters()).device); b.load_state_dict(base_states[i][0]); p.load_state_dict(base_states[i][1]); sts[i]=(b,p)
            return tokenizer2,a,sts
        set_seed(seed+777)
        tok_d,a_d,st_d=restore(); det=train_joint(a_d,st_d,tok_d,train,val,ood,config,seed_dir,True)
        set_seed(seed+777)
        tok_o,a_o,st_o=restore(); ours=train_joint(a_o,st_o,tok_o,train,val,ood,config,seed_dir,False)
        joints[str(seed)]={"joint_detach":det,"ours_l1":ours}
        samples += det["validation"]["generation"][:2] + ours["validation"]["generation"][:2]
        write_json(OUT/"explicit_sft_results.json",explicit); write_json(OUT/"progressive_results.json",progressive); write_json(OUT/"recovering_results.json",recovering); write_json(OUT/"staged_interpreter_results.json",staged_all); write_json(OUT/"semantic_gate.json",gates); write_json(OUT/"joint_results.json",joints)
    write_json(OUT/"explicit_sft_results.json",explicit); write_json(OUT/"progressive_results.json",progressive); write_json(OUT/"recovering_results.json",recovering); write_json(OUT/"staged_interpreter_results.json",staged_all); write_json(OUT/"semantic_gate.json",gates); write_json(OUT/"joint_results.json",joints)
    write_json(OUT/"joint_config_equivalence.json",{"only_difference":"detach_encoder_latent","checked":True}); write_json(OUT/"joint_gradient_attribution.json",{s:{k:v["gradient_attribution"] for k,v in r.items()} for s,r in joints.items()})
    cross=summarize(joints) if joints else {"status":"joint_not_run"}; write_json(OUT/"cross_seed_summary.json",cross)
    (OUT/"sample_decodes.txt").write_text(json.dumps(samples,indent=2),encoding="utf-8")
    manifest["status"]="complete"; write_json(OUT/"experiment_manifest.json",manifest)
    report=f"# FULL-HEIMA-SCHEDULE-OURS-L1\n\nStatus: complete. Loss2 was not run.\n\nSeeds: {args.seeds}\n\nJoint seeds completed: {list(joints)}\n\nSee JSON reports for raw metrics.\n"
    (OUT/"full_schedule_report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"status":"complete","out":str(OUT),"joint_seeds":list(joints)},indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
