#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re, sys, statistics, math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.htext.heima_reuse import HeimaOfficialAbstractProjection, extract_thinking_state, heima_ce_loss, official_embedding_replacement
from src.htext.formal_eval import cot_nll_breakdown, make_latent_variants, hidden_geometry
from src.htext.synthetic_data import generate_synthetic_split, read_jsonl, write_jsonl
from src.htext.trainer import batch_records, set_seed, _grad_norm

OUT=Path('reports/full_heima_diagnostics')
TOKENS=['<THINKING_OF_REASONING_1>','<THINKING_OF_REASONING_2>','<THINKING_OF_REASONING_3>']
MODEL='/mnt/nas/share2/home/zxl/models/openai-community-gpt2'
NUM_RE=re.compile(r'-?\d+')

def write(p,obj): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def tok(tk,text,max_len=None):
 ids=tk(text,add_special_tokens=False)['input_ids']; return ids[:max_len] if max_len else ids

def load(with_b=0):
 kw={'local_files_only':True,'use_safetensors':True}
 tk=AutoTokenizer.from_pretrained(MODEL,**kw); tk.pad_token=tk.eos_token
 ms=[AutoModelForCausalLM.from_pretrained(MODEL,**kw) for _ in range(1+with_b)]
 added=tk.add_special_tokens({'additional_special_tokens':TOKENS})
 if added:
  for m in ms: m.resize_token_embeddings(len(tk))
 for m in ms: m.config.use_cache=False; m.to('cuda' if torch.cuda.is_available() else 'cpu')
 return tk,ms

def split_cot(r):
 s=list(r.get('cot_steps_text') or r.get('cot_steps_raw') or [r['cot']])
 if len(s)>=3: return [s[0],s[1],' '.join(s[2:])]
 if len(s)==2: return [s[0],s[1],f"The answer is {r['answer']}."]
 return [r['question'],r['cot'],f"The answer is {r['answer']}."]

def fullcot_ids(tk,r):
 ids=tok(tk,r['question']+'\n'); labs=[-100]*len(ids)
 for p in split_cot(r):
  x=tok(tk,p+' '); ids+=x; labs+=x
 a=tok(tk,'\nAnswer: '+r['answer']+tk.eos_token); ids+=a; labs+=a
 return ids,labs

def direct_ids(tk,r):
 ids=tok(tk,r['question']+'\nAnswer: '); labs=[-100]*len(ids); a=tok(tk,r['answer']+tk.eos_token); ids+=a; labs+=a; return ids,labs

def batch_loss(model,tk,records,kind):
 rows=[]; labs=[]
 for r in records:
  i,l=(direct_ids if kind=='direct' else fullcot_ids)(tk,r); rows.append(i); labs.append(l)
 dev=next(model.parameters()).device; L=max(map(len,rows))
 inp=torch.full((len(rows),L),tk.pad_token_id,dtype=torch.long,device=dev); lab=torch.full_like(inp,-100); att=torch.zeros_like(inp)
 for k,row in enumerate(rows): inp[k,:len(row)]=torch.tensor(row,device=dev); lab[k,:len(labs[k])]=torch.tensor(labs[k],device=dev); att[k,:len(row)]=1
 out=model(input_ids=inp,attention_mask=att,use_cache=False)
 return heima_ce_loss(out.logits,lab),out.logits,lab

def train_tiny(kind,seed,steps=500):
 set_seed(seed); train,_=generate_synthetic_split(32,8,seed); tk,ms=load(0); m=ms[0]
 opt=torch.optim.AdamW(m.parameters(),lr=5e-5,weight_decay=0.0); logs=[]
 for st in range(1,steps+1):
  opt.zero_grad(set_to_none=True); b=batch_records(train,8,st-1); loss,_,_=batch_loss(m,tk,b,kind); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
  if st in {1,steps} or st%(steps//5)==0: logs.append({'step':st,'loss':float(loss.item()),'grad':_grad_norm(m.parameters())[0]})
 gens=[]; dev=next(m.parameters()).device
 for r in train[:32]:
  if kind=='direct': prefix=tok(tk,r['question']+'\nAnswer: ')
  else: prefix=tok(tk,r['question']+'\n')
  inp=torch.tensor([prefix],dtype=torch.long,device=dev)
  gen=m.generate(input_ids=inp,max_new_tokens=96 if kind=='fullcot' else 16,do_sample=False,pad_token_id=tk.pad_token_id,eos_token_id=tk.eos_token_id)
  new=gen[0,inp.size(1):].tolist(); text=tk.decode(new,skip_special_tokens=True); nums=NUM_RE.findall(text); parsed=nums[-1] if nums else None
  gens.append({'id':r['id'],'question':r['question'],'raw':text,'parsed':parsed,'gold':r['answer'],'match':parsed==r['answer'],'token_ids':new,'stop':'eos' if tk.eos_token_id in new else 'max_new_tokens'})
 return {'kind':kind,'logs':logs,'generation_em':sum(g['match'] for g in gens)/len(gens),'generations':gens}

def load_recover(seed):
 tk,ms=load(0); m=ms[0]; ck=torch.load(Path(f'checkpoints/full_schedule/seed_{seed}/recover_encoder/checkpoint.pt'),map_location=next(m.parameters()).device); m.load_state_dict(ck['model_a'],strict=True); m.eval(); return tk,m

def enc_hidden(model,tk,records,mode):
 dev=next(model.parameters()).device; rows=[]
 for r in records:
  ids=tok(tk,r['question']+'\n')
  for T in TOKENS: ids.append(tk.convert_tokens_to_ids(T))
  ids += tok(tk,'\nAnswer: '+r['answer']+tk.eos_token)
  rows.append(ids)
 L=max(map(len,rows)); inp=torch.full((len(rows),L),tk.pad_token_id,dtype=torch.long,device=dev); att=torch.zeros_like(inp)
 for i,row in enumerate(rows): inp[i,:len(row)]=torch.tensor(row,device=dev); att[i,:len(row)]=1
 out=model(input_ids=inp,attention_mask=att,output_hidden_states=True,use_cache=False)
 hs=[]
 for T in TOKENS:
  st=extract_thinking_state(input_ids=inp,last_hidden_state=out.hidden_states[-1],thinking_token_id=tk.convert_tokens_to_ids(T),mode=mode)
  hs.append(st.hidden)
 return torch.stack(hs,dim=1)

def dec_prompt(r,stage,masked=False,zonly=False):
 q='[MASKED QUESTION]' if masked else r['question']
 if zonly: return f'Latent:\n{TOKENS[stage-1]}\n\nReasoning:\n'
 return f'Question:\n{q}\n\nStage instruction:\nReconstruct cot{stage}.\n\nLatent:\n{TOKENS[stage-1]}\n\nReasoning:\n'

def dec_forward(b,tk,records,z,proj,stage,mode='qz',masked=False,override=None):
 dev=next(b.parameters()).device; use=override if override is not None else z; projected=proj(use) if mode!='q' else None
 rows=[]; labs=[]; pos=[]
 for r in records:
  prompt = dec_prompt(r,stage,masked=masked,zonly=(mode=='z')) if mode!='q' else dec_prompt(r,stage,masked=masked).replace(f'Latent:\n{TOKENS[stage-1]}\n\n','')
  p=tok(tk,prompt); t=tok(tk,split_cot(r)[stage-1]+tk.eos_token); rows.append(p+t); labs.append([-100]*len(p)+t)
  loc=[i for i,x in enumerate(p) if x==tk.convert_tokens_to_ids(TOKENS[stage-1])]; pos.append(locs[0] if (locs:=loc) else None)
 L=max(map(len,rows)); inp=torch.full((len(rows),L),tk.pad_token_id,dtype=torch.long,device=dev); lab=torch.full_like(inp,-100); att=torch.zeros_like(inp)
 for i,row in enumerate(rows): inp[i,:len(row)]=torch.tensor(row,device=dev); lab[i,:len(labs[i])]=torch.tensor(labs[i],device=dev); att[i,:len(row)]=1
 emb=b.get_input_embeddings()(inp)
 if mode!='q':
  mask=torch.zeros_like(inp,dtype=torch.bool)
  for i,p in enumerate(pos): mask[i,p]=True
  emb=official_embedding_replacement(emb,projected.unsqueeze(1),mask)
 out=b(inputs_embeds=emb,attention_mask=att,use_cache=False)
 return heima_ce_loss(out.logits,lab),out.logits,lab

def train_interp(seed,hidden_mode,stage,steps=80):
 train=read_jsonl(f'experiments/htext_gpt2/data/full_schedule/seed_{seed}/train.jsonl'); val=read_jsonl(f'experiments/htext_gpt2/data/full_schedule/seed_{seed}/validation.jsonl')[:48]
 tk,a=load_recover(seed); _,ms=load(1); b=ms[1]; proj=HeimaOfficialAbstractProjection(a.config.n_embd,b.config.n_embd).to(next(a.parameters()).device)
 opt=torch.optim.AdamW(list(b.parameters())+list(proj.parameters()),lr=5e-5,weight_decay=0.0)
 for p in a.parameters(): p.requires_grad_(False)
 for st in range(1,steps+1):
  opt.zero_grad(set_to_none=True); batch=batch_records(train,8,st-1)
  with torch.no_grad(): h=enc_hidden(a,tk,batch,hidden_mode)[:,stage-1,:]
  loss,_,_=dec_forward(b,tk,batch,h.detach(),proj,stage,'qz'); loss.backward(); opt.step()
 with torch.no_grad():
  h=enc_hidden(a,tk,val,hidden_mode)[:,stage-1,:]; vars=make_latent_variants(h); out={}
  for name,z in {'normal':h,'shuffle':vars['shuffled'],'zero':torch.zeros_like(h),'farthest':vars['farthest']}.items():
   loss,log,lab=dec_forward(b,tk,val,h,proj,stage,'qz',override=z); out[name]=cot_nll_breakdown(tk,val,log,lab)
  for name,mode,masked in [('q_only','q',False),('z_only','z',False),('masked_q_normal','qz',True),('masked_q_shuffle','qz',True)]:
   z=h if 'shuffle' not in name else vars['shuffled']; loss,log,lab=dec_forward(b,tk,val,h,proj,stage,mode,masked=masked,override=z); out[name]=cot_nll_breakdown(tk,val,log,lab)
 geom=hidden_geometry(h.unsqueeze(1))
 return {'hidden_mode':hidden_mode,'stage':stage,'eval':out,'margins':{'normal_shuffle_full':out['shuffle']['full']-out['normal']['full'],'q_gain':out['q_only']['full']-out['normal']['full'],'masked_margin':out['masked_q_shuffle']['full']-out['masked_q_normal']['full']},'geometry':geom}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]); ap.add_argument('--steps',type=int,default=80); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
 ans={'direct':train_tiny('direct',42,500),'fullcot':train_tiny('fullcot',42,500)}; write(OUT/'answer_generation_audit.json',ans)
 hidden={}; shortcut={}
 for seed in args.seeds:
  hidden[str(seed)]={}
  for mode in ['predictor','token']:
   hidden[str(seed)][mode]={}
   for stage in [1,2,3]:
    res=train_interp(seed,mode,stage,args.steps); hidden[str(seed)][mode][f'stage_{stage}']=res
    shortcut.setdefault(str(seed),{}).setdefault(mode,{})[f'stage_{stage}']={'q_gain':res['margins']['q_gain'],'normal_shuffle':res['margins']['normal_shuffle_full'],'masked_margin':res['margins']['masked_margin']}
  write(OUT/'hidden_mode_comparison.json',hidden); write(OUT/'question_shortcut_diagnostic.json',shortcut)
 summary={'answer_direct_em':ans['direct']['generation_em'],'answer_fullcot_em':ans['fullcot']['generation_em'],'hidden_mode_direction':{}}
 for mode in ['predictor','token']:
  vals=[]
  for s in hidden.values():
   for st in s[mode].values(): vals.append(st['margins']['normal_shuffle_full'])
  summary['hidden_mode_direction'][mode]={'mean_normal_shuffle_margin':statistics.mean(vals),'positive_count':sum(v>0 for v in vals),'n':len(vals)}
 write(OUT/'diagnostic_summary.json',summary)
 (OUT/'diagnostic_report.md').write_text(f"# Full Heima Diagnostics\n\nDirect-answer tiny EM: {summary['answer_direct_em']}\n\nFull-CoT tiny EM: {summary['answer_fullcot_em']}\n\nHidden mode summary: {summary['hidden_mode_direction']}\n",encoding='utf-8')
 print(json.dumps({'status':'complete','out':str(OUT),'summary':summary},indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
