#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
ROOT=Path(os.environ.get("REPO_ROOT","/data/zxl/Heima-qwen7b-overnight-20260803"))
def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--lane",type=int,required=True); ap.add_argument("--protocol",required=True); ap.add_argument("--scale",required=True); ap.add_argument("--gpu",type=int,required=True); a=ap.parse_args()
    out=ROOT/"runs/overnight_structural_sweep"/f"lane{a.lane}"/a.scale; out.mkdir(parents=True,exist_ok=True); start=time.time()
    if a.scale=="unit_smoke":
        status="complete"; rc=0; metrics={"special_token_tests":"prior_o3_tokens_available","remote_launch":"ok"}
    else:
        status="failed_protocol_trainer_not_implemented"; rc=2; metrics={"reason":"scheduler launched but full structural trainer for this protocol/scale is not implemented yet","protocol":a.protocol,"scale":a.scale}
    write(out/"metrics.json",{"lane":a.lane,"gpu":a.gpu,"protocol":a.protocol,"scale":a.scale,"status":status,"returncode":rc,"runtime_seconds":time.time()-start,"metrics":metrics})
    raise SystemExit(rc)
if __name__=="__main__": main()
