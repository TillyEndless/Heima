#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,time,subprocess
from pathlib import Path
ROOT=Path(os.environ.get('REPO_ROOT','/data/zxl/Heima-qwen7b-teacher-validation-20260803'))

def write(p,o):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(o,indent=2,ensure_ascii=False,sort_keys=True)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--protocol',required=True); ap.add_argument('--scale',required=True); args=ap.parse_args()
    out=ROOT/'runs/teacher_five_points'/f'gpu{args.gpu}'/args.protocol/args.scale; out.mkdir(parents=True,exist_ok=True)
    data=json.loads((ROOT/'reports/teacher_five_points/data_manifest.json').read_text())
    start=time.time(); status='FAILED'; reason=None
    if not data.get(args.scale,{}).get('available'):
        reason='DATA_MISSING: required AM split/source unavailable on this server'
    elif args.scale=='D32' and args.protocol=='P0_CURRENT_O3_DYNAMIC':
        reason='READY_BUT_NOT_STARTED_BY_JOB_WRAPPER: use existing qwen7b_stage1_boundary_o3.py for real D32 O3; wrapper avoids overwriting historical stage1_boundary outputs'
    else:
        reason='TRAINER_NOT_IMPLEMENTED_FOR_PROTOCOL: full teacher-five-points trainer not yet implemented'
    write(out/'metrics.json',{'gpu':args.gpu,'protocol':args.protocol,'scale':args.scale,'status':status,'reason':reason,'runtime_seconds':time.time()-start})
    raise SystemExit(2)
if __name__=='__main__': main()
