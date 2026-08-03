#!/usr/bin/env python3
import json,csv,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; rep=ROOT/'reports/teacher_five_points'; rep.mkdir(parents=True,exist_ok=True)
rows=[]
for p in (ROOT/'runs/teacher_five_points').glob('gpu*/*/*/metrics.json'):
    try: rows.append(json.loads(p.read_text()))
    except Exception: pass
(rep/'live_status.json').write_text(json.dumps({'time':time.time(),'finished_jobs':len(rows),'jobs':rows},indent=2,ensure_ascii=False)+'\n')
fields=['gpu','protocol','scale','status','reason','runtime_seconds']
with (rep/'all_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow({k:r.get(k) for k in fields}) for r in rows]
(rep/'all_metrics.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
(rep/'failure_registry.json').write_text(json.dumps([r for r in rows if r.get('status')!='COMPLETE'],indent=2,ensure_ascii=False)+'\n')
for name in ['teacher_recommendations_status','accuracy_root_cause_closure','model_data_switch_effect','data_conversion_effect','stage1_protocol_comparison','main_joint_twostage','decoder_semantic_evaluation','latent_intervention','final_recommendation']:
    (rep/f'{name}.md').write_text(f'# {name}\n\nCurrent status: {len(rows)} jobs finished/failed. Large AM data source and full protocol trainers are currently missing on reachable 4090 server.\n')
