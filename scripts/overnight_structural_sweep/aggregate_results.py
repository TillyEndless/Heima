#!/usr/bin/env python3
import json,time,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; rep=ROOT/'reports/overnight_structural_sweep'; rep.mkdir(parents=True,exist_ok=True)
rows=[]
for p in (ROOT/'runs/overnight_structural_sweep').glob('lane*/*/metrics.json'):
    try: rows.append(json.loads(p.read_text()))
    except Exception: pass
(rep/'live_status.json').write_text(json.dumps({'time':time.time(),'completed_or_failed_jobs':len(rows),'jobs':rows},indent=2,ensure_ascii=False)+'\n')
with (rep/'all_metrics.csv').open('w',newline='') as f:
    fields=['lane','gpu','protocol','scale','status','returncode','runtime_seconds']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow({k:r.get(k) for k in fields}) for r in rows]
(rep/'all_metrics.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
(rep/'failure_registry.json').write_text(json.dumps([r for r in rows if str(r.get('status','')).startswith('failed')],indent=2)+'\n')
for name in ['protocol_comparison','scale_comparison','progressive_vs_oneshot','typed_vs_repeated','latent_ntp_ablation','fixed_k_comparison','main_joint_twostage','decoder_semantic_evaluation','latent_intervention','final_recommendation']:
    (rep/f'{name}.md').write_text(f'# {name}\n\nUpdated by monitor. Completed/failed jobs: {len(rows)}.\n')
