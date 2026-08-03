from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
def test_job_registry_exists():
    cfg=json.loads((ROOT/'configs/teacher_five_points/jobs.json').read_text())
    assert len(cfg)==8
    assert any(p=='P0_CURRENT_O3_DYNAMIC' for jobs in cfg.values() for p,s in jobs)
def test_debug32_manifest_exists():
    assert (ROOT/'data/manifests/stage1_debug32_no_truncation.json').exists()
def test_scripts_exist():
    for name in ['teacher_job.py','gpu_worker.sh','launch_4090_workers.sh','aggregate.py','monitor.sh','status.sh']:
        assert (ROOT/'scripts/teacher_five_points'/name).exists()
