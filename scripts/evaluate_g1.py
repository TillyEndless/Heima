#!/usr/bin/env python3
import json
from pathlib import Path


def main() -> None:
    reports = {}
    for path in [
        Path("experiments/g1_gpt2/reports/main_only_result.json"),
        Path("experiments/g1_gpt2/reports/main_l1_result.json"),
    ]:
        if path.exists():
            reports[path.stem] = json.loads(path.read_text())
    output = Path("experiments/g1_gpt2/reports/summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v.get("status") for k, v in reports.items()}, indent=2))


if __name__ == "__main__":
    main()
