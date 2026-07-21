#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
"${PYTHON:-python3}" "${SCRIPT_DIR}/pipeline.py" "$@" --skip-eval
