#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  ""|--download-models) ;;
  *) printf 'Usage: bash setup.sh [--download-models]\n' >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  printf 'Usage: bash setup.sh [--download-models]\n' >&2
  exit 2
fi
"${PYTHON:-python3.10}" -m venv .venv
.venv/bin/python -m pip install uv==0.10.2
.venv/bin/uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c 'import torch, torchvision, vllm; print("torch", torch.__version__, "torchvision", torchvision.__version__, "vllm", vllm.__version__)'
if [[ "${1:-}" == --download-models ]]; then
  .venv/bin/python vision_pipeline_demo/download_models.py
fi
