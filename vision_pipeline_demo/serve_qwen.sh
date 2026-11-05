#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
: "${VLLM_API_KEY:?Set VLLM_API_KEY before starting the server}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
../.venv/bin/python download_models.py --only qwen
exec ../.venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$PWD/models/Qwen3-VL-8B-Instruct" --served-model-name qwen3-vl-8b --host 127.0.0.1 --port "${PORT:-8000}" \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":7,"video":0}' \
  --mm-processor-kwargs '{"min_pixels":65536,"max_pixels":131072}' \
  --mm-processor-cache-gb 0 --no-enable-prefix-caching \
  --structured-outputs-config '{"backend":"xgrammar"}' --generation-config vllm \
  --enforce-eager --seed 9803 --no-enable-log-requests
