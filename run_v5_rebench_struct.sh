#!/usr/bin/env bash
# After run_v5_heads_encoders.sh: re-time v5.0, CRF and SPAN on an idle GPU with the batched struct
# decoder (heads.decode_many, 2026-10-10). Output: results/<tag>_latency_ml_gpu_batched.txt
cd "$(dirname "$0")"
until grep -q "HEAD + ENCODER QUEUE COMPLETE" outputs/queue_v5_he/queue.log 2>/dev/null; do sleep 300; done
cd src
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for m in A-v5data:xlmr-base-pii-ml-A-v5data CRF-v5data:pii-ml-CRF-v5data SPAN-v5data:pii-ml-SPAN-v5data; do
  ../.venv/Scripts/python bench_latency.py --model_dir ../outputs/${m#*:} --doc ml --device cuda > ../results/${m%%:*}_latency_ml_gpu_batched.txt 2>&1
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] struct re-bench done" >> ../outputs/queue_v5_he/queue.log
