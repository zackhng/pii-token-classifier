#!/usr/bin/env bash
# Waits for run_v5_deephead.sh to end (complete or failed), then starts run_v5_heads_encoders.sh.
cd "$(dirname "$0")/../.."
until grep -q "DEEP-HEAD QUEUE COMPLETE\|FAILED\|stopping" outputs/queue_v5_deephead/queue.log 2>/dev/null; do sleep 300; done
bash scripts/v5/run_v5_heads_encoders.sh > outputs/queue_v5_he_stdout.log 2>&1
