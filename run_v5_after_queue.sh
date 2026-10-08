#!/usr/bin/env bash
# Waits for run_v5_queue.sh to finish, then starts run_v5_stage2.sh (only if the first queue completed).
cd "$(dirname "$0")"
until grep -q "QUEUE COMPLETE\|FAILED" outputs/queue_v5/queue.log 2>/dev/null; do sleep 300; done
if grep -q "QUEUE COMPLETE" outputs/queue_v5/queue.log; then
  bash run_v5_stage2.sh > outputs/queue_v5_stage2_stdout.log 2>&1
else
  mkdir -p outputs/queue_v5_stage2
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_v5_queue.sh failed - stage 2 not started" >> outputs/queue_v5_stage2/queue.log
fi
