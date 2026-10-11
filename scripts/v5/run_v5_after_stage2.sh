#!/usr/bin/env bash
# Waits for run_v5_stage2.sh to finish, then starts run_v5_deephead.sh (only if stage 2 completed).
cd "$(dirname "$0")/../.."
until grep -q "STAGE 2 COMPLETE\|FAILED" outputs/queue_v5_stage2/queue.log 2>/dev/null; do sleep 300; done
if grep -q "STAGE 2 COMPLETE" outputs/queue_v5_stage2/queue.log; then
  bash scripts/v5/run_v5_deephead.sh > outputs/queue_v5_deephead_stdout.log 2>&1
else
  mkdir -p outputs/queue_v5_deephead
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] run_v5_stage2.sh failed - deep-head queue not started" >> outputs/queue_v5_deephead/queue.log
fi
