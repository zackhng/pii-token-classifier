#!/usr/bin/env bash
# v5 no-retraining experiments that need the model re-run (GPU), after run_v5_heads_encoders.sh and the idle
# re-bench: window merge rule (central / avg) and inference overlap 96 -> 64, for v5.0 and SPAN. Each variant
# changes one inference setting of an existing model; everything else as the queue's evals.
# Log: outputs/queue_v5_noretrain/queue.log
cd "$(dirname "$0")/../.."
until grep -q "struct re-bench done" outputs/queue_v5_he/queue.log 2>/dev/null; do sleep 120; done
cd src
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=../.venv/Scripts/python
Q=../outputs/queue_v5_noretrain; mkdir -p $Q; LOG=$Q/queue.log
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
for m in A-v5data:xlmr-base-pii-ml-A-v5data SPAN-v5data:pii-ml-SPAN-v5data; do
  t=${m%%:*}; d=../outputs/${m#*:}
  for v in "merge-central:--merge central" "merge-avg:--merge avg" "stride64:--stride 64"; do
    n=${v%%:*}; a=${v#*:}; tag=${t}_${n}
    for s in ml_struct_dev ml_wm_dev ml_synth ml_real ml_mixed ml_kiii_test; do
      [ -f ../results/${tag}_$s.json ] && continue
      step "start $tag $s"
      $PY eval_multilingual.py --split $s --model_dir $d $a --default_only --out ../results/${tag}_$s > ../outputs/eval_${tag}_$s.log 2>&1 || step "FAILED $tag $s"
    done
    for s in test stress; do
      [ -f ../results/${tag}_en_$s.json ] && continue
      step "start $tag en_$s"
      $PY evaluate.py --split $s --model_dir $d $a --out ../results/${tag}_en_$s.json > ../results/${tag}_en_$s.txt 2>&1 || step "FAILED $tag en_$s"
    done
  done
  $PY bench_latency.py --model_dir $d --doc ml --device cuda --stride 64 > ../results/${t}_latency_ml_gpu_stride64.txt 2>&1
  $PY bench_latency.py --model_dir $d --doc ml --device cuda --merge avg > ../results/${t}_latency_ml_gpu_mergeavg.txt 2>&1
done
step "NO-RETRAIN GPU EXPERIMENTS COMPLETE"
