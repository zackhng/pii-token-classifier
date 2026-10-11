#!/usr/bin/env bash
# v5: span head on mmBERT (user, 2026-10-10). Each run = its softmax parent (MMBS-v5data / MMBB-v5data) with
# ONE change, the head: GlobalPointer span head at head LR 1e-4, exactly as SPAN-v5data on XLM-R.
# Starts after run_v5_noretrain_gpu.sh. Resumable: outputs/queue_v5_mmspan/<step>.done  Log: .../queue.log
cd "$(dirname "$0")/../.."
until grep -q "NO-RETRAIN GPU EXPERIMENTS COMPLETE" outputs/queue_v5_noretrain/queue.log 2>/dev/null; do sleep 120; done
cd src
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=../.venv/Scripts/python
Q=../outputs/queue_v5_mmspan; mkdir -p $Q; LOG=$Q/queue.log
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
do_step() { local n=$1; shift
  if [ -f "$Q/$n.done" ]; then step "skip $n (done)"; return 0; fi
  step "start $n"
  if "$@"; then touch "$Q/$n.done"; step "done  $n"; else step "FAILED $n: $*"; return 1; fi; }
eval_all() {  # eval_all <tag> <model_dir>
  local t=$1 M=$2
  for s in ml_real ml_synth ml_mixed ml_kiii_test; do
    do_step eval_${t}_$s sh -c "$PY eval_multilingual.py --split $s --model_dir $M --out ../results/${t}_$s > ../outputs/eval_${t}_$s.log 2>&1"
  done
  for s in test stress; do
    do_step eval_${t}_en_$s sh -c "$PY evaluate.py --split $s --model_dir $M --out ../results/${t}_en_$s.json > ../results/${t}_en_$s.txt 2>&1"
  done
  do_step eval_${t}_ml_struct_dev sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir $M --group slice --out ../results/${t}_ml_struct_dev > ../outputs/eval_${t}_ml_struct_dev.log 2>&1"
  do_step eval_${t}_ml_wm_dev sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir $M --group format --out ../results/${t}_ml_wm_dev > ../outputs/eval_${t}_ml_wm_dev.log 2>&1 && $PY eval_wm.py --split ml_wm_dev --pred ../outputs/${t}_ml_wm_dev_predictions.jsonl --out ../results/${t}_ml_wm_dev_grid > /dev/null 2>&1"
  do_step latency_$t sh -c "$PY bench_latency.py --model_dir $M --doc ml --device cuda > ../results/${t}_latency_ml_gpu.txt 2>&1"
}
train_try() {  # batch 32x1, then 16x2, then 8x4 + checkpointing on CUDA OOM (same effective batch)
  local t=$1; shift
  local out=outputs/pii-ml-$t
  for b in "32 1" "16 2" "8 4 --grad_ckpt"; do
    local bs=${b%% *} rest=${b#* }
    rm -rf ../$out
    $PY train.py --config train_ml.yaml --train_split ml_train_v5 --val_split ml_val_v5 --output_dir $out \
      --batch_size $bs --grad_accum $rest "$@" > ../outputs/train_$t.log 2>&1 && return 0
    grep -q "out of memory\|OutOfMemoryError" ../outputs/train_$t.log || return 1
    step "$t: CUDA out of memory at batch $bs - retrying smaller"
  done
  return 1
}
run() { local t=$1; shift
  do_step train_$t train_try $t "$@" || return 0
  eval_all $t ../outputs/pii-ml-$t
  rm -rf ../outputs/pii-ml-$t/checkpoint-*
}
step "mmBERT + span queue started (pid $$)"
run MMBS-SPAN-v5data --model_name jhu-clsp/mmBERT-small --head span --head_lr 1e-4
run MMBB-SPAN-v5data --model_name jhu-clsp/mmBERT-base --head span --head_lr 1e-4
step "MMBERT SPAN QUEUE COMPLETE"
