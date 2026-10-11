#!/usr/bin/env bash
# v5 head + encoder swaps on v5.0's recipe (user, 2026-10-10). Every run = v5.0 (A-v5data: ml_train_v5 /
# ml_val_v5, 384/96 windows, 3 epochs, seed 42, LR 2e-5, effective batch 32, macro-F1 checkpoint,
# same eval suite) with exactly ONE change:
#   heads (XLM-R base):  CRF-v5data   linear-chain CRF on the classifier   (transitions at head LR 5e-3)
#                        SPAN-v5data  GlobalPointer span head              (head LR 1e-4, AL sweep's choice)
#   encoders (softmax head as v5.0):
#                        MMBS-v5data  jhu-clsp/mmBERT-small
#                        MMBB-v5data  jhu-clsp/mmBERT-base
#                        EURO-v5data  EuroBERT/EuroBERT-210m
#                        QWEN-v5data  Qwen/Qwen3-Embedding-0.6B
# Memory: batch 32 x 1; on CUDA OOM the same run restarts at 16 x 2, then 8 x 4 + gradient checkpointing
# (same effective batch, same maths). Latency on the 100k-char multilingual document (GPU) per model.
# Runs after run_v5_deephead.sh. Resumable: outputs/queue_v5_he/<step>.done.  Log: outputs/queue_v5_he/queue.log
set -u
cd "$(dirname "$0")/../../src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # models pre-downloaded (user: no network during the queue)
PY=../.venv/Scripts/python
Q=../outputs/queue_v5_he
mkdir -p $Q
LOG=$Q/queue.log
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
train_try() {  # train_try <tag> <extra args...>: batch 32x1, then 16x2, then 8x4 + checkpointing on OOM
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
run() {  # run <tag> <extra train args...>
  local t=$1; shift
  do_step train_$t train_try $t "$@" || return 0       # a failed model is logged and skipped
  eval_all $t ../outputs/pii-ml-$t
  rm -rf ../outputs/pii-ml-$t/checkpoint-*
}

step "head + encoder queue started (pid $$)"
# AL queue stopped by the user before AL16 (2026-10-10): finish AL8's dev evals here
A8=../outputs/xlmr-base-pii-ml-AL8-v5data
for i in $(seq 40); do [ -f ../results/AL8-v5data_ml_wm_dev_grid.json ] && break; sleep 30; done   # its in-flight eval
[ -f ../results/AL8-v5data_en_stress.json ] || do_step eval_AL8-v5data_en_stress sh -c "$PY evaluate.py --split stress --model_dir $A8 --out ../results/AL8-v5data_en_stress.json > ../results/AL8-v5data_en_stress.txt 2>&1"
[ -f ../results/AL8-v5data_ml_struct_dev.json ] || do_step eval_AL8-v5data_ml_struct_dev sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir $A8 --group slice --out ../results/AL8-v5data_ml_struct_dev > ../outputs/eval_AL8-v5data_ml_struct_dev.log 2>&1"
[ -f ../results/AL8-v5data_ml_wm_dev_grid.json ] || do_step eval_AL8-v5data_ml_wm_dev sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir $A8 --group format --out ../results/AL8-v5data_ml_wm_dev > ../outputs/eval_AL8-v5data_ml_wm_dev.log 2>&1 && $PY eval_wm.py --split ml_wm_dev --pred ../outputs/AL8-v5data_ml_wm_dev_predictions.jsonl --out ../results/AL8-v5data_ml_wm_dev_grid > /dev/null 2>&1"
rm -rf $A8/checkpoint-*
do_step latency_A-v5data sh -c "$PY bench_latency.py --model_dir ../outputs/xlmr-base-pii-ml-A-v5data --doc ml --device cuda > ../results/A-v5data_latency_ml_gpu.txt 2>&1"
run CRF-v5data  --head crf --head_lr 5e-3
run SPAN-v5data --head span --head_lr 1e-4
run MMBS-v5data --model_name jhu-clsp/mmBERT-small
run MMBB-v5data --model_name jhu-clsp/mmBERT-base
run EURO-v5data --model_name EuroBERT/EuroBERT-210m
run QWEN-v5data --model_name Qwen/Qwen3-Embedding-0.6B
step "HEAD + ENCODER QUEUE COMPLETE"
