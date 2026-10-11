#!/usr/bin/env bash
# v5 encoders with the boundary-smoothed span head (user, 2026-10-11; SPAN-BS = per-span softmax CE + boundary
# smoothing eps 0.1, D 1, head LR 1e-4). MiniLM rows: parent = MiniLM + span, change = the SPAN-BS head.
# Encoder rows: parent = SPAN-BS (XLM-R base), change = the encoder. Then GPU + CPU latency for the new models
# and the README tables. Resumable: outputs/queue_v5_bsenc/<step>.done  Log: outputs/queue_v5_bsenc/queue.log
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=../.venv/Scripts/python
Q=../outputs/queue_v5_bsenc; mkdir -p $Q; LOG=$Q/queue.log
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
}
# Memory: effective batch is always 32. Windows does NOT raise CUDA OOM when the GPU is full - it spills into
# shared system RAM and training runs ~10x slower (seen with mDeBERTa at batch 32, 2026-10-11). So models that
# need more than 16 GB start at a configuration that fits (FIT=...), and the OOM fallback stays as a backstop.
train_try() {  # train_try <tag> <first config> <args...>; configs: "32 1" -> "16 2" -> "8 4 --grad_ckpt"
  local t=$1 first=$2; shift 2
  local out=outputs/pii-ml-$t on=""
  for b in "32 1" "16 2" "16 2 --grad_ckpt" "8 4 --grad_ckpt"; do
    [ "$b" = "$first" ] && on=1
    [ -z "$on" ] && continue
    local bs=${b%% *} rest=${b#* }
    rm -rf ../$out
    $PY train.py --config train_ml.yaml --train_split ml_train_v5 --val_split ml_val_v5 --output_dir $out \
      --batch_size $bs --grad_accum $rest "$@" > ../outputs/train_$t.log 2>&1 && return 0
    grep -q "out of memory\|OutOfMemoryError" ../outputs/train_$t.log || return 1
    step "$t: CUDA out of memory at batch $bs - retrying smaller"
  done
  return 1
}
BS="--head span --head_lr 1e-4 --span_loss ce --bs_eps 0.1 --bs_d 1"
run() { local t=$1 fit=$2; shift 2
  do_step train_$t train_try $t "$fit" "$@" $BS || return 0
  eval_all $t ../outputs/pii-ml-$t
  rm -rf ../outputs/pii-ml-$t/checkpoint-*
}
step "SPAN-BS encoder queue started (pid $$)"
run MML12-BS-v5data "32 1"              --model_name nreimers/mMiniLMv2-L12-H384-distilled-from-XLMR-Large
run MML6-BS-v5data  "32 1"              --model_name nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large
# order (user, 2026-10-11): InfoXLM and XLM-V first; mDeBERTa (~8 h) and XLM-R large (~11 h) after them
run INFO-BS-v5data  "32 1"              --model_name microsoft/infoxlm-base
$PY experiments_table.py > /dev/null 2>&1
run XLMV-BS-v5data  "8 4 --grad_ckpt"   --model_name facebook/xlm-v-base
$PY experiments_table.py > /dev/null 2>&1
step "INFOXLM + XLM-V DONE"
run MDEB-BS-v5data  "16 2"              --model_name microsoft/mdeberta-v3-base
$PY experiments_table.py > /dev/null 2>&1
run XLMRL-BS-v5data "16 2 --grad_ckpt"  --model_name FacebookAI/xlm-roberta-large
# latency (GPU, then CPU 6 threads) for the new models, same documents and sizes as results/{gpu,cpu}_latency
for dev in cuda cpu; do
  name=gpu; [ $dev = cpu ] && name=cpu
  OUT=../results/${name}_latency
  step "${name} latency start"
  for t in MML12-BS-v5data MML6-BS-v5data MDEB-BS-v5data INFO-BS-v5data XLMV-BS-v5data XLMRL-BS-v5data; do
    [ -d ../outputs/pii-ml-$t ] || continue
    for n in 1000 5000 50000 100000 200000; do
      runs=3; [ $n -le 5000 ] && runs=15; [ $dev = cuda ] && runs=10
      CUDA_VISIBLE_DEVICES=$([ $dev = cpu ] && echo "" || echo 0) $PY bench_latency.py --model_dir ../outputs/pii-ml-$t \
        --doc ml --device $dev --runs $runs --chars $n > $OUT/${t}_$n.txt 2>&1
    done
  done
done
$PY experiments_table.py > /dev/null 2>&1
step "SPAN-BS ENCODER QUEUE COMPLETE"
