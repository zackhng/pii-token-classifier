#!/usr/bin/env bash
# Resume XLM-V-base + span-BS (user, 2026-10-12). The first run crashed at step 36,000 of 45,411 (2026-10-12
# 03:19): Windows ran out of commit memory (RAM + page file) while the checkpoint save copied the model and
# optimizer state (~9.3 GB with XLM-V's 902k-token vocabulary). checkpoint-32000 is complete; the broken
# checkpoint-36000 was moved to outputs/XLMV-BS-v5data_checkpoint-36000_broken.
# Starts after the SPAN-BS encoder queue ends (XLM-R large + latency), so nothing else is training or timing:
# XLM-V needs ~15.7 GB of GPU memory and the most free commit memory for its saves. Same arguments as the
# first run; the training log is appended to, so the validation history stays in one file.
# Then: the standard evaluation suite, GPU + CPU latency, README tables.
# Resumable: outputs/queue_v5_xlmv/<step>.done   Log: outputs/queue_v5_xlmv/queue.log
cd "$(dirname "$0")/../../src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=../.venv/Scripts/python
T=XLMV-BS-v5data; M=../outputs/pii-ml-$T
Q=../outputs/queue_v5_xlmv; mkdir -p $Q; LOG=$Q/queue.log
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
do_step() { local n=$1; shift
  if [ -f "$Q/$n.done" ]; then step "skip $n (done)"; return 0; fi
  step "start $n"
  if "$@"; then touch "$Q/$n.done"; step "done  $n"; else step "FAILED $n: $*"; return 1; fi; }
commit_free() { powershell -NoProfile -c "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory/1MB,1)"; }

# wait for the encoder queue to finish (or to stop for any reason)
until grep -q "SPAN-BS ENCODER QUEUE COMPLETE" ../outputs/queue_v5_bsenc/queue.log 2>/dev/null \
      || ! ps -ef | grep -q "[b]ash run_v5_bs_encoders.sh"; do sleep 120; done
step "XLM-V resume queue started (pid $$); free commit memory $(commit_free) GB"

resume() {
  local ck; ck=$(ls -d $M/checkpoint-* 2>/dev/null | sort -t- -k2 -n | while read d; do
    [ -f "$d/trainer_state.json" ] && [ -f "$d/scheduler.pt" ] && echo "$d"; done | tail -1)   # newest complete
  [ -n "$ck" ] || { step "no complete checkpoint in $M"; return 1; }
  step "resuming from $ck"
  echo "===== resume from $ck ($(date '+%Y-%m-%d %H:%M:%S')) =====" >> ../outputs/train_$T.log
  $PY train.py --config train_ml.yaml --train_split ml_train_v5 --val_split ml_val_v5 --output_dir outputs/pii-ml-$T \
    --batch_size 8 --grad_accum 4 --grad_ckpt --model_name facebook/xlm-v-base \
    --head span --head_lr 1e-4 --span_loss ce --bs_eps 0.1 --bs_d 1 \
    --resume "$(cd .. && pwd -W)/${ck#../}" >> ../outputs/train_$T.log 2>&1
}
do_step train_resume resume || exit 1
step "free commit memory after training $(commit_free) GB"

for s in ml_real ml_synth ml_mixed ml_kiii_test; do
  do_step eval_$s sh -c "$PY eval_multilingual.py --split $s --model_dir $M --out ../results/${T}_$s > ../outputs/eval_${T}_$s.log 2>&1"
done
for s in test stress; do
  do_step eval_en_$s sh -c "$PY evaluate.py --split $s --model_dir $M --out ../results/${T}_en_$s.json > ../results/${T}_en_$s.txt 2>&1"
done
do_step eval_ml_struct_dev sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir $M --group slice --out ../results/${T}_ml_struct_dev > ../outputs/eval_${T}_ml_struct_dev.log 2>&1"
do_step eval_ml_wm_dev sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir $M --group format --out ../results/${T}_ml_wm_dev > ../outputs/eval_${T}_ml_wm_dev.log 2>&1 && $PY eval_wm.py --split ml_wm_dev --pred ../outputs/${T}_ml_wm_dev_predictions.jsonl --out ../results/${T}_ml_wm_dev_grid > /dev/null 2>&1"
rm -rf $M/checkpoint-*

# latency: same documents, sizes and settings as results/v5/latency/{gpu,cpu}_latency
for dev in cuda cpu; do
  name=gpu; [ $dev = cpu ] && name=cpu
  OUT=../results/v5/latency/${name}_latency; mkdir -p $OUT
  for n in 1000 5000 50000 100000 200000; do
    runs=3; [ $n -le 5000 ] && runs=15; [ $dev = cuda ] && runs=10
    do_step latency_${name}_$n sh -c "CUDA_VISIBLE_DEVICES=$([ $dev = cpu ] && echo '' || echo 0) $PY bench_latency.py \
      --model_dir $M --doc ml --device $dev --runs $runs --chars $n > $OUT/${T}_$n.txt 2>&1"
  done
done
$PY -c "from results_layout import latency_summary, RESULTS
for d in ('gpu', 'cpu'): latency_summary(RESULTS / 'v5/latency' / f'{d}_latency')"
sed -i '/"XLMV-BS-v5data", "XLMV-BS-v5data"/s/, \(QUEUED\|RUN\|"[^"]*"\)),$/, DONE),/' experiments_table.py
$PY experiments_table.py > /dev/null 2>&1
step "XLM-V RESUME QUEUE COMPLETE"
