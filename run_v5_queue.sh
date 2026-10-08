#!/usr/bin/env bash
# v5 training queue (user, 2026-10-09): runs unattended, detached from the Claude session.
# Resumable: each finished step leaves outputs/queue_v5/<step>.done and is skipped on a re-run.
# Progress: outputs/queue_v5/queue.log    (one line per step, with times)
#
#  1. frame-share pilot round 2 (ml_train_v5 already rebuilt with hand-written + wealth-management frames)
#  2. v4.1 / v4.2 / v4.3 baselines on ml_wm_dev (+ context grid)
#  3. choose the frame share (rule below), rebuild ml_train_v5 at that share if it differs from 10%
#  4. A-v5data  = v4.1's recipe on ml_train_v5                (Stage 1: the data-only contrast)
#  5. B1        = v4.2's recipe (adapters b=64) on ml_train_v5 (Stage 2 cell, existing code)
#  6. B'        = adapters on A-v5data's encoder              (Stage 2 cell, existing code)
# Each model is evaluated on ml_real, ml_synth, ml_mixed, ml_kiii_test, English test + stress,
# ml_struct_dev (per slice) and ml_wm_dev (per format + context grid). ml_struct_test / ml_wm_test
# are NOT touched here: they are reported once, at the end of v5.
set -u
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=../.venv/Scripts/python
Q=../outputs/queue_v5
mkdir -p $Q
LOG=$Q/queue.log
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
# do <name> <command...>: run once; on failure log it and stop the queue
do_step() { local n=$1; shift
  if [ -f "$Q/$n.done" ]; then step "skip $n (done)"; return 0; fi
  step "start $n"
  if "$@"; then touch "$Q/$n.done"; step "done  $n"; else step "FAILED $n: $*"; exit 1; fi; }

eval_all() {  # eval_all <tag> <model_dir>
  local t=$1 M=$2
  for s in ml_real ml_synth ml_mixed ml_kiii_test; do
    do_step eval_${t}_$s sh -c "$PY eval_multilingual.py --split $s --model_dir $M --out ../results/${t}_$s > ../outputs/eval_${t}_$s.log 2>&1"
  done
  for s in test stress; do
    do_step eval_${t}_en_$s sh -c "$PY evaluate.py --split $s --model_dir $M --out ../results/${t}_en_$s.json > ../results/${t}_en_$s.txt 2>&1 && $PY eval_confusion.py --split $s --model_dir $M --out ../results/${t}_en_${s}_confusion.json > ../results/${t}_en_${s}_confusion.txt 2>&1"
  done
  do_step eval_${t}_ml_struct_dev sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir $M --group slice --out ../results/${t}_ml_struct_dev > ../outputs/eval_${t}_ml_struct_dev.log 2>&1"
  do_step eval_${t}_ml_wm_dev sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir $M --group format --out ../results/${t}_ml_wm_dev > ../outputs/eval_${t}_ml_wm_dev.log 2>&1 && $PY eval_wm.py --split ml_wm_dev --pred ../outputs/${t}_ml_wm_dev_predictions.jsonl --out ../results/${t}_ml_wm_dev_grid > /dev/null 2>&1"
}

step "queue started (pid $$)"

# ---------------------------------------------------------------- 1. pilot round 2
for f in 00 10 25 40; do
  M=../outputs/pilot2_f$f
  do_step pilot2_f${f}_train sh -c "$PY train.py --config train_ml.yaml --train_split ml_pilot_f$f --val_split ml_val_v5 --output_dir outputs/pilot2_f$f > ../outputs/train_pilot2_f$f.log 2>&1"
  do_step pilot2_f${f}_struct sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir $M --group slice --out ../results/pilot2_f${f}_ml_struct_dev > ../outputs/eval_pilot2_f${f}_ml_struct_dev.log 2>&1"
  do_step pilot2_f${f}_wm sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir $M --group format --out ../results/pilot2_f${f}_ml_wm_dev > ../outputs/eval_pilot2_f${f}_ml_wm_dev.log 2>&1 && $PY eval_wm.py --split ml_wm_dev --pred ../outputs/pilot2_f${f}_ml_wm_dev_predictions.jsonl --out ../results/pilot2_f${f}_ml_wm_dev_grid > /dev/null 2>&1"
  do_step pilot2_f${f}_real sh -c "$PY eval_multilingual.py --split ml_real --limit 300 --model_dir $M --out ../results/pilot2_f${f}_ml_real > ../outputs/eval_pilot2_f${f}_ml_real.log 2>&1"
  rm -rf $M/checkpoint-*          # final model kept; checkpoints are 6 GB per run
done

# ---------------------------------------------------------------- 2. v4 baselines on ml_wm_dev
for m in "A:xlmr-base-pii-ml-A" "B64:xlmr-base-pii-ml-B64" "A-minilm:minilm-pii-ml-A"; do
  t=${m%%:*}; d=${m#*:}
  do_step base_${t}_wm sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir ../outputs/$d --group format --out ../results/${t}_ml_wm_dev > ../outputs/eval_${t}_ml_wm_dev.log 2>&1 && $PY eval_wm.py --split ml_wm_dev --pred ../outputs/${t}_ml_wm_dev_predictions.jsonl --out ../results/${t}_ml_wm_dev_grid > /dev/null 2>&1"
done

# ---------------------------------------------------------------- 3. choose the frame share
# Rule (fixed before seeing round-2 results): score = mean(ml_struct_dev F1, ml_wm_dev F1) over the
# shares whose ml_real F1 is within 0.010 of the 0% run; highest score wins, ties (< 0.002) go to
# the smaller share. Written to outputs/queue_v5/chosen_share.txt with the table.
do_step choose_share $PY -c "
import json
f1 = lambda p: json.load(open(p, encoding='utf-8'))['default']['strict']['ALL']['f1']
rows = []
for f in ['00', '10', '25', '40']:
    real, st, wm = f1(f'../results/pilot2_f{f}_ml_real.json'), f1(f'../results/pilot2_f{f}_ml_struct_dev.json'), f1(f'../results/pilot2_f{f}_ml_wm_dev.json')
    rows.append((int(f) / 100, real, st, wm, (st + wm) / 2))
base = rows[0][1]
ok = [r for r in rows if base - r[1] <= 0.010]
best = max(r[4] for r in ok)
share = min(r[0] for r in ok if best - r[4] < 0.002)
with open('../outputs/queue_v5/chosen_share.txt', 'w') as fh:
    fh.write('share  ml_real  struct_dev  wm_dev  score\n')
    for r in rows:
        fh.write(f'{r[0]:.2f}   {r[1]:.3f}    {r[2]:.3f}       {r[3]:.3f}   {r[4]:.3f}\n')
    fh.write(f'chosen {share}\n')
print('chosen share', share)
"
SHARE=$(tail -1 $Q/chosen_share.txt | awk '{print $2}')
step "frame share = $SHARE"
if [ "$SHARE" != "0.1" ]; then
  do_step rebuild_v5 sh -c "sed -i 's/^  share: 0.10 .*/  share: $SHARE          # chosen by pilot round 2 (run_v5_queue.sh)/' ../configs/ml_sources.yaml && $PY build_ml_train.py --v5 > ../outputs/build_ml_train_v5c.log 2>&1"
fi

# ---------------------------------------------------------------- 4. A-v5data
A=../outputs/xlmr-base-pii-ml-A-v5data
do_step train_A_v5data sh -c "$PY train.py --config train_ml.yaml --train_split ml_train_v5 --val_split ml_val_v5 --output_dir outputs/xlmr-base-pii-ml-A-v5data > ../outputs/train_A_v5data.log 2>&1"
eval_all A-v5data $A
rm -rf $A/checkpoint-*

# ---------------------------------------------------------------- 5. B1 (adapters on the pretrained encoder)
B=../outputs/xlmr-base-pii-ml-B64-v5data
do_step train_B1_v5data sh -c "$PY train.py --config train_ml_B.yaml --train_split ml_train_v5 --val_split ml_val_v5 --output_dir outputs/xlmr-base-pii-ml-B64-v5data > ../outputs/train_B1_v5data.log 2>&1"
eval_all B1-v5data $B
rm -rf $B/checkpoint-*

# ---------------------------------------------------------------- 6. B' (adapters on A-v5data's encoder)
BP=../outputs/xlmr-base-pii-ml-Bprime-v5data
do_step train_Bprime_v5data sh -c "$PY train.py --config train_ml_Bprime_v5.yaml --train_split ml_train_v5 --val_split ml_val_v5 --output_dir outputs/xlmr-base-pii-ml-Bprime-v5data > ../outputs/train_Bprime_v5data.log 2>&1"
eval_all Bprime-v5data $BP
rm -rf $BP/checkpoint-*

step "QUEUE COMPLETE"
