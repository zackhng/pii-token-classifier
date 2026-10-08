#!/usr/bin/env bash
# v5 Stage 2 queue (user, 2026-10-09): runs unattended after run_v5_queue.sh, detached from Claude.
# Resumable: finished steps leave outputs/queue_v5_stage2/<step>.done.   Log: outputs/queue_v5_stage2/queue.log
#
#  1. per-method learning-rate sweep on the 50k pilot set at the chosen frame share
#     (so "method" is compared at each method's best LR, not at one shared LR):
#       H  head only        3e-4 / 1e-3 / 3e-3
#       P4 top 4 layers     2e-5 / 5e-5 / 1e-4
#       S64 shared adapter  5e-5 / 1e-4 / 3e-4
#       B64 per-lang adapt. 5e-5 / 1e-4 / 3e-4
#       A  full fine-tune   1e-5 / 2e-5 / 5e-5   (2e-5 = the pilot round-2 run, reused)
#     rule (fixed before any sweep result): score = mean(best ml_val_v5 macro-F1, ml_struct_dev F1,
#     ml_wm_dev F1); highest wins, ties (< 0.002) go to the LR closest to the method's default.
#  2. full-size grid cells on ml_train_v5 at the chosen LR: H1, P1, S1 (+ A / B1 again if their
#     chosen LR differs from the one used in run_v5_queue.sh)
#  3. B' with a fresh head (only A-v5data's encoder inherited), LR 1e-4 as the queued B' - so the
#     two B' runs differ only in the head initialisation
#  4. gold-language evaluation for every adapter model (separates routing errors from model errors)
# DAPT row (H2, P2, B2, A2) is not here yet: its corpus / pretraining code is still to be written.
set -u
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=../.venv/Scripts/python
Q=../outputs/queue_v5_stage2
mkdir -p $Q
LOG=$Q/queue.log
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
do_step() { local n=$1; shift
  if [ -f "$Q/$n.done" ]; then step "skip $n (done)"; return 0; fi
  step "start $n"
  if "$@"; then touch "$Q/$n.done"; step "done  $n"; else step "FAILED $n: $*"; exit 1; fi; }

eval_all() {  # eval_all <tag> <model_dir> [gold]
  local t=$1 M=$2 gold=${3:-}
  for s in ml_real ml_synth ml_mixed ml_kiii_test; do
    do_step eval_${t}_$s sh -c "$PY eval_multilingual.py --split $s --model_dir $M --out ../results/${t}_$s > ../outputs/eval_${t}_$s.log 2>&1"
  done
  for s in test stress; do
    do_step eval_${t}_en_$s sh -c "$PY evaluate.py --split $s --model_dir $M --out ../results/${t}_en_$s.json > ../results/${t}_en_$s.txt 2>&1 && $PY eval_confusion.py --split $s --model_dir $M --out ../results/${t}_en_${s}_confusion.json > ../results/${t}_en_${s}_confusion.txt 2>&1"
  done
  do_step eval_${t}_ml_struct_dev sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir $M --group slice --out ../results/${t}_ml_struct_dev > ../outputs/eval_${t}_ml_struct_dev.log 2>&1"
  do_step eval_${t}_ml_wm_dev sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir $M --group format --out ../results/${t}_ml_wm_dev > ../outputs/eval_${t}_ml_wm_dev.log 2>&1 && $PY eval_wm.py --split ml_wm_dev --pred ../outputs/${t}_ml_wm_dev_predictions.jsonl --out ../results/${t}_ml_wm_dev_grid > /dev/null 2>&1"
  if [ "$gold" = gold ]; then gold_eval $t $M; fi
}
gold_eval() {  # adapter models routed by the gold language
  local t=$1 M=$2
  for s in ml_real ml_mixed ml_struct_dev ml_wm_dev; do
    do_step gold_${t}_$s sh -c "$PY eval_multilingual.py --split $s --model_dir $M --gold_lang --out ../results/${t}_${s}_gold > ../outputs/eval_${t}_${s}_gold.log 2>&1"
  done
}

step "stage-2 queue started (pid $$)"
grep -q "QUEUE COMPLETE" ../outputs/queue_v5/queue.log || { step "run_v5_queue.sh has not completed - stopping"; exit 1; }
SHARE=$(tail -1 ../outputs/queue_v5/chosen_share.txt | awk '{print $2}')
PF=$(printf "%02d" $(awk -v s="$SHARE" 'BEGIN{printf "%d", s*100 + 0.5}'))
step "frame share $SHARE -> sweep on ml_pilot_f$PF"

# ---------------------------------------------------------------- 1. LR sweep
sweep() {  # sweep <method> <config> <lr>
  local m=$1 c=$2 lr=$3 tag=sweep_${1}_${3}
  local M=../outputs/$tag
  do_step ${tag}_train sh -c "$PY train.py --config $c --train_split ml_pilot_f$PF --val_split ml_val_v5 --learning_rate $lr --output_dir outputs/$tag > ../outputs/train_$tag.log 2>&1"
  do_step ${tag}_struct sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir $M --out ../results/${tag}_ml_struct_dev > ../outputs/eval_${tag}_ml_struct_dev.log 2>&1"
  do_step ${tag}_wm sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir $M --out ../results/${tag}_ml_wm_dev > ../outputs/eval_${tag}_ml_wm_dev.log 2>&1"
  rm -rf $M/checkpoint-*
}
for lr in 3e-4 1e-3 3e-3; do sweep H train_ml_H.yaml $lr; done
for lr in 2e-5 5e-5 1e-4; do sweep P4 train_ml_P4.yaml $lr; done
for lr in 5e-5 1e-4 3e-4; do sweep S64 train_ml_S64.yaml $lr; done
for lr in 5e-5 1e-4 3e-4; do sweep B64 train_ml_B.yaml $lr; done
for lr in 1e-5 5e-5; do sweep A train_ml.yaml $lr; done
# A at 2e-5 on this pilot set = pilot round 2 (same data, LR, seed): reuse its results
cp ../results/pilot2_f${PF}_ml_struct_dev.json ../results/sweep_A_2e-5_ml_struct_dev.json
cp ../results/pilot2_f${PF}_ml_wm_dev.json ../results/sweep_A_2e-5_ml_wm_dev.json
cp ../outputs/train_pilot2_f${PF}.log ../outputs/train_sweep_A_2e-5.log

do_step choose_lr $PY -c "
import json, re
f1 = lambda p: json.load(open(p, encoding='utf-8'))['default']['strict']['ALL']['f1']
grid = {'H': (['3e-4', '1e-3', '3e-3'], '1e-3'), 'P4': (['2e-5', '5e-5', '1e-4'], '5e-5'), 'S64': (['5e-5', '1e-4', '3e-4'], '1e-4'),
        'B64': (['5e-5', '1e-4', '3e-4'], '1e-4'), 'A': (['1e-5', '2e-5', '5e-5'], '2e-5')}
out, lines = {}, ['method  lr     val_macro  struct_dev  wm_dev  score']
for m, (lrs, default) in grid.items():
    rows = []
    for lr in lrs:
        log = open(f'../outputs/train_sweep_{m}_{lr}.log', encoding='utf-8', errors='ignore').read()
        val = max(float(x) for x in re.findall(r\"'eval_macro_f1': '([0-9.]+)'\", log))
        st, wm = f1(f'../results/sweep_{m}_{lr}_ml_struct_dev.json'), f1(f'../results/sweep_{m}_{lr}_ml_wm_dev.json')
        rows.append((lr, val, st, wm, (val + st + wm) / 3))
        lines.append(f'{m:6s}  {lr:5s}  {val:9.3f}  {st:10.3f}  {wm:6.3f}  {(val + st + wm) / 3:.3f}')
    best = max(r[4] for r in rows)
    near = [r[0] for r in rows if best - r[4] < 0.002]
    out[m] = min(near, key=lambda lr: abs(float(lr) - float(default)) / float(default))
lines.append('chosen ' + json.dumps(out))
open('../outputs/queue_v5_stage2/chosen_lr.txt', 'w').write('\n'.join(lines) + '\n')
json.dump(out, open('../outputs/queue_v5_stage2/chosen_lr.json', 'w'))
print(out)
"
lr_of() { $PY -c "import json; print(json.load(open('../outputs/queue_v5_stage2/chosen_lr.json'))['$1'])"; }

# ---------------------------------------------------------------- 2. full-size grid cells at the chosen LR
full() {  # full <tag> <config> <lr> [gold]
  local t=$1 c=$2 lr=$3 g=${4:-}
  do_step train_$t sh -c "$PY train.py --config $c --train_split ml_train_v5 --val_split ml_val_v5 --learning_rate $lr --output_dir outputs/xlmr-base-pii-ml-$t > ../outputs/train_$t.log 2>&1"
  eval_all $t ../outputs/xlmr-base-pii-ml-$t $g
  rm -rf ../outputs/xlmr-base-pii-ml-$t/checkpoint-*
}
full H1-v5data train_ml_H.yaml $(lr_of H)
full P1-v5data train_ml_P4.yaml $(lr_of P4)
full S1-v5data train_ml_S64.yaml $(lr_of S64)
[ "$(lr_of A)" != "2e-5" ] && full A-v5data-lr$(lr_of A) train_ml.yaml $(lr_of A)
[ "$(lr_of B64)" != "1e-4" ] && full B1-v5data-lr$(lr_of B64) train_ml_B.yaml $(lr_of B64) gold

# ---------------------------------------------------------------- 3. B' with a fresh head (LR 1e-4 as the queued B')
do_step train_Bprime-fresh sh -c "$PY train.py --config train_ml_Bprime_v5_fresh.yaml --train_split ml_train_v5 --val_split ml_val_v5 --output_dir outputs/xlmr-base-pii-ml-Bprime-fresh-v5data > ../outputs/train_Bprime_fresh_v5data.log 2>&1"
eval_all Bprime-fresh-v5data ../outputs/xlmr-base-pii-ml-Bprime-fresh-v5data gold
rm -rf ../outputs/xlmr-base-pii-ml-Bprime-fresh-v5data/checkpoint-*

# ---------------------------------------------------------------- 4. gold-language evaluation of the queued adapter models
gold_eval B1-v5data ../outputs/xlmr-base-pii-ml-B64-v5data
gold_eval Bprime-v5data ../outputs/xlmr-base-pii-ml-Bprime-v5data

step "STAGE 2 COMPLETE"
