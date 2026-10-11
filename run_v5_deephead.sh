#!/usr/bin/env bash
# v5 deep-head grid (user, 2026-10-09): N transformer layers between the encoder and the classifier
# (src/deep_head.py), encoder fine-tuned (AL* = A series + deep head). Runs after run_v5_stage2.sh.
# Frozen-encoder HL* dropped (user, 2026-10-09): head-only and the B series were not promising - build on A only.
# Resumable: outputs/queue_v5_deephead/<step>.done.   Log: outputs/queue_v5_deephead/queue.log
#
#  1. head-LR sweep at depth 4 on the 50k pilot set (frame share 25%):
#       head LR 5e-5 / 1e-4 / 3e-4            (encoder at v5.0's 2e-5)
#     rule (fixed in advance): score = mean(best ml_val_v5 macro-F1, ml_struct_dev F1, ml_wm_dev F1);
#     if the best LR is at an edge of the range, one more point beyond it is tried (x3 or /3) before
#     choosing; ties (< 0.002) go to 1e-4.
#  2. full-size models on ml_train_v5 at the chosen head LR: AL2, AL4, AL8
#  3. AL16 only if AL8 beats AL4 by > 0.002 on the rule
# Every model is evaluated on the full suite (as run_v5_queue.sh). Test sets are not touched.
set -u
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=../.venv/Scripts/python
Q=../outputs/queue_v5_deephead
mkdir -p $Q
LOG=$Q/queue.log
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
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
    do_step eval_${t}_en_$s sh -c "$PY evaluate.py --split $s --model_dir $M --out ../results/${t}_en_$s.json > ../results/${t}_en_$s.txt 2>&1"
  done
  do_step eval_${t}_ml_struct_dev sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir $M --group slice --out ../results/${t}_ml_struct_dev > ../outputs/eval_${t}_ml_struct_dev.log 2>&1"
  do_step eval_${t}_ml_wm_dev sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir $M --group format --out ../results/${t}_ml_wm_dev > ../outputs/eval_${t}_ml_wm_dev.log 2>&1 && $PY eval_wm.py --split ml_wm_dev --pred ../outputs/${t}_ml_wm_dev_predictions.jsonl --out ../results/${t}_ml_wm_dev_grid > /dev/null 2>&1"
}
# score <tag> -> mean(best val macro-F1 from ../outputs/train_<tag>.log, struct_dev, wm_dev)
score() { $PY -c "
import json, re, sys
t = sys.argv[1]
log = open(f'../outputs/train_{t}.log', encoding='utf-8', errors='ignore').read()
val = max(float(x) for x in re.findall(r\"'eval_macro_f1': '([0-9.]+)'\", log))
f1 = lambda s: json.load(open(f'../results/{t}_{s}.json', encoding='utf-8'))['default']['strict']['ALL']['f1']
print(f'{(val + f1(\"ml_struct_dev\") + f1(\"ml_wm_dev\")) / 3:.4f}')" "$1"; }

step "deep-head queue started (pid $$)"
grep -q "STAGE 2 COMPLETE" ../outputs/queue_v5_stage2/queue.log || { step "run_v5_stage2.sh has not completed - stopping"; exit 1; }
PILOT=ml_pilot_f25

# ---------------------------------------------------------------- 1. head-LR sweep at depth 4
sweep_point() {  # sweep_point <setting frozen|ft> <head_lr>
  local s=$1 lr=$2 tag=dhsweep_${1}_L4_${2} fz=""
  [ "$s" = frozen ] && fz="--freeze_encoder"
  do_step ${tag}_train sh -c "$PY train.py --config train_ml.yaml --train_split $PILOT --val_split ml_val_v5 --head_layers 4 $fz --head_lr $lr --output_dir outputs/$tag > ../outputs/train_$tag.log 2>&1"
  do_step ${tag}_struct sh -c "$PY eval_multilingual.py --split ml_struct_dev --model_dir ../outputs/$tag --out ../results/${tag}_ml_struct_dev > ../outputs/eval_${tag}_ml_struct_dev.log 2>&1"
  do_step ${tag}_wm sh -c "$PY eval_multilingual.py --split ml_wm_dev --model_dir ../outputs/$tag --out ../results/${tag}_ml_wm_dev > ../outputs/eval_${tag}_ml_wm_dev.log 2>&1"
  rm -rf ../outputs/$tag/checkpoint-*
  echo "$lr $(score $tag)" >> $Q/sweep_$s.txt
}
choose() {  # choose <setting> <lr...>: runs the sweep (+1 edge extension), prints the chosen LR
  local s=$1; shift
  local lrs=("$@")
  : > $Q/sweep_$s.txt
  for lr in "${lrs[@]}"; do sweep_point $s $lr; done
  local best=$(sort -k2 -gr $Q/sweep_$s.txt | head -1 | awk '{print $1}')
  local ext=""
  [ "$best" = "${lrs[0]}" ] && ext=$($PY -c "print(f'{float(\"$best\") / 3:.0e}')")
  [ "$best" = "${lrs[-1]}" ] && ext=$($PY -c "print(f'{float(\"$best\") * 3:.0e}')")
  if [ -n "$ext" ]; then step "best $s head LR $best is at the edge - extending to $ext"; sweep_point $s $ext; fi
  $PY -c "
rows = [l.split() for l in open('$Q/sweep_$s.txt')]
best = max(float(r[1]) for r in rows)
near = [r[0] for r in rows if best - float(r[1]) < 0.002]
print(min(near, key=lambda lr: abs(float(lr) - 1e-4)))"
}
LR_FT=$(choose ft 5e-5 1e-4 3e-4 | tail -1)
$PY -c "float('$LR_FT')" 2>/dev/null || { step "FAILED head-LR sweep (got '$LR_FT') - stopping"; exit 1; }
step "chosen head LR: fine-tuned $LR_FT"
echo "ft $LR_FT" > $Q/chosen_head_lr.txt

# ---------------------------------------------------------------- 2./3. full-size models
full() {  # full <setting> <depth> <head_lr>
  local s=$1 n=$2 lr=$3 fz="" t
  if [ "$s" = frozen ]; then fz="--freeze_encoder"; t=HL$n-v5data; else t=AL$n-v5data; fi
  do_step train_$t sh -c "$PY train.py --config train_ml.yaml --train_split ml_train_v5 --val_split ml_val_v5 --head_layers $n $fz --head_lr $lr --output_dir outputs/xlmr-base-pii-ml-$t > ../outputs/train_$t.log 2>&1"
  eval_all $t ../outputs/xlmr-base-pii-ml-$t
  rm -rf ../outputs/xlmr-base-pii-ml-$t/checkpoint-*
}
for s in ft; do
  lr=$LR_FT
  for n in 2 4 8; do full $s $n $lr; done
  p=H; [ "$s" = ft ] && p=A
  s4=$(score ${p}L4-v5data); s8=$(score ${p}L8-v5data)
  step "$s: depth 4 score $s4, depth 8 score $s8"
  if $PY -c "import sys; sys.exit(0 if $s8 - $s4 > 0.002 else 1)"; then
    step "$s: depth 8 beats depth 4 - running depth 16"
    full $s 16 $lr
  else
    step "$s: depth 8 does not beat depth 4 - depth 16 skipped"
  fi
done
step "DEEP-HEAD QUEUE COMPLETE"
