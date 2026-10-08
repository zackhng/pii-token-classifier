#!/usr/bin/env bash
# v5 frame-share pilot, round 2: ml_train_v5 rebuilt with the wealth-management frames and the
# ml_wm grid docs (user, 2026-10-08). v4.1's recipe on 0 / 10 / 25 / 40% frame documents.
# Scores: ml_real (300 docs / language, i.i.d.), ml_struct_dev and ml_wm_dev (choice sets).
# Progress: outputs/pipeline_v5_pilot2.log
set -u
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=../.venv/Scripts/python
LOG=../outputs/pipeline_v5_pilot2.log
step() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run() { "$@" || { step "FAILED: $*"; exit 1; }; }

step "rebuild ml_train_v5 (hand-written + wealth-management frames)"
run $PY build_ml_train.py --v5 > ../outputs/build_ml_train_v5b.log 2>&1
run $PY pilot_frames.py --total 50000 >> ../outputs/build_ml_train_v5b.log 2>&1
for f in 00 10 25 40; do
  M=../outputs/pilot2_f$f
  step "pilot2 f$f: train"
  run $PY train.py --config train_ml.yaml --train_split ml_pilot_f$f --val_split ml_val_v5 --output_dir outputs/pilot2_f$f > ../outputs/train_pilot2_f$f.log 2>&1
  step "pilot2 f$f: evaluate"
  run $PY eval_multilingual.py --split ml_struct_dev --model_dir $M --group slice --out ../results/pilot2_f${f}_ml_struct_dev > ../outputs/eval_pilot2_f${f}_ml_struct_dev.log 2>&1
  run $PY eval_multilingual.py --split ml_wm_dev --model_dir $M --group format --out ../results/pilot2_f${f}_ml_wm_dev > ../outputs/eval_pilot2_f${f}_ml_wm_dev.log 2>&1
  run $PY eval_multilingual.py --split ml_real --limit 300 --model_dir $M --out ../results/pilot2_f${f}_ml_real > ../outputs/eval_pilot2_f${f}_ml_real.log 2>&1
done
step "ALL DONE"
