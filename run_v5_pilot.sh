#!/usr/bin/env bash
# v5 frame-share pilot (plan 4e): v4.1's recipe (configs/train_ml.yaml) on four equal-size training
# sets with 0 / 10 / 25 / 40% frame documents (src/pilot_frames.py). Everything else is fixed.
# Scores: best ml_val_v5 macro-F1 (trainer_state.json), ml_real (300 docs / language, i.i.d.),
# ml_struct_dev (unseen structures, per slice). Progress: outputs/pipeline_v5_pilot.log
set -u
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=../.venv/Scripts/python
LOG=../outputs/pipeline_v5_pilot.log
step() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run() { "$@" || { step "FAILED: $*"; exit 1; }; }

for f in 00 10 25 40; do
  M=../outputs/pilot_f$f
  step "pilot f$f: train"
  run $PY train.py --config train_ml.yaml --train_split ml_pilot_f$f --val_split ml_val_v5 --output_dir outputs/pilot_f$f > ../outputs/train_pilot_f$f.log 2>&1
  step "pilot f$f: evaluate"
  run $PY eval_multilingual.py --split ml_struct_dev --model_dir $M --group slice --out ../results/pilot_f${f}_ml_struct_dev > ../outputs/eval_pilot_f${f}_ml_struct_dev.log 2>&1
  run $PY eval_multilingual.py --split ml_real --limit 300 --model_dir $M --out ../results/pilot_f${f}_ml_real > ../outputs/eval_pilot_f${f}_ml_real.log 2>&1
done
step "ALL DONE"
