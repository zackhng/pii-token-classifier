#!/usr/bin/env bash
# Final Model A + Model B runs on the fixed data (ml_train v2), each step only if the previous one
# succeeded. Progress: outputs/pipeline_AB.log
set -u
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=../.venv/Scripts/python
LOG=../outputs/pipeline_AB.log
step() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run() { "$@" || { step "FAILED: $*"; exit 1; }; }

step "1/6 re-score v3 and A-pilot on the fixed ml_synth (zh-Hant 代碼 fix)"
run $PY eval_multilingual.py --split ml_synth --model_dir ../outputs/deberta-v3-xsmall-pii-v3 --out ../results/v3_ml_synth > ../outputs/eval_v3_ml_synth.log 2>&1
run $PY eval_multilingual.py --split ml_synth --model_dir ../outputs/deberta-v3-xsmall-pii-v3 --script_boundaries on --out ../results/v3_ml_synth_sb > ../outputs/eval_v3_ml_synth_sb.log 2>&1
run $PY eval_multilingual.py --split ml_synth --model_dir ../outputs/xlmr-base-pii-ml-A-pilot --out ../results/A-pilot_ml_synth > ../outputs/eval_A-pilot_ml_synth.log 2>&1

step "2/6 train Model A (final)"
run $PY train.py --config train_ml.yaml > ../outputs/train_ml_A.log 2>&1

step "3/6 evaluate Model A"
A=../outputs/xlmr-base-pii-ml-A
for s in ml_real ml_synth ml_mixed ml_kiii_test; do
  run $PY eval_multilingual.py --split $s --model_dir $A --out ../results/A_$s > ../outputs/eval_A_$s.log 2>&1
done
for s in test stress; do
  run $PY evaluate.py --split $s --model_dir $A --out ../results/A_en_$s.json > ../results/A_en_$s.txt 2>&1
  run $PY eval_confusion.py --split $s --model_dir $A --out ../results/A_en_${s}_confusion.json > ../results/A_en_${s}_confusion.txt 2>&1
done

step "4/6 retrain the router on the final training texts, train Model B (adapters b=64)"
run $PY router.py --out ../outputs/router > ../outputs/router.log 2>&1
run $PY train.py --config train_ml_B.yaml > ../outputs/train_ml_B.log 2>&1

step "5/6 evaluate Model B (router, and gold language)"
B=../outputs/xlmr-base-pii-ml-B64
for s in ml_real ml_synth ml_mixed ml_kiii_test; do
  run $PY eval_multilingual.py --split $s --model_dir $B --out ../results/B64_$s > ../outputs/eval_B64_$s.log 2>&1
  run $PY eval_multilingual.py --split $s --model_dir $B --gold_lang --out ../results/B64_${s}_gold > ../outputs/eval_B64_${s}_gold.log 2>&1
done
for s in test stress; do
  run $PY evaluate.py --split $s --model_dir $B --out ../results/B64_en_$s.json > ../results/B64_en_$s.txt 2>&1
  run $PY eval_confusion.py --split $s --model_dir $B --out ../results/B64_en_${s}_confusion.json > ../results/B64_en_${s}_confusion.txt 2>&1
done

step "6/6 latency: A and B on the 100k-char document"
for m in $A $B; do $PY bench_latency.py --model_dir $m --doc en --device cuda --runs 20 2>&1 | grep '^{'; done > ../results/latency_A_B.jsonl
for m in $A $B; do CUDA_VISIBLE_DEVICES="" $PY bench_latency.py --model_dir $m --doc en --device cpu --runs 3 2>&1 | grep '^{'; done >> ../results/latency_A_B.jsonl
step "ALL DONE"
