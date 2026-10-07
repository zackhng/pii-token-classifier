#!/usr/bin/env bash
# Model A on Multilingual-MiniLM-L12-H384: train, evaluate on every v4 benchmark, latency.
# Progress: outputs/pipeline_minilm.log
set -u
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=../.venv/Scripts/python
LOG=../outputs/pipeline_minilm.log
step() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
run() { "$@" || { step "FAILED: $*"; exit 1; }; }
M=../outputs/minilm-pii-ml-A

step "1/3 train Model A on MiniLM"
run $PY train.py --config train_ml_minilm.yaml > ../outputs/train_ml_minilm.log 2>&1

step "2/3 evaluate"
for s in ml_real ml_synth ml_mixed ml_kiii_test; do
  run $PY eval_multilingual.py --split $s --model_dir $M --out ../results/A-minilm_$s > ../outputs/eval_A-minilm_$s.log 2>&1
done
for s in test stress; do
  run $PY evaluate.py --split $s --model_dir $M --out ../results/A-minilm_en_$s.json > ../results/A-minilm_en_$s.txt 2>&1
  run $PY eval_confusion.py --split $s --model_dir $M --out ../results/A-minilm_en_${s}_confusion.json > ../results/A-minilm_en_${s}_confusion.txt 2>&1
done

step "3/3 latency: v3, v4.1 (XLM-R) and MiniLM on the 100k-char document, same session"
for m in ../outputs/deberta-v3-xsmall-pii-v3 ../outputs/xlmr-base-pii-ml-A $M; do $PY bench_latency.py --model_dir $m --doc en --device cuda --runs 20 2>&1 | grep '^{'; done > ../results/latency_minilm.jsonl
for m in ../outputs/deberta-v3-xsmall-pii-v3 ../outputs/xlmr-base-pii-ml-A $M; do CUDA_VISIBLE_DEVICES="" $PY bench_latency.py --model_dir $m --doc en --device cpu --runs 3 2>&1 | grep '^{'; done >> ../results/latency_minilm.jsonl
step "ALL DONE"
