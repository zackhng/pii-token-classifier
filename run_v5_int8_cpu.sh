#!/usr/bin/env bash
# v5 no-retraining experiment: dynamic int8 (CPU) vs fp32 (CPU) for v5.0 and SPAN, same 25 docs per language
# of ml_wm_dev / ml_struct_dev, plus CPU latency on the 100k-char multilingual document.
cd "$(dirname "$0")/src"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=
PY=../.venv/Scripts/python
for m in A-v5data:xlmr-base-pii-ml-A-v5data SPAN-v5data:pii-ml-SPAN-v5data; do
  t=${m%%:*}; d=../outputs/${m#*:}
  for prec in fp32 int8; do
    q=""; [ $prec = int8 ] && q="--int8"
    for s in ml_wm_dev ml_struct_dev; do
      $PY eval_multilingual.py --split $s --model_dir $d --device cpu $q --default_only --limit 25 --out ../results/cpu_${t}_${prec}_$s > ../outputs/eval_cpu_${t}_${prec}_$s.log 2>&1
    done
    $PY bench_latency.py --model_dir $d --doc ml --device cpu --runs 3 $q > ../results/${t}_latency_ml_cpu_${prec}.txt 2>&1
  done
done
echo done
