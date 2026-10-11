#!/usr/bin/env bash
# Latency of every v5 candidate on multilingual documents of 1k / 5k / 50k / 100k / 200k characters (ml_real
# text), on GPU and on CPU (6 threads), after ALL training queues (MiniLM, boundary smoothing) so nothing
# competes for the GPU or CPU. GPU first (minutes), then CPU.
# Output: results/{gpu,cpu}_latency/<tag>_<chars>.txt and results/{gpu,cpu}_latency/summary.tsv
cd "$(dirname "$0")"
until grep -q "BOUNDARY SMOOTHING QUEUE COMPLETE" outputs/queue_v5_bs/queue.log 2>/dev/null; do sleep 60; done
cd src
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=../outputs/queue_v5_bs/queue.log
MODELS="A-v5data:xlmr-base-pii-ml-A-v5data SPAN-v5data:pii-ml-SPAN-v5data MMBS-v5data:pii-ml-MMBS-v5data
        MMBS-SPAN-v5data:pii-ml-MMBS-SPAN-v5data MML12-v5data:pii-ml-MML12-v5data MML6-v5data:pii-ml-MML6-v5data
        MML12-SPAN-v5data:pii-ml-MML12-SPAN-v5data MML6-SPAN-v5data:pii-ml-MML6-SPAN-v5data
        SPAN-CE-v5data:pii-ml-SPAN-CE-v5data SPAN-BS-v5data:pii-ml-SPAN-BS-v5data"
summary() {  # summary <dir>
  ../.venv/Scripts/python - "$1" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
rows = ["model\tchars\ttokens\twindows\tmedian_s\tmin_s\tmax_s"]
for f in sorted(out.glob("*_*.txt")):
    try:
        d = json.loads(f.read_text(encoding="utf-8").strip().splitlines()[-1])
    except Exception:
        continue
    tag = f.stem.rsplit("_", 1)[0]
    rows.append(f"{tag}\t{d['chars']}\t{d['tokens']}\t{d['windows']}\t{d['median_s']}\t{d['min_s']}\t{d['max_s']}")
(out / "summary.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
}
for dev in cuda cpu; do
  name=gpu; [ $dev = cpu ] && name=cpu
  OUT=../results/${name}_latency; mkdir -p $OUT
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name^^} LATENCY START" >> $LOG
  for m in $MODELS; do
    [ -d ../outputs/${m#*:} ] || continue
    for n in 1000 5000 50000 100000 200000; do
      runs=3; [ $n -le 5000 ] && runs=15; [ $dev = cuda ] && runs=10
      CUDA_VISIBLE_DEVICES=$([ $dev = cpu ] && echo "" || echo 0) ../.venv/Scripts/python bench_latency.py \
        --model_dir ../outputs/${m#*:} --doc ml --device $dev --runs $runs --chars $n > $OUT/${m%%:*}_$n.txt 2>&1
    done
  done
  summary $OUT
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${name^^} LATENCY DONE" >> $LOG
done
