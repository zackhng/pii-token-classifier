#!/usr/bin/env bash
# Held-out test sets for every v5 model (user, 2026-10-12): ml_struct_test (by slice), ml_wm_test and
# ml_wm_known (by format + context grid). Until now only released models (v4.x, v5.0) were scored on them;
# all v5 choices were made on the dev sets. SPAN (v5.1) and SPAN-BS (v5.2) first, for their release notes.
# Starts after the XLM-V resume queue, so it never competes with training or the latency runs.
# Resumable: outputs/queue_v5_test/<step>.done   Log: outputs/queue_v5_test/queue.log
cd "$(dirname "$0")/../../src"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=../.venv/Scripts/python
Q=../outputs/queue_v5_test; mkdir -p $Q; LOG=$Q/queue.log
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
do_step() { local n=$1; shift
  if [ -f "$Q/$n.done" ]; then step "skip $n (done)"; return 0; fi
  step "start $n"
  if "$@"; then touch "$Q/$n.done"; step "done  $n"; else step "FAILED $n: $*"; return 1; fi; }

running() { ps -ef | grep -q "[b]ash $1"; }
until grep -q "XLM-V RESUME QUEUE COMPLETE" ../outputs/queue_v5_xlmv/queue.log 2>/dev/null \
      || { ! running "scripts/v5/run_v5_xlmv_resume.sh" && ! running "run_v5_bs_encoders.sh"; }; do sleep 120; done
step "test-set queue started (pid $$)"

test_all() {  # test_all <tag> <model_dir>
  local t=$1 M=$2
  [ -f "$M/config.json" ] || { step "skip $t: no model at $M"; return 0; }
  do_step ${t}_ml_struct_test sh -c "$PY eval_multilingual.py --split ml_struct_test --model_dir $M --group slice \
    --out ../results/${t}_ml_struct_test > ../outputs/eval_${t}_ml_struct_test.log 2>&1"
  for s in ml_wm_test ml_wm_known; do
    do_step ${t}_$s sh -c "$PY eval_multilingual.py --split $s --model_dir $M --group format --out ../results/${t}_$s \
      > ../outputs/eval_${t}_$s.log 2>&1 && $PY eval_wm.py --split $s --pred ../outputs/${t}_${s}_predictions.jsonl \
      --out ../results/${t}_${s}_grid > /dev/null 2>&1"
  done
}
O=../outputs
test_all SPAN-v5data          $O/pii-ml-SPAN-v5data
test_all SPAN-BS-v5data       $O/pii-ml-SPAN-BS-v5data
$PY experiments_table.py > /dev/null 2>&1
step "SPAN + SPAN-BS (v5.1 / v5.2) DONE"
test_all SPAN-CE-v5data       $O/pii-ml-SPAN-CE-v5data
test_all CRF-v5data           $O/pii-ml-CRF-v5data
for t in AL2 AL4 AL8 H1 P1 S1 Bprime B1-v5data-lr3e-4; do
  case $t in B1-*) tag=$t;; *) tag=$t-v5data;; esac
  test_all $tag $O/xlmr-base-pii-ml-$tag
done
test_all B1-v5data            $O/xlmr-base-pii-ml-B64-v5data
for t in MMBS MMBB MML12 MML6 MMBS-SPAN MML12-SPAN MML6-SPAN MML12-BS MML6-BS INFO-BS MDEB-BS XLMV-BS XLMRL-BS; do
  test_all $t-v5data $O/pii-ml-$t-v5data
done
$PY experiments_table.py > /dev/null 2>&1
step "TEST-SET QUEUE COMPLETE"
