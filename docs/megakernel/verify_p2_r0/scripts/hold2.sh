#!/bin/bash
# hold 2: profiled structural runs (tracy, private *_prof caches = fresh profiler builds) + alternating / shape switch
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp2r0
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
./sums.sh hold2_start > out/sums_hold2_start.txt
for spec in "whole base 20" "whole libero 20" "expert base 4" "expert libero 4" "off base 3" "off libero 3"; do
  set -- $spec
  rm -rf $S/prof/${1}_${2}
  echo "$(date +%F_%T) START prof $1 $2"
  PI05_MEGAKERNEL=$1 TT_METAL_CACHE=$(cache_for ${1}_prof) timeout 1200 python -m tracy -r -p -v --no-op-info-cache --op-support-count 16000 -o $S/prof/${1}_${2} -m d_prof2 $2 $3 $S/out/P_${1}_${2}.json > $S/out/P_${1}_${2}.log 2>&1
  rc=$?
  echo "$(date +%F_%T) END prof $1 $2 rc=$rc"
  csv=$(ls $S/prof/${1}_${2}/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1); [ -n "$csv" ] && gzip -c "$csv" > $S/out/ops_${1}_${2}.csv.gz
done
echo "$(date +%F_%T) START alt whole"
PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 1200 python d_alt2.py out/ALT_whole.json > out/ALT_whole.log 2>&1
rc=$?; echo "$(date +%F_%T) END alt whole rc=$rc"
echo "$(date +%F_%T) START named repo alt whole"
PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 900 python $REPO/models/experimental/pi0_5/tests/megakernel/verify_alternating.py --out out/ALT_named_whole.json > out/ALT_named_whole.log 2>&1
rc=$?; echo "$(date +%F_%T) END named repo alt whole rc=$rc"
./sums.sh hold2_end > out/sums_hold2_end.txt
