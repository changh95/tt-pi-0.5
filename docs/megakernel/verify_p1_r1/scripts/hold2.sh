#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp1r1
cd $S && source env.sh
./sums.sh hold2_start > out/sums_hold2_start.txt
for spec in "expert base 20" "off base 4" "expert libero 20" "off libero 4"; do
  set -- $spec
  echo "$(date +%F_%T) START prof $1 $2"
  PI05_MEGAKERNEL=$1 TT_METAL_CACHE=$(cache_for $1) timeout 1200 python -m tracy -r -p -v --no-op-info-cache --op-support-count 16000 -o $S/prof/${1}_${2} -m d_prof $2 $3 $S/out/P_${1}_${2}.json > $S/out/P_${1}_${2}.log 2>&1
  echo "$(date +%F_%T) END prof $1 $2 rc=$?"
  csv=$(ls $S/prof/${1}_${2}/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1); [ -n "$csv" ] && gzip -c "$csv" > $S/out/ops_${1}_${2}.csv.gz
  rm -f $S/prof/${1}_${2}/.logs/*.csv $S/prof/${1}_${2}/reports/*/*.csv
done
./sums.sh hold2_end > out/sums_hold2_end.txt
