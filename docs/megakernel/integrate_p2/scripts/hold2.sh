#!/bin/bash
# integrate-p2 hold 2: profiled structural runs (tracy; raw logs deleted right after the CSV is extracted, df logged),
# then the request-path profile (9 calls alternating n 1/128/224, d_prof3).
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip2
cd $S && source env.sh
./sums.sh hold2_start > out/sums_hold2_start.txt
for spec in "whole base 20" "whole libero 20" "expert base 2" "expert libero 2" "off base 1" "off libero 1"; do
  set -- $spec
  rm -rf $S/prof
  echo "$(date +%F_%T) df $(df -h / | tail -1)"
  echo "$(date +%F_%T) START prof $1 $2 $3"
  arm $1 ${1}_prof timeout 1200 python -m tracy -r -p -v --no-op-info-cache --op-support-count 16000 -o $S/prof -m d_prof2 $2 $3 $S/out/P_${1}_${2}.json > $S/out/P_${1}_${2}.log 2>&1
  rc=$?
  echo "$(date +%F_%T) END prof $1 $2 rc=$rc"
  csv=$(ls $S/prof/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1); [ -n "$csv" ] && gzip -c "$csv" > $S/out/ops_${1}_${2}.csv.gz
  rm -rf $S/prof
done
echo "$(date +%F_%T) START prof3 whole base 9 requests"
arm whole whole_prof timeout 900 python -m tracy -r -p -v --no-op-info-cache --op-support-count 16000 -o $S/prof3 -m d_prof3 9 $S/out/P3_whole_base.json > $S/out/P3_whole_base.log 2>&1
rc=$?; echo "$(date +%F_%T) END prof3 rc=$rc"
csv=$(ls $S/prof3/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1); [ -n "$csv" ] && gzip -c "$csv" > $S/out/ops3_whole_base.csv.gz
rm -rf $S/prof3
echo "$(date +%F_%T) df $(df -h / | tail -1)"
./sums.sh hold2_end > out/sums_hold2_end.txt
