#!/bin/bash
# integrate-p2 hold 5: per-layer times from the host clock (d_layers, default); latency round 2, arms alternated.
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip2
cd $S && source env.sh
./sums.sh hold5_start > out/sums_hold5_start.txt
for sh in base libero; do
  echo "$(date +%F_%T) START layers $sh"
  arm whole whole timeout 600 python d_layers.py $sh out/L_$sh.json > out/L_$sh.log 2>&1
  rc=$?; echo "$(date +%F_%T) END layers $sh rc=$rc"
done
for spec in "off base" "expert base" "whole base" "off libero" "expert libero" "whole libero"; do
  set -- $spec
  echo "$(date +%F_%T) START arm $1 $2 r2"
  arm $1 $1 timeout 600 python d_arm2.py --shape $2 --nobs 2 --nokv --out out/A_${1}_${2}_r2 > out/A_${1}_${2}_r2.log 2>&1
  rc=$?; echo "$(date +%F_%T) END arm $1 $2 r2 rc=$rc $(grep -o '\"call_median\": [0-9.]*' out/A_${1}_${2}_r2.log)"
done
./sums.sh hold5_end > out/sums_hold5_end.txt
