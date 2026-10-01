#!/bin/bash
# integrate-p2 hold 1: every arm x shape, round 1 (32 base seeds / 8 LIBERO records, K/V read back, replays, poison, latency).
# whole = PI05_MEGAKERNEL unset (the default). Fresh private caches: run with WITH_DEVICE_RESET_AFTER=1.
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip2
cd $S && source env.sh
./sums.sh hold1_start > out/sums_hold1_start.txt
for spec in "whole base" "expert base" "off base" "whole libero" "expert libero" "off libero"; do
  set -- $spec
  echo "$(date +%F_%T) START arm $1 $2 r1"
  arm $1 $1 timeout 900 python d_arm2.py --shape $2 --out out/A_${1}_${2}_r1 > out/A_${1}_${2}_r1.log 2>&1
  rc=$?
  echo "$(date +%F_%T) END arm $1 $2 r1 rc=$rc $(grep -o '\"call_median\": [0-9.]*' out/A_${1}_${2}_r1.log)"
done
./sums.sh hold1_end > out/sums_hold1_end.txt
