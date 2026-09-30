#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip1
cd $S && source env.sh
./sums.sh hold1_start > out/sums_hold1_start.txt
echo "$(date +%F_%T) START nocut probe"
arm default timeout 600 python d_nocut.py out/NOCUT.json > out/NOCUT.log 2>&1; rc=$?
echo "$(date +%F_%T) END nocut probe rc=$rc"
for spec in "default base" "off base" "default libero" "off libero"; do
  set -- $spec
  echo "$(date +%F_%T) START arm $1 $2"
  arm $1 timeout 900 python d_arm.py --shape $2 --out out/A_${1}_${2} > out/A_${1}_${2}.log 2>&1; rc=$?
  echo "$(date +%F_%T) END arm $1 $2 rc=$rc"
done
for a in default off; do
  echo "$(date +%F_%T) START alt $a"
  arm $a timeout 1200 python d_alt.py out/ALT_$a.json > out/ALT_$a.log 2>&1; rc=$?
  echo "$(date +%F_%T) END alt $a rc=$rc"
done
./sums.sh hold1_end > out/sums_hold1_end.txt
