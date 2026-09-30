#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp1r1
cd $S && source env.sh
./sums.sh hold1_start > out/sums_hold1_start.txt
for spec in "expert base" "off base" "expert libero" "off libero"; do
  set -- $spec
  echo "$(date +%F_%T) START arm $1 $2"
  PI05_MEGAKERNEL=$1 TT_METAL_CACHE=$(cache_for $1) timeout 900 python d_arm.py --shape $2 --out out/A_${1}_${2} > out/A_${1}_${2}.log 2>&1
  echo "$(date +%F_%T) END arm $1 $2 rc=$?"
done
for arm in expert off; do
  echo "$(date +%F_%T) START alt $arm"
  PI05_MEGAKERNEL=$arm TT_METAL_CACHE=$(cache_for $arm) timeout 1200 python d_alt.py out/ALT_$arm.json > out/ALT_$arm.log 2>&1
  echo "$(date +%F_%T) END alt $arm rc=$?"
done
./sums.sh hold1_end > out/sums_hold1_end.txt
