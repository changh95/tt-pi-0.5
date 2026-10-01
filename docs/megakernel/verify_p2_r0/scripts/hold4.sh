#!/bin/bash
# hold 4: (a) seed-707 rebuild sensitivity: two EMPTY caches rbA / rbB compile the same source, each runs the n-64
# seeds (706 707 708 809 810); (b) per-layer times from the host clock (d_layers, base + LIBERO, prefix-only variant =
# first run in my cache); (c) latency round 2, arms alternated, 2 obs, no K/V.
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp2r0
cd $S && source env.sh
./sums.sh hold4_start > out/sums_hold4_start.txt
for c in rbA rbB; do
  rm -rf $(cache_for $c); mkdir -p $(cache_for $c)
  echo "$(date +%F_%T) START rebuild $c"
  PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for $c) timeout 600 python d_arm2.py --shape base --seeds 706,707,708,809,810 --nokv --runs 5 --out out/RB_$c > out/RB_$c.log 2>&1
  rc=$?; echo "$(date +%F_%T) END rebuild $c rc=$rc"
  ./kelf.sh $(cache_for $c) $c > out/kelf_$c.txt
done
for sh in base libero; do
  echo "$(date +%F_%T) START layers $sh"
  PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 600 python d_layers.py $sh out/L_$sh.json > out/L_$sh.log 2>&1
  rc=$?; echo "$(date +%F_%T) END layers $sh rc=$rc"
done
for spec in "off base" "expert base" "whole base" "off libero" "expert libero" "whole libero"; do
  set -- $spec
  echo "$(date +%F_%T) START arm $1 $2 r2"
  PI05_MEGAKERNEL=$1 TT_METAL_CACHE=$(cache_for $1) timeout 600 python d_arm2.py --shape $2 --nobs 2 --nokv --out out/A_${1}_${2}_r2 > out/A_${1}_${2}_r2.log 2>&1
  rc=$?; echo "$(date +%F_%T) END arm $1 $2 r2 rc=$rc $(grep -o '\"call_median\": [0-9.]*' out/A_${1}_${2}_r2.log)"
done
./sums.sh hold4_end > out/sums_hold4_end.txt
