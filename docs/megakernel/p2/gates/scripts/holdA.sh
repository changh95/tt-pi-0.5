#!/bin/bash
set -o pipefail
cd /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates && source env.sh
for spec in "whole base 1" "expert base 1" "whole libero 1" "expert libero 1" "whole base 2" "expert base 2" "whole libero 2" "expert libero 2"; do
  set -- $spec
  echo "$(date +%F_%T) START arm $1 $2 round $3"
  PI05_MEGAKERNEL=$1 TT_METAL_CACHE=$(cache_for $1) timeout 900 python d_arm.py --shape $2 --out out/A_${1}_${2}_r$3 > out/A_${1}_${2}_r$3.log 2>&1
  echo "$(date +%F_%T) END arm $1 $2 round $3 rc=$? $(grep -o '\"call_median\": [0-9.]*' out/A_${1}_${2}_r$3.log)"
done
