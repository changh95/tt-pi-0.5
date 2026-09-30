#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/verify_p1
cd $S && source env.sh
for spec in "expert base" "off base" "expert libero" "off libero"; do
  set -- $spec
  echo "$(date +%F_%T) START $1 $2"
  PI05_MEGAKERNEL=$1 timeout 900 python v_arm.py --arm $1 --shape $2 --out out/A_${1}_${2} 2>&1 | grep -v "^\s*$" | tail -n 40 > out/A_${1}_${2}.log
  echo "$(date +%F_%T) END $1 $2 rc=$?"
done
