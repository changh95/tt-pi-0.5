#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/verify_p1
cd $S && source env.sh
for arm in expert off; do
  echo "$(date +%F_%T) START alt $arm"
  PI05_MEGAKERNEL=$arm timeout 1500 python v_alt.py out/ALT_$arm.json > out/ALT_$arm.log 2>&1
  echo "$(date +%F_%T) END alt $arm rc=$?"
done
