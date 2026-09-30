#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp1r1
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
./sums.sh soak_before > out/sums_soak_before.txt
for i in $(seq 1 20); do
  echo "$(date +%F_%T) soak $i start"
  PI05_MEGAKERNEL=expert TT_METAL_CACHE=$(cache_for expert) timeout 600 python d_soak1.py > out/soak_$i.log 2>&1
  rc=$?
  echo "$(date +%F_%T) soak $i rc=$rc $(grep RESULT out/soak_$i.log)"
done
./sums.sh soak_after > out/sums_soak_after.txt
echo "$(date +%F_%T) named repo alt (expert) start"
PI05_MEGAKERNEL=expert TT_METAL_CACHE=$(cache_for expert) timeout 900 python $REPO/models/experimental/pi0_5/tests/megakernel/verify_alternating.py --out out/ALT_named_repo_expert.json > out/ALT_named_repo_expert.log 2>&1
echo "$(date +%F_%T) named repo alt (expert) rc=$?"
