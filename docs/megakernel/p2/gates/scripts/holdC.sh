#!/bin/bash
set -o pipefail
cd /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates && source env.sh
(cd /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5 && md5sum models/experimental/pi0_5/tt/megakernel/kernels_p2/* models/experimental/pi0_5/tt/megakernel/kernels/*) > out/sums_soak_before.txt
for i in $(seq 1 20); do
  echo "$(date +%F_%T) soak $i start"
  PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 600 python d_soak1.py > out/soak_$i.log 2>&1
  rc=$?
  echo "$(date +%F_%T) soak $i rc=$rc $(grep RESULT out/soak_$i.log)"
done
(cd /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5 && md5sum models/experimental/pi0_5/tt/megakernel/kernels_p2/* models/experimental/pi0_5/tt/megakernel/kernels/*) > out/sums_soak_after.txt
