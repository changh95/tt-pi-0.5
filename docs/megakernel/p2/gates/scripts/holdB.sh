#!/bin/bash
set -o pipefail
cd /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates && source env.sh
echo "$(date +%F_%T) START alt whole"
PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 1200 python d_alt.py out/ALT_whole.json > out/ALT_whole.log 2>&1
echo "$(date +%F_%T) END alt whole rc=$?"
echo "$(date +%F_%T) START named alt whole"
PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 900 python /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/models/experimental/pi0_5/tests/megakernel/verify_alternating.py --out out/ALT_named_whole.json > out/ALT_named_whole.log 2>&1
echo "$(date +%F_%T) END named alt whole rc=$?"
