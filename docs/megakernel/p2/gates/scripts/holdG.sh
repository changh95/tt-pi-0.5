#!/bin/bash
set -o pipefail
cd /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates && source env.sh
cd /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
echo "$(date +%F_%T) START pytest test_pcc_pi05_fused (whole)"
PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 1500 python -m pytest -p no:cacheprovider -q -s models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/pytest_pcc_whole.log 2>&1
echo "$(date +%F_%T) END pytest rc=$? $(tail -1 /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/pytest_pcc_whole.log)"
