#!/bin/bash
# hold 2b (rerun of hold 2's tail after the disk filled at 10:27 with tracy logs): off profiled with 1 replay (raw logs
# deleted right after extraction), alternating / shape switch (own + named repo test).
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp2r0
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
./sums.sh hold2b_start > out/sums_hold2b_start.txt
echo "$(date +%F_%T) df $(df -h / | tail -1)"
rm -rf $S/prof/off_base
echo "$(date +%F_%T) START prof off base 1"
PI05_MEGAKERNEL=off TT_METAL_CACHE=$(cache_for off_prof) timeout 1200 python -m tracy -r -p -v --no-op-info-cache --op-support-count 16000 -o $S/prof/off_base -m d_prof2 base 1 $S/out/P_off_base.json > $S/out/P_off_base.log 2>&1
rc=$?; echo "$(date +%F_%T) END prof off base rc=$rc"
csv=$(ls $S/prof/off_base/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1); [ -n "$csv" ] && gzip -c "$csv" > $S/out/ops_off_base.csv.gz
rm -rf $S/prof
echo "$(date +%F_%T) df $(df -h / | tail -1)"
echo "$(date +%F_%T) START alt whole"
PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 1200 python d_alt2.py out/ALT_whole.json > out/ALT_whole.log 2>&1
rc=$?; echo "$(date +%F_%T) END alt whole rc=$rc"
echo "$(date +%F_%T) START named repo alt whole"
PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 900 python $REPO/models/experimental/pi0_5/tests/megakernel/verify_alternating.py --out out/ALT_named_whole.json > out/ALT_named_whole.log 2>&1
rc=$?; echo "$(date +%F_%T) END named repo alt whole rc=$rc"
./sums.sh hold2b_end > out/sums_hold2b_end.txt
