#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp2r0
cd $S && source env.sh
rm -rf $S/prof3
echo "$(date +%F_%T) START prof3 whole base 9 requests"
PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole_prof) timeout 900 python -m tracy -r -p -v --no-op-info-cache --op-support-count 16000 -o $S/prof3 -m d_prof3 9 $S/out/P3_whole_base.json > $S/out/P3_whole_base.log 2>&1
rc=$?; echo "$(date +%F_%T) END prof3 rc=$rc"
csv=$(ls $S/prof3/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1); [ -n "$csv" ] && gzip -c "$csv" > $S/out/ops3_whole_base.csv.gz
rm -rf $S/prof3
