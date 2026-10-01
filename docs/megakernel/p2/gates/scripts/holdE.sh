#!/bin/bash
set -o pipefail
cd /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates && source env.sh
for spec in "whole base 20" "whole libero 20"; do
  set -- $spec
  rm -rf /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/prof/${1}_${2}
  echo "$(date +%F_%T) START prof $1 $2"
  PI05_MEGAKERNEL=$1 TT_METAL_CACHE=$(cache_for ${1}_prof) timeout 1200 python -m tracy -r -p -v --no-op-info-cache --op-support-count 16000 -o /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/prof/${1}_${2} -m d_prof $2 $3 /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/P_${1}_${2}.json > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/P_${1}_${2}.log 2>&1
  echo "$(date +%F_%T) END prof $1 $2 rc=$?"
  csv=$(ls /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/prof/${1}_${2}/reports/*/ops_perf_results_*.csv 2>/dev/null | head -1); [ -n "$csv" ] && gzip -c "$csv" > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/ops_${1}_${2}.csv.gz
done
