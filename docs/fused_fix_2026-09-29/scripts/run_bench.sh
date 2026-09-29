#!/usr/bin/env bash
# usage: run_bench.sh <label> <code_root> <tree> [extra env...]
set -o pipefail
L=$1; C=$2; T=$3; shift 3
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fusedfix
mkdir -p $S/logs $S/results
echo "$(date +%F_%T) START $L code=$C tree=$T $*"
cd $S
env TT_METAL_HOME=$T PYTHONPATH=$C:$T:$T/ttnn TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-pi05fix-$(basename $(dirname $T)) \
  TT_FUSED=1 "$@" timeout 1700 $T/python_env/bin/python $S/bench_candidate.py --label $L --out $S/results/$L --runs ${RUNS:-60}
rc=$?
echo "$(date +%F_%T) END $L rc=$rc"
exit $rc
