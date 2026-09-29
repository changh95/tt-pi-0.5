#!/usr/bin/env bash
# usage: run_py.sh <code_root> <script> [args]
set -o pipefail
C=$1; shift
T=/home/deepgadget/experiments/gr00t/tt-metal
echo "$(date +%F_%T) START $*"
cd /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fusedfix
env PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME=$T PYTHONPATH=$C:$T:$T/ttnn TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-pi05fix-gr00t TT_FUSED=1 timeout 1700 $T/python_env/bin/python "$@"
rc=$?; echo "$(date +%F_%T) END rc=$rc"; exit $rc
