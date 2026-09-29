#!/usr/bin/env bash
# usage: run_fused_test.sh <which: libero|base> <label> [extra env...]
set -o pipefail
W=$1; L=$2; shift 2
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fusedfix
R=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
T=/home/deepgadget/experiments/gr00t/tt-metal
echo "$(date +%F_%T) START $W $L $*"
cd $S
env PYTHONDONTWRITEBYTECODE=1 TT_METAL_HOME=$T PYTHONPATH=$R:$T:$T/ttnn TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-pi05fix-gr00t \
  PI05_WEIGHTS_DIR=/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba \
  TT_FUSED=1 "$@" timeout 1700 $T/python_env/bin/python $R/models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py $W --out $S/results/$L.json
rc=$?
echo "$(date +%F_%T) END $W $L rc=$rc"
exit $rc
