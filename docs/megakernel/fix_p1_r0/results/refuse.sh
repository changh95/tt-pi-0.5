#!/bin/bash
set -o pipefail
source /home/deepgadget/experiments/gr00t/bin/env.sh
export PYTHONPATH=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pylib:/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1
export TT_METAL_CACHE=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/cache PI05_MEGAKERNEL=expert
cd /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
for combo in "PI05_KV_DTYPE=bf16" "PI05_NUM_STEPS=20" "PI05_NUM_STEPS=5"; do
  echo "[$(date '+%F %T')] REFUSE $combo"
  env $combo timeout --signal=KILL 300 python -m uvicorn models.experimental.pi0_5.server.app:app --host 127.0.0.1 --port 20001 --lifespan on > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/refuse_${combo//=/_}.log 2>&1
  echo "rc=$?"; grep -h "refused at startup\|Opening\|Application startup complete" /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/refuse_${combo//=/_}.log
done
