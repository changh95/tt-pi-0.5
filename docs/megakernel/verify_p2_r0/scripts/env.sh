source /home/deepgadget/experiments/gr00t/bin/env.sh
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp2r0
export PYTHONPATH=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5:$S:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export DEVICE_LOCK_TIMEOUT=14400
export TT_METAL_LOGGER_LEVEL=WARNING
# private per-arm caches (verify-p2-r0 only); the off cache must never contain megakernel kernels
cache_for() { echo $S/ttcache_$1; }
