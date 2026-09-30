source /home/deepgadget/experiments/gr00t/bin/env.sh
export PYTHONPATH=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5:/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp1r1:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export DEVICE_LOCK_TIMEOUT=14400
export TT_METAL_LOGGER_LEVEL=WARNING
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp1r1
# private per-arm caches: the off arm's cache must never contain the megakernel kernels (A/B arms differ)
cache_for() { echo $S/ttcache_$1; }
