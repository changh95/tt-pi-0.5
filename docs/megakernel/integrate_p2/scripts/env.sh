source /home/deepgadget/experiments/gr00t/bin/env.sh
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip2
export PYTHONPATH=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5:$S:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export DEVICE_LOCK_TIMEOUT=14400
export TT_METAL_LOGGER_LEVEL=WARNING
# integrate-p2 private per-arm caches
cache_for() { echo $S/ttcache_$1; }
# arm runner. "whole" = PI05_MEGAKERNEL UNSET (the new DEFAULT resolves to whole; files keep the resolved name);
# "expert" / "off" = the comparator knobs set explicitly. usage: arm ARM CACHE_NAME cmd...
arm() { local a=$1 c=$2; shift 2; if [ "$a" = whole ]; then env -u PI05_MEGAKERNEL IP2_ARM=default TT_METAL_CACHE=$(cache_for $c) "$@"; else env PI05_MEGAKERNEL=$a IP2_ARM=$a TT_METAL_CACHE=$(cache_for $c) "$@"; fi; }
