source /home/deepgadget/experiments/gr00t/bin/env.sh
export PYTHONPATH=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5:/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip1:$PYTHONPATH
export PYTHONDONTWRITEBYTECODE=1
export DEVICE_LOCK_TIMEOUT=14400
export TT_METAL_LOGGER_LEVEL=WARNING
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip1
cache_for() { echo $S/ttcache_$1; }
# arm runner: "default" = PI05_MEGAKERNEL UNSET (the new default), "off" = the comparator knob
arm() { local a=$1; shift; if [ "$a" = default ]; then env -u PI05_MEGAKERNEL IP1_ARM=default TT_METAL_CACHE=$(cache_for default) "$@"; else env PI05_MEGAKERNEL=off IP1_ARM=off TT_METAL_CACHE=$(cache_for off) "$@"; fi; }
