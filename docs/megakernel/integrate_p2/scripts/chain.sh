#!/bin/bash
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip2
export DEVICE_LOCK_TIMEOUT=14400
for h in "$@"; do
  echo "$(date +%F_%T) submit $h" >> $S/out/holds.log
  WITH_DEVICE_RESET_AFTER=1 /home/deepgadget/experiments/gr00t/bin/with-device.sh timeout 1800 $S/$h.sh >> $S/out/$h.log 2>&1
  echo "$(date +%F_%T) done $h rc=$?" >> $S/out/holds.log
done
