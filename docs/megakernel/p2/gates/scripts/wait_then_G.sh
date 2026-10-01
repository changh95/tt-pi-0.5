#!/bin/bash
until grep -q "HOLD F end" /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/holds.log; do sleep 10; done
echo "$(date '+%F %T') HOLD G start" >> /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/holds.log
cd /home/deepgadget/experiments/gr00t && DEVICE_LOCK_TIMEOUT=14400 timeout 1790 bin/with-device.sh /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/holdG.sh > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/holdG.log 2>&1
echo "$(date '+%F %T') HOLD G end rc=$?" >> /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/holds.log
