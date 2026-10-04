#!/usr/bin/env bash
# Host reference for the N16 image bit-identity: tt-metal-pr dd9431fa10f (git archive) on the eth16 runtime
# (~/experiments/tt-metal-eth16 c718b5df9b9 build), both profiles, the same img_check2.py inputs. Runs INSIDE with-device.sh.
set -u -o pipefail
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
source /home/deepgadget/experiments/env-eth16.sh >/dev/null 2>&1
export PYTHONPATH=$SP/n16/ref16:/home/deepgadget/experiments/gr00t/libero_eval/pi05/tt_dispatch_matrix/package:$TT_METAL_HOME/ttnn
export PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$SP/n16/ttcache_host16 TT_METAL_LOGGER_LEVEL=WARNING
H=$HOME/.cache/huggingface/hub; RC=0
for D in eth tensix; do
  echo "=== [$(date +%T)] host reference PI05_DISPATCH=$D"
  PI05_DISPATCH=$D timeout 2400 python $SP/n16/img_check2.py --out $SP/n16/host-$D --parts golden,bitid \
    --base $H/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba \
    --libero $H/models--lerobot--pi05_libero/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd \
    --golden $SP/chk/openloop_golden.pt --norm $SP/chk/openpi_pi05_libero_norm_stats.json --spm $SP/chk/paligemma_tokenizer.model \
    2>&1 | grep -E "PHASE|DEVICE|GOLDEN|BITID|DONE|Traceback|Fatal|Error" | cut -c1-200 || RC=1
done
echo "[$(date +%T)] done RC=$RC"; exit $RC
