#!/usr/bin/env bash
# Reference run of img_check.py on the host: tt-metal-main f856a38a build + tt-metal-pr fae9cd0 models (git archive)
# + the LIBERO server module as packaged by pi05-libero-gpu-base. Runs INSIDE with-device.sh.
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
source /home/deepgadget/experiments/tt-metal-main/env-main.sh >/dev/null 2>&1
export PYTHONPATH=$SP/ref:/home/deepgadget/experiments/gr00t/libero_eval/pi05/tt_mc_matrix/package:$TT_METAL_HOME/ttnn PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$SP/ttcache_host TT_METAL_LOGGER_LEVEL=WARNING
H=$HOME/.cache/huggingface/hub
exec python $SP/img_check.py --out $SP/check_host --base $H/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba --libero $H/models--lerobot--pi05_libero/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd --golden $SP/chk/openloop_golden.pt --norm $SP/chk/openpi_pi05_libero_norm_stats.json --spm $SP/chk/paligemma_tokenizer.model "$@"
