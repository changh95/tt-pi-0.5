#!/usr/bin/env bash
# Device test of the documented config mechanism: `tt-model serve ... --print` -> change the PI05_* env values -> docker run.
# Runs INSIDE with-device.sh. Uses the draft single-profile wire manifest (image 36f651704bf1, unchanged).
set -u -o pipefail
D=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc/draft_ste
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; M=$D/build/tt_kernel_manifest.json; N=tt-model-pi05-base-p150-p150
trap 'docker rm -f $N >/dev/null 2>&1; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
echo "[$(date +%T)] 1) a valid configuration: 3 cameras, action chunk 10, 5 steps"
CMD=$($TTM serve $M --print | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep '^docker run')
CMD=$(echo "$CMD" | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=3/' -e 's/PI05_ACTION_HORIZON=50/PI05_ACTION_HORIZON=10/' -e 's/PI05_NUM_STEPS=10/PI05_NUM_STEPS=5/' -e 's/^docker run /docker run --detach /')
echo "RUN: $CMD"
eval "$CMD" >/dev/null
for i in $(seq 1 90); do curl -s localhost:20000/health 2>/dev/null | grep -q '"ok"' && break; docker ps -q --filter name=$N | grep -q . || break; sleep 3; done
curl -s localhost:20000/info | python3 -c "import json,sys; d=json.load(sys.stdin); p=d['megakernel']['program']; print('info:', 'cameras', p['cameras'], 'action_horizon', p['action_horizon'], 'num_steps', p['num_steps'], 'suffix_rows', p['suffix_rows'], 'device_ops_per_call', p['device_ops_per_call'], 'inputs.num_images', d['inputs']['num_images'], 'outputs.actions', d['outputs']['actions'])"
python3 $SP/ghmain/models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 --timeout 120 | tail -1
docker rm -f $N >/dev/null
echo "[$(date +%T)] 2) an out-of-range value: PI05_NUM_IMAGES=5"
CMD=$($TTM serve $M --print | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep '^docker run' | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=5/')
eval "$CMD" > $D/build/refusal.log 2>&1; echo "container exit code $?"
grep -E "refused|RuntimeError|Application startup failed" $D/build/refusal.log | cut -c1-260 | head -4
docker rm -f $N >/dev/null 2>&1
echo "[$(date +%T)] done"
