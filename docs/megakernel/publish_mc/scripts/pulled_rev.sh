#!/usr/bin/env bash
# The Hub-pulled package by repo id: default serve + smoke; the card's env commands (3 cameras, H 10, N 5) + /info + smoke;
# the 5-camera refusal. Runs INSIDE with-device.sh.
set -u -o pipefail
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
export PATH=/home/deepgadget/experiments/tt-models/.venv/bin:$PATH
SMOKE=$SP/ghmain/models/experimental/pi0_5/server/smoke_test.py
trap 'tt-model stop changh95/pi05-base-p150 2>&1 | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" | tail -1; docker rm -f tt-model-pi05-base-p150-p150 >/dev/null 2>&1; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
RC=0
echo "=== [$(date +%T)] default: tt-model serve changh95/pi05-base-p150"
tt-model serve changh95/pi05-base-p150 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|image|rror" | head -8
curl -s localhost:20000/info | python3 -c "import json,sys; d=json.load(sys.stdin); print('info', d['megakernel']['program'], 'source', d['source']['commit'])"
/usr/bin/python3 $SMOKE --url http://127.0.0.1:20000 --timeout 120 | tail -1; [ ${PIPESTATUS[0]} -ne 0 ] && RC=1
tt-model stop changh95/pi05-base-p150 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -1
echo "=== [$(date +%T)] the card's commands (3 cameras, H 10, N 5)"
CMD=$(tt-model serve changh95/pi05-base-p150 --print | grep '^docker run')
CMD=$(echo "$CMD" | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=3/' \
                        -e 's/PI05_ACTION_HORIZON=50/PI05_ACTION_HORIZON=10/' \
                        -e 's/PI05_NUM_STEPS=10/PI05_NUM_STEPS=5/' -e 's/^docker run /docker run --detach /')
eval "$CMD" >/dev/null
for i in $(seq 1 90); do curl -s localhost:20000/health 2>/dev/null | grep -q '"ok"' && break; sleep 3; done
curl -s localhost:20000/info | python3 -c "import json,sys; print(json.load(sys.stdin)['megakernel']['program'])"
/usr/bin/python3 $SMOKE --url http://127.0.0.1:20000 --timeout 120 | tail -1; [ ${PIPESTATUS[0]} -ne 0 ] && RC=1
docker rm -f tt-model-pi05-base-p150-p150      # stop the server
echo "=== [$(date +%T)] refusal PI05_NUM_IMAGES=5"
CMD=$(tt-model serve changh95/pi05-base-p150 --print | grep '^docker run' | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=5/')
eval "$CMD" > $SP/rev/refusal_pulled.log 2>&1; echo "container exit code $?"
grep "refused at startup" $SP/rev/refusal_pulled.log | cut -c1-230
docker rm -f tt-model-pi05-base-p150-p150 >/dev/null 2>&1
echo "[$(date +%T)] done RC=$RC"; exit $RC
