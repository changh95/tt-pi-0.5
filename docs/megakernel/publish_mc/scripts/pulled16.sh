#!/usr/bin/env bash
# Serve the Hub-pulled N16 package BY REPO ID on both profiles: /info (profile, digest, source) + smoke + 30-request
# bench. Every step under timeout. Runs INSIDE with-device.sh.
set -u -o pipefail
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
L=$SP/n16/pulled; mkdir -p $L
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; R=changh95/pi05-base-p150; PY=/usr/bin/python3
SMOKE=/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc/code/models/experimental/pi0_5/server/smoke_test.py
stopall() { $TTM stop $R >/dev/null 2>&1; docker rm -f $(docker ps -aq --filter label=org.tenstorrent.tt-model=pi05-base-p150) >/dev/null 2>&1; }
trap 'stopall; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
RC=0; fail() { echo "FAIL: $*"; RC=1; }
for PROF in non-scalable scalable; do
  echo "=== [$(date +%T)] serve $R --profile $PROF"; stopall
  timeout 600 $TTM serve $R --profile $PROF 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|rror" | head -8 || fail "serve $PROF"
  CID=$(docker ps -q --filter label=org.tenstorrent.tt-model=pi05-base-p150 | head -1); [ -z "$CID" ] && { fail "no container $PROF"; continue; }
  echo "container $(docker inspect -f '{{.Name}} {{.Image}}' $CID)"
  curl -s --max-time 30 localhost:20000/info > $L/info-$PROF.json
  $PY -c "import json,sys; d=json.load(open('$L/info-$PROF.json')); m=d['megakernel']; print('info', json.dumps(m['profile']), m['program']['kernel_digest'], m['program']['device_ops_per_call'], 'source', d['source']['commit']); sys.exit(0 if m['profile']['name']=='$PROF' and m['program']['kernel_digest']=='26f0c46b7f1721c1' and d['source']['commit'].startswith('33528a8') else 1)" || fail "info $PROF"
  timeout 300 $PY $SMOKE --url http://127.0.0.1:20000 --timeout 120 | tail -1 || fail "smoke $PROF"
  timeout 300 $PY $SP/n16/bench16.py --url http://127.0.0.1:20000 --media /home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc/media --n 30 --warmup 5 --out $L/bench-$PROF.json | tail -1 | cut -c1-260 || fail "bench $PROF"
  stopall
done
echo "[$(date +%T)] done RC=$RC"; exit $RC
