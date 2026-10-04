#!/usr/bin/env bash
# Served-latency bench of the N16 image, both profiles, in impl's quiet slot (after the demo). Runs INSIDE with-device.sh:
#   DEVICE_LOCK_TIMEOUT=600 WITH_DEVICE_RESET_AFTER=1 with-device.sh timeout --signal=TERM --kill-after=60 1500 bench16.sh
# Per profile: tt-model serve --profile (timeout 600) -> /info profile check -> 100 warm requests (5 warm-up), host load in the JSON.
set -u -o pipefail
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
L=$SP/n16/bench; mkdir -p $L
S=/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc
MAN=/home/deepgadget/experiments/tt-models/build/pi05-base-p150/tt_kernel_manifest.json
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; PY=/usr/bin/python3
stopall() { $TTM stop $MAN >/dev/null 2>&1; docker rm -f $(docker ps -aq --filter label=org.tenstorrent.tt-model=pi05-base-p150) >/dev/null 2>&1; }
trap 'stopall; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
RC=0; fail() { echo "FAIL: $*"; RC=1; }
for PROF in non-scalable scalable; do
  echo "=== [$(date +%T)] $PROF: serve"
  stopall
  timeout 600 $TTM serve $MAN --profile $PROF 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|rror" | head -6 || fail "serve $PROF"
  CID=$(docker ps -q --filter label=org.tenstorrent.tt-model=pi05-base-p150 | head -1); [ -z "$CID" ] && { fail "no container $PROF"; continue; }
  echo "[$(date +%T)] bench $PROF"
  timeout 600 $PY $SP/n16/bench16.py --url http://127.0.0.1:20000 --media $S/media --n 100 --warmup 5 --out $L/bench-$PROF.json | tail -1 || fail "bench $PROF"
  $PY -c "import json,sys; j=json.load(open('$L/bench-$PROF.json')); p=j['info_profile']; sys.exit(0 if p and p['name']=='$PROF' else 1)" || fail "profile mismatch $PROF"
  docker logs -t $CID > $L/container-$PROF.log 2>&1
  stopall
done
echo "[$(date +%T)] done RC=$RC"; exit $RC
