#!/usr/bin/env bash
# Validation of the N16 image (both profiles) by manifest path, in ONE hold. Every step runs under `timeout`; the
# in-container checks also dump a faulthandler stack on a hang. Runs INSIDE with-device.sh.
set -u -o pipefail
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
N16=$SP/n16; L=$N16/val; mkdir -p $L
S=/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc
MAN=${MAN:-/home/deepgadget/experiments/tt-models/build/pi05-base-p150/tt_kernel_manifest.json}
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; PY=/usr/bin/python3
SMOKE=$S/code/models/experimental/pi0_5/server/smoke_test.py
stopall() { $TTM stop $MAN >/dev/null 2>&1; docker rm -f $(docker ps -aq --filter label=org.tenstorrent.tt-model=pi05-base-p150) pi05-mccheck >/dev/null 2>&1; }
trap 'stopall; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
RC=0; fail() { echo "FAIL: $*"; RC=1; }
info() { curl -s --max-time 30 localhost:20000/info | $PY -c "import json,sys; d=json.load(sys.stdin); m=d['megakernel']; print('info', json.dumps(m.get('profile')), {k: m['program'][k] for k in ('kernel_digest','cameras','action_horizon','num_steps','device_ops_per_call')}, 'source', d['source']['commit'])"; }
for PROF in non-scalable scalable; do
  echo "=== [$(date +%T)] $PROF: tt-model serve --profile $PROF"
  timeout 600 $TTM serve $MAN --profile $PROF 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|image|rror" | head -8 || fail "serve $PROF"
  CID=$(docker ps -q --filter label=org.tenstorrent.tt-model=pi05-base-p150 | head -1); [ -z "$CID" ] && { fail "no container $PROF"; continue; }
  docker inspect $CID > $L/inspect-$PROF.json
  info | tee $L/info-$PROF.txt
  timeout 300 $PY $SMOKE --url http://127.0.0.1:20000 --timeout 120 | tail -1 || fail "smoke $PROF"
  stopall
  echo "=== [$(date +%T)] $PROF: img_check2 in the image (golden, c1-c4 + c2 N16 outputs, refusals)"
  mkdir -p $N16/check-$PROF; chmod 777 $N16/check-$PROF
  CMD=$($PY $SP/mkrun_mc.py $L/inspect-$PROF.json pi05-mccheck "-v $SP/chk:/chk:ro -v $SP:/sp" "python /sp/n16/img_check2.py --out /sp/n16/check-$PROF --golden /chk/openloop_golden.pt" PYTHONDONTWRITEBYTECODE=1)
  timeout 2400 bash -c "$CMD" > $L/imgcheck-$PROF.log 2>&1 || fail "img_check $PROF (rc $?)"
  grep -E "DEVICE|GOLDEN|BITID|DONE|Traceback|Fatal Python|PHASE" $L/imgcheck-$PROF.log | cut -c1-230
  echo "=== [$(date +%T)] $PROF: the card's env commands (3 cameras, H 10, N 5) and N 16"
  for SUB in "3 10 5" "2 50 16"; do set -- $SUB
    CMD=$($TTM serve $MAN --profile $PROF --print | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep '^docker run')
    CMD=$(echo "$CMD" | sed -e "s/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=$1/" -e "s/PI05_ACTION_HORIZON=50/PI05_ACTION_HORIZON=$2/" -e "s/PI05_NUM_STEPS=10/PI05_NUM_STEPS=$3/" -e 's/^docker run /docker run --detach /')
    eval "$CMD" >/dev/null
    for i in $(seq 1 100); do curl -s localhost:20000/health 2>/dev/null | grep -q '"ok"' && break; sleep 3; done
    info; timeout 300 $PY $SMOKE --url http://127.0.0.1:20000 --timeout 60 | tail -1 || fail "env smoke $PROF $SUB"
    stopall
  done
  echo "=== [$(date +%T)] $PROF: refusals at start (PI05_NUM_IMAGES=5, PI05_NUM_STEPS=17)"
  for SUBST in 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=5/' 's/PI05_NUM_STEPS=10/PI05_NUM_STEPS=17/'; do
    CMD=$($TTM serve $MAN --profile $PROF --print | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep '^docker run' | sed -e "$SUBST")
    timeout 300 bash -c "$CMD" > $L/refusal.log 2>&1; echo "exit $?: $(grep -o 'refused at startup: .*' $L/refusal.log | cut -c1-200)"
    stopall
  done
done
echo "[$(date +%T)] done RC=$RC"; exit $RC
