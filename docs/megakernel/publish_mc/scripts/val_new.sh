#!/usr/bin/env bash
# Full re-validation of image 4dd06e9d6fd3 (single profile p150) in ONE hold. Runs INSIDE with-device.sh.
set -u -o pipefail
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
R=$SP/rev; L=$R/logs; mkdir -p $L
S=/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc
MAN=/home/deepgadget/experiments/tt-models/build/pi05-base-p150/tt_kernel_manifest.json
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; N=tt-model-pi05-base-p150-p150
cleanup() { $TTM stop $MAN 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -1; docker rm -f $N pi05-mccheck >/dev/null 2>&1; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"; }
trap cleanup EXIT
RC=0
echo "=== [$(date +%T)] 1. default profile p150 via tt-model serve"
$TTM serve $MAN 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|image|rror" | head -8
CID=$(docker ps -q --filter label=org.tenstorrent.tt-model=pi05-base-p150 | head -1); [ -z "$CID" ] && { echo "SERVE FAILED"; exit 1; }
echo "container image $(docker inspect -f '{{.Image}}' $CID) profile label $(docker inspect -f '{{index .Config.Labels "org.tenstorrent.tt-model.profile"}}' $CID)"
docker inspect $CID > $L/inspect.json
curl -s localhost:20000/info > $L/info_default.json; python3 -c "import json; d=json.load(open('$L/info_default.json')); p=d['megakernel']['program']; print('info', d['megakernel']['backend'], p['kernel_digest'], 'cameras', p['cameras'], 'H', p['action_horizon'], 'N', p['num_steps'], 'ops', p['device_ops_per_call'], 'source', d['source']['commit'])"
python3 $S/code/models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 --timeout 120 | tail -1; [ ${PIPESTATUS[0]} -ne 0 ] && RC=1
python3 $SP/bench_http_pi05.py --url http://127.0.0.1:20000 --media $S/media --n 100 --warmup 5 --out $L/bench_default.json | tail -1; [ ${PIPESTATUS[0]} -ne 0 ] && RC=1
$TTM stop $MAN 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | tail -1
echo "=== [$(date +%T)] 2. img_check in the image (golden, bit-identity c1-c4, refusals)"
mkdir -p $R/check_new; chmod 777 $R/check_new
CMD=$(python3 $SP/mkrun_mc.py $L/inspect.json pi05-mccheck "-v $SP/chk:/chk:ro -v $SP:/sp" "python /sp/img_check.py --out /sp/rev/check_new" PYTHONDONTWRITEBYTECODE=1)
eval "$CMD" > $L/imgcheck.log 2>&1 || RC=1
grep -E "GOLDEN|BITID|DONE|Traceback" $L/imgcheck.log | cut -c1-260
echo "=== [$(date +%T)] 3. env launch with the card's commands (3 cameras, H 10, N 5)"
CMD=$($TTM serve $MAN --print | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep '^docker run')
CMD=$(echo "$CMD" | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=3/' \
                        -e 's/PI05_ACTION_HORIZON=50/PI05_ACTION_HORIZON=10/' \
                        -e 's/PI05_NUM_STEPS=10/PI05_NUM_STEPS=5/' -e 's/^docker run /docker run --detach /')
eval "$CMD" >/dev/null
for i in $(seq 1 90); do curl -s localhost:20000/health 2>/dev/null | grep -q '"ok"' && break; docker ps -q --filter name=$N | grep -q . || break; sleep 3; done
curl -s localhost:20000/info | python3 -c "import json,sys; print(json.load(sys.stdin)['megakernel']['program'])"
python3 $S/code/models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 --timeout 120 | tail -1; [ ${PIPESTATUS[0]} -ne 0 ] && RC=1
docker rm -f $N >/dev/null
echo "=== [$(date +%T)] 4. refusal PI05_NUM_IMAGES=5"
CMD=$($TTM serve $MAN --print | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep '^docker run' | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=5/')
eval "$CMD" > $L/refusal.log 2>&1; echo "container exit code $?"
grep -E "refused at startup" $L/refusal.log | cut -c1-240
docker rm -f $N >/dev/null 2>&1
echo "[$(date +%T)] done RC=$RC"; exit $RC
