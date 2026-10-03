#!/usr/bin/env bash
# Serve the Hub-pulled package by repo id (default profile), /info + smoke + 30-request bench, then the golden PCC7
# (img_check --parts golden) in the pulled image with the served container's spec. Runs INSIDE with-device.sh.
set -u -o pipefail
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; R=changh95/pi05-base-p150; L=$SP/logs/pulled; mkdir -p $L
trap '$TTM stop $R 2>&1 | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" | tail -1; docker rm -f pi05-mcpulled >/dev/null 2>&1; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
echo "[$(date +%T)] serve $R"
$TTM serve $R 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|image|rror" | head -12
CID=$(docker ps -q --filter label=org.tenstorrent.tt-model=pi05-base-p150 | head -1); [ -z "$CID" ] && { echo "SERVE FAILED"; exit 1; }
echo "container image $(docker inspect -f '{{.Image}}' $CID)"; docker inspect $CID > $L/inspect.json
PORT=$(docker port "$CID" | grep -o '0.0.0.0:[0-9]*' | head -1 | cut -d: -f2); URL=http://127.0.0.1:${PORT:-20000}
curl -s $URL/info > $L/info.json; python3 -c "import json; d=json.load(open('$L/info.json')); print('info', d['megakernel']['backend'], d['megakernel']['program']['kernel_digest'], d['megakernel']['program']['device_ops_per_call'], 'source', d['source']['commit'])"
python3 $SP/ghmain/models/experimental/pi0_5/server/smoke_test.py --url $URL --timeout 300 | tail -1; RC=${PIPESTATUS[0]}
python3 $SP/bench_http_pi05.py --url $URL --media /home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc/media --n 30 --warmup 5 --out $L/bench_pulled.json | tail -1
$TTM stop $R 2>&1 | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" | tail -1
mkdir -p $SP/check_pulled; chmod 777 $SP/check_pulled
CMD=$(python3 $SP/mkrun_mc.py $L/inspect.json pi05-mcpulled "-v $SP/chk:/chk:ro -v $SP:/sp" "python /sp/img_check.py --out /sp/check_pulled --parts golden" PYTHONDONTWRITEBYTECODE=1)
echo "[$(date +%T)] golden in the pulled image"; eval "$CMD" > $L/golden.log 2>&1; G=$?
grep -E "GOLDEN|DONE|Traceback|Error" $L/golden.log | cut -c1-300
echo "[$(date +%T)] done smoke_rc=$RC golden_rc=$G"; [ $RC -eq 0 ] && [ $G -eq 0 ]
