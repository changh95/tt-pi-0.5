#!/usr/bin/env bash
# Serve the Hub-pulled package by repo id, smoke it, stop it (one lock hold).
set -u -o pipefail
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; R=changh95/pi05-base-p150
trap '$TTM stop $R 2>&1 | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" | tail -2; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
$TTM serve $R 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|image|error" | head -12
curl -s localhost:20000/info | python3 -c "import json,sys; d=json.load(sys.stdin); print('info', d['name'], d['source']['commit'], d['weights']['revision'], 'megakernel', d['megakernel']['backend'], d['megakernel']['program']['kernel_digest'])"
python3 /home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused/code/models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 --timeout 300 | tail -1
