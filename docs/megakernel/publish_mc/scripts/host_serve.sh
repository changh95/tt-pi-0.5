#!/usr/bin/env bash
# Host check of the tt-pi-0.5 branch's HTTP app on the tt-metal-main build (f856a38a): serve with the given env,
# smoke_test, HTTP refusals, a short bench. Runs INSIDE with-device.sh. Usage: host_serve.sh <tag> [ENV=VAL ...]
set -u -o pipefail
SP=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc
R=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
TAG=$1; shift
source /home/deepgadget/experiments/tt-metal-main/env-main.sh >/dev/null 2>&1
export PYTHONPATH=$R:$TT_METAL_HOME/ttnn:$SP/pydeps PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=$SP/ttcache_host
export TT_WEIGHTS_REVISION=b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba PI05_SOURCE_COMMIT=host-test
for kv in "$@"; do export "$kv"; done
PORT=20555; URL=http://127.0.0.1:$PORT
echo "[$(date +%T)] start $TAG env: $*"
python -m uvicorn --host 127.0.0.1 --port $PORT --lifespan on models.experimental.pi0_5.server.app:app > $SP/logs/host_$TAG.server.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; echo "[$(date +%T)] server stopped; device users [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
for i in $(seq 1 120); do curl -s $URL/health 2>/dev/null | grep -q '"ok"' && break; kill -0 $SPID 2>/dev/null || break; sleep 3; done
grep -E "Warm|warm|Model built|Opening|refused|Error|error" $SP/logs/host_$TAG.server.log | cut -c1-250 | tail -12
kill -0 $SPID 2>/dev/null || { echo "SERVER EXITED"; exit 1; }
curl -s $URL/info > $SP/logs/host_$TAG.info.json; python3 -c "import json;d=json.load(open('$SP/logs/host_$TAG.info.json'));print('info megakernel', json.dumps(d['megakernel']), 'outputs', d['outputs']['actions'])"
python3 $R/models/experimental/pi0_5/server/smoke_test.py --url $URL --timeout 60 | tail -1
python3 - "$URL" <<'PY' | tee $SP/logs/host_$TAG.refusals.txt
import base64, json, sys, urllib.request, urllib.error
url = sys.argv[1]; info = json.load(urllib.request.urlopen(url + "/info")); n = info["inputs"]["num_images"]
im = base64.b64encode(open("/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused/media/sample_base.png", "rb").read()).decode()
def post(p):
    try:
        r = urllib.request.urlopen(urllib.request.Request(url + "/predict", json.dumps(p).encode(), {"Content-Type": "application/json"}), timeout=300)
        return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
for name, p in [("one image too few", {"images": [im] * (n - 1) if n > 1 else [], "prompt": "x"}),
                ("one image too many", {"images": [im] * (n + 1), "prompt": "x"}),
                ("num_steps mismatch", {"images": [im] * n, "prompt": "x", "num_steps": 3}),
                ("prompt_bucket 48", {"images": [im] * n, "prompt": "x", "prompt_bucket": 48}),
                ("prompt_bucket 32 too small", {"images": [im] * n, "tokens": [2] + [100] * 40, "prompt_bucket": 32}),
                ("prompt_bucket 224 explicit (ok)", {"images": [im] * n, "tokens": [2, 100, 101], "prompt_bucket": 224}),
                ("auto bucket 32 (ok)", {"images": [im] * n, "tokens": [2, 100, 101]})]:
    s, b = post(p)
    print(f"{name}: HTTP {s}: " + (f"bucket={b.get('prompt_bucket')} H={b.get('action_horizon')} inf={b['timing_ms']['inference']}" if s == 200 else str(b)[:240]))
PY
[ -f $SP/media/sample_base.png ] || { mkdir -p $SP/media; cp /home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused/media/sample_*.png $SP/media/; }
python3 $SP/bench_http_pi05.py --url $URL --media $SP/media --n ${NBENCH:-30} --warmup 5 --out $SP/logs/host_$TAG.bench.json | tail -1
echo "[$(date +%T)] done $TAG"
