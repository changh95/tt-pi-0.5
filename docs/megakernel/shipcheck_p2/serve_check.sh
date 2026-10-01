#!/usr/bin/env bash
set -u -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/shipcheck2
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; R=changh95/pi05-base-p150
trap 'echo "[$(date +%T)] stop"; $TTM stop $R 2>&1 | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" | tail -2; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"' EXIT
echo "[$(date +%T)] serve"
$TTM serve $R 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|image|error|rror" | head -15
echo "[$(date +%T)] info"
curl -s localhost:20000/info > $S/info.json
python3 -c "import json; d=json.load(open('$S/info.json')); print('INFO', d['source']['commit'], 'backend', d['megakernel']['backend'], d['megakernel']['program']['kernel_digest'])"
docker exec $(docker ps -q --filter name=tt-model-pi05) sh -c 'ls /opt/tt-metal/models/__init__.py /opt/tt-metal/models/experimental/__init__.py; env | grep PI05_' 2>&1 | head
echo "[$(date +%T)] smoke"
python3 $S/hf/code/models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 --timeout 300 2>&1 | tail -2
echo "[$(date +%T)] bench"
python3 - <<PY
import base64,json,time,urllib.request,statistics as st
S='$S'
im=[base64.b64encode(open(S+'/hf/media/'+f,'rb').read()).decode() for f in ('sample_base.png','sample_wrist.png')]
req=json.dumps({"images":im,"prompt":"pick up the cube","state":[0.1,-0.2,0.3,0,0,0,0.5,-0.5]}).encode()
inf=[];acts=[]
for i in range(35):
    r=urllib.request.Request('http://127.0.0.1:20000/predict',data=req,headers={'Content-Type':'application/json'})
    d=json.loads(urllib.request.urlopen(r,timeout=300).read())
    if i>=5: inf.append(d['timing_ms']['inference']); acts.append(json.dumps(d['actions']))
print('BENCH n',len(inf),'median',st.median(inf),'min',min(inf),'max',max(inf),'identical',len(set(acts))==1,'head',d['actions'][0][:4])
PY
