#!/usr/bin/env bash
# final-critic: profiled run of the PULLED shipped image (tt-model/pi05-base-p150:6fb244df57ff, HF 990e22b5), unmodified
# code, device profiler on: count device programs per trace replay on the real served request path.
set -u -o pipefail
FC=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fc
source /home/deepgadget/experiments/tt-models/bin/docker-env.sh
TTM=/home/deepgadget/experiments/tt-models/.venv/bin/tt-model; R=changh95/pi05-base-p150
cleanup() { echo "[$(date +%T)] trap"; docker rm -f pi05-mkprof >/dev/null 2>&1; $TTM stop $R 2>&1 | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" | tail -1; echo "[trap] device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"; }
trap cleanup EXIT
echo "[$(date +%T)] serve (to read the container spec tt-model uses)"
$TTM serve $R 2>&1 | sed 's/\x1b\[[0-9;]*[A-Za-z]//g' | grep -E "✓|✗|image|rror" | head -12
CID=$(docker ps -q --filter label=org.tenstorrent.tt-model=pi05-base-p150 | head -1)
docker inspect $CID > $FC/inspect.json; echo "image $(docker inspect -f '{{.Image}}' $CID)"
curl -s localhost:20000/info > $FC/info_unprofiled.json
$TTM stop $R 2>&1 | sed "s/\x1b\[[0-9;]*[A-Za-z]//g" | tail -1
sleep 3; echo "device users: [$(fuser /dev/tenstorrent/0 2>/dev/null)]"
mkdir -p $FC/prof $FC/ttcache; chmod 777 $FC $FC/prof $FC/ttcache
CMD=$(python3 $FC/mkrun.py $FC/inspect.json $FC); echo "RUN $CMD" | tee $FC/run_cmd.txt
eval "$CMD"
PORT=20000; grep -q -- "-p 20001" $FC/run_cmd.txt && PORT=20001
echo "[$(date +%T)] waiting for ready on $PORT"
for i in $(seq 1 180); do
  r=$(curl -s localhost:$PORT/health 2>/dev/null); echo "$r" | grep -q '"status": *"ok"' && break
  docker ps -q --filter name=pi05-mkprof | grep -q . || { echo "container exited"; break; }
  sleep 5
done
echo "[$(date +%T)] health: $r"
curl -s localhost:$PORT/info > $FC/info_profiled.json
python3 - <<PY
import base64,json,urllib.request,statistics as st
med='/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused/media'
im=[base64.b64encode(open(med+'/'+f,'rb').read()).decode() for f in ('sample_base.png','sample_wrist.png')]
prompts=['pick up the cube','put the bowl on the plate and then open the top drawer of the cabinet slowly','x']
res=[]
for i in range(12):
    req=json.dumps({"images":im,"prompt":prompts[i%3],"state":[0.1,-0.2,0.3,0,0,0,0.5,-0.5]}).encode()
    r=urllib.request.Request('http://127.0.0.1:$PORT/predict',data=req,headers={'Content-Type':'application/json'})
    d=json.loads(urllib.request.urlopen(r,timeout=300).read())
    res.append({"i":i,"prompt":i%3,"inference_ms":d['timing_ms']['inference'],"head":d['actions'][0][:4]})
json.dump(res,open('$FC/requests.json','w'),indent=1)
print('REQS', len(res), [round(x['inference_ms'],1) for x in res])
PY
echo "[$(date +%T)] graceful stop (SIGTERM -> lifespan close_device -> profiler dump)"
docker stop -t 300 pi05-mkprof; docker logs -t pi05-mkprof > $FC/container.log 2>&1
grep -E "Closing device|Warmup|megakernel|rror" $FC/container.log | tail -12
ls -la $FC/prof/.logs 2>&1 | head
echo "[$(date +%T)] done"
