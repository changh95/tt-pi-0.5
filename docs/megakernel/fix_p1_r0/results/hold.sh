#!/bin/bash
set -o pipefail
source /home/deepgadget/experiments/gr00t/bin/env.sh
export PYTHONPATH=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pylib:/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5:$PYTHONPATH PYTHONDONTWRITEBYTECODE=1 TT_METAL_CACHE=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/cache TT_METAL_LOGGER_LEVEL=WARNING
cd /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
T="models/experimental/pi0_5/tests/megakernel"
echo "[$(date '+%F %T')] HEAD $(git rev-parse --short HEAD) kernels md5 $(cat models/experimental/pi0_5/tt/megakernel/kernels/* | md5sum | cut -c1-16)"
echo "[$(date '+%F %T')] soak_one expert start"; (cd /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/verify_p1_r0/scripts && PI05_MEGAKERNEL=expert timeout 600 python v_soak_one.py) > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/soak_one.log 2>&1; echo "[$(date '+%F %T')] soak_one rc=$? $(grep RESULT /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/soak_one.log)"
echo "[$(date '+%F %T')] E2 libero n_lang 32,1 start"; timeout 900 python $T/mk_vs_shipped.py --shape libero --n-real 32,1 --out /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/E2.json > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/E2.log 2>&1; echo "[$(date '+%F %T')] E2 rc=$?"
echo "[$(date '+%F %T')] L1 guard start"; PI05_MEGAKERNEL=expert timeout 600 python $T/mk_l1_guard.py /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/G.json > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/G.log 2>&1; echo "[$(date '+%F %T')] L1 guard rc=$?"
for mk in expert off expert off; do
  echo "[$(date '+%F %T')] SERVE $mk"
  PI05_MEGAKERNEL=$mk timeout --signal=KILL 1200 python -m uvicorn models.experimental.pi0_5.server.app:app --host 127.0.0.1 --port 20000 --lifespan on > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/S_$mk.server.log 2>&1 &
  P=$!
  for i in $(seq 1 240); do curl -s http://127.0.0.1:20000/health 2>/dev/null | grep -q '"ok"' && break; kill -0 $P 2>/dev/null || { echo "server died"; break; }; sleep 5; done
  timeout 600 python models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/S_$mk.smoke.log 2>&1; echo "smoke rc=$? $(tail -1 /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/S_$mk.smoke.log)"
  timeout 600 python $T/served_latency.py --n 30 --out /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/fixp1/S_${mk}_$(date +%H%M%S).lat.json 2>&1 | tail -1
  curl -s http://127.0.0.1:20000/info | python -c "import json,sys; d=json.load(sys.stdin); print('info megakernel', d.get('megakernel'))"
  kill $P; wait $P 2>/dev/null
done
echo "[$(date '+%F %T')] END"
