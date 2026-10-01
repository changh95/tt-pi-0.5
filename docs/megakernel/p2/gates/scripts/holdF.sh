#!/bin/bash
# served A/B (P2-5): whole vs expert alternated; each server in its own session, stopped by pid, port and process
# table confirmed empty before the next arm, /info's backend must equal the arm before anything is measured.
set -o pipefail
cd /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates && source env.sh
export PYTHONPATH=$PYTHONPATH:/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pylib
cd /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
up() { ps -eo pid,args | awk '/[u]vicorn models.experimental.pi0_5.server.app/ {print $1}'; }
for a in whole expert whole expert; do
  tag=${a}_$(date +%H%M%S)
  echo "$(date +%F_%T) SERVE $a ($tag)"
  setsid bash -c "source /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/env.sh; export PYTHONPATH=\$PYTHONPATH:/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pylib; PI05_MEGAKERNEL=$a TT_METAL_CACHE=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/ttcache_$a timeout --signal=KILL 1200 python -m uvicorn models.experimental.pi0_5.server.app:app --host 127.0.0.1 --port 20000 --lifespan on" > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/S_$tag.server.log 2>&1 &
  P=$!
  ok=0
  for i in $(seq 1 240); do curl -s http://127.0.0.1:20000/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }; kill -0 $P 2>/dev/null || { echo "server died"; break; }; sleep 5; done
  curl -s http://127.0.0.1:20000/info > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/S_$tag.info.json
  got=$(python -c "import json;print(json.load(open('/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/S_$tag.info.json'))['megakernel']['backend'])" 2>/dev/null)
  echo "$(date +%F_%T) health=$ok info backend=$got pid=$P"
  if [ "$ok" = 1 ] && [ "$got" = "$a" ]; then
    timeout 600 python models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/S_$tag.smoke.log 2>&1; rc=$?; echo "smoke rc=$rc $(tail -1 /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/S_$tag.smoke.log)"
    timeout 600 python models/experimental/pi0_5/tests/megakernel/served_latency.py --n 30 --out /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/S_$tag.lat.json 2>&1 | tail -1
    timeout 300 python /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/served_mask_probe.py http://127.0.0.1:20000 /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/S_$tag.mask.json 2>&1 | tail -1
  else
    echo "ARM MISMATCH or not healthy: skipping measurements for $tag"
  fi
  U=$(up); [ -n "$U" ] && kill -TERM $U 2>/dev/null
  for i in $(seq 1 30); do [ -z "$(up)" ] && break; sleep 2; done
  U=$(up); [ -n "$U" ] && { echo "TERM did not stop $U; KILL"; kill -KILL $U 2>/dev/null; sleep 3; }
  kill -KILL $P 2>/dev/null
  for i in $(seq 1 30); do ps -eo args | grep -q "[u]vicorn models.experimental.pi0_5" || break; sleep 2; done
  ps -eo pid,args | grep "[u]vicorn models.experimental.pi0_5" && echo "LEFTOVER server process" || echo "$(date +%F_%T) server stopped, no uvicorn left"
  curl -s -m 2 http://127.0.0.1:20000/health >/dev/null 2>&1 && echo "PORT STILL ANSWERS" || true
done
for combo in "PI05_KV_DTYPE=bf16" "PI05_NUM_STEPS=20" "PI05_BATCH_SIZES=1,2"; do
  echo "$(date +%F_%T) REFUSE whole $combo"
  env PI05_MEGAKERNEL=whole $combo TT_METAL_CACHE=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/ttcache_whole timeout --signal=KILL 300 python -m uvicorn models.experimental.pi0_5.server.app:app --host 127.0.0.1 --port 20001 --lifespan on > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/refuse_whole_${combo//[=,]/_}.log 2>&1
  echo "rc=$?"; grep -h "refused at startup\|Opening\|Application startup complete" /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/refuse_whole_${combo//[=,]/_}.log | cut -c1-250
done
echo "$(date +%F_%T) END"
