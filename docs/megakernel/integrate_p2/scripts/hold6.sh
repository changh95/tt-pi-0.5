#!/bin/bash
# integrate-p2 hold 6: served A/B, default (PI05_MEGAKERNEL unset -> whole) vs expert (phase 1), alternated; each
# server in its own session, stopped by pid, port + process table confirmed empty, /info backend must equal the arm.
# Then startup refusals of the DEFAULT for unsupported serving configs (no device open expected).
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip2
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
export PYTHONPATH=$PYTHONPATH:/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pylib
./sums.sh hold6_start > out/sums_hold6_start.txt
cd $REPO
up() { ps -eo pid,args | awk '/[u]vicorn models.experimental.pi0_5.server.app/ {print $1}'; }
for a in whole expert whole expert; do
  tag=${a}_$(date +%H%M%S)
  echo "$(date +%F_%T) SERVE $a ($tag) want backend=$a"
  setsid bash -c "source $S/env.sh; export PYTHONPATH=\$PYTHONPATH:$S/../pylib; arm $a $a timeout --signal=KILL 1200 python -m uvicorn models.experimental.pi0_5.server.app:app --host 127.0.0.1 --port 20000 --lifespan on" > $S/out/S_$tag.server.log 2>&1 &
  P=$!
  ok=0
  for i in $(seq 1 240); do curl -s http://127.0.0.1:20000/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }; kill -0 $P 2>/dev/null || { echo "server died"; break; }; sleep 5; done
  curl -s http://127.0.0.1:20000/info > $S/out/S_$tag.info.json
  got=$(python -c "import json;print(json.load(open('$S/out/S_$tag.info.json'))['megakernel']['backend'])" 2>/dev/null)
  echo "$(date +%F_%T) health=$ok info backend=$got pid=$P"
  if [ "$ok" = 1 ] && [ "$got" = "$a" ]; then
    timeout 600 python models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 > $S/out/S_$tag.smoke.log 2>&1; rc=$?; echo "smoke rc=$rc $(tail -1 $S/out/S_$tag.smoke.log)"
    timeout 600 python models/experimental/pi0_5/tests/megakernel/served_latency.py --n 30 --out $S/out/S_$tag.lat.json 2>&1 | tail -1
    timeout 300 python $S/served_mask_probe.py http://127.0.0.1:20000 $S/out/S_$tag.mask.json 2>&1 | tail -1
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
for combo in "PI05_NUM_IMAGES=1" "PI05_BATCH_SIZES=1,2" "PI05_KV_DTYPE=bf16"; do
  f=$S/out/refuse_default_${combo//[=,]/_}.log
  echo "$(date +%F_%T) REFUSE default(unset) $combo"
  env -u PI05_MEGAKERNEL $combo TT_METAL_CACHE=$(cache_for whole) timeout --signal=KILL 300 python -m uvicorn models.experimental.pi0_5.server.app:app --host 127.0.0.1 --port 20001 --lifespan on > $f 2>&1
  echo "rc=$?"; grep -h "refused at startup\|Opening\|Application startup complete" $f | cut -c1-250
done
cd $S; ./sums.sh hold6_end > out/sums_hold6_end.txt
echo "$(date +%F_%T) END"
