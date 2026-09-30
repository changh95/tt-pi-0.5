#!/bin/bash
# Served A/B re-run (hold4's serve loop was invalid: `kill $P` hit the subshell of the `arm` function, so the first
# (default) server kept serving port 20000 and every later arm measured it; the off servers blocked on CHIP_IN_USE).
# Here each server runs in its own session (setsid), is stopped by process group, the port and the process table are
# confirmed empty before the next arm, and /info's backend must equal the arm before anything is measured.
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip1
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
export PYTHONPATH=$PYTHONPATH:/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pylib
export S REPO
./sums.sh hold5_start > out/sums_hold5_start.txt
cd $REPO
for a in default off default off; do
  tag=${a}_$(date +%H%M%S)
  want=$([ $a = default ] && echo expert || echo off)
  echo "$(date +%F_%T) SERVE $a ($tag) want backend=$want"
  setsid bash -c "source $S/env.sh; export PYTHONPATH=\$PYTHONPATH:$S/../pylib; arm $a timeout --signal=KILL 1200 python -m uvicorn models.experimental.pi0_5.server.app:app --host 127.0.0.1 --port 20000 --lifespan on" > $S/out/S_$tag.server.log 2>&1 &
  P=$!
  ok=0
  for i in $(seq 1 240); do curl -s http://127.0.0.1:20000/health 2>/dev/null | grep -q '"ok"' && { ok=1; break; }; kill -0 $P 2>/dev/null || { echo "server died"; break; }; sleep 5; done
  curl -s http://127.0.0.1:20000/info > $S/out/S_$tag.info.json
  got=$(python -c "import json;print(json.load(open('$S/out/S_$tag.info.json'))['megakernel']['backend'])" 2>/dev/null)
  echo "$(date +%F_%T) health=$ok info backend=$got pid=$P"
  if [ "$ok" = 1 ] && [ "$got" = "$want" ]; then
    timeout 600 python models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 > $S/out/S_$tag.smoke.log 2>&1; rc=$?; echo "smoke rc=$rc $(tail -1 $S/out/S_$tag.smoke.log)"
    timeout 600 python models/experimental/pi0_5/tests/megakernel/served_latency.py --n 30 --out $S/out/S_$tag.lat.json 2>&1 | tail -1
    timeout 300 python $S/served_mask_probe.py http://127.0.0.1:20000 $S/out/S_$tag.mask.json 2>&1 | tail -1
  else
    echo "ARM MISMATCH or not healthy: skipping measurements for $tag"
  fi
  kill -TERM -- -$P 2>/dev/null; for i in $(seq 1 30); do kill -0 -- -$P 2>/dev/null || break; sleep 2; done; kill -KILL -- -$P 2>/dev/null
  for i in $(seq 1 30); do ps -eo args | grep -q "[u]vicorn models.experimental.pi0_5" || break; sleep 2; done
  ps -eo pid,args | grep "[u]vicorn models.experimental.pi0_5" && echo "LEFTOVER server process" || echo "$(date +%F_%T) server stopped, no uvicorn left"
  curl -s -m 2 http://127.0.0.1:20000/health >/dev/null 2>&1 && echo "PORT STILL ANSWERS" || true
done
cd $S; ./sums.sh hold5_end > out/sums_hold5_end.txt
echo "$(date +%F_%T) END"
