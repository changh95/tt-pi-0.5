#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip1
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
export PYTHONPATH=$PYTHONPATH:/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pylib
./sums.sh hold4_start > out/sums_hold4_start.txt
echo "$(date +%F_%T) nocut probe (final code) start"
arm default timeout 600 python d_nocut.py out/NOCUT_final.json > out/NOCUT_final.log 2>&1; rc=$?
echo "$(date +%F_%T) nocut probe rc=$rc"
for a in default off; do
  echo "$(date +%F_%T) pytest test_pcc_pi05_fused ($a) start"
  (cd $REPO && arm $a timeout 1200 python -m pytest -p no:cacheprovider -q -s models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py) > out/pytest_pcc_$a.log 2>&1; rc=$?
  echo "$(date +%F_%T) pytest ($a) rc=$rc $(grep -E '[0-9]+ (passed|failed)' out/pytest_pcc_$a.log | tail -1)"
done
cd $REPO
for a in default off default off; do
  tag=${a}_$(date +%H%M%S)
  echo "$(date +%F_%T) SERVE $a ($tag)"
  arm $a timeout --signal=KILL 1200 python -m uvicorn models.experimental.pi0_5.server.app:app --host 127.0.0.1 --port 20000 --lifespan on > $S/out/S_$tag.server.log 2>&1 &
  P=$!
  for i in $(seq 1 240); do curl -s http://127.0.0.1:20000/health 2>/dev/null | grep -q '"ok"' && break; kill -0 $P 2>/dev/null || { echo "server died"; break; }; sleep 5; done
  timeout 600 python models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000 > $S/out/S_$tag.smoke.log 2>&1; rc=$?; echo "smoke rc=$rc $(tail -1 $S/out/S_$tag.smoke.log)"
  timeout 600 python models/experimental/pi0_5/tests/megakernel/served_latency.py --n 30 --out $S/out/S_$tag.lat.json 2>&1 | tail -1
  timeout 300 python $S/served_mask_probe.py http://127.0.0.1:20000 $S/out/S_$tag.mask.json 2>&1 | tail -1
  curl -s http://127.0.0.1:20000/info > $S/out/S_$tag.info.json
  kill $P; wait $P 2>/dev/null
done
cd $S; ./sums.sh hold4_end > out/sums_hold4_end.txt
echo "$(date +%F_%T) END"
