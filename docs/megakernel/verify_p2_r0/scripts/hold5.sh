#!/bin/bash
# hold 5: LIBERO edge cases (whole, off); repo suites with whole selected: pytest test_pcc_pi05_fused.py.
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp2r0
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
./sums.sh hold5_start > out/sums_hold5_start.txt
for arm in whole off; do
  echo "$(date +%F_%T) START edge $arm"
  PI05_MEGAKERNEL=$arm TT_METAL_CACHE=$(cache_for $arm) timeout 600 python d_edge.py out/EDGE_$arm.pt > out/EDGE_$arm.log 2>&1
  rc=$?; echo "$(date +%F_%T) END edge $arm rc=$rc $(grep RESULT out/EDGE_$arm.log)"
done
echo "$(date +%F_%T) START pytest test_pcc_pi05_fused (whole)"
cd $REPO && PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 1200 python -m pytest -p no:cacheprovider -q -s models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py > $S/out/pytest_pcc_whole.log 2>&1
rc=$?; echo "$(date +%F_%T) END pytest rc=$rc $(grep -E '[0-9]+ (passed|failed)' $S/out/pytest_pcc_whole.log | tail -1)"
cd $S; ./sums.sh hold5_end > out/sums_hold5_end.txt
