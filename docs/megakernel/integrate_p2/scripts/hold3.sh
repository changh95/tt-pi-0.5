#!/bin/bash
# integrate-p2 hold 3: alternating / shape switch (own d_alt2 + the named repo verify_alternating.py), LIBERO edge
# cases (whole, off), pytest test_pcc_pi05_fused.py with PI05_MEGAKERNEL UNSET.
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/ip2
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
./sums.sh hold3_start > out/sums_hold3_start.txt
echo "$(date +%F_%T) START alt whole(default)"
arm whole whole timeout 1200 python d_alt2.py out/ALT_whole.json > out/ALT_whole.log 2>&1
rc=$?; echo "$(date +%F_%T) END alt rc=$rc"
echo "$(date +%F_%T) START named repo alt whole(default)"
arm whole whole timeout 900 python $REPO/models/experimental/pi0_5/tests/megakernel/verify_alternating.py --out out/ALT_named_whole.json > out/ALT_named_whole.log 2>&1
rc=$?; echo "$(date +%F_%T) END named repo alt rc=$rc $(grep -o 'OVERALL.*' out/ALT_named_whole.log | tail -1)"
for a in whole off; do
  echo "$(date +%F_%T) START edge $a"
  arm $a $a timeout 600 python d_edge.py out/EDGE_$a.pt > out/EDGE_$a.log 2>&1
  rc=$?; echo "$(date +%F_%T) END edge $a rc=$rc $(grep RESULT out/EDGE_$a.log)"
done
echo "$(date +%F_%T) START pytest test_pcc_pi05_fused (PI05_MEGAKERNEL unset)"
cd $REPO && arm whole whole timeout 1200 python -m pytest -p no:cacheprovider -q -s models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py > $S/out/pytest_pcc_default.log 2>&1
rc=$?; echo "$(date +%F_%T) END pytest rc=$rc $(grep -E '[0-9]+ (passed|failed)' $S/out/pytest_pcc_default.log | tail -1)"
cd $S; ./sums.sh hold3_end > out/sums_hold3_end.txt
