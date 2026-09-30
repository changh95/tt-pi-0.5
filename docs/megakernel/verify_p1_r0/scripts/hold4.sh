#!/bin/bash
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/verify_p1
R=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/models/experimental/pi0_5/tt
cd $S && source env.sh
echo "$(date +%F_%T) named alt start"; PI05_MEGAKERNEL=expert timeout 900 python /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/models/experimental/pi0_5/tests/megakernel/verify_alternating.py --out out/ALT_named_expert.json > out/ALT_named_expert.log 2>&1; echo "$(date +%F_%T) named alt rc=$?"
(date; find $R -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.py' \) | sort | xargs md5sum) > out/soak_md5_before.txt
for i in $(seq 1 20); do
  echo "$(date +%F_%T) run $i start"
  PI05_MEGAKERNEL=expert timeout 600 python v_soak_one.py > out/soak_$i.log 2>&1
  rc=$?
  echo "$(date +%F_%T) run $i rc=$rc $(grep RESULT out/soak_$i.log)"
done
echo "$(date +%F_%T) named alt start"; PI05_MEGAKERNEL=expert timeout 900 python /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/models/experimental/pi0_5/tests/megakernel/verify_alternating.py --out out/ALT_named_expert.json > out/ALT_named_expert.log 2>&1; echo "$(date +%F_%T) named alt rc=$?"
(date; find $R -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.py' \) | sort | xargs md5sum) > out/soak_md5_after.txt
