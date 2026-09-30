#!/bin/bash
# (1) the task's named test file (scratchpad, 09-29) as written, off arm (it opens the device WITHOUT the 64 KiB cut);
# (2) tests/pcc/test_pcc_pi05_fused.py both arms; (3) LAST: the named file with PI05_MEGAKERNEL=expert, i.e. the
# megakernel on a device opened without the cut = probe of DESIGN §4.12 refusal (d). May fail; must not hang.
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp1r1
T=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/verify_alternating.py
REPO=/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
cd $S && source env.sh
echo "$(date +%F_%T) named scratch alt (off) start"
PI05_MEGAKERNEL=off TT_METAL_CACHE=$(cache_for off) timeout 900 python $T --out out/ALT_named_scratch_off.json > out/ALT_named_scratch_off.log 2>&1
echo "$(date +%F_%T) named scratch alt (off) rc=$?"
for arm in expert off; do
  echo "$(date +%F_%T) pytest test_pcc_pi05_fused ($arm) start"
  cd $REPO && PI05_MEGAKERNEL=$arm TT_METAL_CACHE=$(cache_for $arm) timeout 1200 python -m pytest -p no:cacheprovider -q -s models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py > $S/out/pytest_pcc_$arm.log 2>&1
  echo "$(date +%F_%T) pytest test_pcc_pi05_fused ($arm) rc=$? $(tail -1 $S/out/pytest_pcc_$arm.log)"
  cd $S
done
echo "$(date +%F_%T) named scratch alt (expert, NO cut) start"
PI05_MEGAKERNEL=expert TT_METAL_CACHE=$(cache_for expert) timeout 600 python $T --out out/ALT_named_scratch_expert_nocut.json > out/ALT_named_scratch_expert_nocut.log 2>&1
echo "$(date +%F_%T) named scratch alt (expert, NO cut) rc=$?"
