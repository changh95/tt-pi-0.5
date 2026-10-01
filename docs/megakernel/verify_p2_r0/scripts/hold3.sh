#!/bin/bash
# hold 3: fresh 20-run soak of PI05_MEGAKERNEL=whole (one process each: open, build, warm-up + capture, 30 more calls
# cycling n 1 / 128 / 224); source md5s + compiled-ELF md5s before and after.
set -o pipefail
S=/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/vp2r0
cd $S && source env.sh
./sums.sh soak_before > out/sums_soak_before.txt
./kelf.sh $(cache_for whole) soak_before > out/kelf_soak_before.txt
for i in $(seq 1 20); do
  echo "$(date +%F_%T) soak $i start"
  PI05_MEGAKERNEL=whole TT_METAL_CACHE=$(cache_for whole) timeout 600 python d_soak2.py > out/soak_$i.log 2>&1
  rc=$?
  echo "$(date +%F_%T) soak $i rc=$rc $(grep RESULT out/soak_$i.log)"
done
./sums.sh soak_after > out/sums_soak_after.txt
./kelf.sh $(cache_for whole) soak_after > out/kelf_soak_after.txt
