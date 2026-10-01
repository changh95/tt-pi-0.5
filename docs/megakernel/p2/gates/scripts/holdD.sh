#!/bin/bash
set -o pipefail
cd /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates && source env.sh
cd /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5
for arm in whole off; do
  for spec in "base 11:1,12:224,13:128,14:150 ref_base_edge.pt" "libero 11:32,12:1 ref_libero_edge.pt"; do
    set -- $spec
    echo "$(date +%F_%T) START edge $arm $1"
    PI05_MEGAKERNEL=$arm TT_METAL_CACHE=$(cache_for $arm) timeout 600 python /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/models/experimental/pi0_5/tests/megakernel/pe_edge_gate.py --shape $1 --spec $2 --refs /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/$3 --arm-out /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/E_${arm}_$1.pt > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/E_${arm}_$1.log 2>&1
    echo "$(date +%F_%T) END edge $arm $1 rc=$?"
  done
done
echo "$(date +%F_%T) START layer_vs_ttnn"
TT_METAL_CACHE=$(cache_for whole) TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1 timeout 600 python /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/models/experimental/pi0_5/tests/megakernel/pe_layer_vs_ttnn.py --out /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/L_layer_vs_ttnn.json > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/L_layer_vs_ttnn.log 2>&1
echo "$(date +%F_%T) END layer_vs_ttnn rc=$?"
for sh in base libero; do
  echo "$(date +%F_%T) START prefix $sh"
  TT_METAL_CACHE=$(cache_for whole) timeout 600 python /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/models/experimental/pi0_5/tests/megakernel/pe_prefix_run.py --shape $sh --reps 5 --out /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/P23_$sh.json > /tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/p2/gates/out/P23_$sh.log 2>&1
  echo "$(date +%F_%T) END prefix $sh rc=$?"
done
