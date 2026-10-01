"""Staging README.md = HF front matter (kept from head 2cac5ec1) + render_model_card(final yaml, round-1 built) with the
generated pull paragraph replaced by the two bullets of the live card + the new LIBERO demo section."""
import json, re, sys
sys.path.insert(0, "/home/deepgadget/experiments/tt-models/tools/tt-model-manager/src")
from tt_kernel.container_manifest import load_container_manifest
from tt_kernel.build import render_model_card
import os
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"
B = "/home/deepgadget/experiments/tt-models/build/pi05-base-p150"
os.chdir(S)
m = load_container_manifest("tt-model.yaml", check_sources=True)
built = json.load(open(f"{B}/tt_kernel_manifest.json"))["container"]["built"]
card = render_model_card(m, built)
old = open("/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pub1/bak/README.md").read()
fm_old = old.split("---\n", 2)[1]
body = card.split("---\n", 2)[2]
# generated pull paragraph -> the live card's two bullets
i = body.index("`pull --with-weights` downloads"); j = body.index("\n\n", i)
bullets = old[old.index("- Weights [`lerobot/pi05_base`]"):]
bullets = bullets[: bullets.index("\n\n")]
body = body[:i] + bullets + body[j:]
demo = open(sys.argv[1]).read()
out = "---\n" + fm_old + "---\n" + body.rstrip() + "\n\n" + demo.strip() + "\n"
open(f"{S}/README.md", "w").write(out)
print(len(out)); print(body[body.index("## Provenance"):])
