"""Staging README.md = HF front matter (kept from HF head 990e22b5) + render_model_card(final yaml, built manifest) with
the generated pull paragraph replaced by the live card's two bullets + the LIBERO demo section."""
import json, os, sys
sys.path.insert(0, "/home/deepgadget/experiments/tt-models/tools/tt-model-manager/src")
from tt_kernel.container_manifest import load_container_manifest
from tt_kernel.build import render_model_card
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc"
B = "/home/deepgadget/experiments/tt-models/build/pi05-base-p150"
SP = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc"
os.chdir(S)
m = load_container_manifest("tt-model.yaml", check_sources=True)
built = json.load(open(f"{B}/tt_kernel_manifest.json"))["container"]["built"]
card = render_model_card(m, built)
old = open(f"{SP}/hf_head/README.md").read()
fm_old = old.split("---\n", 2)[1]
body = card.split("---\n", 2)[2]
if "`pull --with-weights` downloads" in body:
    i = body.index("`pull --with-weights` downloads"); j = body.index("\n\n", i)
    bullets = old[old.index("- Weights [`lerobot/pi05_base`]"):]
    bullets = bullets[: bullets.index("\n\n")]
    body = body[:i] + bullets + body[j:]
import re
runs = [l for l in body.split("\n") if l.startswith("Runs on **p150** or")]
assert len(runs) == 1
body = body.replace(runs[0], "Runs on **p150** (mesh `P150`, one p150a); six serve profiles pick the cameras / horizon / steps (below).")
i = body.index("| profile | hardware | mesh | max_num_seqs | max_model_len |"); j = body.index("\n\n", i)
rows = ["| profile | hardware | mesh | configuration |", "| --- | --- | --- | --- |"]
for pr in m.effective_profiles():
    star = " *(default)*" if pr.name == m.default_profile else ""
    rows.append(f"| `{pr.name}`{star} | p150 | P150 | {pr.description} |")
body = body[:i] + "\n".join(rows) + body[j:]
demo = open(f"{SP}/demo_section_mc.md").read()
out = "---\n" + fm_old + "---\n" + body.rstrip() + "\n\n" + demo.strip() + "\n"
open(f"{S}/README.md", "w").write(out)
print(len(out)); print(body[body.index("## Provenance"):] if "## Provenance" in body else "(no provenance section)")
