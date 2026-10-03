"""DRAFT (ASD-STE100 prose): render draft_ste/README.md = HF front matter of the published README (unchanged) +
render_model_card(staging manifest with the card of draft_ste/tt-model.yaml, built manifest), the generated tool
sentences rewritten in STE, + draft_ste/demo_section_ste.md. Writes nothing outside draft_ste/."""
import json, os, sys, yaml
sys.path.insert(0, "/home/deepgadget/experiments/tt-models/tools/tt-model-manager/src")
from tt_kernel.container_manifest import load_container_manifest, CardSettings
from tt_kernel.build import render_model_card
HERE = os.path.dirname(os.path.abspath(__file__))
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc"
B = "/home/deepgadget/experiments/tt-models/build/pi05-base-p150"
os.chdir(S)
m = load_container_manifest("tt-model.yaml", check_sources=True)
DY = yaml.safe_load(open(f"{HERE}/tt-model.yaml"))
m.card = CardSettings(**DY["card"])
from tt_kernel.manifest import ServeProfile
m.serve_profiles = [ServeProfile(**pp) for pp in DY["serve_profiles"]]
m.default_profile = DY["default_profile"]
built = json.load(open(f"{B}/tt_kernel_manifest.json"))["container"]["built"]
card = render_model_card(m, built)
pub = open(f"{S}/README.md").read()
fm = pub.split("---\n", 2)[1]
body = card.split("---\n", 2)[2]

def rep(a, b):
    global body
    assert body.count(a) == 1, a[:80]
    body = body.replace(a, b)

runs = [l for l in body.split("\n") if l.startswith("Runs on **p150**")]; assert len(runs) == 1
rep(runs[0], "- This package runs on **p150** (mesh `P150`, one p150a).\n- The package has one serve profile, `p150`. Three environment variables change its configuration (see \"How to change the configuration\").")
pk = [l for l in body.split("\n") if l.startswith("Packaged and published with")]; assert len(pk) == 1
rep(pk[0], pk[0].replace("Packaged and published with ", "- The package uses "))
i = body.index("`pull --with-weights` downloads"); j = body.index("\n\n", i)
body = body[:i] + ("- `tt-model pull` downloads the weights [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) at `b211f3d44c36` into your HF cache. The image does not contain the weights.\n"
                   "- The server uses port 20000. If that port is busy, the server uses the next free port.\n"
                   "- The server is ready when the log shows `Application startup complete`.") + body[j:]
if "One image serves every profile below" in body:
    rep("One image serves every profile below; pick one with `--profile`.", "- One image supports each serve profile below.\n- To select a serve profile, use `--profile`.")
prov = [l for l in body.split("\n") if l.startswith("The exact sources the image was built from")]; assert len(prov) == 1
rep(prov[0], "- The next table shows the sources of the image.\n- `code/` in this repository is byte-identical to the model code in the image.")
assert "| profile | hardware | mesh |" not in body  # one profile: the tool renders no profile table
demo = open(f"{HERE}/demo_section_ste.md").read()
out = "---\n" + fm + "---\n" + body.rstrip() + "\n\n" + demo.strip() + "\n"
open(f"{HERE}/README.md", "w").write(out)
assert out.split("---\n", 2)[1] == fm
print(len(out), "->", f"{HERE}/README.md")
