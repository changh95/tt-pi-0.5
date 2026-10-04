"""Replace the card: block (the last top-level key) of the staging tt-model.yaml with the final README's text:
description = title .. before '## Quickstart' (intro, profiles, demo); quickstart = '### Run with tt-cli' .. before
'## Provenance'. The README itself is published as written; the block only carries the same text."""
import re, sys
README = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc/card_n16/README.md"
Y = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc/tt-model.yaml"
r = open(README).read(); y = open(Y).read()
assert "PENDING[" not in r
desc = r.split("\n# pi05-base-p150\n\n", 1)[1].split("\n## Quickstart\n", 1)[0].strip("\n")
qs = r[r.index("### Run with tt-cli"):r.index("\n## Provenance")].strip("\n")
assert "\t" not in desc + qs
blk = lambda t: "\n".join(("    " + l) if l else "" for l in t.split("\n"))
i = y.index("\ncard:\n"); assert not re.search(r"\n[a-z_]+:", y[i + 7:]), "card: is not the last top-level key"
y2 = y[:i] + "\ncard:\n  description: |\n" + blk(desc) + "\n  quickstart: |\n" + blk(qs) + "\n"
open(Y, "w").write(y2)
import yaml; c = yaml.safe_load(open(Y))["card"]
assert c["description"].rstrip("\n") == desc and c["quickstart"].rstrip("\n") == qs
sys.path.insert(0, "/home/deepgadget/experiments/tt-models/tools/tt-model-manager/src")
from tt_kernel.container_manifest import ContainerManifest
ContainerManifest.model_validate(yaml.safe_load(open(Y)))
print("card block synced:", len(desc), "+", len(qs), "chars; yaml validates")
