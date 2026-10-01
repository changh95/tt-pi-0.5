"""CPU fp32 reference for edge.EDGE + per-input comparison of whole / off (base edges n 1/224/128/150 are seeds 701 /
6,716,715 / 712 / 4 of the main seed gate)."""
import json, sys, torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from edge import edge_obs  # noqa: E402
from vc2 import OUT, pcc  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config
torch.set_grad_enabled(False)
ref = PI0Model(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS))
A = {a: torch.load(f"{OUT}/EDGE_{a}.pt", weights_only=False) for a in ("whole", "off")}
rows = []
for tag, (im, tk, nz, mk) in edge_obs():
    ref.denoising.sample_noise = lambda *x, _n=nz, **k: _n.clone()
    r = ref.sample_actions(im, [torch.ones(1, dtype=torch.bool)] * 2, tk, mk, torch.zeros(1, 32)).float()
    row = {"tag": tag, **{a: pcc(A[a]["outs"][tag], r) for a in A}}
    row["margin"] = row["whole"] - row["off"]
    rows.append(row)
    print(row, flush=True)
json.dump({"backends": {a: A[a]["backend"] for a in A}, "rows": rows, "pass": all(r["margin"] >= 0 for r in rows)},
          open(f"{OUT}/EDGE_libero.json", "w"), indent=1)
