"""verify-p1-r1 alternating prompts + shape switch, one arm per process (PI05_MEGAKERNEL picks it).

References: (1) a FRESH model per prompt in this process (sees only that prompt), (2) the outputs of the separate
d_arm.py process of the same arm for the same seeds / records (another process, another call history).
Sequence: base  A B C D D2 A D2 D C B  (D / D2 = same length 224, different content: the attention inputs are NOT
rewritten between them) -> release -> LIBERO E F E F F E -> release -> base again B D2 A C.
Every call must be BIT-identical to its fresh reference. Positive control: the references differ pairwise."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc import BASE_WEIGHTS, OUT, SPEC, base_config, now, obs_for, pcc  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.device_open import device_kwargs  # noqa: E402
from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, pcc7, record_inputs  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

out_path = sys.argv[1]
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
N = dict(SPEC)
BSEED = {"A": 703, "B": 712, "C": 701, "D": 6, "D2": 716}
base_obs = {k: obs_for(s, N[s]) for k, s in BSEED.items()}
recs = load_records()
by_n = {}
for r in recs:
    im, tk, mk, nz = record_inputs(r, 32)
    by_n.setdefault(int(mk.sum()), (r, (im, tk, nz, mk)))
ns = sorted(by_n)
lib = {"E": by_n[ns[0]], "F": by_n[ns[-1]]}
lib_obs = {k: v[1] for k, v in lib.items()}
BL, LL = PI0WeightLoader(BASE_WEIGHTS), PI0WeightLoader(LIBERO_WEIGHTS)


def model(kind):
    torch.manual_seed(42)
    return PI0ModelTTNN(base_config() if kind == "base" else libero_config(), BL if kind == "base" else LL, dev, fused=env)


def call(m, o):
    return m.sample_actions_fused(o[0], o[1], o[2], lang_masks=o[3]).clone()


dev = ttnn.open_device(**device_kwargs(env))
dev.enable_program_cache()
res = {"t0": now(), "arm": env.megakernel, "base_seeds": BSEED, "libero_n_lang": [ns[0], ns[-1]],
       "libero_tags": {k: v[0]["tag"] for k, v in lib.items()}, "checks": []}
try:
    refs = {}
    for k, o in list(base_obs.items()) + list(lib_obs.items()):
        m = model("base" if k in base_obs else "libero")
        refs[k] = call(m, o)
        res.setdefault("stamps", {})[k] = m.megakernel_backend
        m.release_trace()
        del m
        print(now(), "fresh ref", k, flush=True)
    ks = list(refs)
    res["ref_pairs_max_pcc"] = max(pcc(refs[a], refs[b]) for i, a in enumerate(ks) for b in ks[i + 1:]
                                   if refs[a].shape == refs[b].shape)
    res["ref_pairs_any_equal"] = any(torch.equal(refs[a], refs[b]) for i, a in enumerate(ks) for b in ks[i + 1:]
                                     if refs[a].shape == refs[b].shape)
    res["pcc7_fresh"] = {k: pcc7(refs[k], lib[k][0]["actions_model_norm"]) for k in lib}

    def run(m, seq, phase):
        for k in seq:
            y = call(m, base_obs.get(k) if k in base_obs else lib_obs[k])
            ok = bool(torch.equal(y, refs[k]))
            res["checks"].append({"phase": phase, "prompt": k, "bit_identical": ok, "pcc": pcc(y, refs[k]),
                                  "maxabs": float((y - refs[k]).abs().max())})
            print(now(), phase, k, ok, flush=True)

    m = model("base")
    run(m, ["A", "B", "C", "D", "D2", "A", "D2", "D", "C", "B"], "alt_base")
    m.release_trace()
    del m
    m = model("libero")
    run(m, list("EFEFFE"), "libero_after_switch")
    m.release_trace()
    del m
    m = model("base")
    run(m, ["B", "D2", "A", "C"], "base_switch_back")
    m.release_trace()
    del m
finally:
    ttnn.close_device(dev)
res["all_bit_identical"] = all(c["bit_identical"] for c in res["checks"])
res["n_checks"] = len(res["checks"])
# cross-process reference: the d_arm run of the same arm
xp = {}
for shape in ("base", "libero"):
    p = f"{OUT}/A_{os.environ['IP1_ARM']}_{shape}.pt"
    if os.path.exists(p):
        D = torch.load(p, weights_only=False)
        for k in (BSEED if shape == "base" else lib):
            tag = BSEED[k] if shape == "base" else lib[k][0]["tag"]
            if tag in D["tags"]:
                xp[k] = bool(torch.equal(refs[k], D["outs"][D["tags"].index(tag)]))
res["fresh_ref_eq_other_process"] = xp
res["t1"] = now()
json.dump(res, open(out_path, "w"), indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "checks"}), flush=True)
