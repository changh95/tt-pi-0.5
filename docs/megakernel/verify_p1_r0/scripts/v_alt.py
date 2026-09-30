"""Alternating prompts + shape switch vs FRESH-model references (verify-p1-r0), one arm per process.

Fresh refs: for each of 4 prompts (base n_real 12 / 150 / 1 / 224, LIBERO the 2 records with the most distinct n_lang)
a new model that sees only that prompt (first call = compile + capture). Then ONE base model serves
A B C D A B C D (base) -> released -> ONE LIBERO model serves E F E F -> released -> a second base model serves D A,
i.e. a shape switch back. Every call must be BIT-identical to its fresh ref. Positive control: the refs differ."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import BASE_WEIGHTS, base_config, now, obs_for, pcc  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, pcc7, record_inputs  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

out_path = sys.argv[1]
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
kw = dict(device_id=0, l1_small_size=24576, trace_region_size=env.trace_region_size)
if env.megakernel != "off":
    kw["worker_l1_size"] = 1_395_712
BL, LL = PI0WeightLoader(BASE_WEIGHTS), PI0WeightLoader(LIBERO_WEIGHTS)
base_obs = {"A": obs_for(3, 12), "B": obs_for(5, 150), "C": obs_for(7, 1), "D": obs_for(8, 224)}
recs = load_records()
by_n = {}
for r in recs:
    im, tk, mk, nz = record_inputs(r, 32)
    by_n.setdefault(int(mk.sum()), (r, (im, tk, nz, mk)))
ns = sorted(by_n)
lib_obs = {"E": by_n[ns[0]][1], "F": by_n[ns[-1]][1]}
lib_rec = {"E": by_n[ns[0]][0], "F": by_n[ns[-1]][0]}


def model(kind):
    torch.manual_seed(42)
    return PI0ModelTTNN(base_config() if kind == "base" else libero_config(), BL if kind == "base" else LL, dev, fused=env)


def call(m, o):
    return m.sample_actions_fused(o[0], o[1], o[2], lang_masks=o[3]).clone()


dev = ttnn.open_device(**kw)
dev.enable_program_cache()
res = {"t0": now(), "arm": env.megakernel, "n_lang_libero": [ns[0], ns[-1]], "checks": []}
try:
    refs = {}
    for k, o in list(base_obs.items()) + list(lib_obs.items()):
        m = model("base" if k in base_obs else "libero")
        refs[k] = call(m, o)
        res.setdefault("stamps", []).append(m.megakernel_backend)
        m.release_trace()
        del m
        print(now(), "fresh ref", k, flush=True)
    keys = list(refs)
    res["positive_control_max_pcc_between_refs"] = max(
        pcc(refs[a], refs[b]) for i, a in enumerate(keys) for b in keys[i + 1:] if refs[a].shape == refs[b].shape)
    res["pcc7_fresh"] = {k: pcc7(refs[k], lib_rec[k]["actions_model_norm"]) for k in lib_obs}

    def run(m, seq, phase):
        for k in seq:
            y = call(m, base_obs.get(k) or lib_obs.get(k))
            ok = bool(torch.equal(y, refs[k]))
            res["checks"].append({"phase": phase, "prompt": k, "bit_identical": ok,
                                  "pcc": pcc(y, refs[k]), "maxabs": float((y - refs[k]).abs().max())})
            print(now(), phase, k, ok, flush=True)

    m = model("base")
    run(m, "ABCDABCD", "alt_base")
    m.release_trace()
    del m
    m = model("libero")
    run(m, "EFEFFE", "alt_libero_after_switch")
    m.release_trace()
    del m
    m = model("base")
    run(m, "DACB", "switch_back_base")
    m.release_trace()
    del m
finally:
    ttnn.close_device(dev)
res["all_bit_identical"] = all(c["bit_identical"] for c in res["checks"])
res["t1"] = now()
json.dump(res, open(out_path, "w"), indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items() if k != "checks"}), flush=True)
