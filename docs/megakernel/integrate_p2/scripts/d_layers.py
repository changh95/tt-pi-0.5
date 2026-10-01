"""verify-p2-r0 SigLIP / VLM per-layer device time of the engine, measured from the HOST wall clock, not from the
engine's own stamps: inside the real whole model (real weights, real activations from one sample_actions_fused call),
the prefix-only program (PE_WHOLE=0) runs op ranges with in-kernel reps R; per-layer = marginal between a 27- (17-)
layer range and a 1-layer range: (T_many - T_one) / ((n_many - 1) * R), launch cost cancels. Also: the full prefix
(ops 0..314, all K/V) per rep (WP-P2-3 stack time), and the engine's hub stamps for comparison.
  python d_layers.py SHAPE OUT.json"""
import json, os, statistics, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc2 import BASE_WEIGHTS, base_config, now, obs_for  # noqa: E402
import ttnn  # noqa: E402
from models.experimental.pi0_5.common.device_open import open_pi05_device  # noqa: E402
from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, record_inputs  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P  # noqa: E402

shape, outp = sys.argv[1], sys.argv[2]
R = int(os.environ.get("VP_REPS", "10"))
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
assert env.megakernel == "whole"
dev = open_pi05_device(env)
res = {"t0": now(), "shape": shape, "reps": R}
try:
    torch.manual_seed(42)
    if shape == "base":
        m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=env)
        im, tk, nz, mk = obs_for(4, 150)
    else:
        m = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), dev, fused=env)
        im, tk, mk, nz = record_inputs(load_records()[3], 32)
    y0 = m.sample_actions_fused(im, tk, nz, lang_masks=mk).clone()
    plan = m.backbone.kv_cache_plan
    wm = m._whole_for(plan["prefix_len"], plan["batch"])
    a_in = m._fused_attn_in()
    args = (m.backbone.kv_caches, a_in["exp_mask"], a_in["tables"], m._fused_in_noise)

    def wall(first, stop, reps, n=5):
        ts = []
        for _ in range(n):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            wm.run(*args, out=m._fused_out, whole=False, first=first, stop=stop, reps=reps)
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts), ts

    wall(P.OP_S0, P.OP_S0 + 7, 1, n=1)  # compile the prefix-only variant
    out = {}
    for name, f0, nl in (("siglip", P.OP_S0, 27), ("vlm", P.OP_V0, 17)):
        t1, a1 = wall(f0, f0 + 7, R)
        tn, an = wall(f0, f0 + 7 * nl, R)
        t1b, _ = wall(f0, f0 + 7, 1)
        out[name] = {"ms_1layer_Rreps": t1, "ms_nlayers_Rreps": tn, "n_layers": nl, "all_1": a1, "all_n": an,
                     "ms_1layer_1rep": t1b,
                     "per_layer_us_marginal": (tn - t1) / ((nl - 1) * R) * 1e3,
                     "per_layer_us_naive_nlayers": tn / (nl * R) * 1e3}
        tt = ttnn.to_torch(wm.t.times).to(torch.int64) & 0xFFFFFFFF  # engine stamps of the LAST launch (the 1-rep one)
        out[name]["engine_stamps_present"] = int((tt[:8, 0] != 0).sum())
        print(now(), name, {k: v for k, v in out[name].items() if not k.startswith("all")}, flush=True)
    tp, ap_ = wall(0, P.N_OPS, R, n=3)
    tp1, _ = wall(0, P.N_OPS, 1, n=3)
    out["prefix"] = {"ms_Rreps": tp, "ms_1rep": tp1, "per_rep_ms_marginal": (tp - tp1) / (R - 1), "all": ap_}
    print(now(), "prefix", out["prefix"], flush=True)
    res["timing"] = out
    # the whole model still gives the same output after the prefix-only launches (the trace re-reads its inputs)
    y1 = m.sample_actions_fused(im, tk, nz, lang_masks=mk)
    res["after_eq_before"] = bool(torch.equal(y0, y1))
    m.release_trace()
finally:
    ttnn.close_device(dev)
res["t1"] = now()
json.dump(res, open(outp, "w"), indent=1)
print("RESULT", json.dumps({k: v for k, v in res.items()}, default=str)[:3000], flush=True)
