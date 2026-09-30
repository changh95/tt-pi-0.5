# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy of the replaced component alone: the shipped expert loop and the megakernel against an fp32 host oracle
of the expert loop (host_model.loop_reference with the checkpoint's fp32 weights) fed with the DEVICE's prefix K / V
caches, mask, RoPE tables and noise (identical in both paths: the ttnn prefix is shared and deterministic). Also
reports each path vs the full fp32 torch reference, as test_pcc_pi05_fused.py does.

    python models/experimental/pi0_5/tests/megakernel/mk_expert_oracle.py --seeds 18 --out r.json
"""
import argparse
import dataclasses
import json
import time

import torch
import ttnn

from models.experimental.pi0_5.common.device_open import open_pi05_device
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0_5.tests.megakernel.mk_vs_shipped import prompt
from models.experimental.pi0_5.tests.pcc.golden_openpi import pcc
from models.experimental.pi0_5.tests.pcc.test_pcc_pi05_fused import BASE_WEIGHTS, N_REAL, base_config
from models.experimental.pi0_5.tt.megakernel import host_model as hm
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN


def device_inputs(m):
    P = m.backbone.kv_cache_plan["prefix_len"]
    kv = [(ttnn.to_torch(k).float()[0, 0, :P], ttnn.to_torch(v).float()[0, 0, :P]) for k, v in m.backbone.kv_caches]
    att = {k: ttnn.to_torch(t).float() for k, t in m._fused_attn_dev.items()}
    noise = ttnn.to_torch(m._fused_in_noise).float()[0]
    return kv, hm.attn_inputs_from(att), noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=18)
    ap.add_argument("--shape", default="base", choices=["base", "libero"])
    ap.add_argument("--n-real", default=None, help="comma list of prompt lengths (seeds 11, 12, ...)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.shape == "base":
        loader, mcfg, tok_len, H = PI0WeightLoader(BASE_WEIGHTS), base_config, 224, 50
    else:
        from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config

        loader, mcfg, tok_len, H = PI0WeightLoader(LIBERO_WEIGHTS), libero_config, 32, 10
    if a.n_real:
        spec = [(11 + i, int(n)) for i, n in enumerate(a.n_real.split(","))]
    else:
        extra = [(100 + i, n) for i, n in enumerate([40, 97, 12, 150, 201, 224] * 4)][: max(0, a.seeds - len(N_REAL))]
        spec = [(i + 1, n) for i, n in enumerate(N_REAL)] + extra
    obs = [prompt(sd, n, tok_len, H) for sd, n in spec]
    ref = PI0ModelTorch(mcfg(), loader)
    refs = []
    for images, tokens, noise, mask in obs:
        ref.denoising.sample_noise = lambda *x, _n=noise, **k: _n.clone()
        with torch.no_grad():
            refs.append(ref.sample_actions(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask, torch.zeros(1, 32)).float())
    del ref
    cw = loader.categorized_weights
    params = hm.expert_params(cw["action_expert"], cw["pi0_projections"])
    env = FusedConfig.from_env()
    dev = open_pi05_device(dataclasses.replace(env, megakernel="expert"))
    res = {"time": time.strftime("%F %T"), "shape": a.shape, "obs": spec}
    try:
        outs, oracle = {}, []
        for name in ("off", "expert"):
            torch.manual_seed(42)
            m = PI0ModelTTNN(mcfg(), loader, dev, fused=dataclasses.replace(env, megakernel=name))
            outs[name] = []
            for i, (im, t, n, k) in enumerate(obs):
                outs[name].append(m.sample_actions_fused(im, t, n, lang_masks=k))
                if name == "off":
                    kv, ai, nz = device_inputs(m)
                    with torch.no_grad():
                        oracle.append(hm.loop_reference(params, kv, ai, nz)[:H][None])
            m.release_trace()
            del m
        for name in ("off", "expert"):
            res[f"pcc_{name}_vs_expert_oracle"] = [pcc(o, r) for o, r in zip(outs[name], oracle)]
            res[f"pcc_{name}_vs_ref"] = [pcc(o, r) for o, r in zip(outs[name], refs)]
            res[f"mean_{name}_vs_expert_oracle"] = sum(res[f"pcc_{name}_vs_expert_oracle"]) / len(obs)
            res[f"mean_{name}_vs_ref"] = sum(res[f"pcc_{name}_vs_ref"]) / len(obs)
        res["pcc_oracle_vs_ref"] = [pcc(o, r) for o, r in zip(oracle, refs)]
        print("RESULT " + json.dumps(res), flush=True)
    finally:
        ttnn.close_device(dev)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
