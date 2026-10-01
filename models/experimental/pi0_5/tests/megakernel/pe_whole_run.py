# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""sample_actions_fused end to end under PI05_MEGAKERNEL=<arm> (whole / expert / off): per-seed actions on synthetic
requests (saved for cross-arm comparison), replay bit-identity, and per-call latency.

    PI05_MEGAKERNEL=whole python .../pe_whole_run.py --shape base --seeds 0,1,2 --calls 20 --out whole.pt
"""
import argparse
import os
import time

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.device_open import open_pi05_device
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN


def stamp(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def request(seed, shape, n_lang):
    g = torch.Generator().manual_seed(seed)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    ntok = 224 if shape == "base" else 32
    horizon = 50 if shape == "base" else 10
    tokens = torch.zeros(1, ntok, dtype=torch.int64)
    tokens[0, :n_lang] = torch.randint(1, 257152, (n_lang,), generator=g)
    lmask = torch.zeros(1, ntok, dtype=torch.bool)
    lmask[0, :n_lang] = True
    noise = torch.randn(1, horizon, 32, generator=g)
    return images, tokens, noise, lmask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="base")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-lang", default=None, help="comma list aligned with seeds (default 150 / 20)")
    ap.add_argument("--calls", type=int, default=20)
    ap.add_argument("--weights", default=os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(16)
    horizon = 50 if a.shape == "base" else 10
    seeds = [int(s) for s in a.seeds.split(",")]
    nl = [int(x) for x in a.n_lang.split(",")] if a.n_lang else [150 if a.shape == "base" else 20] * len(seeds)
    config = PI0ModelConfig(action_dim=32, action_horizon=horizon, state_dim=32, pi05=True)
    config.siglip_config = SigLIPConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                                        num_attention_heads=16, image_size=224, patch_size=14)
    dev = open_pi05_device()
    res = {"arm": os.environ.get("PI05_MEGAKERNEL", "expert"), "shape": a.shape, "seeds": seeds, "n_lang": nl}
    try:
        t0 = time.time()
        model = PI0ModelTTNN(config, PI0WeightLoader(a.weights), dev)
        res["build_s"] = round(time.time() - t0, 1)
        res["stamp"] = {"backend": model.megakernel_backend, "program": model.megakernel_program}
        stamp(f"model built in {res['build_s']} s, backend {model.megakernel_backend}")
        outs = {}
        for s, n in zip(seeds, nl):
            t0 = time.time()
            outs[s] = model.sample_actions_fused(*request(s, a.shape, n)).clone()
            stamp(f"seed {s} n_lang {n}: {time.time() - t0:.3f} s, absmax {float(outs[s].abs().max()):.3f}, "
                  f"finite {bool(torch.isfinite(outs[s]).all())}")
        # replays: the last request again, n times; bit-identity and latency
        req = request(seeds[-1], a.shape, nl[-1])
        lat, same = [], True
        for _ in range(a.calls):
            t0 = time.perf_counter()
            o = model.sample_actions_fused(*req)
            lat.append((time.perf_counter() - t0) * 1e3)
            same &= bool(torch.equal(o, outs[seeds[-1]]))
        lat.sort()
        res["latency_ms_median"] = round(lat[len(lat) // 2], 3)
        res["latency_ms"] = [round(x, 3) for x in lat]
        res["replays_identical"] = same
        # the first seed again after the others (alternation): identical to its first call
        o0 = model.sample_actions_fused(*request(seeds[0], a.shape, nl[0]))
        res["first_seed_again_identical"] = bool(torch.equal(o0, outs[seeds[0]]))
        stamp(f"median {res['latency_ms_median']} ms over {a.calls} calls, identical {same}, "
              f"alternation {res['first_seed_again_identical']}")
        res["actions"] = outs
        if model.megakernel_l1 is not None:
            res["l1"] = model.megakernel_l1
    finally:
        ttnn.close_device(dev)
    torch.save(res, a.out)
    stamp("done")


if __name__ == "__main__":
    main()
