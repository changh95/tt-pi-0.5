# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shipped path (PI05_MEGAKERNEL=off) and the phase-1 megakernel (expert) in ONE process under the 64 KiB worker-L1
cut, on the base padded-prompt observations of test_pcc_pi05_fused.py plus extra seeds: per seed PCC of each path vs
the fp32 torch reference and of the two device paths vs each other, and both paths' latency.

    python models/experimental/pi0_5/tests/megakernel/mk_vs_shipped.py --seeds 12 --out r.json
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
from models.experimental.pi0_5.tests.pcc.golden_openpi import pcc
from models.experimental.pi0_5.tests.pcc.test_pcc_pi05_fused import BASE_WEIGHTS, N_REAL, base_config, bench, padded_prompt
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    loader = PI0WeightLoader(BASE_WEIGHTS)
    extra = [(100 + i, n) for i, n in enumerate([40, 97, 12, 150, 201, 224] * 4)][: max(0, a.seeds - len(N_REAL))]
    obs_spec = [(i + 1, n) for i, n in enumerate(N_REAL)] + extra
    obs = [padded_prompt(sd, n) for sd, n in obs_spec]
    ref = PI0ModelTorch(base_config(), loader)
    refs = []
    for images, tokens, noise, mask in obs:
        ref.denoising.sample_noise = lambda *x, _n=noise, **k: _n.clone()
        with torch.no_grad():
            refs.append(ref.sample_actions(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask, torch.zeros(1, 32)).float())
    del ref
    env = FusedConfig.from_env()
    cfg_mk = dataclasses.replace(env, megakernel="expert")
    cfg_off = dataclasses.replace(env, megakernel="off")
    dev = open_pi05_device(cfg_mk)  # the cut applies to both models in this process
    res = {"time": time.strftime("%F %T"), "obs": obs_spec, "worker_l1_cut": True}
    try:
        outs = {}
        for name, cfg in (("off", cfg_off), ("expert", cfg_mk)):
            torch.manual_seed(42)
            m = PI0ModelTTNN(base_config(), loader, dev, fused=cfg)
            outs[name] = [m.sample_actions_fused(i, t, n, lang_masks=k) for (i, t, n, k) in obs]
            res[f"latency_{name}"] = bench(m, dev, obs[0], runs=30)
            res[f"stamp_{name}"] = getattr(m, "megakernel_backend", None)
            m.release_trace()
            print(f"[{time.strftime('%F %T')}] {name} done {res[f'latency_{name}']}", flush=True)
        res["pcc_off_vs_ref"] = [pcc(o, r) for o, r in zip(outs["off"], refs)]
        res["pcc_mk_vs_ref"] = [pcc(o, r) for o, r in zip(outs["expert"], refs)]
        res["pcc_mk_vs_off"] = [pcc(o, r) for o, r in zip(outs["expert"], outs["off"])]
        res["mean_off_vs_ref"] = sum(res["pcc_off_vs_ref"]) / len(refs)
        res["mean_mk_vs_ref"] = sum(res["pcc_mk_vs_ref"]) / len(refs)
        print("RESULT " + json.dumps(res), flush=True)
    finally:
        ttnn.close_device(dev)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
