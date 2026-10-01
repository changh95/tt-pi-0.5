# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-length edge cases (DESIGN.md §7 exit gate): per observation, the selected arm vs the fp32 torch reference of
the whole model, "no worse than the current path". Base n_lang 1 / 224 / 128 / 150, LIBERO n_lang 32 / 1 (seeds 11..,
the phase-1 E1 / E2 construction: tests/megakernel/mk_expert_oracle.py prompt()). One process per arm (PI05_MEGAKERNEL
from the env); --compare prints the per-observation table.

    python pe_edge_gate.py --shape libero --spec 11:32,12:1 --refs ref_libero_edge.pt --arm-out whole.pt
    python pe_edge_gate.py --shape libero --spec 11:32,12:1 --refs ref_libero_edge.pt --compare whole.pt off.pt
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VC = os.path.join(HERE, "../../../../../docs/megakernel/verify_p1_r1/scripts")
sys.path.insert(0, os.path.abspath(VC))
from vc import BASE_WEIGHTS, base_config, obs_for, pcc  # noqa: E402


def now():
    return time.strftime("%F %T")


def setup(shape):
    if shape == "base":
        return BASE_WEIGHTS, base_config, 224, 50
    from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config

    return LIBERO_WEIGHTS, libero_config, 32, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="base", choices=["base", "libero"])
    ap.add_argument("--spec", required=True, help="seed:n_lang,... e.g. 11:1,12:224,13:128,14:150")
    ap.add_argument("--refs", required=True)
    ap.add_argument("--arm-out", default=None)
    ap.add_argument("--compare", nargs=2, default=None, metavar=("ARM_PT", "OFF_PT"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.set_num_threads(16)
    spec = [(int(s), int(n)) for s, n in (x.split(":") for x in a.spec.split(","))]
    weights, cfg, tok_len, horizon = setup(a.shape)
    obs = {s: obs_for(s, n, tok_len, horizon) for s, n in spec}
    from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader

    if not os.path.exists(a.refs):
        from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model

        torch.set_grad_enabled(False)
        ref = PI0Model(cfg(), PI0WeightLoader(weights))
        refs = {}
        for s, n in spec:
            images, tokens, noise, mask = obs[s]
            ref.denoising.sample_noise = lambda *x, _n=noise, **k: _n.clone()
            refs[s] = ref.sample_actions(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask,
                                         torch.zeros(1, 32)).float()
            print(now(), "ref", s, n, flush=True)
        torch.save({"refs": refs, "shape": a.shape, "spec": spec}, a.refs)
    refs = torch.load(a.refs)["refs"]
    if a.arm_out:
        import ttnn

        from models.experimental.pi0_5.common.device_open import open_pi05_device
        from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

        dev = open_pi05_device()
        try:
            torch.manual_seed(42)
            model = PI0ModelTTNN(cfg(), PI0WeightLoader(weights), dev)
            outs = {}
            for s, n in spec:
                images, tokens, noise, mask = obs[s]
                outs[s] = model.sample_actions_fused(images, tokens, noise, lang_masks=mask).clone()
                print(now(), model.megakernel_backend, s, n, round(pcc(outs[s], refs[s]), 6), flush=True)
            torch.save({"outs": outs, "backend": model.megakernel_backend, "program": model.megakernel_program},
                       a.arm_out)
        finally:
            ttnn.close_device(dev)
    if a.compare:
        w, o = torch.load(a.compare[0]), torch.load(a.compare[1])
        rows = []
        for s, n in spec:
            pw, po = pcc(w["outs"][s], refs[s]), pcc(o["outs"][s], refs[s])
            rows.append({"seed": s, "n_lang": n, "arm_vs_ref": round(pw, 6), "off_vs_ref": round(po, 6),
                         "margin": round(pw - po, 6), "pass": pw >= po})
            print(rows[-1])
        summ = {"shape": a.shape, "n": len(rows), "passes": sum(r["pass"] for r in rows),
                "backends": [w["backend"], o["backend"]]}
        print(summ)
        if a.json:
            json.dump({"rows": rows, "summary": summ, "program": w.get("program")}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
