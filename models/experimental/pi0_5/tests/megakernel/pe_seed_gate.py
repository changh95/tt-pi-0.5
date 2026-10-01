# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Amended phase-2 base gate (DESIGN.md §7, user decision 2026-09-30): per seed, the whole-model megakernel is at least
as close as the shipped path to the fp32 torch reference of the WHOLE model on the same inputs, over >= 18 seeds.

The 22 inputs are verify_p1_r1's SPEC (docs/megakernel/verify_p1_r1/scripts/vc.py: the 6 test seeds + 16 prompt
lengths on tile edges); the fp32 references are computed on the CPU once (--refs, cached). One process per arm
(PI05_MEGAKERNEL=whole / off): each saves its per-seed outputs; --compare prints the per-seed table.
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
from vc import BASE_WEIGHTS, SPEC, base_config, obs_for, pcc  # noqa: E402


def now():
    return time.strftime("%F %T")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, help="ref_base.pt (the fp32 outputs per seed; computed if missing)")
    ap.add_argument("--arm-out", default=None, help="run the device arm (PI05_MEGAKERNEL from the env) and save here")
    ap.add_argument("--compare", nargs=2, default=None, metavar=("WHOLE_PT", "OFF_PT"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    torch.set_num_threads(16)
    if not os.path.exists(a.refs):
        from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
        from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model

        torch.set_grad_enabled(False)
        ref = PI0Model(base_config(), PI0WeightLoader(BASE_WEIGHTS))
        refs = {}
        for seed, n in SPEC:
            images, tokens, noise, mask = obs_for(seed, n)
            ref.denoising.sample_noise = lambda *x, _n=noise, **k: _n.clone()
            refs[seed] = ref.sample_actions(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask,
                                            torch.zeros(1, 32)).float()
            print(now(), "ref", seed, n, flush=True)
        torch.save({"refs": refs}, a.refs)
    refs = torch.load(a.refs)["refs"]
    if a.arm_out:
        import ttnn

        from models.experimental.pi0_5.common.device_open import open_pi05_device
        from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN
        from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader

        dev = open_pi05_device()
        try:
            model = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev)
            outs = {}
            for seed, n in SPEC:
                images, tokens, noise, mask = obs_for(seed, n)
                outs[seed] = model.sample_actions_fused(images, tokens, noise, lang_masks=mask).clone()
                print(now(), model.megakernel_backend, seed, n, round(pcc(outs[seed], refs[seed]), 6), flush=True)
            torch.save({"outs": outs, "backend": model.megakernel_backend, "program": model.megakernel_program},
                       a.arm_out)
        finally:
            ttnn.close_device(dev)
    if a.compare:
        w, o = torch.load(a.compare[0]), torch.load(a.compare[1])
        rows = []
        for seed, n in SPEC:
            pw, po = pcc(w["outs"][seed], refs[seed]), pcc(o["outs"][seed], refs[seed])
            rows.append({"seed": seed, "n_lang": n, "whole_vs_ref": round(pw, 6), "off_vs_ref": round(po, 6),
                         "margin": round(pw - po, 6), "pass": pw >= po})
            print(rows[-1])
        summ = {"n": len(rows), "passes": sum(r["pass"] for r in rows),
                "whole_mean": sum(r["whole_vs_ref"] for r in rows) / len(rows),
                "off_mean": sum(r["off_vs_ref"] for r in rows) / len(rows),
                "min_margin": min(r["margin"] for r in rows), "backends": [w["backend"], o["backend"]]}
        print(summ)
        if a.json:
            json.dump({"rows": rows, "summary": summ, "program": w.get("program")}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
