# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The WHOLE prefix (all 314 ops: SigLIP x2 -> projector -> embedding -> VLM prefill -> the 18 K / V caches) on the
prefix engine, against (a) the shipped ttnn prefix of the same inputs in the same process (PI05_MEGAKERNEL=off model,
its caches after one sample_actions_fused) and (b) the host fp32 decomposition (pe_host). WP-P2-3.

    python models/experimental/pi0_5/tests/megakernel/pe_prefix_run.py --shape base --n-lang 150 --reps 5 --out r.json
"""
import argparse
import json
import os
import time

import torch
import ttnn

os.environ.setdefault("PI05_MEGAKERNEL", "off")

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_host import im2col_patches, prefix_valid_mask
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tests.megakernel.pe_bringup import AICLK_MHZ, Rig, pcc, rel, stamp
from models.experimental.pi0_5.tt.megakernel import geometry as G
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P
from models.experimental.pi0_5.tt.megakernel import pe_host as H
from models.experimental.pi0_5.tt.megakernel.pe_program import kernel_digest2


def base_config(horizon):
    config = PI0ModelConfig(action_dim=32, action_horizon=horizon, state_dim=32, pi05=True)
    config.siglip_config = SigLIPConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                                        num_attention_heads=16, image_size=224, patch_size=14)
    return config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="base")
    ap.add_argument("--n-lang", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--no-ship", action="store_true", help="skip the shipped-path comparison")
    ap.add_argument("--weights", default=os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sh = G.SHAPES[a.shape]
    ps = P.pshape_for(sh)
    n_lang = a.n_lang if a.n_lang is not None else (150 if a.shape == "base" else 20)
    res = {"shape": a.shape, "n_lang": n_lang, "seed": a.seed, "digest": kernel_digest2(), "time": time.strftime("%F %T")}
    g = torch.Generator().manual_seed(a.seed)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    tokens = torch.zeros(1, ps.ntok, dtype=torch.int64)
    tokens[0, :n_lang] = torch.randint(1, 257152, (n_lang,), generator=g)
    lmask = torch.zeros(1, ps.ntok, dtype=torch.bool)
    lmask[0, :n_lang] = True
    valid = prefix_valid_mask(2, lmask)[0]
    stamp("loading weights")
    wl = PI0WeightLoader(a.weights)
    cat = wl.categorized_weights
    from models.experimental.pi0_5.common.fused_config import FusedConfig

    torch.set_num_threads(16)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1395712,
                           trace_region_size=FusedConfig.from_env().trace_region_size)
    dev.enable_program_cache()
    try:
        Pr = ps.ptv * 32
        ship = None
        emb = None
        if not a.no_ship:
            from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

            stamp("building the shipped model (PI05_MEGAKERNEL=off)")
            model = PI0ModelTTNN(base_config(sh.horizon), wl, dev)
            noise = torch.randn(1, sh.horizon, 32, generator=g)
            model.sample_actions_fused(images, tokens, noise, lmask)
            ship = [(ttnn.to_torch(k).float()[0, 0, :Pr], ttnn.to_torch(v).float()[0, 0, :Pr])
                    for k, v in model.backbone.kv_caches]
            emb = model.backbone.vlm_embed_tokens
            # free the shipped path's L1 residents (trace, persistent inputs, caches): the megakernel's static CB
            # region needs the L1 below them (a whole-mode process has no ttnn co-tenants at all)
            model.release_trace()
            model._fused_release_inputs()
            for k, v in model.backbone.kv_caches:
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            model.backbone.kv_caches = None
            model.backbone.__dict__.get("_kv_sets", {}).clear()
            stamp("shipped prefix done")
        if emb is None:
            ew = cat["vlm_language"].get("model.embed_tokens.weight")
            ew = cat["vlm_language"]["lm_head.weight"] if ew is None else ew
            emb = ttnn.from_torch(ew, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
        stamp("prefix params + arenas")
        t0 = time.time()
        pp = H.prefix_params(cat)
        rig = Rig(dev, a.shape, pp, emb)
        res["upload_s"] = round(time.time() - t0, 1)
        stamp(f"uploaded in {res['upload_s']} s")
        pix = torch.cat(images, 0)
        im = im2col_patches(pix, 14, pad_to=608).reshape(512, 608)
        rig.put(rig.t.im2col, im, ttnn.bfloat16)
        rig.put(rig.t.tokens, tokens.to(torch.int32), ttnn.uint32)
        rig.put(rig.t.vmask, H.vlm_key_mask(valid, ps), ttnn.bfloat16)
        stamp("run the whole prefix")
        wall = rig.run(0, P.N_OPS)
        res["first_run_wall_s"] = round(wall, 3)
        mine = [(rig.get(k)[0, 0, :Pr], rig.get(v)[0, 0, :Pr]) for k, v in rig.kv]
        d = rig.diag()
        res["diag_core0"] = [int(v) for v in d[0, :8]]
        # host fp32 decomposition on the same (bf16-rounded) inputs
        stamp("host model")
        imb = im.to(torch.bfloat16).float()
        img = H.host_siglip(pp, imb) @ pp.proj_w + pp.proj_b
        ew = cat["vlm_language"].get("model.embed_tokens.weight")
        ew = cat["vlm_language"]["lm_head.weight"] if ew is None else ew
        x0 = torch.zeros(ps.mt * 32, 2048)
        x0[:512] = img
        x0[512:Pr] = ew[tokens[0]].to(torch.bfloat16).float() * (2048 ** 0.5)
        _, hkv = H.host_vlm(pp, x0, valid, ps)
        vr = valid.clone()
        rows = {"mine_vs_host": [], "ship_vs_host": [], "mine_vs_ship": []}
        for l in range(18):
            for j, name in ((0, "K"), (1, "V")):
                m, h = mine[l][j][vr], hkv[l][j][vr]
                r = {"layer": l, "t": name, "pcc": round(pcc(m, h), 6), "rel": round(rel(m, h), 5)}
                rows["mine_vs_host"].append(r)
                if ship is not None:
                    s_ = ship[l][j][vr]
                    rows["ship_vs_host"].append({"layer": l, "t": name, "pcc": round(pcc(s_, h), 6), "rel": round(rel(s_, h), 5)})
                    rows["mine_vs_ship"].append({"layer": l, "t": name, "pcc": round(pcc(m, s_), 6), "rel": round(rel(m, s_), 5)})
        res["per_layer"] = rows
        for k_, v_ in rows.items():
            if v_:
                stamp(f"{k_}: min pcc {min(r['pcc'] for r in v_)} max rel {max(r['rel'] for r in v_)}; "
                      f"L0 K {v_[0]['pcc']} L17 V {v_[-1]['pcc']}")
        if a.reps:
            rig.run(0, P.N_OPS, a.reps)
            tt = ttnn.to_torch(rig.t.times).to(torch.int64) & 0xFFFFFFFF
            n = P.N_OPS * a.reps
            st = [int(tt[4095, 0])] + [int(tt[k, 0]) for k in range(min(n, 4095))]
            per = [((st[i + 1] - st[i]) & 0xFFFFFFFF) / (AICLK_MHZ * 1e3) for i in range(len(st) - 1)]
            rep_ms = [sum(per[r * P.N_OPS:(r + 1) * P.N_OPS]) for r in range(min(a.reps, len(per) // P.N_OPS))]
            res["prefix_ms_per_rep"] = [round(x, 3) for x in rep_ms]
            ops = P.all_ops(ps)
            by = {}
            for i, dt in enumerate(per[:P.N_OPS]):
                w = ops[i].what
                by[w] = by.get(w, 0.0) + dt
            names = {v: k for k, v in P.PD.items() if k.startswith("W_")}
            res["ms_by_what_rep0"] = {names[w]: round(v, 3) for w, v in sorted(by.items())}
            stamp(f"prefix per rep {res['prefix_ms_per_rep']} ms; by op {res['ms_by_what_rep0']}")
    finally:
        ttnn.close_device(dev)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    stamp("done")


if __name__ == "__main__":
    main()
