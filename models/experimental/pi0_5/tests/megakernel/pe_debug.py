# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Localisation runs for pe_bringup failures (VLM attention, VLM down): per-head / per-row-tile / per-column errors."""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tests.megakernel.pe_bringup import Rig, cmp, pcc, stamp
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P
from models.experimental.pi0_5.tt.megakernel import pe_host as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="base")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only-down", action="store_true")
    ap.add_argument("--save", default=None, help="torch.save the down op's tensors here")
    a = ap.parse_args()
    res = {}
    wl = PI0WeightLoader(os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    cat = wl.categorized_weights
    pp = H.prefix_params(cat, n_sig=1, n_vlm=1)
    torch.set_num_threads(16)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1395712)
    try:
        emb = ttnn.from_torch(cat["vlm_language"]["lm_head.weight"][:256], dtype=ttnn.bfloat16,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rig = Rig(dev, a.shape, pp, emb)
        ps, t = rig.ps, rig.t
        rows = ps.mt * 32
        P_rows = ps.ptv * 32
        g = torch.Generator().manual_seed(11)
        base = P.OP_V0
        # ---- attention with synthetic q / K / V written straight into the scratch and the layer-0 caches
        for tag, n_valid in (() if a.only_down else (("allvalid", P_rows), ("masked", 512 + 150))):
            valid = torch.zeros(P_rows, dtype=torch.bool)
            valid[:n_valid] = True
            rig.put(t.vmask, H.vlm_key_mask(valid, ps), ttnn.bfloat16)
            q = torch.randn(8, rows, 256, generator=g) * 0.5
            k = torch.randn(P_rows, 256, generator=g)
            v = torch.randn(P_rows, 256, generator=g)
            rig.put(t.q_v, q.reshape(8 * rows, 256), ttnn.bfloat16)
            kc = torch.zeros(1, 1, rig.kv[0][0].shape[2], 256)
            vc = torch.zeros_like(kc)
            kc[0, 0, :P_rows] = k
            vc[0, 0, :P_rows] = v
            rig.put(rig.kv[0][0], kc, ttnn.bfloat8_b)
            rig.put(rig.kv[0][1], vc, ttnn.bfloat8_b)
            qd = rig.get(t.q_v).reshape(8, rows, 256)
            kd = rig.get(rig.kv[0][0])[0, 0, :P_rows]
            vd = rig.get(rig.kv[0][1])[0, 0, :P_rows]
            rig.run(base + 2, base + 3)
            ctx = rig.get(t.ctx_v)
            bias = torch.where(valid, 0.0, -1.0e9)[None, :]
            cref = (torch.softmax(qd @ kd.T + bias, -1) @ vd).transpose(0, 1).reshape(rows, 2048)
            cmp(f"attn_{tag}", ctx[:P_rows], cref[:P_rows], res)
            per_head = [round(pcc(ctx[:P_rows, h * 256:(h + 1) * 256], cref[:P_rows, h * 256:(h + 1) * 256]), 5)
                        for h in range(8)]
            per_rt = [round(pcc(ctx[r * 32:(r + 1) * 32], cref[r * 32:(r + 1) * 32]), 4) for r in range(ps.ptv)]
            per_d = [round(pcc(ctx[:P_rows].reshape(P_rows, 8, 8, 32)[:, :, d], cref[:P_rows].reshape(P_rows, 8, 8, 32)[:, :, d]), 4)
                     for d in range(8)]
            res[f"attn_{tag}_per_head"], res[f"attn_{tag}_per_rt"], res[f"attn_{tag}_per_d"] = per_head, per_rt, per_d
            stamp(f"{tag} per head {per_head}\n per rt {per_rt}\n per d {per_d}")
            # hypotheses: V ignored / only one chunk / wrong head K
            for hname, alt in (("first_chunk_only", 6 * 32), ("last_chunk_only", None)):
                if alt is None:
                    sl = slice(18 * 32, P_rows)
                else:
                    sl = slice(0, alt)
                cr = (torch.softmax(qd @ kd[sl].T + bias[:, sl], -1) @ vd[sl]).transpose(0, 1).reshape(rows, 2048)
                res[f"attn_{tag}_vs_{hname}"] = round(pcc(ctx[:P_rows], cr[:P_rows]), 5)
            stamp({k_: v_ for k_, v_ in res.items() if k_.startswith(f"attn_{tag}_vs")})
        # ---- down with a synthetic h (bfp8) and residual
        vp = pp.vlm[0]
        h = torch.randn(rows, 16384, generator=g) * 0.1
        x = torch.randn(rows, 2048, generator=g)
        rig.put(t.h_v, h, ttnn.bfloat8_b)
        rig.put(t.x_v, x, ttnn.float32)
        hd = rig.get(t.h_v)
        xd = rig.get(t.x_v)
        rig.run(base + 6, base + 7)
        out = rig.get(t.x_v)
        ref = xd + hd @ vp.wd
        cmp("down", out, ref, res)
        if a.save:
            torch.save({"h": hd, "x": xd, "out": out, "wd": vp.wd}, a.save)
        o = P.describe(base + 6, ps)
        per_col = []
        for c in range(11):
            p0, npr = P.mm_pair0(o, c), P.mm_pairs(o, c)
            cs = slice(p0 * 64, (p0 + npr) * 64)
            per_col.append(round(pcc(out[:, cs] - xd[:, cs], ref[:, cs] - xd[:, cs]), 4))
        per_band = [round(pcc(out[b * 96:(b + 1) * 96] - xd[b * 96:(b + 1) * 96], ref[b * 96:(b + 1) * 96] - xd[b * 96:(b + 1) * 96]), 4)
                    for b in range(ps.rv)]
        res["down_per_col"], res["down_per_band"] = per_col, per_band
        stamp(f"down per col {per_col} per band {per_band}")
        # hypotheses on the K blocks: only block 0 / only the last block / residual missing
        d_only = out - xd
        for kname, ks in (("kb0", slice(0, 512)), ("kblast", slice(16384 - 512, 16384))):
            r_ = hd[:, ks] @ vp.wd[ks]
            res[f"down_vs_{kname}"] = round(pcc(d_only, r_), 5)
        res["down_delta_vs_full"] = round(pcc(d_only, hd @ vp.wd), 5)
        res["down_out_vs_resid"] = round(pcc(out, xd), 5)
        stamp({k_: v_ for k_, v_ in res.items() if k_.startswith("down_")})
    finally:
        ttnn.close_device(dev)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
