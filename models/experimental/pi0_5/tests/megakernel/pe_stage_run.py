# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage-by-stage check of the whole prefix on the prefix engine vs the host decomposition (localisation tool):
SigLIP after each of the first layers / all 27 + post-LN, projector + embedding rows, VLM layers."""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.pi0_5.common.fused_host import im2col_patches, prefix_valid_mask
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tests.megakernel.pe_bringup import Rig, cmp, stamp
from models.experimental.pi0_5.tt.megakernel import geometry as G
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P
from models.experimental.pi0_5.tt.megakernel import pe_host as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="base")
    ap.add_argument("--n-lang", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sh = G.SHAPES[a.shape]
    ps = P.pshape_for(sh)
    res = {}
    g = torch.Generator().manual_seed(0)
    pix = torch.rand(2, 3, 224, 224, generator=g) * 2 - 1
    tokens = torch.zeros(1, ps.ntok, dtype=torch.int64)
    tokens[0, : a.n_lang] = torch.randint(1, 257152, (a.n_lang,), generator=g)
    lmask = torch.zeros(1, ps.ntok, dtype=torch.bool)
    lmask[0, : a.n_lang] = True
    valid = prefix_valid_mask(2, lmask)[0]
    wl = PI0WeightLoader(os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    cat = wl.categorized_weights
    pp = H.prefix_params(cat)
    ew = cat["vlm_language"].get("model.embed_tokens.weight")
    ew = cat["vlm_language"]["lm_head.weight"] if ew is None else ew
    torch.set_num_threads(16)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1395712)
    try:
        emb = ttnn.from_torch(ew, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rig = Rig(dev, a.shape, pp, emb)
        t = rig.t
        im = im2col_patches(pix, 14, pad_to=608).reshape(512, 608)
        rig.put(t.im2col, im, ttnn.bfloat16)
        rig.put(t.tokens, tokens.to(torch.int32), ttnn.uint32)
        rig.put(t.vmask, H.vlm_key_mask(valid, ps), ttnn.bfloat16)
        imb = im.to(torch.bfloat16).float()
        # SigLIP: after 1, 2, 3 layers and all 27 (+ post-LN)
        for nl in ((27,) if os.environ.get("PE_STAGE_FAST") else (1, 2, 3, 27)):
            rig.run(0, P.OP_S0 + nl * 7)
            x = rig.get(t.x_s)
            xr = imb @ pp.patch_w + pp.pos_b.repeat(2, 1)
            pp_n = H.PrefixParams(**{**pp.__dict__, "sig": pp.sig[:nl]})
            # host_siglip applies post-LN at the end: recompute the residual by hand
            for s in pp.sig[:nl]:
                xn = H._ln(xr, s.ln1w, s.ln1b, pp.eps_s)
                qkv = xn @ s.wqkv + s.bqkv
                q, k, v = qkv.split(1536, dim=1)
                ctx = torch.zeros(512, 1536)
                for img in range(2):
                    rs = slice(img * 256, (img + 1) * 256)
                    qh, kh, vh = (u[rs].reshape(256, 16, 96).transpose(0, 1) for u in (q, k, v))
                    ctx[rs] = (torch.softmax(qh @ kh.transpose(1, 2), -1) @ vh).transpose(0, 1).reshape(256, 1536)
                xr = xr + ctx @ s.wo + s.bo
                xn = H._ln(xr, s.ln2w, s.ln2b, pp.eps_s)
                xr = xr + F.gelu(xn @ s.wfc1 + s.bfc1, approximate="tanh") @ s.wfc2 + s.bfc2
            cmp(f"siglip_x_after_{nl}", x, xr, res)
            del pp_n
        rig.run(P.OP_POSTLN, P.OP_POSTLN + 1)
        xn = rig.get(t.xn_s)
        cmp("siglip_postln", xn, H._ln(rig.get(t.x_s), pp.post_w, pp.post_b, pp.eps_s), res)
        rig.run(P.OP_PROJ, P.OP_EMBED + 1)
        xv = rig.get(t.x_v)
        cmp("proj_rows", xv[:512], xn @ pp.proj_w + pp.proj_b, res)
        lang = ew[tokens[0]].to(torch.bfloat16).float() * (2048 ** 0.5)
        Pr = ps.ptv * 32
        cmp("embed_rows", xv[512:Pr], lang, res)
        res["pad_rows_absmax"] = float(xv[Pr:].abs().max())
        stamp(f"pad rows absmax {res['pad_rows_absmax']}")
        # VLM: layer by layer from the device's own x
        x_in = xv.clone()
        for l in range(3):
            base = P.OP_V0 + l * 7
            rig.put(t.x_v, x_in, ttnn.float32)
            rig.run(base, base + 7)
            xo = rig.get(t.x_v)
            two = H.PrefixParams(**{**pp.__dict__, "vlm": [pp.vlm[l], pp.vlm[l]]})
            xr, kvr = H.host_vlm(two, x_in, valid, ps, layers=1)
            k = rig.get(rig.kv[l][0])[0, 0, :Pr]
            cmp(f"vlm_x_after_layer{l}", xo[:Pr], xr[:Pr], res)
            cmp(f"vlm_k_layer{l}", k[valid], kvr[0][0][valid], res)
            x_in = xo
    finally:
        ttnn.close_device(dev)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
