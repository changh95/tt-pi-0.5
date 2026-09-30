# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WP-P2-1 / P2-2 accuracy halves (DESIGN.md §7): one SigLIP layer and one VLM layer of the prefix engine vs the
SHIPPED ttnn layer on the same real input, in one process: PCC(engine, ttnn) >= 0.9995 and the engine's rel-L2 vs the
fp32 host decomposition <= the ttnn layer's own rel-L2 vs fp32 + 20 % (the P1-3 rule).

Inputs: the real patch + position embedding of seed-0 images (SigLIP layer 0), the real prefix embeddings (fp32 host
SigLIP + projector, language embedding; bf16-rounded) of the same request (VLM layer 0). Both paths get the same
bf16 values; the fp32 reference is the host decomposition (pe_host, the checkpoint in fp32).

    python models/experimental/pi0_5/tests/megakernel/pe_layer_vs_ttnn.py --out r.json
"""
import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
import ttnn

os.environ.setdefault("PI05_MEGAKERNEL", "off")

from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.fused_host import im2col_patches, prefix_valid_mask  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.megakernel.pe_bringup import Rig, pcc, rel, stamp  # noqa: E402
from models.experimental.pi0_5.tests.megakernel.pe_prefix_run import base_config  # noqa: E402
from models.experimental.pi0_5.tt.megakernel import geometry as G  # noqa: E402
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P  # noqa: E402
from models.experimental.pi0_5.tt.megakernel import pe_host as H  # noqa: E402
from models.experimental.pi0_5.tt.megakernel.pe_program import kernel_digest2  # noqa: E402


def sig_layer_fp32(pp, s, x):
    xn = H._ln(x, s.ln1w, s.ln1b, pp.eps_s)
    qkv = xn @ s.wqkv + s.bqkv
    q, k, v = qkv.split(1536, dim=1)
    ctx = torch.zeros(512, 1536)
    for img in range(2):
        rs = slice(img * 256, (img + 1) * 256)
        qh, kh, vh = (u[rs].reshape(256, 16, 96).transpose(0, 1) for u in (q, k, v))
        ctx[rs] = (torch.softmax(qh @ kh.transpose(1, 2), -1) @ vh).transpose(0, 1).reshape(256, 1536)
    x = x + ctx @ s.wo + s.bo
    xn = H._ln(x, s.ln2w, s.ln2b, pp.eps_s)
    return x + F.gelu(xn @ s.wfc1 + s.bfc1, approximate="tanh") @ s.wfc2 + s.bfc2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-lang", type=int, default=150)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(16)
    sh = G.SHAPES["base"]
    ps = P.pshape_for(sh)
    res = {"digest": kernel_digest2(), "time": time.strftime("%F %T"), "n_lang": a.n_lang}
    g = torch.Generator().manual_seed(0)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    tokens = torch.zeros(1, ps.ntok, dtype=torch.int64)
    tokens[0, : a.n_lang] = torch.randint(1, 257152, (a.n_lang,), generator=g)
    lmask = torch.zeros(1, ps.ntok, dtype=torch.bool)
    lmask[0, : a.n_lang] = True
    noise = torch.randn(1, sh.horizon, 32, generator=g)
    valid = prefix_valid_mask(2, lmask)[0]
    wl = PI0WeightLoader(os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    cat = wl.categorized_weights
    pp = H.prefix_params(cat)
    Pr = ps.ptv * 32
    # the real inputs (host, bf16-rounded)
    im = im2col_patches(torch.cat(images, 0), 14, pad_to=608).reshape(512, 608).to(torch.bfloat16).float()
    x0 = (im @ pp.patch_w + pp.pos_b.repeat(2, 1)).to(torch.bfloat16).float()
    img = H.host_siglip(pp, im) @ pp.proj_w + pp.proj_b
    ew = cat["vlm_language"].get("model.embed_tokens.weight")
    ew = cat["vlm_language"]["lm_head.weight"] if ew is None else ew
    xv0 = torch.zeros(ps.mt * 32, 2048)
    xv0[:512] = img
    xv0[512:Pr] = ew[tokens[0]].to(torch.bfloat16).float() * (2048 ** 0.5)
    xv0 = xv0.to(torch.bfloat16).float()
    # fp32 references
    ref_s = sig_layer_fp32(pp, pp.sig[0], x0)
    two = H.PrefixParams(**{**pp.__dict__, "vlm": [pp.vlm[0], pp.vlm[0]]})
    ref_v, _ = H.host_vlm(two, xv0, valid, ps, layers=1)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1395712,
                           trace_region_size=FusedConfig.from_env().trace_region_size)
    dev.enable_program_cache()
    try:
        stamp("shipped model (PI05_MEGAKERNEL=off)")
        from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

        model = PI0ModelTTNN(base_config(sh.horizon), wl, dev)
        model.sample_actions_fused(images, tokens, noise, lmask)
        attn_in = model._fused_attn_in()
        xt = ttnn.from_torch(x0.reshape(2, 256, 1152), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.L1_MEMORY_CONFIG)
        ys = model.backbone.vision_tower.blocks[0].forward_fused(xt)
        tt_s = ttnn.to_torch(ys).float().reshape(512, 1152)
        ttnn.deallocate(ys)
        S = model.backbone.kv_cache_plan["prefix_len"]
        assert S == Pr, (S, Pr)
        xvt = ttnn.from_torch(xv0[:S].reshape(1, S, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        k0, v0 = model.backbone.kv_caches[0]
        yv = model.backbone.vlm_blocks[0].forward_fused_vlm(xvt, k0, v0, attn_mask=attn_in["vlm_mask"])
        tt_v = ttnn.to_torch(yv).float().reshape(S, 2048)
        ttnn.deallocate(yv)
        emb = model.backbone.vlm_embed_tokens
        model.release_trace()
        model._fused_release_inputs()
        for k, v in model.backbone.kv_caches:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        model.backbone.kv_caches = None
        model.backbone.__dict__.get("_kv_sets", {}).clear()
        stamp("prefix engine")
        rig = Rig(dev, "base", pp, emb)
        rig.put(rig.t.x_s, x0, ttnn.float32)
        rig.run(P.OP_S0, P.OP_S0 + 7)
        pe_s = rig.get(rig.t.x_s)
        rig.put(rig.t.vmask, H.vlm_key_mask(valid, ps), ttnn.bfloat16)
        rig.put(rig.t.x_v, xv0, ttnn.float32)
        rig.run(P.OP_V0, P.OP_V0 + 7)
        pe_v = rig.get(rig.t.x_v)[:S]
    finally:
        ttnn.close_device(dev)
    vr = valid.bool()
    for name, pe, tt, ref in (("siglip_layer0", pe_s, tt_s, ref_s), ("vlm_layer0", pe_v[vr], tt_v[vr], ref_v[:S][vr])):
        r = {"pcc_engine_vs_ttnn": round(pcc(pe, tt), 7), "rel_engine_vs_fp32": round(rel(pe, ref), 6),
             "rel_ttnn_vs_fp32": round(rel(tt, ref), 6), "pcc_engine_vs_fp32": round(pcc(pe, ref), 7),
             "pcc_ttnn_vs_fp32": round(pcc(tt, ref), 7)}
        r["pass"] = r["pcc_engine_vs_ttnn"] >= 0.9995 and r["rel_engine_vs_fp32"] <= 1.2 * r["rel_ttnn_vs_fp32"]
        res[name] = r
        stamp(f"{name}: {r}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
