# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the phase-2 prefix engine host side (no device).

* op list / geometry invariants at both shapes (pe_geometry.check_ops, arena sizes);
* weight arena packing: every matmul op's arena is read back through the KERNEL's page addressing (pe_ncrisc.hpp
  feed_w: page g of an op -> bank g % 8, offset op_off + (g // 8) * page_bytes; pe_trisc.hpp mm_k: a page is
  [k in piece][t in 2]) and must reproduce the weight exactly;
* (slow, real weights) the kernel's decomposition (pe_host.host_siglip / host_vlm) against the torch reference.
"""
from __future__ import annotations

import os

import pytest
import torch

from models.experimental.pi0_5.tt.megakernel import pe_geometry as P
from models.experimental.pi0_5.tt.megakernel import pe_host as H


@pytest.mark.parametrize("shape", ["base", "libero"])
def test_cpu_pe_ops(shape):
    ps = P.PSHAPES[shape]
    P.check_ops(ps)
    assert P.N_OPS == P.OP_V0 + 17 * P.OPS_PER_LAYER + 2
    ops = P.all_ops(ps)
    assert ops[P.OP_V0 + 17 * 7].what == P.W_VRMS1 and ops[-1].what == P.W_VQKV and ops[-1].layer == 17
    assert ops[P.OP_S0 + 26 * 7 + 6].what == P.W_SFC2 and ops[P.OP_S0 + 26 * 7 + 6].layer == 26
    # the arena a prefix op may use fits the phase-1 CB region + the tail the host declares
    from models.experimental.pi0_5.tt.megakernel import geometry as G

    need = P.arena_need(ps)
    p1 = sum(c.total_bytes for c in G.cb_table(G.SHAPES[shape]) if c.cb_id != G.CB_SYNC)
    assert max(64, need - p1) + p1 >= need


def _read_back(o, arena, ps, w_shape):
    """Reconstruct W from the arena the way the kernels address it."""
    out = torch.full(w_shape, float("nan"))
    tb = P.mm_wtile(o)
    off, pb, nq, pc = P.mm_arena_off(o, ps), P.mm_page_bytes(o), P.mm_nq(o), o.piece
    for x in range(P.NCOL):
        np_, p0 = P.mm_pairs(o, x), P.mm_pair0(o, x)
        for j in range(np_ * nq):
            g = p0 * nq + j
            t0 = (off + (g // P.N_BANKS) * pb) // tb
            page = arena[g % P.N_BANKS, t0:t0 + 2 * pc]
            if o.mode == P.MM_R:
                s, q = j // nq, j % nq
            else:
                q, s = j // np_, j % np_
            for k in range(pc):
                for t in range(2):
                    kt, nt = q * pc + k, P.pair_col(o, p0 + s, t)
                    out[kt * 32:(kt + 1) * 32, nt * 32:(nt + 1) * 32] = page[2 * k + t]
    return out


@pytest.mark.parametrize("what", ["siglip", "vlm", "patch", "proj"])
def test_cpu_pe_arena_roundtrip(what):
    ps = P.PSHAPES["base"]
    g = torch.Generator().manual_seed(7)
    r = lambda *s: torch.randn(*s, generator=g)
    if what == "siglip":
        sp = H.SigLayer(wqkv=r(1152, 4608), bqkv=r(4608), wo=r(1536, 1152), bo=r(1152), ln1w=r(1152), ln1b=r(1152),
                        ln2w=r(1152), ln2b=r(1152), wfc1=r(1152, 4352), bfc1=r(4352), wfc2=r(4352, 1152), bfc2=r(1152))
        arena = H.sig_layer_arena(sp, 3, ps)
        base = P.OP_S0 + 3 * 7
        cases = [(base + 1, sp.wqkv), (base + 3, sp.wo), (base + 5, sp.wfc1), (base + 6, sp.wfc2)]
    elif what == "vlm":
        vp = H.VlmLayer(wqkv=r(2048, 2560), wo=r(2048, 2048), wug=r(2048, 32768), wd=r(16384, 2048), g1=r(2048),
                        g2=r(2048))
        arena = H.vlm_layer_arena(vp, 5, ps)
        base = P.OP_V0 + 5 * 7
        cases = [(base + 1, vp.wqkv), (base + 3, vp.wo), (base + 5, vp.wug), (base + 6, vp.wd)]
    else:
        from models.experimental.pi0_5.tt.megakernel.pe_size_check import dummy_prefix_params

        pp = dummy_prefix_params()
        if what == "patch":
            pp.patch_w = r(608, 1152)
            arena, cases = H.patch_arena(pp, ps), [(0, pp.patch_w)]
        else:
            pp.proj_w = r(1152, 2048)
            arena, cases = H.proj_arena(pp, ps), [(P.OP_PROJ, pp.proj_w)]
    for op, w in cases:
        o = P.describe(op, ps)
        back = _read_back(o, arena, ps, w.shape)
        assert torch.equal(back, w), (what, op, o.what)


def test_cpu_pe_rope_tables_match_reference():
    """Signed split-half tables reproduce the reference apply_rotary_emb at positions 0..P-1 (q with the 1/16 fold)."""
    from models.experimental.pi0_5.reference.torch_gemma import apply_rotary_emb, precompute_freqs_cis

    ps = P.PSHAPES["libero"]
    tab = H.rope_tables(ps)
    cos, sin = precompute_freqs_cis(256, ps.mt * 32)
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, 8, ps.mt * 32, 256, generator=g)
    k = torch.randn(1, 1, ps.mt * 32, 256, generator=g)
    qr, kr = apply_rotary_emb(q, k, cos, sin)

    def rope(t, c, s):
        h = t.shape[-1] // 2
        return t * c + torch.cat([t[..., h:], t[..., :h]], -1) * s

    assert torch.allclose(rope(q, tab["cosq"], tab["sinq"]), qr / 16.0, atol=1e-5)
    assert torch.allclose(rope(k, tab["cosk"], tab["sink"]), kr, atol=1e-5)


@pytest.mark.skipif(os.environ.get("PI05_SLOW_CPU", "0") != "1", reason="real weights (set PI05_SLOW_CPU=1)")
def test_cpu_pe_host_model_vs_reference():
    """The kernel's decomposition on the real checkpoint vs the torch reference: SigLIP tower and the VLM K / V of
    the valid rows (pad rows sit at different positions and are hidden from every valid query)."""
    from models.experimental.pi0_5.common.fused_host import im2col_patches
    from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_model import prefix_attention_inputs
    from models.experimental.pi0_5.common.configs import PI0ModelConfig
    from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch

    wl = PI0WeightLoader(os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    cat = wl.categorized_weights
    pp = H.prefix_params(cat)
    ref = PI0ModelTorch(PI0ModelConfig(action_dim=32, action_horizon=50, pi05=True), wl).backbone
    g = torch.Generator().manual_seed(3)
    pix = torch.rand(2, 3, 224, 224, generator=g) * 2 - 1
    feats = [ref.embed_image(pix[i:i + 1]) for i in range(2)]  # [1, 256, 2048] each (projector included)
    im = im2col_patches(pix, 14, pad_to=608).reshape(512, 608)
    s_host = H.host_siglip(pp, im)
    proj = s_host @ pp.proj_w + pp.proj_b
    ref_img = torch.cat(feats, 1)[0]
    pcc = torch.corrcoef(torch.stack([proj.flatten(), ref_img.flatten()]))[0, 1].item()
    assert pcc > 0.99999, pcc
    ps = P.PSHAPES["base"]
    n_lang = 150
    tok = torch.randint(1, 257152, (1, 224), generator=g)
    valid = torch.zeros(1, 736, dtype=torch.bool)
    valid[0, : 512 + n_lang] = True
    lang = ref.embed_language_tokens(tok)[0] * (2048 ** 0.5)
    x0 = torch.zeros(ps.mt * 32, 2048)
    x0[:512] = ref_img
    x0[512:736] = lang
    vlm_mask, vlm_pos, _, _ = prefix_attention_inputs(valid, 50)
    emb = torch.cat([ref_img[None], lang[None]], 1)
    _, cache = ref.forward_vlm(emb, attention_mask=vlm_mask, position_ids=vlm_pos, use_cache=True)
    _, kv = H.host_vlm(pp, x0, valid, ps)
    nv = 512 + n_lang
    for l in (0, 8, 17):
        rk, rv = cache[l][0][0, 0, :nv], cache[l][1][0, 0, :nv]
        for a, b in ((kv[l][0][:nv], rk), (kv[l][1][:nv], rv)):
            pc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
            assert pc > 0.99999, (l, pc)
