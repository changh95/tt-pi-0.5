# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CPU tests (no device) of the pi0.5 megakernel's host side.

* the kernel headers parse, the core roles / CB table / DRAM bank plan hold the invariants the kernels rely on;
* the weight arenas are packed in the tile orders the TRISC loops consume (expert) and read back exactly through the
  kernels' page addressing (prefix engine);
* the expert loop's decomposition (folded adaRMS, chunked flash attention, 2-D MLP) equals the plain formulas;
* the attention inputs (key row, RoPE rows) reproduce openpi's pi0.5 attention; adaRMS units;
* the request checks and device refusals of ``PI05MegakernelTTNN``;
* (real weights, when available) the prefix engine's decomposition equals the torch reference.

    pytest models/experimental/pi0/tests/pcc/test_pi05_megakernel_host.py
"""

import dataclasses
import math
import os
import re
import types

import pytest
import torch

from models.experimental.pi0.common import pi05_host as ph
from models.experimental.pi0.common.configs import GemmaConfig
from models.experimental.pi0.reference.torch_gemma import (
    GemmaBlock,
    adarms_norm,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from models.experimental.pi0.tt.megakernel import arena as A
from models.experimental.pi0.tt.megakernel import geometry as G
from models.experimental.pi0.tt.megakernel import host_model as hm
from models.experimental.pi0.tt.megakernel import pe_geometry as P
from models.experimental.pi0.tt.megakernel import pe_host as H
from models.experimental.pi0.tt.megakernel import presets as PS

PI05_BASE_WEIGHTS = os.environ.get("PI05_BASE_WEIGHTS", "lerobot/pi05_base")


# ============================================================================ expert loop geometry
def test_kernel_defs_parse_and_dense():
    d = G.parse_defs()
    cbs = sorted(v for k, v in d.items() if k.startswith("CB_"))
    assert cbs == list(range(d["N_CBS"]))
    words = [v for k, v in d.items() if k.startswith("S_") and k not in ("S_SRC0", "S_DBG", "SYNC_STRIDE")]
    assert len(set(words)) == len(words)
    assert d["SYNC_STRIDE"] == 16  # every sync word on its own 16 B line


@pytest.mark.parametrize("name", ["base", "libero"])
def test_core_roles(name):
    G.check_roles(G.SHAPES[name])


@pytest.mark.parametrize("name", ["base", "libero"])
def test_cb_full_capacity_cycles(name):
    """Landing / staging CBs: every transaction the kernels issue equals the CB capacity (pointers return to the
    base), and multi-page ring waits never straddle a ring end."""
    sh = G.SHAPES[name]
    t = {c.cb_id: c for c in G.cb_table(sh)}
    rt, nch, cht = sh.rt, sh.nch, sh.chunk_tiles
    full = {
        G.CB_IN0: 64 * rt, G.CB_Q: 8 * rt, G.CB_KSV: 16 * rt, G.CB_KV: 2 * cht * 8, G.CB_MASK: cht,
        G.CB_PART: 8 * (nch - 1), G.CB_PM: nch - 1, G.CB_PL: nch - 1, G.CB_QKVO: 2 * rt, G.CB_XS: rt,
        G.CB_H: 2 * rt, G.CB_HG: 16 * rt, G.CB_DP: 4 * rt, G.CB_RED: 8 * rt, G.CB_CTXO: 8, G.CB_OP: 8,
        G.CB_XRES: rt, G.CB_ROUT: rt, G.CB_SCR16: rt, G.CB_CONST: 4, G.CB_TAB: 4 * rt, G.CB_D: nch,
    }  # fmt: skip
    for cb, pages in full.items():
        assert t[cb].pages == pages, (t[cb].name, t[cb].pages, pages)
    # H0 x_new: row 0 in CB_PART (>= 32 tiles), row 1 in CB_HG (32 tiles at RT = 2)
    assert t[G.CB_PART].pages >= 32 and (rt == 1 or t[G.CB_HG].pages >= 32)
    # rings: W8 16 pages consumed in 8-page (qkv) and 2-page (up|gate) waits; W16 page by page; H0 16-page steps
    assert G.W8_PAGES % 8 == 0 and G.W16_PAGES == 8
    h0 = G.core_streams(G.build_roles(sh)[G.H0], sh)["w16"]
    assert sum(p for _, p in h0) % G.W16_PAGES == 0


@pytest.mark.parametrize("name", ["base", "libero"])
def test_bank_plan_covers_streams(name):
    sh = G.SHAPES[name]
    plan = G.plan_banks(sh)
    for xy, p in plan.pages.items():
        assert plan.off8[xy] + p["w8"] * 8 * 1088 <= plan.tiles8_per_bank * 1088
        assert plan.offc[xy] + p["wc"] * 8 * 2048 <= plan.tiles16_per_bank * 2048
    # the streams of different cores in one bank never overlap
    for key in ("off8", "off16"):
        spans = {}
        for xy, b in plan.bank.items():
            size = plan.pages[xy]["w8" if key == "off8" else "w16"] * 8 * (1088 if key == "off8" else 2048)
            spans.setdefault(b, []).append((getattr(plan, key)[xy], size))
        for sp in spans.values():
            sp = sorted(s for s in sp if s[1])
            for (o0, s0), (o1, _) in zip(sp, sp[1:]):
                assert o0 + s0 <= o1


def test_share_orders_match_trisc_loops():
    """The TRISC reads qkv tile (2*kb + c)*8 + k, up|gate (jj, kb) pages [u | g], o_proj page kb tile k, down page
    (kb, n) tile k: the packers must produce exactly those orders."""
    w = torch.arange(1024 * 2560, dtype=torch.float32).reshape(1024, 2560)
    t = A.w_tiles(w)
    share = A.qkv_share(t, 5, 9)
    for kb in range(4):
        for c, col in enumerate((5, 9)):
            for k in range(8):
                assert torch.equal(share[(2 * kb + c) * 8 + k], t[kb * 8 + k, col])
    tu = A.w_tiles(torch.randn(1024, 8192))
    ug = A.ug_share(tu, 17, 18)
    for jj, j in enumerate((17, 18)):
        for kb in range(4):
            base = (jj * 4 + kb) * 16
            for k in range(8):
                assert torch.equal(ug[base + k], tu[kb * 8 + k, j])
                assert torch.equal(ug[base + 8 + k], tu[kb * 8 + k, 128 + j])
    td = A.w_tiles(torch.randn(4096, 1024))
    dn = A.wd_share(td, 3, 5)
    for kb in range(2):
        for n in range(4):
            for k in range(8):
                assert torch.equal(dn[(kb * 4 + n) * 8 + k], td[3 * 16 + kb * 8 + k, 4 * 5 + n])


def test_shape_contract_and_step_refusal(expect_error):
    assert G.shape_for(736, 64).name == "base" and G.shape_for(544, 32).name == "libero"
    with expect_error(RuntimeError, "no program for prefix 768"):
        G.shape_for(768, 64)
    assert G.N_STEPS == 10
    for n in range(1, 11):
        assert G.megakernel_refusal(n) is None
    for n in (0, 11, 20):
        assert "runs 1..10 denoising steps" in G.megakernel_refusal(n)
    for h in (1, 10, 32, 33, 50, 64):
        assert G.megakernel_refusal(10, h) is None
    for h in (0, 65, 100):
        assert "1..64 action rows" in G.megakernel_refusal(10, h)


def test_merge_dst_budget_and_multicast_counts(expect_error):
    """NCH <= 6 (merge(): M / L in DST 0, weights in DST 1..NCH, temporary in DST 7); the kernel's multicast
    destination counts equal the rectangle-derived counts. Positive controls: a doctored count and NCH = 7 fail."""
    for s in G.SHAPES.values():
        assert s.nch <= 6
    with expect_error(ValueError, "NCH <= 6"):
        G.Shape("nch7", prefix_len=32 * 7 * 3 - 32, suffix_rows=32, chunk_tiles=3).check()
    brisc = open(os.path.join(G.KDIR, "mk_brisc.cpp")).read()
    for name, lit in (("rm", "64"), ("ro", "32"), ("row", "7")):
        pat = re.compile(r"(me\." + name + r" = \{[^;]*?,\s*)" + lit + r"\};")
        assert pat.search(brisc), name
        bad = pat.sub(lambda m: m.group(1) + str(int(lit) + 1) + "};", brisc, count=1)
        c = G.kernel_mcast_counts(G.SHAPES["base"], bad)
        lo, hi, snd = G.mcast_rects(G.SHAPES["base"])[name]
        assert c[name] != G.mcast_dest_count(lo, hi, snd[0])


def _synthetic_expert_params(seed: int = 0, n_steps: int = hm.N_STEPS) -> hm.ExpertParams:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc
    W, D, NH, M, L = hm.WIDTH, hm.DH, hm.NH, hm.MLP, hm.N_LAYERS
    mods = [
        [(r(W, sc=0.1), r(W, sc=0.1), r(W, sc=0.3), r(W, sc=0.1), r(W, sc=0.1), r(W, sc=0.3)) for _ in range(L)]
        for _ in range(n_steps)
    ]
    return hm.ExpertParams(
        wqkv=[r(W, (NH + 2) * D, sc=W**-0.5) for _ in range(L)],
        wo=[r(NH * D, W, sc=(NH * D) ** -0.5) for _ in range(L)],
        wug=[r(W, 2 * M, sc=W**-0.5) for _ in range(L)],
        wd=[r(M, W, sc=M**-0.5) for _ in range(L)],
        mods=mods,
        final=[(r(W, sc=0.1), r(W, sc=0.1)) for _ in range(n_steps)],
        w_in=r(32, W, sc=32**-0.5),
        b_in=r(W, sc=0.1),
        w_out=r(W, 32, sc=W**-0.5),
        b_out=r(32, sc=0.1),
        eps=1e-6,
        dts=tuple([-1.0 / n_steps] * n_steps),
    )


@pytest.mark.parametrize("name, n_steps", [("base", 10), ("libero", 10), ("base", 1), ("libero", 5)])
def test_expert_decomposition_equals_reference_loop(name, n_steps):
    """host_model.loop_decomposed (the kernel's folds, chunked flash parts + diag merge, 2-D MLP, K-split reduce)
    equals loop_reference (plain fp32 formulas) over the whole N x 18 loop, with a padded prompt. Positive control:
    dropping one prefix chunk's keys from the decomposed arm only must move the output well past the tolerance."""
    sh = G.SHAPES[name]
    p = _synthetic_expert_params(n_steps=n_steps)
    g = torch.Generator().manual_seed(1)
    Pn, S, Hn = sh.prefix_len, sh.suffix_rows, {"base": 50, "libero": 10}[name]
    kv = [(torch.randn(Pn, hm.DH, generator=g) * 2, torch.randn(Pn, hm.DH, generator=g)) for _ in range(hm.N_LAYERS)]
    mask = torch.zeros(Pn + S)
    mask[Pn - 100 : Pn] = -1e9  # padded prompt: masked prefix keys
    mask[Pn + Hn :] = -1e9  # tile-pad action rows as keys
    ang = torch.arange(S).float()[:, None] * (1.0 / 10000 ** (torch.arange(0, hm.DH, 2).float() / hm.DH))[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1)
    sin = torch.cat([-ang.sin(), ang.sin()], -1)
    a = hm.AttnInputs(mask=mask, cosq=cos / 16, sinq=sin / 16, cosk=cos, sink=sin)
    noise = torch.zeros(S, 32)
    noise[:Hn] = torch.randn(Hn, 32, generator=g)
    ref = hm.loop_reference(p, kv, a, noise)[:Hn]
    dec = hm.loop_decomposed(p, kv, a, noise, sh.chunk_tiles)[:Hn]
    assert torch.isfinite(ref).all() and ref.abs().max() > 0.1
    err = float((ref - dec).abs().max() / ref.abs().max())
    assert hm.pcc(ref, dec) > 0.999999 and err < 1e-4, (hm.pcc(ref, dec), err)
    m2 = mask.clone()
    m2[: sh.chunk_tiles * 32] = -1e9  # chunk 0's keys removed in the decomposed arm only
    bad = hm.loop_decomposed(p, kv, hm.AttnInputs(m2, a.cosq, a.sinq, a.cosk, a.sink), noise, sh.chunk_tiles)[:Hn]
    assert float((ref - bad).abs().max() / ref.abs().max()) > 100 * max(err, 1e-6)


# ============================================================================ prefix engine
@pytest.mark.parametrize("shape", ["base", "libero"])
def test_prefix_ops(shape):
    ps = P.PSHAPES[shape]
    P.check_ops(ps)
    assert P.N_OPS == P.OP_V0 + 17 * P.OPS_PER_LAYER + 2
    ops = P.all_ops(ps)
    assert ops[P.OP_V0 + 17 * 7].what == P.W_VRMS1 and ops[-1].what == P.W_VQKV and ops[-1].layer == 17
    assert ops[P.OP_S0 + 26 * 7 + 6].what == P.W_SFC2 and ops[P.OP_S0 + 26 * 7 + 6].layer == 26
    # the arena a prefix op may use fits the expert CB region + the tail the host declares
    need = P.arena_need(ps)
    p1 = sum(c.total_bytes for c in G.cb_table(G.SHAPES[shape]) if c.cb_id != G.CB_SYNC)
    assert max(64, need - p1) + p1 >= need


def _read_back(o, arena, ps, w_shape):
    """Reconstruct W from the arena the way the kernels address it (pe_ncrisc.hpp: page g of an op -> bank g % 8 at
    op_off + (g // 8) * page_bytes; pe_trisc.hpp: a page is [k in piece][t in 2])."""
    out = torch.full(w_shape, float("nan"))
    tb = P.mm_wtile(o)
    off, pb, nq, pc = P.mm_arena_off(o, ps), P.mm_page_bytes(o), P.mm_nq(o), o.piece
    for x in range(P.NCOL):
        np_, p0 = P.mm_pairs(o, x), P.mm_pair0(o, x)
        for j in range(np_ * nq):
            g = p0 * nq + j
            t0 = (off + (g // P.N_BANKS) * pb) // tb
            page = arena[g % P.N_BANKS, t0 : t0 + 2 * pc]
            s, q = (j // nq, j % nq) if o.mode == P.MM_R else (j % np_, j // np_)
            for k in range(pc):
                for t in range(2):
                    kt, nt = q * pc + k, P.pair_col(o, p0 + s, t)
                    out[kt * 32 : (kt + 1) * 32, nt * 32 : (nt + 1) * 32] = page[2 * k + t]
    return out


@pytest.mark.parametrize("what", ["siglip", "vlm", "patch", "proj"])
def test_prefix_arena_roundtrip(what):
    from models.experimental.pi0.tt.megakernel.size_check import dummy_prefix_params

    ps = P.PSHAPES["base"]
    g = torch.Generator().manual_seed(7)
    r = lambda *s: torch.randn(*s, generator=g)
    if what == "siglip":
        sp = H.SigLayer(wqkv=r(1152, 4608), bqkv=r(4608), wo=r(1536, 1152), bo=r(1152), ln1w=r(1152), ln1b=r(1152),
                        ln2w=r(1152), ln2b=r(1152), wfc1=r(1152, 4352), bfc1=r(4352), wfc2=r(4352, 1152), bfc2=r(1152))  # fmt: skip
        arena = H.sig_layer_arena(sp, 3, ps)
        base = P.OP_S0 + 3 * 7
        cases = [(base + 1, sp.wqkv), (base + 3, sp.wo), (base + 5, sp.wfc1), (base + 6, sp.wfc2)]
    elif what == "vlm":
        vp = H.VlmLayer(wqkv=r(2048, 2560), wo=r(2048, 2048), wug=r(2048, 32768), wd=r(16384, 2048), g1=r(2048),
                        g2=r(2048))  # fmt: skip
        arena = H.vlm_layer_arena(vp, 5, ps)
        base = P.OP_V0 + 5 * 7
        cases = [(base + 3, vp.wo), (base + 5, vp.wug), (base + 6, vp.wd)]
        o = P.describe(base + 1, ps)
        assert torch.equal(_read_back(o, H.vlm_qkv_arena(vp, 5, ps), ps, vp.wqkv.shape), vp.wqkv)
    else:
        pp = dummy_prefix_params()
        if what == "patch":
            pp.patch_w = r(608, 1152)
            arena, cases = H.patch_arena(pp, ps), [(0, pp.patch_w)]
        else:
            pp.proj_w = r(1152, 2048)
            arena, cases = H.proj_arena(pp, ps), [(P.OP_PROJ, pp.proj_w)]
    for op, w in cases:
        o = P.describe(op, ps)
        assert torch.equal(_read_back(o, arena, ps, w.shape), w), (what, op, o.what)


def test_prefix_rope_tables_match_reference():
    """Signed split-half tables reproduce the reference apply_rotary_emb at positions 0..P-1 (q with the 1/16 fold)."""
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


# ============================================================================ attention inputs, adaRMS
def test_im2col_is_the_patch_embedding_order():
    """im2col rows @ the (kh, kw, c) patch weight == the SigLIP conv2d patch embedding."""
    x = torch.randn(2, 3, 224, 224, dtype=torch.float64)
    w = torch.randn(1152, 3, 14, 14, dtype=torch.float64)
    conv = torch.nn.functional.conv2d(x, w, stride=14).flatten(2).transpose(1, 2)  # [2, 256, 1152]
    im = ph.im2col_patches(x, 14, pad_to=608)
    assert im.shape == (2, 256, 608) and torch.count_nonzero(im[..., 588:]) == 0
    lin = im[..., :588] @ w.permute(0, 2, 3, 1).reshape(1152, -1).T
    assert torch.allclose(lin, conv, atol=1e-9)


def test_expert_attention_inputs_match_openpi_attention():
    """The key row and RoPE rows (rotate-half sign and 1/sqrt(dh) folded) reproduce openpi's attention for the action
    tokens: positions n_valid + [0, H), attending the valid prefix keys and the H real action rows (fp64, two
    right-padded prompts)."""
    Hn, dh, heads = 50, 256, 8
    plan = ph.kv_cache_plan(2 * 256 + 224, Hn)
    Pn, S = plan["prefix_len"], plan["suffix_rows"]
    assert plan["cache_len"] == 800 and ph.kv_cache_plan(544, 10)["cache_len"] == 640
    lang = torch.zeros(2, 224, dtype=torch.bool)
    lang[0, :40] = True
    lang[1, :97] = True
    valid = ph.prefix_valid_mask(2, lang)
    c_half, s_half = precompute_freqs_cis(dh, 2048)
    cos_full, sin_full = torch.cat([c_half, c_half], -1), torch.cat([s_half, s_half], -1)
    att = ph.attention_inputs(valid, plan, cos_full, sin_full, 1.0 / math.sqrt(dh))
    assert att["n_valid"].tolist() == [552, 609]
    f64 = torch.float64
    q = torch.randn(2, heads, S, dh, dtype=f64)
    k = torch.randn(2, 1, S, dh, dtype=f64)
    v = torch.randn(2, 1, S, dh, dtype=f64)
    kp = torch.randn(2, 1, Pn, dh, dtype=f64)  # the cache prefix (already rotated by the VLM)
    vp = torch.randn(2, 1, Pn, dh, dtype=f64)
    swap = lambda x: torch.cat([x[..., dh // 2 :], x[..., : dh // 2]], dim=-1)
    qr = q * att["cosq"].double() + swap(q) * att["sinq"].double()
    kr = k * att["cosk"].double() + swap(k) * att["sink"].double()
    scores = qr @ torch.cat([kp, kr], 2).transpose(-1, -2) + att["exp_mask"].double()[:, :, :1]
    fused = torch.softmax(scores, -1) @ torch.cat([vp, v], 2)

    pos = att["n_valid"][:, None] + torch.arange(Hn)[None]
    qh, kh = apply_rotary_emb(q[:, :, :Hn], k[:, :, :Hn], c_half.double(), s_half.double(), pos)
    mask = torch.cat([torch.where(valid, 0.0, -1e9).double(), torch.zeros(2, Hn, dtype=f64)], 1)[:, None, None]
    ref_scores = qh @ torch.cat([kp, kh], 2).transpose(-1, -2) / math.sqrt(dh) + mask
    ref = torch.softmax(ref_scores, -1) @ torch.cat([vp, v[:, :, :Hn]], 2)
    assert torch.allclose(fused[:, :, :Hn], ref, rtol=1e-10, atol=1e-10)


def test_attention_inputs_reject_bad_prefixes(expect_error):
    plan = ph.kv_cache_plan(544, 10)
    cos = torch.zeros(2048, 256)
    with expect_error(ValueError, "at least one valid prefix token"):
        ph.attention_inputs(torch.zeros(1, 544, dtype=torch.bool), plan, cos, cos, 1.0)
    with expect_error(ValueError, "512 prefix tokens"):
        ph.attention_inputs(torch.ones(1, 512, dtype=torch.bool), plan, cos, cos, 1.0)


def test_adarms_zero_modulation_is_rmsnorm_with_zero_gate():
    x = torch.randn(2, 5, 64)
    cond = torch.randn(2, 32)
    normed, gate = adarms_norm(x, torch.zeros(192, 32), torch.zeros(192), cond)
    assert torch.allclose(normed, x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6), atol=1e-5)
    assert gate.shape == (2, 1, 64) and torch.count_nonzero(gate) == 0
    normed2, gate2 = adarms_norm(x, torch.randn(192, 32) * 0.1, torch.zeros(192), cond)
    assert not torch.allclose(normed, normed2) and torch.count_nonzero(gate2) > 0


def test_expert_block_zero_adarms_is_identity():
    """gate = 0 in both gated residuals: the adaRMS expert block (pi0.5) returns its input."""
    cfg = GemmaConfig.gemma_300m(use_adarms=True)
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g) * 0.02
    w = {
        "self_attn.q_proj.weight": r(cfg.num_heads * cfg.head_dim, cfg.width),
        "self_attn.k_proj.weight": r(cfg.head_dim, cfg.width),
        "self_attn.v_proj.weight": r(cfg.head_dim, cfg.width),
        "self_attn.o_proj.weight": r(cfg.width, cfg.num_heads * cfg.head_dim),
        "mlp.gate_proj.weight": r(cfg.mlp_dim, cfg.width),
        "mlp.up_proj.weight": r(cfg.mlp_dim, cfg.width),
        "mlp.down_proj.weight": r(cfg.width, cfg.mlp_dim),
    }
    for n in ("input_layernorm", "post_attention_layernorm"):
        w[f"{n}.dense.weight"] = torch.zeros(3 * cfg.width, cfg.adarms_cond_dim)
        w[f"{n}.dense.bias"] = torch.zeros(3 * cfg.width)
    cos, sin = precompute_freqs_cis(cfg.head_dim, 4, cfg.rope_base)
    x = torch.randn(1, 4, cfg.width)
    out, _ = GemmaBlock(cfg, w, layer_idx=0).forward(x, cos, sin, adarms_cond=torch.randn(1, cfg.adarms_cond_dim))
    assert torch.allclose(out, x, atol=1e-5)
    assert not GemmaConfig.gemma_300m().use_adarms  # pi0 keeps the plain RMSNorm expert


# ============================================================================ presets and the program split
def test_presets_and_split():
    from models.experimental.pi0.tt.megakernel.pe_program import PREFIX_OPS, VISION_OPS

    for p in PS.PRESETS.values():
        p.check()
        assert p.prefix_len == 256 * p.cameras + p.prompt_len
    assert [p.key for p in PS.presets_for(2, 64)] == [(2, 32, 64), (2, 64, 64), (2, 128, 64), (2, 224, 64)]
    assert [p.key for p in PS.presets_for(2, 32)] == [(2, 32, 32), (2, 64, 32), (2, 128, 32), (2, 224, 32)]
    assert [p.key for p in PS.presets_for(1, 32)] == [(1, 32, 32), (1, 64, 32), (1, 128, 32), (1, 224, 32)]
    assert [p.key for p in PS.presets_for(3, 64)] == [(3, 32, 64), (3, 64, 64), (3, 128, 64), (3, 224, 64)]
    assert [p.key for p in PS.presets_for(4, 32)] == [(4, 32, 32), (4, 64, 32), (4, 128, 32), (4, 224, 32)]
    assert PS.presets_for(5, 32) == []
    # the two shapes the single program had are the rules' output
    assert PS.PRESETS[(2, 224, 64)].shape.chunk_tiles == G.SHAPES["base"].chunk_tiles
    assert PS.PRESETS[(2, 32, 32)].shape.chunk_tiles == G.SHAPES["libero"].chunk_tiles
    for name, key in (("base", (2, 224, 64)), ("libero", (2, 32, 32))):
        p, ps = PS.PRESETS[key], P.PSHAPES[name]
        assert (p.pshape.mt, p.pshape.ptv, p.pshape.rv, p.pshape.lt, p.pshape.ntok) == (
            ps.mt,
            ps.ptv,
            ps.rv,
            ps.lt,
            ps.ntok,
        )
        assert p.shape.prefix_len == G.SHAPES[name].prefix_len
    a = PS.alloc_pshape([p for p in PS.PRESETS.values() if p.cameras <= 2])
    assert (a.mt, a.ptv, a.ntok) == (24, 23, 224)
    a = PS.alloc_pshape(list(PS.PRESETS.values()))
    assert (a.mt, a.ptv, a.ntok, a.img) == (40, 39, 224, 4)
    # SigLIP in groups of <= 2 cameras (one VISION program each)
    assert [P.vision_groups(c) for c in (1, 2, 3, 4)] == [[(0, 1)], [(0, 2)], [(0, 2), (2, 1)], [(0, 2), (2, 2)]]
    # vision | prefix cover the op list in order: the projector (vision) writes the image rows of x_v, the language
    # embedding is the first prefix op
    assert VISION_OPS[0] == 0 and VISION_OPS[1] == PREFIX_OPS[0] == P.OP_EMBED and PREFIX_OPS[1] == P.N_OPS
    assert P.describe(VISION_OPS[1] - 1, P.PSHAPES["base"]).what == P.W_PROJ
    for ps in P.PSHAPES.values():  # each program reserves the arena of its own ops
        v, pr = P.arena_need(ps, VISION_OPS), P.arena_need(ps, PREFIX_OPS)
        assert max(v, pr) == P.arena_need(ps) and min(v, pr) < P.arena_need(ps)
    # the split programs run the prefix engine only: the expert loop is reachable in whole_*.cpp only through
    # PA_EXPERT, inside the TRISC's common-arg list
    assert G.N_COMMON_ARGS <= P.PA_EXPERT < P.PA0 <= P.PA_TRISC_N
    for r in ("brisc", "ncrisc", "trisc"):
        src = open(os.path.join(P.KDIR2, f"whole_{r}.cpp")).read()
        body = src[src.index("void kernel_main()") :]
        assert re.search(r"if \(get_common_arg_val<uint32_t>\(pe::PA_EXPERT\)\) \{\s*mk_expert_kernel_main\(\);", body)


# ============================================================================ PI05MegakernelTTNN host checks
def _host_only_model(horizon: int):
    """A PI05MegakernelTTNN with only the attributes host_inputs reads (no device, no weights)."""
    from models.experimental.pi0.common.configs import PI0ModelConfig
    from models.experimental.pi0.tt.ttnn_pi05_model import PI05MegakernelTTNN

    m = object.__new__(PI05MegakernelTTNN)
    m.config = PI0ModelConfig(action_horizon=horizon, pi05=True)
    m.cameras = 2
    m.horizon = horizon
    m.suffix_rows = ph.round_up(horizon)
    m.presets = PS.presets_for(2, m.suffix_rows)
    m.default_noise = torch.zeros(1, horizon, 32)
    return m


def test_request_checks(expect_error):
    m = _host_only_model(10)
    images = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(2)]
    tokens = torch.zeros(1, 32, dtype=torch.long)
    tokens[0, :12] = torch.arange(1, 13)
    host = m.host_inputs(images, None, tokens, None, None)
    assert host["preset"].key == (2, 32, 32)
    assert host["im2col"].shape == (2, 256, 608) and host["noise"].shape == (1, 32, 32)
    assert host["valid"].shape == (1, 544) and int(host["valid"].sum()) == 512 + 12
    with expect_error(RuntimeError, "serves 2 cameras"):
        m.host_inputs(images[:1], None, tokens, None, None)
    with expect_error(RuntimeError, "drop the masked image slots and pass only the real cameras"):
        m.host_inputs(images, [torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)], tokens, None, None)
    with expect_error(RuntimeError, "serves batch 1"):
        m.host_inputs(images, None, tokens.repeat(2, 1), None, None)
    with expect_error(RuntimeError, "225 real tokens"):
        m.host_inputs(images, None, torch.ones(1, 225, dtype=torch.long), None, None)
    left_padded = torch.zeros(1, 32, dtype=torch.bool)
    left_padded[0, 20:] = True
    with expect_error(ValueError, "right-padded"):
        m.host_inputs(images, None, tokens, left_padded, None)
    assert [p.prompt_len for p in _host_only_model(50).presets] == [32, 64, 128, 224]


@pytest.mark.parametrize("horizon", [1, 10, 32, 33, 50, 64])
def test_request_horizons(horizon):
    """Any H in 1..64: the suffix bucket (32 / 64 rows) picks the presets, the noise is zero-padded to the bucket and
    the expert key row masks the pad action rows (keys >= H)."""
    m = _host_only_model(horizon)
    bucket = 32 if horizon <= 32 else 64
    assert m.suffix_rows == bucket and {p.suffix_rows for p in m.presets} == {bucket}
    prompt_len = m.presets[0].prompt_len
    images = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(2)]
    tokens = torch.zeros(1, prompt_len, dtype=torch.long)
    tokens[0, :5] = torch.arange(1, 6)
    noise = torch.randn(1, horizon, 32)
    host = m.host_inputs(images, None, tokens, None, noise)
    assert host["noise"].shape == (1, bucket, 32)
    assert torch.equal(host["noise"][0, :horizon], noise[0]) and not host["noise"][0, horizon:].any()
    p = host["preset"]
    plan = ph.kv_cache_plan(p.prefix_len, horizon)
    rope = torch.zeros(2048, 256)
    att = ph.attention_inputs(host["valid"], plan, rope, rope, 1.0, prefix_keys=p.prefix_keys)
    assert (att["exp_mask"][0, 0, 0, p.prefix_len : p.prefix_keys] == ph.MASK_NEG).all()
    suffix = att["exp_mask"][0, 0, 0, p.prefix_keys :]
    assert suffix.shape == (bucket,) and not suffix[:horizon].any() and (suffix[horizon:] == ph.MASK_NEG).all()


def test_prompt_buckets(expect_error):
    """A request runs in the smallest prompt bucket holding its real tokens; the token tensor is truncated (its tail is
    padding) or padded with masked <pad> ids; ``prompt_bucket=`` overrides the bucket; the expert chunking per preset is
    the design table's."""
    m = _host_only_model(50)
    images = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(2)]
    for n_real, length, bucket in (
        (12, 224, 32),
        (32, 32, 32),
        (33, 224, 64),
        (97, 224, 128),
        (150, 160, 224),
        (3, 3, 32),
    ):
        tokens = torch.zeros(1, length, dtype=torch.long)
        tokens[0, :n_real] = torch.arange(1, n_real + 1)
        host = m.host_inputs(images, None, tokens, None, None)
        assert host["preset"].prompt_len == bucket and host["tokens"].shape == (1, bucket)
        assert host["valid"].shape == (1, 512 + bucket) and int(host["valid"].sum()) == 512 + n_real
    # an explicit bucket overrides the routing; it must hold the prompt and be compiled
    tokens = torch.zeros(1, 224, dtype=torch.long)
    tokens[0, :12] = torch.arange(1, 13)
    assert m.host_inputs(images, None, tokens, None, None, prompt_bucket=224)["preset"].prompt_len == 224
    assert m.host_inputs(images, None, tokens, None, None, prompt_bucket=64)["tokens"].shape == (1, 64)
    tokens[0, :40] = torch.arange(1, 41)
    with expect_error(RuntimeError, "cannot hold 40 real tokens"):
        m.host_inputs(images, None, tokens, None, None, prompt_bucket=32)
    with expect_error(RuntimeError, "prompt_bucket=100"):
        m.host_inputs(images, None, tokens, None, None, prompt_bucket=100)
    # (cht, nch, padded prefix key tiles) per c = 2 preset (MULTICONFIG_DESIGN.md table 4.2)
    table = {
        (32, 32): (3, 6, 0),
        (32, 64): (4, 5, 1),
        (64, 32): (4, 5, 1),
        (64, 64): (4, 5, 0),
        (128, 32): (4, 6, 3),
        (128, 64): (5, 5, 3),
        (224, 32): (4, 6, 0),
        (224, 64): (5, 5, 0),
    }
    table1 = {
        (32, 32): (2, 5, 0),
        (32, 64): (3, 4, 1),
        (64, 32): (2, 6, 1),
        (64, 64): (3, 4, 0),
        (128, 32): (3, 5, 2),
        (128, 64): (3, 5, 1),
        (224, 32): (3, 6, 2),
        (224, 64): (4, 5, 3),
    }
    mt1 = {32: (10, 5), 64: (10, 5), 128: (12, 6), 224: (16, 8)}  # c = 1: (mt, bands) with 2 row tiles per band
    for (l, s), v in table1.items():
        p = PS.PRESETS[(1, l, s)]
        assert (p.shape.chunk_tiles, p.shape.nch, p.shape.pt - p.pshape.ptv) == v, p.key
        assert (p.pshape.mt, p.pshape.rv, p.pshape.img, p.pshape.s_m) == (*mt1[l], 1, 8), p.key
    for (l, s), (cht, nch, pad) in table.items():
        p = PS.PRESETS[(2, l, s)]
        assert (p.shape.chunk_tiles, p.shape.nch, p.shape.pt - p.pshape.ptv) == (cht, nch, pad), p.key
        assert p.cache_rows >= max(p.pshape.mt, p.shape.nkt) * 32


@pytest.mark.parametrize("key", [(2, 128, 32), (2, 64, 32)])
def test_padded_prefix_keys_are_exact(key):
    """The expert's prefix padded to its chunking (masked keys over finite cache rows) equals the unpadded loop."""
    p = PS.PRESETS[key]
    sh, S, Hn = p.shape, p.suffix_rows, 10
    params = _synthetic_expert_params(n_steps=2)
    g = torch.Generator().manual_seed(4)
    n = p.prefix_len
    kv = [(torch.randn(n, hm.DH, generator=g), torch.randn(n, hm.DH, generator=g)) for _ in range(hm.N_LAYERS)]
    junk = [(torch.randn(p.prefix_keys - n, hm.DH, generator=g) * 3,) * 2 for _ in range(hm.N_LAYERS)]
    kvp = [(torch.cat([k, j[0]]), torch.cat([v, j[1]])) for (k, v), j in zip(kv, junk)]
    valid = torch.ones(1, n, dtype=torch.bool)
    valid[0, n - 40 :] = False
    rope = torch.ones(4096, 256) * 0.5
    att = ph.attention_inputs(valid, ph.kv_cache_plan(n, Hn), rope, rope, 1.0 / 16, prefix_keys=p.prefix_keys)
    a = hm.attn_inputs_from(att)
    att0 = ph.attention_inputs(valid, ph.kv_cache_plan(n, Hn), rope, rope, 1.0 / 16)
    noise = torch.zeros(S, 32)
    noise[:Hn] = torch.randn(Hn, 32, generator=g)
    ref = hm.loop_reference(params, kv, hm.attn_inputs_from(att0), noise)[:Hn]
    dec = hm.loop_decomposed(params, kvp, a, noise, sh.chunk_tiles)[:Hn]
    assert hm.pcc(ref, dec) > 0.999999 and float((ref - dec).abs().max() / ref.abs().max()) < 1e-4


@pytest.mark.parametrize("key", sorted(PS.PRESETS))
def test_preset_prefix_ops(key):
    """Every preset's prefix op list: kernel invariants, item counts <= one per core, the arena of each program."""
    p = PS.PRESETS[key]
    ps = p.pshape
    P.check_ops(ps)
    ops = P.all_ops(ps)
    # VLM norms: 4 column groups always; two rounds over 108 cores once mt x 4 items exceed the cores (c = 3)
    assert ps.ncg == P.NCG_V and ps.mt * ps.ncg <= ps.norm_rounds * ps.norm_cores
    assert (ps.norm_rounds, ps.norm_cores) == ((2, 108) if p.cameras >= 3 else (1, P.NCORES))
    assert ops[P.OP_V0].items == ps.mt * ps.ncg and ops[P.OP_EMBED].items == (ps.lt + ps.mt - ps.ptv) * 4
    assert ops[P.OP_V0 + 2].items == ps.mt * 4 <= 2 * P.NCORES  # VLM attention: (q row tile, head pair), <= 2 rounds
    assert P.arena_need(ps, (P.OP_EMBED, P.N_OPS)) > 0
    for _, n in P.vision_groups(p.cameras):  # SigLIP per vision group
        g = dataclasses.replace(ps, img=n)
        P.check_ops(g, group=True)
        gops = P.all_ops(g)
        assert gops[0].mt == g.s_m == 8 * n and gops[P.OP_PROJ].mt == g.s_m
        assert gops[P.OP_S0 + 2].items == 48 * n <= P.NCORES  # SigLIP attention: (image, head, q row group)
        assert P.arena_need(g, (0, P.OP_EMBED)) > 0
    # rpb 4 at c = 3 (RoPE per 3-row sub-block), rpb 5 at c = 4 (every VLM matmul per 3-row sub-block)
    assert ps.mt // ps.rv == {1: 2, 2: 3, 3: 4, 4: 5}[p.cameras]
    # the VLM attention merge spills its weights only at 7 parts (c4 L224: ptv 39)
    assert ps.merge_spill == (ps.v_np == 7) == (p.key[:2] == (4, 224))
    # the expert row loop exactly where one query row per unit cannot fit the grid's rows (c4 S64, L >= 64)
    assert p.shape.row_loop == (p.cameras == 4 and p.suffix_rows == 64 and p.prompt_len >= 64)
    assert p.shape.nu <= G.GRID[1] and p.shape.nch <= 6 and p.shape.chunk_tiles <= 7
    G.check_roles(p.shape)


def test_request_one_camera(expect_error):
    m = _host_only_model(10)
    m.cameras, m.presets = 1, PS.presets_for(1, 32)
    images = [torch.rand(1, 3, 224, 224) * 2 - 1]
    tokens = torch.zeros(1, 64, dtype=torch.long)
    tokens[0, :40] = torch.arange(1, 41)
    host = m.host_inputs(images, None, tokens, None, None)
    assert host["preset"].key == (1, 64, 32) and host["im2col"].shape == (1, 256, 608)
    assert host["valid"].shape == (1, 256 + 64) and int(host["valid"].sum()) == 256 + 40
    with expect_error(RuntimeError, "serves 1 cameras"):
        m.host_inputs(images * 2, None, tokens, None, None)


def test_request_three_cameras(expect_error):
    m = _host_only_model(50)
    m.cameras, m.presets = 3, PS.presets_for(3, 64)
    images = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(3)]
    tokens = torch.zeros(1, 224, dtype=torch.long)
    tokens[0, :100] = torch.arange(1, 101)
    host = m.host_inputs(images, None, tokens, None, None)
    assert host["preset"].key == (3, 128, 64) and host["im2col"].shape == (3, 256, 608)
    assert host["valid"].shape == (1, 768 + 128) and int(host["valid"].sum()) == 768 + 100
    assert m.host_inputs(images, None, tokens, None, None, 224)["preset"].key == (3, 224, 64)
    with expect_error(RuntimeError, "serves 3 cameras"):
        m.host_inputs(images[:2], None, tokens, None, None)


def test_close_and_live_models(monkeypatch):
    """close() releases the traces and deallocates every device tensor of the model (L1 first) even while it is
    referenced, is idempotent, and later calls raise; building a model while another live one holds device memory on
    the same device is refused by name (it used to fail at the first call with an anonymous CB / L1 clash)."""
    from models.experimental.pi0.common.configs import PI0ModelConfig
    from models.experimental.pi0.tt import ttnn_pi05_model as TM

    events = []

    class Buf:
        def __init__(self, name, l1):
            self.name, self.l1 = name, l1

        def memory_config(self):
            return types.SimpleNamespace(buffer_type=TM.ttnn.BufferType.L1 if self.l1 else TM.ttnn.BufferType.DRAM)

    monkeypatch.setattr(TM.ttnn, "Tensor", Buf)
    monkeypatch.setattr(TM.ttnn, "release_trace", lambda dev, tid: events.append(("trace", tid)))
    monkeypatch.setattr(TM.ttnn, "deallocate", lambda t: events.append(("free", t.name)))
    dev, other = object(), object()

    def model(device, traces):
        m = object.__new__(TM.PI05MegakernelTTNN)
        m.device, m._traces, m._l1_at_capture, m._closed = device, dict(traces), "sig", False
        m.cameras, m.suffix_rows, m.l1_bytes = 3, 64, 129_792
        TM._LIVE.add(m)
        return m

    a = model(dev, {(3, 32, 64): 7})
    shared = Buf("kv0", True)
    a.kv_caches = [(shared, Buf("kv1", True))]
    a.noise, a.embed_tokens = Buf("noise", True), Buf("embed", False)
    prog = object.__new__(TM._Programs)  # this package's objects are walked; a shared tensor is freed once
    prog.prefix_dram, prog.again = Buf("arena", False), shared
    a.programs = {"k": prog}
    b = model(other, {})
    assert TM.live_models(dev) == [a] and TM.live_models(other) == [b]
    with pytest.raises(RuntimeError, match=r"1 other model\(s\) on this device still hold device memory.*close\(\)"):
        TM.PI05MegakernelTTNN(PI0ModelConfig(action_horizon=50, pi05=True), None, dev)
    a.close()
    frees = [n for k, n in events if k == "free"]
    assert events[0] == ("trace", 7) and sorted(frees) == ["arena", "embed", "kv0", "kv1", "noise"]
    assert set(frees[:3]) == {"kv0", "kv1", "noise"}  # L1 first
    assert a._traces == {} and a._closed and TM.live_models(dev) == []
    n = len(events)
    a.close()  # idempotent
    assert len(events) == n
    for call in (lambda: a.capture((3, 32, 64)), lambda: a.replay((3, 32, 64)), lambda: a.write_inputs({})):
        with pytest.raises(RuntimeError, match="closed"):
            call()
    with b:
        pass
    assert b._closed and TM.live_models(other) == []


def test_set_horizon_invalidates_attention_inputs():
    """A new action horizon changes the expert key row (rows >= H masked): the cached attention inputs must be rewritten
    on the next request (they are keyed by preset and prompt validity only)."""
    m = _host_only_model(50)
    m._inputs_key = ("preset", "validity")
    m._set_horizon(33)
    assert m._inputs_key is None and m.horizon == 33
    with pytest.raises(ValueError, match="outside"):
        m._set_horizon(10)  # another suffix bucket


def test_wrapper_close(monkeypatch):
    """PI0ModelTTNN.close() closes the megakernel, keeps its own closed state and refuses later calls."""
    from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN

    closed = []
    w = object.__new__(PI0ModelTTNN)
    w._closed, w.pi05_model = False, types.SimpleNamespace(close=lambda: closed.append(1))
    with w:
        pass
    assert closed == [1] and w._closed
    with pytest.raises(RuntimeError, match="closed"):
        w.sample_actions([], [], None, None, None)


def test_kv_placement():
    """K / V in L1 unless a program of the model's presets would keep < 16 KB of L1 free (then DRAM + the expert's
    row multicast): only the 4-camera models (their prefix arena with L1 caches overflows)."""
    from models.experimental.pi0.tt.megakernel import program as PR

    for c in PS.CAMERAS:
        for s in (32, 64):
            ps = PS.presets_for(c, s)
            assert PS.kv_in_dram(ps) == (c == 4), (c, s)
            assert min(PS.l1_free(ps, kv_in_l1=not PS.kv_in_dram(ps)).values()) >= PS.L1_MIN_FREE
    assert PS.kv_l1_bytes(1120) == 117_504  # the c3 L224 S64 caches (WP4 table)
    assert PR.kv_mcast(PS.PRESETS[(4, 32, 32)].shape, kv_dram=True) and not PR.kv_mcast(PS.PRESETS[(4, 32, 32)].shape)


def test_request_four_cameras():
    m = _host_only_model(50)
    m.cameras, m.presets = 4, PS.presets_for(4, 64)
    images = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(4)]
    tokens = torch.zeros(1, 224, dtype=torch.long)
    tokens[0, :30] = torch.arange(1, 31)
    host = m.host_inputs(images, None, tokens, None, None)
    assert host["preset"].key == (4, 32, 64) and host["im2col"].shape == (4, 256, 608)
    assert host["valid"].shape == (1, 1024 + 32) and int(host["valid"].sum()) == 1024 + 30


def test_extra_defines_override_production():
    """Test-only A / B defines must OVERRIDE a production value of the same name (the JIT keeps the first definition
    of a repeated name, so a plain append silently kept the production fidelity). Positive control: a HiFi2 override
    yields PE_MM_FID 2 / MK_FID 2 exactly once; production lists are unchanged."""
    from models.experimental.pi0.tt.megakernel import pe_program as PP
    from models.experimental.pi0.tt.megakernel import program as PR

    assert PR.merge_defines([("A", "1"), ("B", "2")], [("B", "9"), ("C", "3")]) == [("A", "1"), ("B", "9"), ("C", "3")]

    def prog(ops, extra):
        p = object.__new__(PP.PrefixEngineProgram)
        p.first, p.stop, p.extra_defines, p.ps = ops[0], ops[1], list(extra), P.PSHAPES["base"]
        return p

    vis = prog(PP.VISION_OPS, [])
    cg = [("PE_NO_EXPERT", "1"), ("PE_ATT_FLAT", "2")]
    assert vis.defines(7) == [("PE_CT0", "7"), ("MK_FID8_HIFI2", "1")] + cg + [("PE_MM_FID", "3"), ("PE_ATT_FID", "3")]
    assert prog(PP.PREFIX_OPS, []).defines(7) == [("PE_CT0", "7"), ("MK_FID8_HIFI2", "1")] + cg + [("PE_ATT_FID", "3")]
    d = prog(PP.VISION_OPS, [("PE_MM_FID", "2"), ("PE_ATT_FID", "2")]).defines(7)
    names = [k for k, _ in d]
    assert len(names) == len(set(names)) and dict(d)["PE_MM_FID"] == "2" and dict(d)["PE_ATT_FID"] == "2"
    mk = object.__new__(PR.ExpertMegakernel)
    mk.extra_defines, mk.shape, mk.kv_dram = [], PS.PRESETS[(2, 224, 64)].shape, False  # S64: the K / V row multicast
    assert mk.defines() == [("MK_FID8_HIFI2", "1"), ("MK_FID", "3"), ("MK_KV_MCAST", "1")]
    mk.extra_defines = [("MK_FID", "2"), ("MK_KV_MCAST", "0")]
    assert mk.defines() == [("MK_FID8_HIFI2", "1"), ("MK_FID", "2"), ("MK_KV_MCAST", "0")]
    mk.extra_defines, mk.shape = [], PS.PRESETS[(2, 32, 32)].shape  # S32: per-unit reads
    assert mk.defines() == [("MK_FID8_HIFI2", "1"), ("MK_FID", "3")]
    assert all(PR.kv_mcast(p.shape) == (p.suffix_rows == 64) for p in PS.PRESETS.values())


def test_device_refusals(monkeypatch):
    from models.experimental.pi0.tt import ttnn_pi05_model as TM

    class FakeDevice:
        def __init__(self, n=1, grid=(11, 10)):
            self.n, self.grid = n, grid

        def get_num_devices(self):
            return self.n

        def compute_with_storage_grid_size(self):
            return types.SimpleNamespace(x=self.grid[0], y=self.grid[1])

    def memory_view(l1_total):
        small = 24_576
        return lambda _dev, bt: types.SimpleNamespace(
            total_bytes_per_bank=l1_total if bt == TM.ttnn.BufferType.L1 else small
        )

    monkeypatch.setattr(TM.ttnn.device, "is_blackhole", lambda _d: True)
    cut = TM.MEGAKERNEL_WORKER_L1_SIZE - 24_576
    monkeypatch.setattr(TM.ttnn, "get_memory_view", memory_view(cut))
    assert TM.PI05MegakernelTTNN.device_refusal(FakeDevice()) is None
    assert "single chip" in TM.PI05MegakernelTTNN.device_refusal(FakeDevice(n=4))
    assert "core grid" in TM.PI05MegakernelTTNN.device_refusal(FakeDevice(grid=(8, 8)))
    monkeypatch.setattr(TM.ttnn, "get_memory_view", memory_view(cut + 65_536))
    assert "worker-L1 cut" in TM.PI05MegakernelTTNN.device_refusal(FakeDevice())
    monkeypatch.setattr(TM.ttnn.device, "is_blackhole", lambda _d: False)
    assert "Blackhole only" in TM.PI05MegakernelTTNN.device_refusal(FakeDevice())


# ============================================================================ real weights
def _weights_available(path: str) -> bool:
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "model.safetensors"))
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(path, "model.safetensors"), str)
    except Exception:
        return False


@pytest.mark.skipif(not _weights_available(PI05_BASE_WEIGHTS), reason=f"{PI05_BASE_WEIGHTS} not available")
def test_prefix_host_model_vs_reference():
    """The prefix engine's decomposition on the real checkpoint (folded norms, padded heads) vs the torch reference:
    the SigLIP tower + projector, and the VLM K / V of the valid rows of a padded prompt."""
    from models.experimental.pi0.common.configs import PI0ModelConfig
    from models.experimental.pi0.common.weight_loader import PI0WeightLoader
    from models.experimental.pi0.reference.torch_pi0_model import PI0Model as PI0ModelTorch
    from models.experimental.pi0.reference.torch_pi0_model import prefix_attention_inputs

    wl = PI0WeightLoader(PI05_BASE_WEIGHTS)
    pp = H.prefix_params(wl.categorized_weights)
    ref = PI0ModelTorch(PI0ModelConfig(action_dim=32, action_horizon=50, pi05=True), wl).backbone
    g = torch.Generator().manual_seed(3)
    pix = torch.rand(2, 3, 224, 224, generator=g) * 2 - 1
    with torch.no_grad():
        ref_img = torch.cat([ref.embed_image(pix[i : i + 1]) for i in range(2)], 1)[0]  # [512, 2048]
        im = ph.im2col_patches(pix, 14, pad_to=608).reshape(512, 608)
        proj = H.host_siglip(pp, im) @ pp.proj_w + pp.proj_b
        assert hm.pcc(proj, ref_img) > 0.99999
        ps = P.PSHAPES["base"]
        n_lang = 150
        tok = torch.randint(1, 257152, (1, 224), generator=g)
        valid = torch.zeros(1, 736, dtype=torch.bool)
        valid[0, : 512 + n_lang] = True
        lang = ref.embed_language_tokens(tok)[0] * (2048**0.5)
        x0 = torch.zeros(ps.mt * 32, 2048)
        x0[:512] = ref_img
        x0[512:736] = lang
        vlm_mask, vlm_pos, _, _ = prefix_attention_inputs(valid, 50)
        _, cache = ref.forward_vlm(
            torch.cat([ref_img[None], lang[None]], 1), attention_mask=vlm_mask, position_ids=vlm_pos, use_cache=True
        )
        _, kv = H.host_vlm(pp, x0, valid, ps)
    nv = 512 + n_lang
    for l in (0, 8, 17):
        for a, b in ((kv[l][0][:nv], cache[l][0][0, 0, :nv]), (kv[l][1][:nv], cache[l][1][0, 0, :nv])):
            assert hm.pcc(a, b) > 0.99999, l
