# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the expert megakernel's host side (no device): the kernel-header constants, the role tables, the CB
table invariants the kernels rely on, the arena share tile orders against the TRISC loops, the knob and refusals.

    pytest models/experimental/pi0_5/tests/megakernel/test_cpu_mk.py -k cpu
"""
import os
import re

import pytest
import torch

from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.tt.megakernel import arena as A
from models.experimental.pi0_5.tt.megakernel import geometry as G


def test_cpu_defs_parse_and_dense():
    d = G.parse_defs()
    cbs = sorted(v for k, v in d.items() if k.startswith("CB_"))
    assert cbs == list(range(d["N_CBS"]))
    words = [v for k, v in d.items() if k.startswith("S_") and k not in ("S_SRC0", "S_DBG", "SYNC_STRIDE")]
    assert len(set(words)) == len(words)
    assert d["SYNC_STRIDE"] == 16  # every sync word on its own 16 B line (F0 device fact 6)


@pytest.mark.parametrize("name", ["base", "libero"])
def test_cpu_roles(name):
    G.check_roles(G.SHAPES[name])


@pytest.mark.parametrize("name", ["base", "libero"])
def test_cpu_cb_full_capacity_cycles(name):
    """Landing / staging CBs: every transaction the kernels issue equals the CB capacity (pointers return to the
    base), and multi-page ring waits never straddle a ring end."""
    sh = G.SHAPES[name]
    t = {c.cb_id: c for c in G.cb_table(sh)}
    rt, nch, cht = sh.rt, sh.nch, sh.chunk_tiles
    full = {G.CB_IN0: 64 * rt, G.CB_Q: 8 * rt, G.CB_KSV: 16 * rt, G.CB_KV: 2 * cht * 8, G.CB_MASK: cht,
            G.CB_PART: 8 * (nch - 1), G.CB_PM: nch - 1, G.CB_PL: nch - 1, G.CB_QKVO: 2 * rt, G.CB_XS: rt,
            G.CB_H: 2 * rt, G.CB_HG: 16 * rt, G.CB_DP: 4 * rt, G.CB_RED: 8 * rt, G.CB_CTXO: 8, G.CB_OP: 8,
            G.CB_XRES: rt, G.CB_ROUT: rt, G.CB_SCR16: rt, G.CB_CONST: 4, G.CB_TAB: 4 * rt, G.CB_D: nch}
    for cb, pages in full.items():
        assert t[cb].pages == pages, (t[cb].name, t[cb].pages, pages)
    # H0 x_new: row 0 in CB_PART (>= 32 tiles), row 1 in CB_HG (32 tiles at RT = 2)
    assert t[G.CB_PART].pages >= 32 and (rt == 1 or t[G.CB_HG].pages >= 32)
    # rings: W8 16 pages consumed in 8-page (qkv) and 2-page (up|gate) waits; W16 page by page; H0 16-page steps
    assert G.W8_PAGES % 8 == 0 and G.W16_PAGES == 8
    h0 = G.core_streams(G.build_roles(sh)[G.H0], sh)["w16"]
    assert sum(p for _, p in h0) % G.W16_PAGES == 0


@pytest.mark.parametrize("name", ["base", "libero"])
def test_cpu_bank_plan_covers_streams(name):
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
        for b, sp in spans.items():
            sp = sorted(s for s in sp if s[1])
            for (o0, s0), (o1, _) in zip(sp, sp[1:]):
                assert o0 + s0 <= o1


def test_cpu_share_orders_match_trisc_loops():
    """The TRISC reads qkv tile (2*kb + c)*8 + k, up|gate (jj, kb) pages [u | g], o_proj page kb tile k, down page
    (kb, n) tile k: the packers must produce exactly those orders."""
    w = torch.arange(1024 * 2560, dtype=torch.float32).reshape(1024, 2560)
    t = A.w_tiles(w)
    share = A.qkv_share(t, 5, 9)
    for kb in range(4):
        for c, col in enumerate((5, 9)):
            for k in range(8):
                assert torch.equal(share[(2 * kb + c) * 8 + k], t[kb * 8 + k, col])
    wu = torch.randn(1024, 8192)
    tu = A.w_tiles(wu)
    ug = A.ug_share(tu, 17, 18)
    for jj, j in enumerate((17, 18)):
        for kb in range(4):
            base = (jj * 4 + kb) * 16
            for k in range(8):
                assert torch.equal(ug[base + k], tu[kb * 8 + k, j])
                assert torch.equal(ug[base + 8 + k], tu[kb * 8 + k, 128 + j])
    wd = torch.randn(4096, 1024)
    td = A.w_tiles(wd)
    dn = A.wd_share(td, 3, 5)
    for kb in range(2):
        for n in range(4):
            for k in range(8):
                assert torch.equal(dn[(kb * 4 + n) * 8 + k], td[3 * 16 + kb * 8 + k, 4 * 5 + n])


def test_cpu_knob_and_refusals():
    # the whole-model megakernel is the DEFAULT since 2026-10-01; ``expert`` (phase 1) and ``off`` (the previous
    # path) are the comparator knobs
    assert FusedConfig.from_env({}).megakernel == "whole"
    assert FusedConfig().megakernel == "whole"
    assert not FusedConfig.from_env({}).megakernel_explicit
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": "whole"}).megakernel == "whole"
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": "expert"}).megakernel == "expert"
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": "off"}).megakernel == "off"
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": ""}).megakernel == "whole"
    # the resolved default: single chip keeps whole; on a mesh the UNSET default is off, an explicit choice stays
    # (and the model refuses it on a mesh)
    assert FusedConfig.from_env({}).resolved(1).megakernel == "whole"
    assert FusedConfig.from_env({}).resolved(4).megakernel == "off"
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": "whole"}).resolved(4).megakernel == "whole"
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": "expert"}).resolved(4).megakernel == "expert"
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": "expert"}).resolved(1).megakernel == "expert"
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": "off"}).resolved(1).megakernel == "off"
    assert FusedConfig.from_env({}).resolved(1).resolved(1) == FusedConfig.from_env({}).resolved(1)
    with pytest.raises(ValueError):
        FusedConfig.from_env({"PI05_MEGAKERNEL": "bogus"})
    assert G.shape_for(736, 64).name == "base" and G.shape_for(544, 32).name == "libero"
    with pytest.raises(RuntimeError, match="PI05_MEGAKERNEL"):
        G.shape_for(768, 64)
    from models.experimental.pi0_5.common.device_open import device_kwargs

    assert device_kwargs(FusedConfig.from_env({"PI05_MEGAKERNEL": "expert"}))["worker_l1_size"] == 1_395_712
    assert device_kwargs(FusedConfig.from_env({"PI05_MEGAKERNEL": "whole"}))["worker_l1_size"] == 1_395_712
    assert device_kwargs(FusedConfig.from_env({}))["worker_l1_size"] == 1_395_712  # the default path needs the cut
    assert "worker_l1_size" not in device_kwargs(FusedConfig.from_env({"PI05_MEGAKERNEL": "off"}))


def test_cpu_refuses_a_device_without_the_worker_l1_cut(monkeypatch):
    """DESIGN.md §4.12 refusal (d): without the 64 KiB worker-L1 cut the program misses the kernel-config ring by
    ~100 B (verify_p1_r1: an unnamed TT_FATAL at the first launch). The model names it before any upload."""
    import types

    import ttnn

    from models.experimental.pi0_5.common.device_open import MEGAKERNEL_WORKER_L1_SIZE
    from models.experimental.pi0_5.tt import ttnn_pi0_model as TM

    def fake_view(l1_total, small):
        def view(_dev, bt):
            return types.SimpleNamespace(total_bytes_per_bank=l1_total if bt == ttnn.BufferType.L1 else small)
        return view

    small = 24_576
    monkeypatch.setattr(TM.ttnn, "get_memory_view", fake_view(MEGAKERNEL_WORKER_L1_SIZE - small, small))
    assert TM.PI0ModelTTNN.megakernel_device_refusal(object()) is None
    monkeypatch.setattr(TM.ttnn, "get_memory_view", fake_view(MEGAKERNEL_WORKER_L1_SIZE - small + 65_536, small))
    why = TM.PI0ModelTTNN.megakernel_device_refusal(object())
    assert why and "worker-L1 cut" in why and "PI05_MEGAKERNEL=off" in why
    # the model constructor path raises it by name (after the kv / steps / mesh checks, before any parameter build)
    monkeypatch.setattr(TM, "_is_mesh", lambda _d: False)
    stub = types.SimpleNamespace(fused_cfg=FusedConfig.from_env({}), denoise_config=types.SimpleNamespace(num_steps=10),
                                 device=object(), megakernel_device_refusal=TM.PI0ModelTTNN.megakernel_device_refusal)
    with pytest.raises(RuntimeError, match="worker-L1 cut"):
        TM.PI0ModelTTNN._init_megakernel(stub)


def test_cpu_one_producer_per_cb_on_h0():
    """Every CB has ONE producer RISC and ONE consumer RISC (the TRISC packer / unpacker keep local copies of the CB
    counters: a second producer hung H0 on 2026-09-30). Pinned on the H0 paths that once violated it."""
    kd = G.KDIR
    brisc = open(os.path.join(kd, "mk_brisc.cpp")).read()
    trisc = open(os.path.join(kd, "mk_trisc.cpp")).read()
    assert "read_tiles_into(CB_SCR16" not in brisc  # noise goes through CB_Q
    assert not re.search(r"pack_to\([^;]*CB_IN0", trisc)  # the TRISC never produces CB_IN0
    assert "cb_push_back(CB_RTOK" not in trisc


# ------------------------------------------------------------------ refusals of configurations the kernels do not compile
def test_cpu_refuses_bf16_kv_and_other_step_counts():
    """PI05_KV_DTYPE=bf16 (bf16 caches landed at the bfp8 page stride would overrun CB_KV) and a step count other than
    the compiled N_STEPS (20 steps silently returned x at t = 0.5; < 10 raised an unnamed IndexError) must refuse by
    name, at every entry: the pure check, the model constructor, the program's cache dtype check and the step check."""
    import types

    import ttnn

    from models.experimental.pi0_5.tt.megakernel import program as P
    from models.experimental.pi0_5.tt.megakernel.host_model import ExpertParams
    from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

    assert G.N_STEPS == 10
    assert G.megakernel_refusal("bf8", 10) is None
    assert "PI05_KV_DTYPE=bf16" in G.megakernel_refusal("bf16", 10)
    for n in (1, 5, 9, 11, 16, 20):
        assert "PI05_NUM_STEPS" in G.megakernel_refusal("bf8", n), n
        # the old guard (one distinct fp32 dt) passes for these schedules: it cannot be the refusal
        dts = [(1.0 - (i + 1) / n) - (1.0 - i / n) for i in range(n)]
        assert len({G.f32_bits(d) for d in dts}) == 1 or n in (9, 11), n

    # model constructor: refused before any parameter build or device use (device=None would fail later)
    for kv_dtype, steps, what in (("bf16", 10, "PI05_KV_DTYPE"), ("bf8", 20, "PI05_NUM_STEPS"), ("bf8", 5, "PI05_NUM_STEPS")):
        stub = types.SimpleNamespace(fused_cfg=FusedConfig.from_env({"PI05_MEGAKERNEL": "expert", "PI05_KV_DTYPE": kv_dtype}),
                                     denoise_config=types.SimpleNamespace(num_steps=steps), device=None)
        with pytest.raises(RuntimeError, match=what):
            PI0ModelTTNN._init_megakernel(stub)
    ok = types.SimpleNamespace(fused_cfg=FusedConfig.from_env({"PI05_MEGAKERNEL": "off"}), denoise_config=types.SimpleNamespace(num_steps=20),
                               device=None)
    PI0ModelTTNN._init_megakernel(ok)  # PI05_MEGAKERNEL=off: any step count / dtype is the shipped path's business
    assert ok.megakernel_backend == "off"

    # ExpertMegakernel: the step count is checked before any upload
    z = torch.zeros(1)
    for n in (5, 20):
        params = ExpertParams(wqkv=[], wo=[], wug=[], wd=[], mods=[], final=[], w_in=z, b_in=z, w_out=z, b_out=z,
                              dts=tuple([-1.0 / n] * n))
        with pytest.raises(RuntimeError, match="PI05_NUM_STEPS"):
            P.ExpertMegakernel(None, params, G.SHAPES["base"])

    # program(): every K / V cache must be bfp8 (18 layers)
    mk = object.__new__(P.ExpertMegakernel)
    mk.shape = G.SHAPES["base"]
    t = lambda d: types.SimpleNamespace(dtype=d)
    bf16 = [(t(ttnn.bfloat16), t(ttnn.bfloat16)) for _ in range(G.N_LAYERS)]
    with pytest.raises(RuntimeError, match="PI05_KV_DTYPE"):
        mk.program(bf16, None, None, None, None)
    mixed = [(t(ttnn.bfloat8_b), t(ttnn.bfloat8_b)) for _ in range(G.N_LAYERS - 1)] + [(t(ttnn.bfloat8_b), t(ttnn.bfloat16))]
    with pytest.raises(RuntimeError, match="PI05_KV_DTYPE"):
        mk.program(mixed, None, None, None, None)
    with pytest.raises(RuntimeError, match="18 bfp8"):
        mk.program(mixed[:-1], None, None, None, None)


def test_cpu_merge_dst_budget_and_host_invariants():
    """Shape.check allows NCH <= 6 (merge(): M / L in DST 0, weights in DST 1..NCH, temporary in DST 7); the kernel's
    multicast destination counts equal the rectangle-derived counts; mergers and units with y < 8 are MLP cores (the
    K / V prefetch path). Positive controls: a doctored kernel count and NCH = 7 are rejected."""
    for s in G.SHAPES.values():
        assert s.nch <= 6
        G.check_roles(s)
    with pytest.raises(ValueError, match="NCH <= 6"):
        G.Shape("nch7", prefix_len=32 * 7 * 3 - 32, suffix_rows=32, horizon=10, chunk_tiles=3).check()
    brisc = open(os.path.join(G.KDIR, "mk_brisc.cpp")).read()
    for name, lit in (("rm", "64"), ("ro", "32"), ("row", "7")):
        pat = re.compile(r"(me\." + name + r" = \{[^;]*?,\s*)" + lit + r"\};")
        assert pat.search(brisc), name
        bad = pat.sub(lambda m: m.group(1) + str(int(lit) + 1) + "};", brisc, count=1)
        c = G.kernel_mcast_counts(G.SHAPES["base"], bad)
        lo, hi, snd = G.mcast_rects(G.SHAPES["base"])[name]
        assert c[name] != G.mcast_dest_count(lo, hi, snd[0])


def _synthetic_params(seed: int = 0):
    from models.experimental.pi0_5.tt.megakernel import host_model as hm

    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=1.0: torch.randn(*s, generator=g) * sc
    W, D, NH, M, L = hm.WIDTH, hm.DH, hm.NH, hm.MLP, hm.N_LAYERS
    mods = [[(r(W, sc=0.1), r(W, sc=0.1), r(W, sc=0.3), r(W, sc=0.1), r(W, sc=0.1), r(W, sc=0.3)) for _ in range(L)]
            for _ in range(hm.N_STEPS)]
    return hm.ExpertParams(
        wqkv=[r(W, (NH + 2) * D, sc=W ** -0.5) for _ in range(L)], wo=[r(NH * D, W, sc=(NH * D) ** -0.5) for _ in range(L)],
        wug=[r(W, 2 * M, sc=W ** -0.5) for _ in range(L)], wd=[r(M, W, sc=M ** -0.5) for _ in range(L)], mods=mods,
        final=[(r(W, sc=0.1), r(W, sc=0.1)) for _ in range(hm.N_STEPS)], w_in=r(32, W, sc=32 ** -0.5), b_in=r(W, sc=0.1),
        w_out=r(W, 32, sc=W ** -0.5), b_out=r(32, sc=0.1), eps=1e-6, dts=tuple([-0.1] * hm.N_STEPS))


@pytest.mark.parametrize("name", ["base", "libero"])
def test_cpu_decomposition_equals_reference_loop(name):
    """host_model.loop_decomposed (the kernel's folds, chunked flash parts + diag merge, 2-D MLP, K-split reduce)
    equals loop_reference (plain fp32 formulas) over the whole 10 x 18 loop at the shape's chunking, with a padded
    prompt (masked prefix keys). Positive control: dropping one prefix chunk's keys from the decomposed arm only
    (mask -> -inf) must move the output well past the tolerance."""
    from models.experimental.pi0_5.tt.megakernel import host_model as hm

    sh = G.SHAPES[name]
    p = _synthetic_params()
    g = torch.Generator().manual_seed(1)
    P, S, H = sh.prefix_len, sh.suffix_rows, sh.horizon
    kv = [(torch.randn(P, hm.DH, generator=g) * 2, torch.randn(P, hm.DH, generator=g)) for _ in range(hm.N_LAYERS)]
    mask = torch.zeros(P + S)
    mask[P - 100:P] = -1e9  # padded prompt: 100 masked prefix keys inside the last prefix tiles
    mask[P + H:] = -1e9  # tile-pad action rows as keys
    ang = torch.arange(S).float()[:, None] * (1.0 / 10000 ** (torch.arange(0, hm.DH, 2).float() / hm.DH))[None]
    cos = torch.cat([ang.cos(), ang.cos()], -1)
    sin = torch.cat([-ang.sin(), ang.sin()], -1)
    a = hm.AttnInputs(mask=mask, cosq=cos / 16, sinq=sin / 16, cosk=cos, sink=sin)
    noise = torch.zeros(S, 32)
    noise[:H] = torch.randn(H, 32, generator=g)
    ref = hm.loop_reference(p, kv, a, noise)[:H]
    dec = hm.loop_decomposed(p, kv, a, noise, sh.chunk_tiles)[:H]
    assert torch.isfinite(ref).all() and ref.abs().max() > 0.1
    err = float((ref - dec).abs().max() / ref.abs().max())
    assert hm.pcc(ref, dec) > 0.999999 and err < 1e-4, (hm.pcc(ref, dec), err)
    m2 = mask.clone()
    m2[:sh.chunk_tiles * 32] = -1e9  # chunk 0's keys removed in the decomposed arm only
    bad = hm.loop_decomposed(p, kv, hm.AttnInputs(m2, a.cosq, a.sinq, a.cosk, a.sink), noise, sh.chunk_tiles)[:H]
    assert float((ref - bad).abs().max() / ref.abs().max()) > 100 * max(err, 1e-6)
