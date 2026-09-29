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
    assert FusedConfig.from_env({}).megakernel == "off"
    assert FusedConfig.from_env({"PI05_MEGAKERNEL": "expert"}).megakernel == "expert"
    with pytest.raises(ValueError):
        FusedConfig.from_env({"PI05_MEGAKERNEL": "bogus"})
    assert G.shape_for(736, 64).name == "base" and G.shape_for(544, 32).name == "libero"
    with pytest.raises(RuntimeError, match="PI05_MEGAKERNEL"):
        G.shape_for(768, 64)
    from models.experimental.pi0_5.common.device_open import device_kwargs

    assert device_kwargs(FusedConfig.from_env({"PI05_MEGAKERNEL": "expert"}))["worker_l1_size"] == 1_395_712
    assert "worker_l1_size" not in device_kwargs(FusedConfig.from_env({}))


def test_cpu_one_producer_per_cb_on_h0():
    """Every CB has ONE producer RISC and ONE consumer RISC (the TRISC packer / unpacker keep local copies of the CB
    counters: a second producer hung H0 on 2026-09-30). Pinned on the H0 paths that once violated it."""
    kd = G.KDIR
    brisc = open(os.path.join(kd, "mk_brisc.cpp")).read()
    trisc = open(os.path.join(kd, "mk_trisc.cpp")).read()
    assert "read_tiles_into(CB_SCR16" not in brisc  # noise goes through CB_Q
    assert not re.search(r"pack_to\([^;]*CB_IN0", trisc)  # the TRISC never produces CB_IN0
    assert "cb_push_back(CB_RTOK" not in trisc
