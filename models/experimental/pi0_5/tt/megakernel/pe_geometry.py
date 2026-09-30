# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Geometry of the phase-2 prefix engine (host mirror of ``kernels_p2/pe_common.hpp``).

Pure Python (no ttnn). Constants are parsed from ``kernels_p2/pe_defs.hpp`` (single source of truth, like
``geometry.py`` parses ``kernels/mk_defs.hpp``). ``describe`` / ``layout`` / ``mm_arena_off`` / ``pair_col`` reproduce the
kernel's integer code line by line; ``tests/megakernel/test_cpu_pe.py`` pins them (op list, splits, arena offsets, L1
layouts against the arena size).

Execution model: DESIGN.md §11 (phase-2 implementation design, 2026-09-30).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import geometry as G

KDIR2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels_p2")
PE_DEFS_PATH = os.path.join(KDIR2, "pe_defs.hpp")


def parse_pe_defs(path: str = PE_DEFS_PATH) -> Dict[str, int]:
    out: Dict[str, int] = {}
    pat = re.compile(r"^\s*constexpr\s+uint32_t\s+([A-Z_0-9]+)\s*=\s*([0-9]+)\s*;")
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                if m.group(1) in out:
                    raise ValueError(f"pe_defs.hpp defines {m.group(1)} twice")
                out[m.group(1)] = int(m.group(2))
    return out


PD = parse_pe_defs()
globals().update(PD)

T16, T8, T32 = 2048, 1088, 4096
SV_LN1W, SV_LN1B, SV_BQKV, SV_BO, SV_LN2W, SV_LN2B, SV_BFC1, SV_BFC2, SV_N = 0, 36, 72, 216, 252, 288, 324, 460, 496
VV_G1, VV_G2, VV_N = 0, 64, 128
GV_PLNW, GV_PLNB, GV_BPROJ, GV_N = 0, 36, 72, 136
S_NK, S_CH, V_CH = 8, 8, 6
N_BANKS = 8
W_BUDGET = 8 * 34816
IN0_SLOTS = 4
IN0_CHUNK = 16


@dataclass(frozen=True)
class PShape:
    """Prefix shape: base (P 736 -> 24 row tiles in 8 bands) or LIBERO (P 544 -> 18 row tiles in 6 bands)."""
    name: str
    mt: int  # VLM row tiles (tile-padded prefix)
    ptv: int  # prefix key tiles
    rv: int  # VLM bands
    lt: int  # language row tiles
    ntok: int  # prompt tokens

    @property
    def v_np(self) -> int:
        return (self.ptv + V_CH - 1) // V_CH


PSHAPES = {
    "base": PShape("base", mt=24, ptv=23, rv=8, lt=7, ntok=224),
    "libero": PShape("libero", mt=18, ptv=17, rv=6, lt=1, ntok=32),
}


def pshape_for(shape: G.Shape) -> PShape:
    ps = PSHAPES[shape.name]
    assert ps.ptv * 32 == shape.prefix_len, (ps, shape)
    return ps


@dataclass
class Op:
    op: int
    kind: int = 0
    what: int = 0
    layer: int = 0
    mode: int = 0
    mt: int = 0
    nb: int = 0
    rpb: int = 0
    kt: int = 0
    np: int = 0
    piece: int = 0
    wbf16: int = 0
    epi: int = 0
    fid_hi: int = 0
    nkind: int = 0
    nk: int = 0
    items: int = 0


def split_n(total: int, c: int) -> int:
    return total // NCOL + (1 if c < total % NCOL else 0)


def split_0(total: int, c: int) -> int:
    return c * (total // NCOL) + min(c, total % NCOL)


def describe(op: int, ps: PShape) -> Op:
    o = Op(op)
    if op == 0:
        what, layer = W_PATCH, 0
    elif op < OP_POSTLN:
        layer, what = (op - OP_S0) // OPS_PER_LAYER, W_SLN1 + (op - OP_S0) % OPS_PER_LAYER
    elif op == OP_POSTLN:
        what, layer = W_POSTLN, 0
    elif op == OP_PROJ:
        what, layer = W_PROJ, 0
    elif op == OP_EMBED:
        what, layer = W_EMBED, 0
    else:
        layer, what = (op - OP_V0) // OPS_PER_LAYER, W_VRMS1 + (op - OP_V0) % OPS_PER_LAYER
    o.what, o.layer = what, layer

    def mm(mode, mt, nb, kt, np_, piece, epi, wbf16, fid_hi):
        o.kind, o.mode, o.mt, o.nb, o.rpb, o.kt, o.np, o.piece = K_MM, mode, mt, nb, mt // nb, kt, np_, piece
        o.epi, o.wbf16, o.fid_hi = epi, wbf16, fid_hi

    if what == W_PATCH:
        mm(MM_R, S_M, S_R, S_KP, S_D // 2, S_KP, E_POS, 1, 1)
    elif what == W_SQKV:
        mm(MM_R, S_M, S_R, S_D, S_NQKV // 2, S_PIECE, E_BIAS, 0, 1)
    elif what == W_SO:
        mm(MM_R, S_M, S_R, S_DCTX, S_D // 2, S_PIECE_O, E_RES_BIAS, 0, 1)
    elif what == W_SFC1:
        mm(MM_R, S_M, S_R, S_D, S_I // 2, S_PIECE, E_BIAS_GELU, 0, 1)
    elif what == W_SFC2:
        mm(MM_R, S_M, S_R, S_I, S_D // 2, S_PIECE_2, E_RES_BIAS, 0, 1)
    elif what == W_PROJ:
        mm(MM_R, S_M, S_R, S_D, V_D // 2, S_PIECE, E_BIAS, 0, 1)
    elif what == W_VQKV:
        mm(MM_R, ps.mt, ps.rv, V_D, V_NQKV // 2, V_PIECE, E_ROPE, 0, 0)
    elif what == W_VO:
        mm(MM_R, ps.mt, ps.rv, V_D, V_D // 2, V_PIECE, E_RES, 0, 0)
    elif what == W_VGU:
        mm(MM_R, ps.mt, ps.rv, V_D, V_I, V_PIECE, E_GEGLU, 0, 0)
    elif what == W_VDOWN:
        mm(MM_S, ps.mt, ps.rv, V_I, V_D // 2, V_PIECE, E_RES, 0, 0)
    elif what in (W_SLN1, W_SLN2, W_POSTLN):
        o.kind, o.nkind, o.nk, o.items = K_NORM, N_LN, S_D, S_M
    elif what in (W_VRMS1, W_VRMS2):
        o.kind, o.nkind, o.nk, o.items = K_NORM, N_RMS, V_D, ps.mt
    elif what == W_SATTN:
        o.kind, o.items = K_ATTN, 2 * S_HEADS * 3
    elif what == W_VATTN:
        o.kind, o.items = K_ATTN, ps.mt * (V_H // 2)
    elif what == W_EMBED:
        o.kind, o.items = K_EMBED, (ps.lt + (ps.mt - ps.ptv)) * 4
    else:
        o.kind = K_NOP
    return o


def mm_pairs(o: Op, x: int) -> int:
    return split_n(o.np, x)


def mm_pair0(o: Op, x: int) -> int:
    return split_0(o.np, x)


def mm_nq(o: Op) -> int:
    return o.kt // o.piece


def mm_wtile(o: Op) -> int:
    return T16 if o.wbf16 else T8


def mm_page_tiles(o: Op) -> int:
    return 2 * o.piece


def mm_page_bytes(o: Op) -> int:
    return mm_page_tiles(o) * mm_wtile(o)


def mm_in0_page_tiles(o: Op) -> int:
    return o.rpb * o.piece if o.mode == MM_S else o.rpb * o.kt


def op_pages_total(o: Op) -> int:
    return o.np * mm_nq(o)


def op_bank_bytes(o: Op) -> int:
    return -(-op_pages_total(o) // N_BANKS) * mm_page_bytes(o)


def mm_arena_off(o: Op, ps: PShape) -> int:
    if o.what in (W_PATCH, W_PROJ):
        return 0
    first = W_VQKV if o.what >= W_VRMS1 else W_SQKV
    off = 0
    for w in range(first, o.what):
        if w in (W_SATTN, W_SLN2, W_VATTN, W_VRMS2):
            continue
        off += op_bank_bytes(describe(o.op - (o.what - w), ps))
    return off


def layer_arena_bank_bytes(model: str, ps: PShape) -> int:
    """Bytes per DRAM bank of one layer arena ("siglip" / "vlm")."""
    op0 = OP_S0 if model == "siglip" else OP_V0
    last = W_SFC2 if model == "siglip" else W_VDOWN
    o = describe(op0 + (last - (W_SLN1 if model == "siglip" else W_VRMS1)), ps)
    return mm_arena_off(o, ps) + op_bank_bytes(o)


def pair_col(o: Op, p: int, t: int) -> int:
    if o.what == W_VQKV:
        if p < 32:
            return (p // 4) * 8 + (p % 4) + 4 * t
        j = (p - 32) % 4
        return (64 if p < 36 else 72) + j + 4 * t
    if o.what == W_VGU:
        return p + V_I * t
    return 2 * p + t


def side16(o: Op, p: int) -> int:
    if o.epi == E_POS:
        return 2 * o.rpb
    if o.epi in (E_BIAS, E_BIAS_GELU, E_RES_BIAS):
        return 2
    if o.epi == E_ROPE:
        return 4 * o.rpb if p < 36 else 0
    return 0


def side32(o: Op) -> int:
    return 2 * o.rpb if o.epi in (E_RES, E_RES_BIAS) else 0


def _al(b: int) -> int:
    return (b + 63) & ~63


def layout(o: Op, ps: PShape) -> Dict[str, int]:
    """Byte offsets from the arena base (pe_common.hpp layout); 'end' = bytes used."""
    y: Dict[str, int] = {}
    a = 0

    def take(name, nbytes):
        nonlocal a
        y[name] = a
        a += _al(nbytes)

    if o.kind == K_MM:
        rp = o.rpb
        y["in0_bytes"] = IN0_SLOTS * mm_in0_page_tiles(o) * T8 if o.mode == MM_S else rp * o.kt * T16
        take("in0", y["in0_bytes"])
        y["w_slots"] = min(8, W_BUDGET // mm_page_bytes(o))
        take("w", y["w_slots"] * mm_page_bytes(o))
        take("s16", 24 * T16)
        take("s32", 4 * rp * T32)
        take("o16", 4 * rp * T16)
        take("o8", 4 * rp * T8)
        take("o32", 4 * rp * T32)
        if o.mode == MM_S:
            take("part", 3 * 2 * rp * T32)
        stage = max(mm_page_bytes(o), rp * o.piece * T16)
        if o.mode == MM_S:
            stage = max(stage, mm_in0_page_tiles(o) * T8)
        y["stage_bytes"] = stage
        y["stage"] = 0
    elif o.kind == K_NORM:
        take("x32", o.nk * T32)
        take("s16", 2 * o.nk * T16)
        take("o16", 8 * T16)
        take("scr", 2 * T32)
        take("r", 2 * T32)
        take("cst", PC_N * T16)
    elif o.kind == K_ATTN:
        v = o.what == W_VATTN
        take("q", (2 * V_DH if v else 3 * S_DH) * T16)
        take("kv", 2 * ps.ptv * V_DH * T8 if v else 2 * S_NK * S_DH * T16)
        take("msk", (ps.ptv if v else 1) * T16)
        take("ss", 8 * T16)
        take("m", T16)
        take("mf", T16)
        take("l", T32)
        np_ = ps.v_np if v else -(-S_NK // S_CH)
        dh = V_DH if v else S_DH
        take("op", 2 * np_ * dh * T16)
        take("pm", 2 * np_ * T16)
        take("pl", 2 * np_ * T32)
        take("d", np_ * T32)
        take("o16", 2 * dh * T16)
        take("cst", PC_N * T16)
    elif o.kind == K_EMBED:
        take("tok", max(ps.ntok * 4, 64))
        take("rm", 16 * T16)
        take("s16", 16 * T16)
        take("o32", 16 * T32)
        take("cst", PC_N * T16)
    y["end"] = a
    # a feeder stages in its own arena from offset 0 (it computes nothing in a matmul op)
    if o.kind == K_MM:
        y["end"] = max(a, 2 * y["stage_bytes"])
    return y


def all_ops(ps: PShape) -> List[Op]:
    return [describe(i, ps) for i in range(N_OPS)]


def arena_need(ps: PShape) -> int:
    return max(layout(o, ps)["end"] for o in all_ops(ps))


# ======================================================================================================== CB table
# placeholder page formats of the prefix ids (they are re-pointed per op in-kernel; the host only fixes id + format)
PFMT = {
    P_IN0: "bf16", P_IN08: "bfp8", P_W8: "bfp8", P_W16: "bf16", P_S16: "bf16", P_S32: "fp32", P_PART: "fp32",
    P_O16: "bf16", P_O8: "bfp8", P_O32: "fp32", P_X32: "fp32", P_SCR: "fp32", P_R: "fp32", P_Q: "bf16",
    P_KV8: "bfp8", P_KV16: "bf16", P_MSK: "bf16", P_SS: "bf16", P_M: "bf16", P_MF: "bf16", P_L: "fp32",
    P_OP: "bf16", P_PM: "bf16", P_PL: "fp32", P_D: "fp32", P_CONST: "bf16", P_RM: "bf16", P_TOK: "raw",
}
# P_SYNC, P_OPD (static) and P_TAIL (the arena extension) are declared separately (pe_program.py)
# UnpackToDestFp32 (exact fp32 copies into DST; never FPU operands)
P_FP32_UNPACK = (P_S32, P_PART, P_L, P_PL)  # P_R, P_X32, P_SCR are FPU operands of the norms (default unpack)
PSYNC_BYTES = 48 * PSTRIDE  # sync words (PS_N) + diagnostics staging (PS_DIAG..)


def check_ops(ps: PShape) -> None:
    """Host invariants of the op list (the kernels rely on them)."""
    ops = all_ops(ps)
    assert len(ops) == N_OPS
    for o in ops:
        if o.kind == K_MM:
            assert o.mt % o.nb == 0 and o.kt % o.piece == 0, o
            assert o.nb <= 8, o  # compute rows 0..7 (rows 8 / 9 are feeders)
            assert 2 * o.rpb <= 6 or o.epi in (E_NONE, E_GEGLU, E_POS), o  # epilogue temporaries in DST 6, 7
            assert max(mm_pairs(o, x) for x in range(NCOL)) <= 64
            if o.mode == MM_S:
                assert max(mm_pairs(o, x) for x in range(NCOL)) <= 3, o  # P_PART holds 3 pairs
            lay = layout(o, ps)
            assert lay["w_slots"] >= 2, o
        if o.kind == K_ATTN and o.what == W_VATTN:
            assert ps.v_np <= 6
    assert ps.mt == ps.ptv + 1
    assert S_M + ps.lt == ps.ptv, "images + language rows = the prefix"
