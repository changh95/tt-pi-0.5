# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Geometry of the pi0.5 expert megakernel (phase 1): shapes, the core map, CB table, stream plans, host checks.

Pure Python (no ttnn): everything here is testable on the CPU (``tests/megakernel/test_cpu_mk.py``).
Constants shared with the kernels are parsed from ``kernels/mk_defs.hpp`` (single source of truth).

Core map (logical coordinates, grid 11 x 10; see docs/megakernel/JOURNAL.md "implementation v1" for the deviations
from DESIGN.md §4.2 and why):

* pair cores (x, y), x in 0..9, y in 0..3: qkv for two dh tiles (y, y + 4) of Q head x (x < 8), of K (x = 8) or of V
  (x = 9), both rows, with the adaRMS r / c epilogue and RoPE (Q, K) done locally (no pair exchange).
* attention units (h, kc*RT + r): head h = x < 8, q row tile r, key chunk kc; NCH*RT <= 10 rows.
* mergers = the kc = 0 units (h, r); Q leader of head h = (h, 0); KL = (8, 4).
* owners n = 0..31 at (n // 4, 4 + n % 4): residual column n, o_proj column n, the column reduce.
* MLP cores (kg, ng) at (x = ng, y = kg), 8 x 8: up|gate + GeGLU for h columns kg*16 + 2*ng + {0, 1}, the down partial
  for K tiles kg*16..+16 and N tiles 4*ng..+4; row leaders (0, kg).
* hubs H0 (10, 9) (x / x_mid rounds, r, the per-step in / out projections and Euler) and H1 (10, 8) (ctx round).
"""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TILE = 32
TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "fp32": 4096}
KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
DEFS_PATH = os.path.join(KDIR, "mk_defs.hpp")

GRID = (11, 10)
H0 = (10, 9)
H1 = (10, 8)
KL = (8, 4)
N_BANKS = 8


def parse_defs(path: str = DEFS_PATH) -> Dict[str, int]:
    """``constexpr uint32_t NAME = <int>;`` lines of mk_defs.hpp -> {NAME: value}."""
    out: Dict[str, int] = {}
    pat = re.compile(r"^\s*constexpr\s+uint32_t\s+([A-Z_0-9]+)\s*=\s*([0-9]+)\s*;")
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m:
                if m.group(1) in out:
                    raise ValueError(f"mk_defs.hpp defines {m.group(1)} twice")
                out[m.group(1)] = int(m.group(2))
    return out


DEFS = parse_defs()
globals().update(DEFS)  # D_T, CB_IN0, S_X_FLAG, A_ROLE, ... as module constants


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


# ======================================================================================================== shapes
@dataclass(frozen=True)
class Shape:
    name: str
    prefix_len: int  # P (valid + pad prefix rows)
    suffix_rows: int  # S (tile-padded action rows)
    horizon: int  # H
    chunk_tiles: int  # key tiles per attention unit

    @property
    def rt(self) -> int:
        return self.suffix_rows // TILE

    @property
    def pt(self) -> int:
        return self.prefix_len // TILE

    @property
    def nkt(self) -> int:
        return self.pt + self.rt

    @property
    def nch(self) -> int:
        return self.nkt // self.chunk_tiles

    @property
    def nu(self) -> int:
        """Attention units per head column (rows of the column)."""
        return self.nch * self.rt

    def check(self) -> None:
        if self.prefix_len % TILE or self.suffix_rows % TILE:
            raise ValueError(f"{self.name}: P and S must be tile multiples")
        if self.nkt % self.chunk_tiles:
            raise ValueError(f"{self.name}: {self.nkt} key tiles do not split into chunks of {self.chunk_tiles}")
        if self.nu > GRID[1]:
            raise ValueError(f"{self.name}: {self.nu} units per head column exceed the {GRID[1]} grid rows")
        if self.chunk_tiles > 7:
            raise ValueError("a chunk's scores + one temporary must fit the 8 fp32 DST tiles")
        if self.nch > 6:
            # merge() (mk_trisc.cpp) holds M / L in DST 0, the NCH weights in DST 1..NCH and a temporary in DST 7
            raise ValueError("the merge keeps NCH weights in DST 1..NCH + temporaries in DST 0 and 7: NCH <= 6")
        if self.rt not in (1, 2):
            raise ValueError("RT must be 1 or 2 (DST budget of the MLP epilogue)")
        # the last chunk must contain every suffix key tile (the KL round feeds only the last-chunk units)
        if (self.nch - 1) * self.chunk_tiles > self.pt:
            raise ValueError("suffix key tiles fall outside the last chunk")


SHAPES: Dict[str, Shape] = {
    "base": Shape("base", prefix_len=736, suffix_rows=64, horizon=50, chunk_tiles=5),
    "libero": Shape("libero", prefix_len=544, suffix_rows=32, horizon=10, chunk_tiles=3),
}
for _s in SHAPES.values():
    _s.check()


def shape_for(prefix_len: int, suffix_rows: int) -> Shape:
    """The megakernel's shape contract (DESIGN.md §4.12 refusal (a)): only the two served shapes exist."""
    for s in SHAPES.values():
        if s.prefix_len == prefix_len and s.suffix_rows == suffix_rows:
            return s
    raise RuntimeError(
        f"PI05_MEGAKERNEL: no megakernel program for prefix {prefix_len} / suffix {suffix_rows} rows "
        f"(supported: {[(s.prefix_len, s.suffix_rows) for s in SHAPES.values()]})")


# ======================================================================================================== roles
@dataclass
class CoreRole:
    xy: Tuple[int, int]
    bits: int = 0
    pair_kind: int = 0
    pair_j: int = 0
    head: int = 0
    unit_r: int = 0
    unit_kc: int = 0
    owner_n: int = 0
    mlp_kg: int = 0
    mlp_ng: int = 0

    def has(self, bit: int) -> bool:
        return bool(self.bits & bit)


def owner_xy(n: int) -> Tuple[int, int]:
    return (n // 4, 4 + n % 4)


def build_roles(shape: Shape) -> Dict[Tuple[int, int], CoreRole]:
    roles = {(x, y): CoreRole((x, y)) for x in range(GRID[0]) for y in range(GRID[1])}
    rt = shape.rt
    for x in range(10):
        for y in range(4):
            c = roles[(x, y)]
            c.bits |= R_PAIR | R_XRECV
            c.pair_kind = PK_Q if x < 8 else (PK_K if x == 8 else PK_V)
            c.pair_j = y
            c.head = x if x < 8 else 0
    for h in range(NH):
        for kc in range(shape.nch):
            for r in range(rt):
                c = roles[(h, kc * rt + r)]
                c.bits |= R_UNIT
                c.head, c.unit_r, c.unit_kc = h, r, kc
                if kc == 0:
                    c.bits |= R_MERGER
                if kc == shape.nch - 1:
                    c.bits |= R_LASTCH
        roles[(h, 0)].bits |= R_QLEAD
    roles[KL].bits |= R_KL
    for n in range(D_T):
        c = roles[owner_xy(n)]
        c.bits |= R_OWNER
        c.owner_n = n
    for kg in range(8):
        for ng in range(8):
            c = roles[(ng, kg)]
            c.bits |= R_MLP
            c.mlp_kg, c.mlp_ng = kg, ng
        roles[(0, kg)].bits |= R_ROWLEAD
    roles[H0].bits |= R_H0
    roles[H1].bits |= R_H1
    return roles


def qkv_col(kind: int, head: int, d: int) -> int:
    """qkv output tile column of (kind, head, dh tile d): [Q heads 0..7 | K | V], 8 tiles each."""
    if kind == PK_Q:
        return head * DH_T + d
    return NH * DH_T + (0 if kind == PK_K else DH_T) + d


def mlp_hcols(kg: int, ng: int) -> Tuple[int, int]:
    """h column tiles (= up / gate output columns) of MLP core (kg, ng)."""
    base = kg * 16 + 2 * ng
    return base, base + 1


def unit_chunk(shape: Shape, kc: int) -> Tuple[int, int]:
    """(first key tile, prefix key tiles) of chunk kc."""
    kt0 = kc * shape.chunk_tiles
    npre = max(0, min(shape.chunk_tiles, shape.pt - kt0))
    return kt0, npre


# ======================================================================================================== CBs
@dataclass(frozen=True)
class CBSpec:
    cb_id: int
    name: str
    fmt: str  # bf16 | bfp8 | fp32 | raw
    pages: int
    page_bytes: int
    fp32_unpack: bool = False

    @property
    def total_bytes(self) -> int:
        return self.pages * self.page_bytes


def cb_table(shape: Shape) -> List[CBSpec]:
    rt, cht, nch = shape.rt, shape.chunk_tiles, shape.nch
    b16, b8, f32 = TILE_BYTES["bf16"], TILE_BYTES["bfp8"], TILE_BYTES["fp32"]
    t = [
        CBSpec(CB_IN0, "in0", "bf16", 64 * rt, b16),
        CBSpec(CB_W8, "w8", "bfp8", W8_PAGES * PAGE_TILES, b8),
        CBSpec(CB_W16, "w16", "bf16", W16_PAGES * PAGE_TILES, b16),
        CBSpec(CB_WC, "wc", "bf16", WC_PAGES * PAGE_TILES, b16),
        CBSpec(CB_RTOK, "rtok", "bf16", 1, 32),
        CBSpec(CB_QKVO, "qkvo", "bf16", 2 * rt, b16),
        CBSpec(CB_TAB, "tab", "bf16", 4 * rt, b16),
        CBSpec(CB_Q, "q", "bf16", DH_T * rt, b16),
        CBSpec(CB_KSV, "ksv", "bf16", 2 * DH_T * rt, b16),
        CBSpec(CB_KV, "kv", "bfp8", 2 * cht * DH_T, b8),
        CBSpec(CB_MASK, "mask", "bf16", cht, b16),
        CBSpec(CB_S, "s", "bf16", cht, b16),
        CBSpec(CB_M, "m", "bf16", 1, b16),
        CBSpec(CB_MF, "mf", "bf16", 1, b16),
        CBSpec(CB_L, "l", "fp32", 1, f32, fp32_unpack=True),
        CBSpec(CB_OP, "op", "bf16", DH_T, b16),
        CBSpec(CB_PART, "part", "bf16", DH_T * (nch - 1), b16),
        CBSpec(CB_PM, "pm", "bf16", nch - 1, b16),
        CBSpec(CB_PL, "pl", "fp32", nch - 1, f32, fp32_unpack=True),
        CBSpec(CB_D, "d", "fp32", nch, f32),
        CBSpec(CB_CTXO, "ctxo", "bf16", DH_T, b16),
        CBSpec(CB_XRES, "xres", "fp32", rt, f32, fp32_unpack=True),
        CBSpec(CB_XS, "xs", "bf16", rt, b16),
        CBSpec(CB_H, "h", "bf16", 2 * rt, b16),
        CBSpec(CB_HG, "hg", "bf16", 16 * rt, b16),
        CBSpec(CB_DP, "dp", "fp32", 4 * rt, f32),
        CBSpec(CB_RED, "red", "fp32", 8 * rt, f32),
        CBSpec(CB_CONST, "const", "bf16", 4, b16),
        CBSpec(CB_SCR, "scr", "fp32", 2, f32),
        CBSpec(CB_SCR16, "scr16", "bf16", rt, b16),
        CBSpec(CB_ROUT, "rout", "bf16", rt, b16),
        CBSpec(CB_SYNC, "sync", "raw", 1, N_SYNC * SYNC_STRIDE),
    ]
    ids = [c.cb_id for c in t]
    if sorted(ids) != list(range(N_CBS)):
        raise AssertionError(f"CB ids must be dense 0..{N_CBS - 1}: {sorted(ids)}")
    return t


def cb_union_bytes(shape: Shape) -> int:
    return sum(c.total_bytes for c in cb_table(shape))


# ======================================================================================================== streams
# Weight / constant items per core in consumption order. An item = (ring, page_count, tag); tags are resolved to
# tiles by arena.py. Per (step, layer) unless noted.
def core_streams(role: CoreRole, shape: Shape) -> Dict[str, List[Tuple[str, int]]]:
    """{ring: [(tag, pages)]} of ONE (step, layer) for a compute core, or of ONE step for H0 (ring 'w16')."""
    w8: List[Tuple[str, int]] = []
    w16: List[Tuple[str, int]] = []
    wc: List[Tuple[str, int]] = []
    if role.has(R_H0):
        w16 = [("inproj", 8), ("w_out", 4), ("c_out", 1), ("pad", 3)]  # 16 pages: every step starts at ring page 0
        return {"w8": w8, "w16": w16, "wc": wc}
    if role.has(R_PAIR):
        w8.append(("qkv", 8))
    if role.has(R_MLP):
        w8.append(("ug", 16))
    if role.has(R_PAIR) or role.has(R_MLP) or role.has(R_OWNER):
        wc.append(("const", 1))
    if role.has(R_OWNER):
        w16.append(("wo", 8))
    if role.has(R_MLP):
        w16.append(("wd", 8))
    return {"w8": w8, "w16": w16, "wc": wc}


def pages_per_step(role: CoreRole, shape: Shape) -> Dict[str, int]:
    st = core_streams(role, shape)
    mult = 1 if role.has(R_H0) else N_LAYERS
    return {k: mult * sum(p for _, p in v) for k, v in st.items()}


@dataclass
class BankPlan:
    """Per-core (bank, byte offsets) of the three streams inside every per-step arena tensor pair."""
    bank: Dict[Tuple[int, int], int] = field(default_factory=dict)
    off8: Dict[Tuple[int, int], int] = field(default_factory=dict)
    off16: Dict[Tuple[int, int], int] = field(default_factory=dict)
    offc: Dict[Tuple[int, int], int] = field(default_factory=dict)
    pages: Dict[Tuple[int, int], Dict[str, int]] = field(default_factory=dict)
    tiles8_per_bank: int = 0  # shard height of the w8 tensor, tiles
    tiles16_per_bank: int = 0  # shard height of the w16 tensor (w16 + wc streams), tiles


def plan_banks(shape: Shape) -> BankPlan:
    """Greedy byte balance of the streaming cores over the 8 DRAM banks (direct mode: one bank per core)."""
    roles = build_roles(shape)
    plan = BankPlan()
    items = []
    for xy, r in roles.items():
        p = pages_per_step(r, shape)
        if sum(p.values()) == 0:
            continue
        b = p["w8"] * PAGE_TILES * TILE_BYTES["bfp8"] + (p["w16"] + p["wc"]) * PAGE_TILES * TILE_BYTES["bf16"]
        items.append((b, xy, p))
    items.sort(key=lambda t: (-t[0], t[1]))
    load = [0] * N_BANKS
    t8 = [0] * N_BANKS
    t16 = [0] * N_BANKS
    for b, xy, p in items:
        bank = min(range(N_BANKS), key=lambda i: (load[i], i))
        load[bank] += b
        plan.bank[xy] = bank
        plan.pages[xy] = p
        plan.off8[xy] = t8[bank] * TILE_BYTES["bfp8"]
        t8[bank] += p["w8"] * PAGE_TILES
        plan.off16[xy] = t16[bank] * TILE_BYTES["bf16"]
        t16[bank] += p["w16"] * PAGE_TILES
        plan.offc[xy] = t16[bank] * TILE_BYTES["bf16"]
        t16[bank] += p["wc"] * PAGE_TILES
    plan.tiles8_per_bank = max(t8)
    plan.tiles16_per_bank = max(t16)
    return plan


# ======================================================================================================== checks
def l1_budget(shape: Shape) -> Dict[str, int]:
    union = cb_union_bytes(shape)
    return {"cb_union": union, "cbs": {c.name: c.total_bytes for c in cb_table(shape)}}


def megakernel_refusal(kv_dtype: str, num_steps: int) -> Optional[str]:
    """Why ``PI05_MEGAKERNEL=expert`` cannot serve this configuration (None = it can). The kernels hard-code
    ``N_STEPS`` denoising steps (mk_defs.hpp; the NCRISC / BRISC / TRISC loops and the per-step arenas) and land every
    K / V cache page at the bfp8 tile stride into the bfp8 ``CB_KV`` (mk_brisc.cpp), so a bf16 cache
    (PI05_KV_DTYPE=bf16) or another step count would give silent wrong actions, never an error."""
    if kv_dtype != "bf8":
        return (f"PI05_KV_DTYPE={kv_dtype} (the megakernel reads bfp8 K / V caches at the {TILE_BYTES['bfp8']} B bfp8 "
                "page stride; use PI05_KV_DTYPE=bf8)")
    if int(num_steps) != N_STEPS:
        return (f"PI05_NUM_STEPS / num_denoising_steps={num_steps} (the megakernel compiles exactly {N_STEPS} "
                f"denoising steps; use {N_STEPS})")
    return None


# multicast rectangles (logical corners as ExpertMegakernel.common_args / core_args build them), the sender, and the
# destination count the kernels hard-code (mk_brisc.cpp setup: literal or common arg). count = cores in the rectangle
# minus the sender when the sender lies inside it (the multicasts are issued without loopback).
def mcast_rects(shape: Shape) -> Dict[str, Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]]:
    rt = shape.rt
    return {
        "rx": ((0, 0), (9, 7), [H0]),
        "rm": ((0, 0), (7, 7), [H0]),
        "ro": ((0, 4), (7, 7), [H1]),
        "rk": ((0, (shape.nch - 1) * rt), (7, shape.nch * rt - 1), [KL]),
        "row": ((0, 0), (7, 0), [(0, 0)]),  # row kg = 0 (every row kg has the same form, sender = row leader (0, kg))
        "ra": ((0, 0), (GRID[0] - 1, GRID[1] - 1), [H0]),
        "col": ((0, 0), (0, shape.nu - 1), [(0, 0)]),  # head 0's column (every head h: (h, 0)..(h, NU-1), sender (h, 0))
    }


def mcast_dest_count(lo, hi, sender) -> int:
    n = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1)
    return n - (1 if lo[0] <= sender[0] <= hi[0] and lo[1] <= sender[1] <= hi[1] else 0)


def kernel_mcast_counts(shape: Shape, brisc_src: Optional[str] = None) -> Dict[str, int]:
    """The destination counts mk_brisc.cpp's setup assigns to each rectangle (parsed from the source)."""
    if brisc_src is None:
        with open(os.path.join(KDIR, "mk_brisc.cpp")) as f:
            brisc_src = f.read()
    n_rx, _ = x_round_receivers(shape)
    env = {"RT": shape.rt, "NU": shape.nu, "me.n_cores": GRID[0] * GRID[1], "ct_arg(C_N_RX)": n_rx}
    out = {}
    for name in ("rx", "rm", "ro", "rk", "row", "ra", "col"):
        m = re.search(r"me\." + name + r" = \{[^;]*?,\s*([^,;{}]+)\};", brisc_src)
        if m is None:
            raise AssertionError(f"mk_brisc.cpp: no 'me.{name} = {{...}}' setup line")
        expr = m.group(1).strip()
        for k, v in env.items():
            expr = expr.replace(k, str(v))
        if not re.fullmatch(r"[0-9 *+\-]+", expr):
            raise AssertionError(f"mk_brisc.cpp: me.{name} count {m.group(1)!r} is not a known expression")
        out[name] = int(eval(expr))  # digits and + - * only (checked above)
    return out


def check_roles(shape: Shape) -> None:
    """Host invariants of the role tables (fail loudly before any device time)."""
    roles = build_roles(shape)
    rt = shape.rt
    pairs = [r for r in roles.values() if r.has(R_PAIR)]
    assert len(pairs) == 40
    cols = sorted(qkv_col(p.pair_kind, p.head, d) for p in pairs for d in (p.pair_j, p.pair_j + 4))
    assert cols == list(range(80)), "every qkv column produced exactly once"
    units = [r for r in roles.values() if r.has(R_UNIT)]
    assert len(units) == NH * shape.nu
    keys = sorted((u.head, u.unit_r, t) for u in units
                  for t in range(u.unit_kc * shape.chunk_tiles, (u.unit_kc + 1) * shape.chunk_tiles))
    assert keys == sorted((h, r, t) for h in range(NH) for r in range(rt) for t in range(shape.nkt))
    owners = sorted(r.owner_n for r in roles.values() if r.has(R_OWNER))
    assert owners == list(range(D_T))
    mlp = [r for r in roles.values() if r.has(R_MLP)]
    hc = sorted(c for m in mlp for c in mlp_hcols(m.mlp_kg, m.mlp_ng))
    assert hc == list(range(MLP_T)), "every h column produced exactly once"
    for m in mlp:  # the down partial of (kg, ng) reads exactly the h columns of row kg
        row = sorted(c for mm in mlp if mm.mlp_kg == m.mlp_kg for c in mlp_hcols(mm.mlp_kg, mm.mlp_ng))
        assert row == list(range(m.mlp_kg * 16, m.mlp_kg * 16 + 16))
        for n in range(4 * m.mlp_ng, 4 * m.mlp_ng + 4):  # owners of its N tiles are in its own column
            assert owner_xy(n)[0] == m.mlp_ng
    for xy, r in roles.items():  # role exclusions the kernels rely on
        if r.has(R_PAIR):
            assert not r.has(R_OWNER), xy
        if r.has(R_H0) or r.has(R_H1) or r.has(R_KL):
            assert (r.bits & ~(R_H0 | R_H1 | R_KL)) == 0, xy
        if r.has(R_OWNER):
            assert r.has(R_MLP) and r.xy[1] >= 4, xy
        if r.has(R_MERGER):
            assert r.unit_kc == 0 and r.xy == (r.head, r.unit_r)
            # a merger prefetches the next generation's K / V on the MLP path (mk_brisc.cpp step 9); off it, the
            # prefetch never runs and its TRISC waits on CB_KV forever at generation 1
            assert r.has(R_MLP), xy
        if r.has(R_UNIT) and xy[1] < 8:
            assert r.has(R_MLP), xy  # same prefetch rule: only units at y >= 8 take the non-MLP (step 5) path
    # multicast destination counts: kernel literals == counts derived from the rectangles and the sender
    kc = kernel_mcast_counts(shape)
    for name, (lo, hi, senders) in mcast_rects(shape).items():
        for snd in senders:
            assert kc[name] == mcast_dest_count(lo, hi, snd), (name, kc[name], lo, hi, snd)
    for kg in range(8):  # every row-leader rectangle has the same count as row 0
        assert mcast_dest_count((0, kg), (7, kg), (0, kg)) == kc["row"]
    for h in range(NH):  # and every Q-leader column the same as head 0's
        assert mcast_dest_count((h, 0), (h, shape.nu - 1), (h, 0)) == kc["col"]
    lo, hi, _ = mcast_rects(shape)["rx"]
    inside = [roles[(x, y)] for x in range(lo[0], hi[0] + 1) for y in range(lo[1], hi[1] + 1)]
    n_rx, n_xrdy = x_round_receivers(shape)
    assert n_rx == len(inside)
    assert n_xrdy == sum(1 for r in inside if r.has(R_PAIR) or r.has(R_OWNER))
    assert all(r.xy in {c.xy for c in inside} for r in roles.values() if r.has(R_PAIR) or r.has(R_OWNER))
    # multicast rectangles: every receiver of each rectangle is a consumer or explicitly idle
    assert all(roles[(x, y)].has(R_MLP) for x in range(8) for y in range(8))
    # sync-word slots are distinct lines
    words = [v for k, v in DEFS.items() if k.startswith("S_") and k not in ("S_SRC0", "S_DBG")]
    assert len(set(words)) == len(words) and max(words) < DEFS["S_SRC0"]
    assert DEFS["S_DBG"] + 4 <= DEFS["N_SYNC"]
    # rt args fit
    assert max(v for k, v in DEFS.items() if k.startswith("A_")) < DEFS["N_RT_ARGS"]
    assert max(v for k, v in DEFS.items() if k.startswith("C_")) < DEFS["N_COMMON_ARGS"]
    assert DEFS["C_V_ADDR"] + N_LAYERS <= DEFS["N_COMMON_ARGS"]
    assert DEFS["C_W16_ADDR"] + N_STEPS == DEFS["C_K_ADDR"] and DEFS["C_W8_ADDR"] + N_STEPS == DEFS["C_W16_ADDR"]


def x_round_receivers(shape: Shape) -> Tuple[int, int]:
    """(cores in the x rectangle (0,0)-(9,7), receivers that announce ready = pair cores + owners)."""
    return 80, 40 + 32


def describe(shape: Shape) -> Dict[str, object]:
    return {
        "shape": shape.name, "P": shape.prefix_len, "S": shape.suffix_rows, "H": shape.horizon,
        "RT": shape.rt, "PT": shape.pt, "NKT": shape.nkt, "CHT": shape.chunk_tiles, "NCH": shape.nch,
        "cb_union_bytes": cb_union_bytes(shape),
    }
