# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-step DRAM weight arenas of the expert megakernel (direct mode, one DRAM bank per streaming core).

Per denoising step s two tensors, both WIDTH_SHARDED over the 8 DRAM banks with shard ``[T*32, 32]`` (bank b's shard
is one contiguous tile stream):

* ``w8[s]``  (bfp8): per core, in consumption order over the 18 layers: the pair core's folded qkv share
  (``[kb][col][k]``, 8 pages) and the MLP core's folded up|gate share (``[j][kb][u | g][k]``, 16 pages);
* ``w16[s]`` (bf16): per core the o_proj share (owner n: ``[kb][k]`` of column n, 8 pages) and the down share
  (MLP (kg, ng): ``[kb][n][k]``, 8 pages) per layer, then (same tensor, own offset) the constant stream: one page per
  layer ``[c_qkv j, c_qkv j+4, c_u j0, c_u j1, c_g j0, c_g j1, g_attn n, g_mlp n]`` (row-broadcast tiles); H0's
  per-step stream ``4 x (W_in page, b_in page), Wout'[s] (4 pages), c_out (1 page), 3 pad pages``.

The byte offsets come from ``geometry.plan_banks`` and are identical in every step tensor (a runtime-arg table per
core, not per step). Tile orders are pinned against the TRISC loops by ``tests/pcc/test_pi05_megakernel_host.py``.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from . import geometry as G
from .host_model import ExpertParams, fold

T = G.TILE


def w_tiles(w: torch.Tensor) -> torch.Tensor:
    """[K, N] -> [Kt, Nt, 32, 32] (tile (kt, nt) = w[kt*32:(kt+1)*32, nt*32:(nt+1)*32])."""
    k, n = w.shape
    return w.reshape(k // T, T, n // T, T).permute(0, 2, 1, 3)


def row_bcast_tiles(v: torch.Tensor) -> torch.Tensor:
    """[N] -> [Nt, 32, 32] with every row of tile j equal to v[32j:32j+32]."""
    return v.reshape(-1, 1, T).expand(-1, T, T)


def qkv_share(tq: torch.Tensor, c0: int, c1: int) -> torch.Tensor:
    """Pair core: [kb][c][k] over 32 K tiles of columns (c0, c1) -> [64, 32, 32]."""
    sel = tq[:, [c0, c1]]  # [32, 2, 32, 32]
    return sel.reshape(4, 8, 2, T, T).permute(0, 2, 1, 3, 4).reshape(-1, T, T)


def ug_share(tu: torch.Tensor, j0: int, j1: int) -> torch.Tensor:
    """MLP core: [jj][kb][u | g][k] -> [128, 32, 32] (u column j, g column 128 + j of the [up | gate] weight)."""
    out = []
    for j in (j0, j1):
        sel = tu[:, [j, G.MLP_T + j]]  # [32, 2, 32, 32]
        out.append(sel.reshape(4, 8, 2, T, T).permute(0, 2, 1, 3, 4).reshape(-1, T, T))
    return torch.cat(out, 0)


def wo_share(to: torch.Tensor, n: int) -> torch.Tensor:
    """Owner n: [kb][k] of o_proj column n (64 K tiles) -> [64, 32, 32]."""
    return to[:, n].contiguous()


def wd_share(td: torch.Tensor, kg: int, ng: int) -> torch.Tensor:
    """MLP (kg, ng): [kb][n][k], K tiles kg*16..+16, N tiles 4 ng..+4 -> [64, 32, 32]."""
    sel = td[kg * 16 : (kg + 1) * 16, 4 * ng : 4 * ng + 4]  # [16, 4, 32, 32]
    return sel.reshape(2, 8, 4, T, T).permute(0, 2, 1, 3, 4).reshape(-1, T, T)


class ArenaBuilder:
    """Host packing of the per-step arenas for one shape (torch only; ``upload`` needs ttnn)."""

    def __init__(self, params: ExpertParams, shape: G.Shape):
        self.p = params
        self.shape = shape
        self.roles = G.build_roles(shape)
        self.plan = G.plan_banks(shape)
        self.n_steps = len(params.dts)

    # ------------------------------------------------------------------ per-(step, layer) folded constants
    def _layer(self, s: int, l: int):
        si, hi, gi, sp, hp, gp = self.p.mods[s][l]
        wq, cq = fold(self.p.wqkv[l], si, hi)
        wu, cu = fold(self.p.wug[l], sp, hp)
        return (
            w_tiles(wq),
            row_bcast_tiles(cq[0]),
            w_tiles(wu),
            row_bcast_tiles(cu[0]),
            row_bcast_tiles(gi),
            row_bcast_tiles(gp),
        )

    def step(self, s: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(w8 [8, T8, 32, 32] fp32, w16 [8, T16, 32, 32] fp32) of step s."""
        plan = self.plan
        a8 = torch.zeros(G.N_BANKS, plan.tiles8_per_bank, T, T)
        a16 = torch.zeros(G.N_BANKS, plan.tiles16_per_bank, T, T)
        pos8 = {xy: plan.off8[xy] // G.TILE_BYTES["bfp8"] for xy in plan.bank}
        pos16 = {xy: plan.off16[xy] // G.TILE_BYTES["bf16"] for xy in plan.bank}
        posc = {xy: plan.offc[xy] // G.TILE_BYTES["bf16"] for xy in plan.bank}
        to = [w_tiles(w) for w in self.p.wo]
        td = [w_tiles(w) for w in self.p.wd]

        def put(arena, pos, xy, tiles):
            b = plan.bank[xy]
            n = tiles.shape[0]
            arena[b, pos[xy] : pos[xy] + n] = tiles
            pos[xy] += n

        for l in range(G.N_LAYERS):
            tq, cq, tu, cu, ga, gm = self._layer(s, l)
            for xy, r in self.roles.items():
                if xy not in plan.bank or r.has(G.R_H0):
                    continue
                const = torch.zeros(G.PAGE_TILES, T, T)
                if r.has(G.R_PAIR):
                    c0 = G.qkv_col(r.pair_kind, r.head, r.pair_j)
                    c1 = G.qkv_col(r.pair_kind, r.head, r.pair_j + 4)
                    put(a8, pos8, xy, qkv_share(tq, c0, c1))
                    const[G.WC_CQKV] = cq[c0]
                    const[G.WC_CQKV + 1] = cq[c1]
                if r.has(G.R_MLP):
                    j0, j1 = G.mlp_hcols(r.mlp_kg, r.mlp_ng)
                    put(a8, pos8, xy, ug_share(tu, j0, j1))
                    const[G.WC_CUG + 0] = cu[j0]
                    const[G.WC_CUG + 1] = cu[j1]
                    const[G.WC_CUG + 2] = cu[G.MLP_T + j0]
                    const[G.WC_CUG + 3] = cu[G.MLP_T + j1]
                if r.has(G.R_OWNER):
                    put(a16, pos16, xy, wo_share(to[l], r.owner_n))
                    const[G.WC_GATTN] = ga[r.owner_n]
                    const[G.WC_GMLP] = gm[r.owner_n]
                if r.has(G.R_MLP):
                    put(a16, pos16, xy, wd_share(td[l], r.mlp_kg, r.mlp_ng))
                if r.has(G.R_PAIR) or r.has(G.R_MLP) or r.has(G.R_OWNER):
                    put(a16, posc, xy, const)
        # H0: in-projection pages, folded out-projection, c_out, pad
        xy = G.H0
        tin = w_tiles(self.p.w_in)[0]  # [32 N tiles, 32, 32] (one K tile)
        bin_ = row_bcast_tiles(self.p.b_in)
        for nb in range(4):
            put(a16, pos16, xy, tin[nb * 8 : (nb + 1) * 8])
            put(a16, pos16, xy, bin_[nb * 8 : (nb + 1) * 8])
        sf, hf = self.p.final[s]
        wout, cout = fold(self.p.w_out, sf, hf)
        put(a16, pos16, xy, w_tiles(wout)[:, 0])  # [32 K tiles, 32, 32]
        page = torch.zeros(G.PAGE_TILES, T, T)
        page[0] = row_bcast_tiles(cout[0] + self.p.b_out)[0]
        put(a16, pos16, xy, page)
        put(a16, pos16, xy, torch.zeros(3 * G.PAGE_TILES, T, T))
        # every stream filled exactly its planned length
        for xy2 in plan.bank:
            p = plan.pages[xy2]
            assert (
                pos8[xy2] * G.TILE_BYTES["bfp8"] - plan.off8[xy2] == p["w8"] * G.PAGE_TILES * G.TILE_BYTES["bfp8"]
            ), xy2
            assert (
                pos16[xy2] * G.TILE_BYTES["bf16"] - plan.off16[xy2] == p["w16"] * G.PAGE_TILES * G.TILE_BYTES["bf16"]
            ), xy2
            assert (
                posc[xy2] * G.TILE_BYTES["bf16"] - plan.offc[xy2] == p["wc"] * G.PAGE_TILES * G.TILE_BYTES["bf16"]
            ), xy2
        return a8, a16


def to_device_layout(arena: torch.Tensor) -> torch.Tensor:
    """[8, T, 32, 32] -> [T*32, 8*32]: the WIDTH_SHARDED tensor whose bank-b shard is bank b's tile stream."""
    nb, t, th, tw = arena.shape
    return arena.permute(1, 2, 0, 3).reshape(t * th, nb * tw)


def memory_config(n_tiles_per_bank: int):
    import ttnn

    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(G.N_BANKS - 1, 0))])
    spec = ttnn.ShardSpec(grid, (n_tiles_per_bank * T, T), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def upload(builder: ArenaBuilder, device) -> Tuple[List, List]:
    """Every denoising step of the builder's parameters -> (w8 tensors, w16 tensors) on the device."""
    import ttnn

    w8, w16 = [], []
    for s in range(builder.n_steps):
        a8, a16 = builder.step(s)
        w8.append(
            ttnn.from_torch(
                to_device_layout(a8),
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=memory_config(a8.shape[1]),
            )
        )
        del a8
        w16.append(
            ttnn.from_torch(
                to_device_layout(a16).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=memory_config(a16.shape[1]),
            )
        )
        del a16
    return w8, w16
