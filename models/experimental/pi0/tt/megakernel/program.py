# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The EXPERT program of a pi0.5 call: weight arenas, runtime arguments and the program of the denoising loop.

``ExpertMegakernel(device, params, shape)`` uploads the per-step weight arenas and the constant tiles once. Its program
(``kernels/mk_*.cpp``) runs the N-step x 18-layer expert loop (plus the action in / out projections and the Euler
steps): it reads the prefix K / V caches the PREFIX program wrote, the key-mask row, the q / k RoPE tables and the
noise, and writes x_0 ``[1, S, 32]`` bf16. Every per-request value is tensor data at a fixed address and the runtime
args hold only addresses and shape constants, so the program is trace-capturable.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch

from . import geometry as G
from .host_model import ExpertParams

KERNELS = {r: os.path.join(G.KDIR, f"mk_{r}.cpp") for r in ("ncrisc", "brisc", "trisc")}


def merge_defines(production, extra=()) -> List[Tuple[str, str]]:
    """The JIT keeps the FIRST definition of a repeated name: merge by name so ``extra`` (test-only A / B builds)
    overrides a production value; production order otherwise, new names appended."""
    out = dict(production)
    out.update(dict(extra))
    return list(out.items())


EXPERT_DEFINES = [("MK_FID8_HIFI2", "1"), ("MK_FID", "3")]


def kv_mcast(shape: G.Shape, kv_dram: bool = False) -> bool:
    """The K / V chunk row multicast (MK_KV_MCAST) for this expert shape: on at two query row tiles (S64), off at one
    (S32) with L1 caches, where it measured slower (MULTICONFIG WP4 hold lat1: +0.3 to +2.7 ms at c2 / c3 S32, -0.5 to
    -1.7 ms at c3 S64; outputs bit-identical either way); always on with DRAM caches (the lead's DRAM + mcast
    fallback)."""
    return shape.rt == 2 or kv_dram


def core_range_set():
    import ttnn

    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(G.GRID[0] - 1, G.GRID[1] - 1))])


def compute_config(unpack_modes):
    """The TRISC config of every program: HiFi2, fp32 DST accumulation, the given per-CB unpack-to-DST modes."""
    import ttnn

    cc = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, dst_full_sync_en=True, math_approx_mode=False
    )
    cc.unpack_to_dest_mode = unpack_modes
    return cc


def semaphores():
    """Semaphores 0 / 1: the expert loop's boot barrier; 2 / 3: the prefix engine's."""
    import ttnn

    return [ttnn.SemaphoreDescriptor(id=i, core_ranges=core_range_set(), initial_value=0) for i in (0, 1, 2, 3)]


def constants_tensor(device):
    import ttnn

    ones = torch.ones(32, 32)
    t = torch.cat([ones, ones / 1024.0, torch.eye(32), torch.zeros(32, 32)], dim=1)  # [32, 128]: 4 tiles in a row
    return ttnn.from_torch(
        t.reshape(1, 1, 32, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class ExpertMegakernel:
    def __init__(
        self,
        device,
        params: ExpertParams,
        shape: G.Shape,
        arenas: Optional[Tuple[List, List]] = None,
        extra_defines: Sequence[Tuple[str, str]] = (),
        kv_dram: bool = False,
    ):
        import ttnn

        self.kv_dram = kv_dram  # the K / V caches are in DRAM (presets.kv_in_dram): read with the row multicast

        from .arena import ArenaBuilder, upload

        self.extra_defines = list(extra_defines)  # test-only A / B builds (override by name); empty in production
        G.check_roles(shape)
        if not 1 <= len(params.dts) <= G.N_STEPS:
            raise RuntimeError(
                f"pi0.5 megakernel: the schedule has {len(params.dts)} denoising steps, the kernels run 1..{G.N_STEPS}"
            )
        dts = {G.f32_bits(d) for d in params.dts}
        if len(dts) != 1:
            raise RuntimeError(f"the megakernel runs one Euler dt; the schedule has {len(dts)} distinct fp32 values")
        self.n_steps = len(params.dts)
        self.ngen = G.N_LAYERS * self.n_steps
        self.dt_bits = dts.pop()
        self.eps_bits = G.f32_bits(params.eps)
        self.device = device
        self.shape = shape
        self.roles = G.build_roles(shape)
        self.plan = G.plan_banks(shape)
        self.w8, self.w16 = arenas if arenas is not None else upload(ArenaBuilder(params, shape), device)
        self.consts = constants_tensor(device)
        self.dbg = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, shape.suffix_rows, 1024]),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        self._noc = {}
        for x in range(G.GRID[0]):
            for y in range(G.GRID[1]):
                c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
                self._noc[(x, y)] = (int(c.x), int(c.y))
        self.program_hash = None
        self.cb_union = G.cb_union_bytes(shape)

    # ------------------------------------------------------------------ args
    def noc(self, xy) -> Tuple[int, int]:
        return self._noc[tuple(xy)]

    def rect(self, lo, hi) -> List[int]:
        a, b = self.noc(lo), self.noc(hi)
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]

    def core_args(self, xy) -> List[int]:
        r = self.roles[xy]
        sh = self.shape
        a = [0] * G.N_RT_ARGS
        a[G.A_ROLE] = r.bits
        a[G.A_PAIR_KIND], a[G.A_PAIR_J], a[G.A_HEAD] = r.pair_kind, r.pair_j, r.head
        if r.has(G.R_PAIR):
            dst = (r.head, 0) if r.pair_kind == G.PK_Q else G.KL
            a[G.A_PAIR_DX], a[G.A_PAIR_DY] = self.noc(dst)
            a[G.A_PAIR_DT] = r.pair_j + (G.DH_T * sh.rt if r.pair_kind == G.PK_V else 0)
        if r.has(G.R_UNIT):
            a[G.A_UNIT_R], a[G.A_UNIT_KC] = r.unit_r, r.unit_kc
            kt0, npre = G.unit_chunk(sh, r.unit_kc)
            a[G.A_UNIT_KT0], a[G.A_UNIT_NPRE] = kt0, npre
            a[G.A_MERGER_X], a[G.A_MERGER_Y] = self.noc((r.head, r.unit_r))
            a[G.A_QLEAD_X], a[G.A_QLEAD_Y] = self.noc((r.head, 0))
        if r.has(G.R_QLEAD):
            a[G.A_COL_X0 : G.A_COL_Y1 + 1] = self.rect((r.head, 0), (r.head, sh.nu - 1))
        if r.has(G.R_OWNER):
            a[G.A_OWNER_N] = r.owner_n
        if r.has(G.R_MLP):
            a[G.A_MLP_KG], a[G.A_MLP_NG] = r.mlp_kg, r.mlp_ng
            a[G.A_ROWL_X], a[G.A_ROWL_Y] = self.noc((0, r.mlp_kg))
            for i in range(4):
                a[G.A_OWNX0 + i], a[G.A_OWNY0 + i] = self.noc(G.owner_xy(4 * r.mlp_ng + i))
        if r.has(G.R_ROWLEAD):
            a[G.A_ROW_X0 : G.A_ROW_Y1 + 1] = self.rect((0, r.mlp_kg), (7, r.mlp_kg))
        if xy in self.plan.bank:
            p = self.plan.pages[xy]
            a[G.A_BANK] = self.plan.bank[xy]
            a[G.A_W8_OFF], a[G.A_W8_PAGES] = self.plan.off8[xy], p["w8"]
            a[G.A_W16_OFF], a[G.A_W16_PAGES] = self.plan.off16[xy], p["w16"]
            a[G.A_WC_OFF], a[G.A_WC_PAGES] = self.plan.offc[xy], p["wc"]
        return a

    def kvm_args(self, xy) -> List[int]:
        """A_KVM_* (MK_KV_MCAST): the NoC coordinates of the unit row's head-0 unit and the row rectangle."""
        r = self.roles[xy]
        if not r.has(G.R_UNIT):
            return [0] * 6
        y = xy[1]
        return list(self.noc((0, y))) + list(self.rect((0, y), (7, y)))

    def common_args(self, kv: Sequence[Tuple], mask, tables, noise, out, ngen: int) -> List[int]:
        sh = self.shape
        c = [0] * G.N_COMMON_ARGS
        c[G.C_H0_X], c[G.C_H0_Y] = self.noc(G.H0)
        c[G.C_H1_X], c[G.C_H1_Y] = self.noc(G.H1)
        c[G.C_KL_X], c[G.C_KL_Y] = self.noc(G.KL)
        c[G.C_RX_X0 : G.C_RX_Y1 + 1] = self.rect((0, 0), (9, 7))
        c[G.C_RM_X0 : G.C_RM_Y1 + 1] = self.rect((0, 0), (7, 7))
        c[G.C_RO_X0 : G.C_RO_Y1 + 1] = self.rect((0, 4), (7, 7))
        last = sh.nu - sh.nu // sh.nch  # the last chunk's first unit row
        c[G.C_RK_X0 : G.C_RK_Y1 + 1] = self.rect((0, last), (7, sh.nu - 1))
        c[G.C_RA_X0 : G.C_RA_Y1 + 1] = self.rect((0, 0), (G.GRID[0] - 1, G.GRID[1] - 1))
        c[G.C_N_RX], c[G.C_N_XRDY] = G.x_round_receivers(sh)
        c[G.C_N_CORES] = G.GRID[0] * G.GRID[1]
        c[G.C_MASK] = mask.buffer_address()
        c[G.C_COSQ], c[G.C_SINQ], c[G.C_COSK], c[G.C_SINK] = [t.buffer_address() for t in tables]
        c[G.C_NOISE] = noise.buffer_address()
        c[G.C_OUT] = out.buffer_address()
        c[G.C_CONSTS] = self.consts.buffer_address()
        c[G.C_DEBUG] = int(ngen)
        c[G.C_DBGOUT] = self.dbg.buffer_address()
        c[G.C_DT_BITS] = self.dt_bits
        if len(self.w8) != self.n_steps or len(self.w16) != self.n_steps:
            raise RuntimeError(f"{len(self.w8)} / {len(self.w16)} step arenas for {self.n_steps} steps")
        for s in range(self.n_steps):
            c[G.C_W8_ADDR + s] = self.w8[s].buffer_address()
            c[G.C_W16_ADDR + s] = self.w16[s].buffer_address()
        for l in range(G.N_LAYERS):
            c[G.C_K_ADDR + l] = kv[l][0].buffer_address()
            c[G.C_V_ADDR + l] = kv[l][1].buffer_address()
        return c

    @staticmethod
    def check_kv(kv: Sequence[Tuple]) -> None:
        """The kernels read ``N_LAYERS`` bfp8 (K, V) caches at the bfp8 page stride: anything else is wrong silently."""
        import ttnn

        bad = [
            (l, i, str(t.dtype)) for l, pair in enumerate(kv) for i, t in enumerate(pair) if t.dtype != ttnn.bfloat8_b
        ]
        if len(kv) != G.N_LAYERS or bad:
            raise RuntimeError(
                f"pi0.5 megakernel: the kernels read {G.N_LAYERS} bfp8 (K, V) caches; got {len(kv)} "
                f"layers, non-bfp8 (layer, K0/V1, dtype): {bad[:4]}"
            )

    def compile_time_args(self, kv, mask, tables, noise, out) -> List[int]:
        import ttnn

        sh = self.shape
        ct = [sh.rt, sh.pt, sh.chunk_tiles, sh.nch, self.eps_bits]
        for t in (kv[0][0], mask, tables[0], noise, out, self.consts, self.dbg):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return ct

    def io_tensors(self, kv, mask, tables, noise, out) -> List:
        ts = list(self.w8) + list(self.w16) + [t for pair in kv for t in pair]
        return ts + [mask] + list(tables) + [noise, self.consts, self.dbg, out]

    # ------------------------------------------------------------------ program
    def program(self, kv, mask, tables, noise, out, ngen: Optional[int] = None):
        import ttnn

        sh = self.shape
        self.check_kv(kv)
        cores = core_range_set()
        fmt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32, "raw": ttnn.bfloat16}
        cbs = [
            ttnn.CBDescriptor(
                total_size=c.total_bytes,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=c.cb_id, data_format=fmt[c.fmt], page_size=c.page_bytes)
                ],
            )
            for c in sorted(G.cb_table(sh), key=lambda c: c.cb_id != G.CB_SYNC)  # CB_SYNC first, then 0..30
        ]
        ct = self.compile_time_args(kv, mask, tables, noise, out)
        common = self.common_args(kv, mask, tables, noise, out, self.ngen if ngen is None else ngen)
        defines = self.defines()
        kvm = dict(defines).get("MK_KV_MCAST", "0") != "0"  # the K / V chunk multicast along the unit row
        rt = ttnn.RuntimeArgs()
        for x in range(G.GRID[0]):
            for y in range(G.GRID[1]):
                rt[x][y] = self.core_args((x, y)) + (self.kvm_args((x, y)) if kvm else [])
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for c in G.cb_table(sh):
            if c.fp32_unpack:
                modes[c.cb_id] = ttnn.UnpackToDestMode.UnpackToDestFp32
        fp = ttnn.KernelDescriptor.SourceType.FILE_PATH
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=KERNELS["ncrisc"],
                source_type=fp,
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                common_runtime_args=common,
                defines=defines,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNELS["brisc"],
                source_type=fp,
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                common_runtime_args=common,
                defines=defines,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNELS["trisc"],
                source_type=fp,
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt,
                common_runtime_args=common,
                defines=defines,
                config=compute_config(modes),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores(), cbs=cbs)

    def defines(self) -> List[Tuple[str, str]]:
        # MK_FID = 3: every expert matmul at HiFi3 (MULTICONFIG WP-P, 2026-10-02: HiFi2 truncates the in0 operand's
        # last mantissa bit, which dominated the expert error at 1 / 5 denoising steps; +1.1 ms at 10 steps)
        # MK_KV_MCAST = 1 (S64 shapes, kv_mcast): one read per K / V chunk per unit row, multicast to its 8 head units
        mcast = [("MK_KV_MCAST", "1")] if kv_mcast(self.shape, self.kv_dram) else []
        return merge_defines(EXPERT_DEFINES + mcast, self.extra_defines)

    def run(self, kv, mask, tables, noise, out) -> None:
        """Enqueue the expert program: reads ``kv`` (18 bfp8 (K, V) caches), ``mask`` (the key row), ``tables``
        (cosq, sinq, cosk, sink), ``noise`` [1, S, 32] bf16; writes ``out`` x_0 [1, S, 32] bf16."""
        import ttnn

        ttnn.generic_op(self.io_tensors(kv, mask, tables, noise, out), self.program(kv, mask, tables, noise, out))
