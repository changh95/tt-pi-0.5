# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The expert megakernel as ONE ``ttnn.generic_op`` (phase 1): program construction, inputs, launch.

``ExpertMegakernel(device, params, shape)`` uploads the per-step weight arenas and the constant tiles once; ``run``
enqueues the single persistent program for the whole 10-step x 18-layer loop (plus the action in / out projections and
the Euler steps) reading the backbone's prefix K / V caches, the key-mask row, the q / k RoPE tables and the noise
(the fused graph's persistent inputs), and writes x_0 ``[1, S, 32]`` bf16. It is trace-capturable: every per-request
value is tensor data at a fixed address, the runtime args hold only addresses and shape constants.
"""
from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import geometry as G
from .host_model import ExpertParams

KERNELS = {
    "ncrisc": os.path.join(G.KDIR, "mk_ncrisc.cpp"),
    "brisc": os.path.join(G.KDIR, "mk_brisc.cpp"),
    "trisc": os.path.join(G.KDIR, "mk_trisc.cpp"),
}
KERNEL_SOURCES = sorted([os.path.join(G.KDIR, f) for f in os.listdir(G.KDIR) if f.endswith((".cpp", ".hpp"))])


def kernel_digest() -> str:
    h = hashlib.sha256()
    for p in KERNEL_SOURCES:
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def constants_tensor(device):
    import ttnn

    ones = torch.ones(32, 32)
    t = torch.cat([ones, ones / 1024.0, torch.eye(32), torch.zeros(32, 32)], dim=1)  # [32, 128]: 4 tiles in a row
    return ttnn.from_torch(t.reshape(1, 1, 32, 128), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


class ExpertMegakernel:
    def __init__(self, device, params: ExpertParams, shape: G.Shape, arenas: Optional[Tuple[List, List]] = None):
        import ttnn

        from .arena import ArenaBuilder, upload

        G.check_roles(shape)
        if len(params.dts) != G.N_STEPS:
            raise RuntimeError(f"PI05_MEGAKERNEL=expert refused: the schedule has {len(params.dts)} denoising steps, the "
                               f"kernels compile exactly {G.N_STEPS} (PI05_NUM_STEPS)")
        dts = {G.f32_bits(d) for d in params.dts}
        if len(dts) != 1:
            raise RuntimeError(f"the megakernel compiles one Euler dt; the schedule has {len(dts)} distinct fp32 values")
        self.dt_bits = dts.pop()
        self.eps_bits = G.f32_bits(params.eps)
        self.device = device
        self.shape = shape
        self.roles = G.build_roles(shape)
        self.plan = G.plan_banks(shape)
        self.w8, self.w16 = arenas if arenas is not None else upload(ArenaBuilder(params, shape), device)
        self.consts = constants_tensor(device)
        self.dbg = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, shape.suffix_rows, 1024]), ttnn.bfloat16,
                                                  ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
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
            a[G.A_COL_X0:G.A_COL_Y1 + 1] = self.rect((r.head, 0), (r.head, sh.nu - 1))
        if r.has(G.R_OWNER):
            a[G.A_OWNER_N] = r.owner_n
        if r.has(G.R_MLP):
            a[G.A_MLP_KG], a[G.A_MLP_NG] = r.mlp_kg, r.mlp_ng
            a[G.A_ROWL_X], a[G.A_ROWL_Y] = self.noc((0, r.mlp_kg))
            for i in range(4):
                a[G.A_OWNX0 + i], a[G.A_OWNY0 + i] = self.noc(G.owner_xy(4 * r.mlp_ng + i))
        if r.has(G.R_ROWLEAD):
            a[G.A_ROW_X0:G.A_ROW_Y1 + 1] = self.rect((0, r.mlp_kg), (7, r.mlp_kg))
        if xy in self.plan.bank:
            p = self.plan.pages[xy]
            a[G.A_BANK] = self.plan.bank[xy]
            a[G.A_W8_OFF], a[G.A_W8_PAGES] = self.plan.off8[xy], p["w8"]
            a[G.A_W16_OFF], a[G.A_W16_PAGES] = self.plan.off16[xy], p["w16"]
            a[G.A_WC_OFF], a[G.A_WC_PAGES] = self.plan.offc[xy], p["wc"]
        return a

    def common_args(self, kv: Sequence[Tuple], mask, tables, noise, out, ngen: int) -> List[int]:
        sh = self.shape
        c = [0] * G.N_COMMON_ARGS
        c[G.C_H0_X], c[G.C_H0_Y] = self.noc(G.H0)
        c[G.C_H1_X], c[G.C_H1_Y] = self.noc(G.H1)
        c[G.C_KL_X], c[G.C_KL_Y] = self.noc(G.KL)
        c[G.C_RX_X0:G.C_RX_Y1 + 1] = self.rect((0, 0), (9, 7))
        c[G.C_RM_X0:G.C_RM_Y1 + 1] = self.rect((0, 0), (7, 7))
        c[G.C_RO_X0:G.C_RO_Y1 + 1] = self.rect((0, 4), (7, 7))
        c[G.C_RK_X0:G.C_RK_Y1 + 1] = self.rect((0, (sh.nch - 1) * sh.rt), (7, sh.nch * sh.rt - 1))
        c[G.C_RA_X0:G.C_RA_Y1 + 1] = self.rect((0, 0), (G.GRID[0] - 1, G.GRID[1] - 1))
        c[G.C_N_RX], c[G.C_N_XRDY] = G.x_round_receivers(sh)
        c[G.C_N_CORES] = G.GRID[0] * G.GRID[1]
        c[G.C_MASK] = mask.buffer_address()
        c[G.C_COSQ], c[G.C_SINQ], c[G.C_COSK], c[G.C_SINK] = [t.buffer_address() for t in tables]
        c[G.C_NOISE] = noise.buffer_address()
        c[G.C_OUT] = out.buffer_address()
        c[G.C_CONSTS] = self.consts.buffer_address()
        c[G.C_DEBUG] = int(ngen)
        c[G.C_DBGOUT] = self.dbg.buffer_address()
        for s in range(G.N_STEPS):
            c[G.C_W8_ADDR + s] = self.w8[s].buffer_address()
            c[G.C_W16_ADDR + s] = self.w16[s].buffer_address()
        for l in range(G.N_LAYERS):
            c[G.C_K_ADDR + l] = kv[l][0].buffer_address()
            c[G.C_V_ADDR + l] = kv[l][1].buffer_address()
        return c

    # ------------------------------------------------------------------ program
    def program(self, kv, mask, tables, noise, out, ngen: int = G.N_GEN):
        import ttnn

        sh = self.shape
        bad = [(l, i, str(t.dtype)) for l, pair in enumerate(kv) for i, t in enumerate(pair) if t.dtype != ttnn.bfloat8_b]
        if len(kv) != G.N_LAYERS or bad:
            raise RuntimeError(f"PI05_MEGAKERNEL=expert refused: the kernels read {G.N_LAYERS} bfp8 (K, V) caches at the "
                               f"bfp8 page stride; got {len(kv)} layers, non-bfp8 (layer, K0/V1, dtype): {bad[:4]} "
                               "(PI05_KV_DTYPE must be bf8)")
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(G.GRID[0] - 1, G.GRID[1] - 1))])
        fmt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32, "raw": ttnn.bfloat16}
        cbs = []
        for cb in G.cb_table(sh):
            cbs.append(ttnn.CBDescriptor(
                total_size=cb.total_bytes, core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb.cb_id, data_format=fmt[cb.fmt],
                                                            page_size=cb.page_bytes)]))
        ct = [sh.rt, sh.pt, sh.chunk_tiles, sh.nch, self.dt_bits, self.eps_bits]
        assert len(ct) == G.CT_ACC0
        for t in (kv[0][0], mask, tables[0], noise, out, self.consts, self.dbg):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        common = self.common_args(kv, mask, tables, noise, out, ngen)
        rt = ttnn.RuntimeArgs()
        for x in range(G.GRID[0]):
            for y in range(G.GRID[1]):
                rt[x][y] = self.core_args((x, y))
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True,
                                          dst_full_sync_en=True, math_approx_mode=False)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in G.cb_table(sh):
            if cb.fp32_unpack:
                modes[cb.cb_id] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = modes
        defines = [("MK_TRACE", "1")] if os.environ.get("PI05_MK_TRACE", "0") == "1" else []
        if os.environ.get("PI05_MK_FID8", "hifi2").lower() == "hifi2":  # default: measured free, mean PCC up (JOURNAL)
            defines.append(("MK_FID8_HIFI2", "1"))
        fp = ttnn.KernelDescriptor.SourceType.FILE_PATH
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=KERNELS["ncrisc"], source_type=fp, core_ranges=cores,
                                  compile_time_args=ct, runtime_args=rt, common_runtime_args=common, defines=defines,
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=KERNELS["brisc"], source_type=fp, core_ranges=cores,
                                  compile_time_args=ct, runtime_args=rt, common_runtime_args=common, defines=defines,
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=KERNELS["trisc"], source_type=fp, core_ranges=cores,
                                  compile_time_args=ct, runtime_args=rt, common_runtime_args=common, defines=defines, config=cc),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in (0, 1)]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    def io_tensors(self, kv, mask, tables, noise, out) -> List:
        ts = list(self.w8) + list(self.w16) + [t for pair in kv for t in pair]
        return ts + [mask] + list(tables) + [noise, self.consts, self.dbg, out]

    def run(self, kv, mask, tables, noise, out=None, ngen: int = G.N_GEN):
        """Enqueue the megakernel. ``kv`` = 18 (K, V) cache tensors, ``mask`` = exp_mask, ``tables`` = (cosq, sinq,
        cosk, sink), ``noise`` [1, S, 32] bf16 TILE. Returns ``out`` [1, S, 32] bf16 (allocated in L1 when None)."""
        import ttnn

        if out is None:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, self.shape.suffix_rows, 32]), ttnn.bfloat16,
                                                 ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG)
        prog = self.program(kv, mask, tables, noise, out, ngen)
        ttnn.generic_op(self.io_tensors(kv, mask, tables, noise, out), prog)
        return out
