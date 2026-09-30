# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The WHOLE ``sample_actions`` as ONE ``ttnn.generic_op`` (phase 2): prefix engine + the phase-1 expert loop.

``WholeMegakernel(device, expert_mk, prefix_params, shape)`` uploads the prefix engine's weight arenas, vectors,
tables and DRAM scratch once; ``program(...)`` builds ONE ProgramDescriptor whose three kernels (kernels_p2/whole_*.cpp)
run the prefix (SigLIP x2 -> projector -> language embedding -> VLM prefill writing the 18 K / V caches) and then the
phase-1 expert kernel unchanged. ``whole=False`` builds the prefix-only test program (same binaries minus the expert
entry). Every per-request value is tensor data at a fixed address (im2col, token ids, key mask, expert mask / RoPE rows,
noise), so the op is trace-capturable.

CB descriptor order fixes the L1 arena (kernels_p2/pe_common.hpp arena_lo / arena_hi): P_SYNC, the placeholders of
the re-pointed ids 32.., phase-1 CB_SYNC, phase-1 CBs 0..30, P_TAIL (the arena extension).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import geometry as G
from . import pe_geometry as P
from . import pe_host as H

KERNELS2 = {r: os.path.join(P.KDIR2, f"whole_{r}.cpp") for r in ("ncrisc", "brisc", "trisc")}
KERNEL_SOURCES2 = sorted(
    [os.path.join(P.KDIR2, f) for f in os.listdir(P.KDIR2) if f.endswith((".cpp", ".hpp"))]
    + [os.path.join(G.KDIR, f) for f in os.listdir(G.KDIR) if f.endswith((".cpp", ".hpp"))])


def kernel_digest2() -> str:
    import hashlib

    h = hashlib.sha256()
    for p in KERNEL_SOURCES2:
        h.update(os.path.relpath(p, os.path.dirname(P.KDIR2)).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _f32_bits(x: float) -> int:
    return G.f32_bits(x)


class PrefixTensors:
    """Device tensors of the prefix engine for one prefix shape (weights shared across shapes where possible)."""

    def __init__(self, device, pp: H.PrefixParams, ps: P.PShape, n_sig: Optional[int] = None,
                 n_vlm: Optional[int] = None, embed=None, shared: Optional["PrefixTensors"] = None):
        import ttnn

        from .arena import memory_config, to_device_layout

        self.device, self.ps = device, ps
        dram = ttnn.DRAM_MEMORY_CONFIG

        def up(t, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=dram)

        def arena(a, dtype):
            return ttnn.from_torch(to_device_layout(a), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device,
                                   memory_config=memory_config(a.shape[1]))

        n_sig = len(pp.sig) if n_sig is None else n_sig
        n_vlm = len(pp.vlm) if n_vlm is None else n_vlm
        if shared is not None:  # the arenas depend on the op geometry only (np / kt / piece), not on the shape
            self.ws, self.wv = shared.ws, shared.wv
        else:
            self.ws = [arena(H.sig_layer_arena(pp.sig[i], i, ps), ttnn.bfloat8_b) for i in range(n_sig)]
            self.wv = [arena(H.vlm_layer_arena(pp.vlm[i], i, ps), ttnn.bfloat8_b) for i in range(n_vlm)]
        if shared is not None:
            self.wpatch, self.wproj = shared.wpatch, shared.wproj
            self.svec, self.gvec, self.vvec, self.consts, self.pos = (shared.svec, shared.gvec, shared.vvec,
                                                                        shared.consts, shared.pos)
        else:
            self.wpatch = arena(H.patch_arena(pp, ps), ttnn.bfloat16)
            self.wproj = arena(H.proj_arena(pp, ps), ttnn.bfloat8_b)
            self.svec = up(H.svec(pp), ttnn.bfloat16)
            self.gvec = up(H.gvec(pp), ttnn.bfloat16)
            self.vvec = up(H.vvec(pp), ttnn.bfloat16)
            self.consts = up(H.consts(), ttnn.bfloat16)
            self.pos = up(pp.pos_b, ttnn.bfloat16)
        tab = H.rope_tables(ps)
        self.rope = [up(tab[k], ttnn.bfloat16) for k in ("cosq", "sinq", "cosk", "sink")]
        rows = ps.mt * 32

        def scratch(shape, dtype):
            return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, dram)

        f32, b16, b8 = ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b
        self.x_s = scratch([512, 1152], f32)
        self.xn_s = scratch([512, 1152], b16)
        self.qkv_s = scratch([512, 4608], b16)
        self.ctx_s = scratch([512, 1536], b16)
        self.h_s = scratch([512, 4352], b16)
        self.x_v = scratch([rows, 2048], f32)
        self.xn_v = scratch([rows, 2048], b16)
        self.q_v = scratch([8 * rows, 256], b16)
        self.ctx_v = scratch([rows, 2048], b16)
        self.h_v = scratch([rows, 16384], b8)
        self.diag = ttnn.from_torch(torch.zeros(P.NCORES, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                    layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram)
        self.times = ttnn.from_torch(torch.zeros(4096, 16, dtype=torch.int32), dtype=ttnn.uint32,
                                     layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram)
        self.embed = embed
        # request inputs (fixed addresses; the host rewrites their contents)
        self.im2col = scratch([512, P.S_KP * 32], b16)
        self.tokens = ttnn.from_torch(torch.zeros(1, ps.ntok, dtype=torch.int32), dtype=ttnn.uint32,
                                      layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=dram)
        self.vmask = up(torch.zeros(32, ps.ptv * 32), ttnn.bfloat16)

    def all(self) -> List:
        return (list(self.ws) + list(self.wv) + [self.wpatch, self.wproj, self.svec, self.gvec, self.vvec, self.consts,
                                                  self.pos] + list(self.rope)
                + [self.x_s, self.xn_s, self.qkv_s, self.ctx_s, self.h_s, self.x_v, self.xn_v, self.q_v, self.ctx_v,
                   self.h_v, self.diag, self.times, self.embed, self.im2col, self.tokens, self.vmask])


class WholeMegakernel:
    """Program builder for the whole-model generic_op; ``expert`` is the phase-1 ExpertMegakernel of this shape."""

    def __init__(self, device, expert, tensors: PrefixTensors):
        import ttnn

        self.device, self.mk, self.t = device, expert, tensors
        self.ps = tensors.ps
        self.shape = expert.shape
        P.check_ops(self.ps)
        self.need = P.arena_need(self.ps)
        self.p1_bytes = sum(c.total_bytes for c in G.cb_table(self.shape) if c.cb_id != G.CB_SYNC)
        self.tail = max(64, self.need - self.p1_bytes)
        self.arena_bytes = self.p1_bytes + self.tail

    # ------------------------------------------------------------------ args
    def core_args(self, xy) -> List[int]:
        mk = self.mk
        x, y = xy
        a = list(mk.core_args(xy)) + [0] * (P.PR0 - G.N_RT_ARGS)
        assert len(a) == P.PR0
        p = [0] * (P.PR_N - P.PR0)

        def put(i, v):
            p[i - P.PR0] = int(v)

        put(P.PR_X, x)
        put(P.PR_Y, y)
        put(P.PR_LIN, y * 11 + x)
        wf = mk.noc((x, P.WF_Y))
        put(P.PR_WFX, wf[0])
        put(P.PR_WFY, wf[1])
        if y < 8:
            f = mk.noc((y, P.IF_Y))
            put(P.PR_IFX, f[0])
            put(P.PR_IFY, f[1])
        hub = mk.noc((P.HUB_X, P.HUB_Y))
        put(P.PR_HUBX, hub[0])
        put(P.PR_HUBY, hub[1])
        put(P.PR_COLX, mk.noc((x, 0))[0])
        put(P.PR_COLY0, mk.noc((x, 0))[1])
        if y == P.IF_Y and x < 8:
            put(P.PR_ROWY, mk.noc((0, x))[1])
        put(P.PR_ROWX0, mk.noc((0, 0))[0])
        put(P.PR_ROWX1, mk.noc((10, 0))[0])
        g = mk.rect((0, 0), (10, 9))
        put(P.PR_GX0, g[0])
        put(P.PR_GY0, g[1])
        put(P.PR_GX1, g[2])
        put(P.PR_GY1, g[3])
        for yy in range(P.GRID_Y):
            put(P.PR_NOCY0 + yy, mk.noc((x, yy))[1])
        return a + p

    def common_args(self, p1_common: List[int], kv, first: int, stop: int, reps: int = 1) -> List[int]:
        t = self.t
        c = list(p1_common) + [0] * (P.PA0 - G.N_COMMON_ARGS)
        assert len(c) == P.PA0
        c += [0] * (P.PA_N - P.PA0)
        addr = lambda x: int(x.buffer_address())
        for i, ten in ((P.PA_X_S, t.x_s), (P.PA_XN_S, t.xn_s), (P.PA_QKV_S, t.qkv_s), (P.PA_CTX_S, t.ctx_s),
                       (P.PA_H_S, t.h_s), (P.PA_X_V, t.x_v), (P.PA_XN_V, t.xn_v), (P.PA_Q_V, t.q_v),
                       (P.PA_CTX_V, t.ctx_v), (P.PA_H_V, t.h_v), (P.PA_IM2COL, t.im2col), (P.PA_TOK, t.tokens),
                       (P.PA_EMB, t.embed), (P.PA_VMASK, t.vmask), (P.PA_POS, t.pos), (P.PA_SVEC, t.svec),
                       (P.PA_VVEC, t.vvec), (P.PA_GVEC, t.gvec), (P.PA_CONST, t.consts), (P.PA_WPATCH, t.wpatch),
                       (P.PA_WPROJ, t.wproj), (P.PA_DIAG, t.diag), (P.PA_TIMES, t.times)):
            c[i] = addr(ten)
        for i, ten in enumerate(t.rope):
            c[P.PA_COSQ + i] = addr(ten)
        for i, ten in enumerate(t.ws):
            c[P.PA_WS + i] = addr(ten)
        for i, ten in enumerate(t.wv):
            c[P.PA_WV + i] = addr(ten)
        for l in range(G.N_LAYERS):  # the prefix writes the caches phase 1 reads (its C_K_ADDR / C_V_ADDR)
            assert c[G.C_K_ADDR + l] == addr(kv[l][0]) and c[G.C_V_ADDR + l] == addr(kv[l][1])
        c[P.PA_OPFIRST] = int(first)
        c[P.PA_DBGSTOP] = int(stop)
        c[P.PA_REPS] = int(reps)
        return c

    # ------------------------------------------------------------------ program
    def program(self, kv, mask, tables, noise, out, whole: bool = True, first: int = 0, stop: int = P.N_OPS,
                ngen: int = G.N_GEN, reps: int = 1):
        import ttnn

        mk, sh, ps = self.mk, self.shape, self.ps
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(G.GRID[0] - 1, G.GRID[1] - 1))])
        fmt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32, "raw": ttnn.bfloat16}
        pbytes = {"bf16": P.T16, "bfp8": P.T8, "fp32": P.T32, "raw": 64}

        def cbd(total, ids_fmts):
            return ttnn.CBDescriptor(total_size=total, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt[f], page_size=pb) for i, f, pb in ids_fmts])

        cbs = [cbd(P.PSYNC_BYTES, [(P.P_SYNC, "raw", P.PSYNC_BYTES)]),
               cbd(2 * P.OPD_BYTES, [(P.P_OPD, "raw", P.OPD_BYTES)])]
        for f in ("bf16", "bfp8", "fp32", "raw"):
            ids = [i for i, ff in P.PFMT.items() if ff == f]
            cbs.append(cbd(pbytes[f], [(i, f, pbytes[f]) for i in ids]))
        p1 = {c.cb_id: c for c in G.cb_table(sh)}
        order = [G.CB_SYNC] + [i for i in range(G.N_CBS) if i != G.CB_SYNC]
        for i in order:
            c = p1[i]
            cbs.append(cbd(c.total_bytes, [(c.cb_id, c.fmt, c.page_bytes)]))
        cbs.append(cbd(self.tail, [(P.P_TAIL, "raw", self.tail)]))
        # compile-time args: phase 1's list, then the prefix's
        ct = [sh.rt, sh.pt, sh.chunk_tiles, sh.nch, mk.dt_bits, mk.eps_bits]
        for t in (kv[0][0], mask, tables[0], noise, out, mk.consts, mk.dbg):
            ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        pe_ct0 = len(ct)
        ct += [ps.mt, ps.ptv, ps.rv, ps.lt, ps.ntok, _f32_bits(1e-6), _f32_bits(1e-6), _f32_bits(2048 ** 0.5),
               self.arena_bytes]
        assert len(ct) - pe_ct0 == P.PT_ACC
        ct.extend(ttnn.TensorAccessorArgs(self.t.x_s).get_compile_time_args())
        ct.extend(ttnn.TensorAccessorArgs(kv[0][0]).get_compile_time_args())
        common = self.common_args(mk.common_args(kv, mask, tables, noise, out, ngen), kv, first, stop, reps)
        rt = ttnn.RuntimeArgs()
        rt_t = ttnn.RuntimeArgs()  # the TRISC reads phase 1's per-core args only
        for x in range(G.GRID[0]):
            for y in range(G.GRID[1]):
                a_ = self.core_args((x, y))
                rt[x][y] = a_
                rt_t[x][y] = a_[:G.N_RT_ARGS]
        cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True,
                                          dst_full_sync_en=True, math_approx_mode=False)
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for c in G.cb_table(sh):
            if c.fp32_unpack:
                modes[c.cb_id] = ttnn.UnpackToDestMode.UnpackToDestFp32
        for i in P.P_FP32_UNPACK:
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        cc.unpack_to_dest_mode = modes
        defines = [("PE_CT0", str(pe_ct0)), ("PE_WHOLE", "1" if whole else "0")]
        if os.environ.get("PI05_MK_TRACE", "0") == "1":
            defines.append(("MK_TRACE", "1"))
        for kv_ in filter(None, os.environ.get("PI05_PE_DEFINES", "").split(",")):  # debug arms only
            k_, _, v_ = kv_.partition("=")
            defines.append((k_, v_ or "1"))
        if os.environ.get("PI05_MK_FID8", "hifi2").lower() == "hifi2":
            defines.append(("MK_FID8_HIFI2", "1"))
        fp = ttnn.KernelDescriptor.SourceType.FILE_PATH
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(kernel_source=KERNELS2["ncrisc"], source_type=fp, core_ranges=cores,
                                  compile_time_args=ct, runtime_args=rt, common_runtime_args=common, defines=defines,
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0)),
            ttnn.KernelDescriptor(kernel_source=KERNELS2["brisc"], source_type=fp, core_ranges=cores,
                                  compile_time_args=ct, runtime_args=rt, common_runtime_args=common, defines=defines,
                                  config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1)),
            ttnn.KernelDescriptor(kernel_source=KERNELS2["trisc"], source_type=fp, core_ranges=cores,
                                  compile_time_args=ct, runtime_args=rt_t, common_runtime_args=common[:P.PA_TRISC_N],
                                  defines=defines, config=cc),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=0) for i in (0, 1, 2, 3)]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs)

    def io_tensors(self, kv, mask, tables, noise, out) -> List:
        return [t for t in self.t.all() if t is not None] + self.mk.io_tensors(kv, mask, tables, noise, out)

    def run(self, kv, mask, tables, noise, out=None, whole: bool = True, first: int = 0, stop: int = P.N_OPS,
            reps: int = 1):
        import ttnn

        if out is None:
            out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, self.shape.suffix_rows, 32]), ttnn.bfloat16,
                                                 ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG)
        prog = self.program(kv, mask, tables, noise, out, whole=whole, first=first, stop=stop, reps=reps)
        ttnn.generic_op(self.io_tensors(kv, mask, tables, noise, out), prog)
        return out
