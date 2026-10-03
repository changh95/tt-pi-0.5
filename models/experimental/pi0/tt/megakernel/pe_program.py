# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The VISION and PREFIX programs of a pi0.5 call: the prefix engine over an op range.

``PrefixTensors`` uploads the prefix engine's weight arenas, vectors, tables and DRAM scratch once (every buffer at
the largest size the model's presets need, so every program of every preset reads fixed addresses).
``PrefixEngineProgram(device, expert_mk, tensors, pshape, ops)`` builds the ProgramDescriptor of one launch of the
prefix engine (``kernels_p2/whole_*.cpp`` with ``PA_EXPERT`` = 0) over the ops ``[first, stop)``:

* ``VISION_OPS``: the patch embedding, 27 SigLIP layers on the cameras, the post-LN and the projector (writes the image
  rows of the VLM residual ``x_v``);
* ``PREFIX_OPS``: the language embedding and the VLM prefill, which writes the 18 K / V caches.

The expert loop is the third program (``program.ExpertMegakernel``). Every per-request value is tensor data at a fixed
address (im2col, token ids, key masks), so the programs are trace-capturable.

The CB descriptor order fixes the L1 arena (``kernels_p2/pe_common.hpp`` ``arena_lo`` / ``arena_hi``):

1. ``P_SYNC``, ``P_OPD`` and the placeholders of the re-pointed ids 32..;
2. ``CB_SYNC`` and the expert ids 0..30, one page each in their expert formats (the arena starts at CB 0; the formats
   are compiled into the TRISC binaries, so the prefix engine's binary is the one the single-program build had);
3. ``P_TAIL``, which extends the arena to what the op range needs.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from . import geometry as G
from . import pe_geometry as P
from . import pe_host as H
from .program import compute_config, core_range_set, merge_defines, semaphores

KERNELS = {r: os.path.join(P.KDIR2, f"whole_{r}.cpp") for r in ("ncrisc", "brisc", "trisc")}
VISION_OPS = (0, P.OP_EMBED)  # patch embedding, SigLIP x 27, post-LN, projector
# Math fidelity of every build (MULTICONFIG WP-P, 2026-10-02): HiFi2 drops the last mantissa bit of a matmul's in0
# operand (truncation toward zero); that bias was the dominant prefix error. The attention (q.K, P.V) runs at HiFi3 in
# both programs (no measured cost), the SigLIP matmuls at HiFi3 in the VISION program (+0.95 ms); the VLM matmuls keep
# HiFi2 (HiFi3 there costs +5.7 ms and was not needed).
FIDELITY_DEFINES = {"vision": [("PE_MM_FID", "3"), ("PE_ATT_FID", "3")], "prefix": [("PE_ATT_FID", "3")]}
PREFIX_OPS = (P.OP_EMBED, P.N_OPS)  # language embedding, VLM prefill (writes the K / V caches)
# Codegen of the VISION / PREFIX binaries (MULTICONFIG WP4 step 1, 2026-10-02): the expert loop is compiled out of
# whole_*.cpp (PE_NO_EXPERT: the ring footprint halves, 129,868 -> ~72 KB) and the norm / attention bodies are flattened
# (PE_ATT_FLAT = 2). Compiling the expert out alone made attention 7-37 us per op slower: the dead expert code had
# decided which LLK helpers GCC specialised (constprop clones of llk_unpack_AB_matmul with the dims propagated); with
# flatten the hot bodies inline and specialise their LLK calls themselves (bit-identical outputs, every op kind as fast
# or faster: .val/mc_impl/r_ring2).
CODEGEN_DEFINES = [("PE_NO_EXPERT", "1"), ("PE_ATT_FLAT", "2")]
KERNEL_SOURCES = sorted(
    [os.path.join(P.KDIR2, f) for f in os.listdir(P.KDIR2) if f.endswith((".cpp", ".hpp"))]
    + [os.path.join(G.KDIR, f) for f in os.listdir(G.KDIR) if f.endswith((".cpp", ".hpp"))]
)


def kernel_digest() -> str:
    """sha256 (16 hex) over the kernel sources: names which kernels a model instance runs."""
    import hashlib

    h = hashlib.sha256()
    for p in KERNEL_SOURCES:
        h.update(os.path.relpath(p, os.path.dirname(P.KDIR2)).encode())
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _f32_bits(x: float) -> int:
    return G.f32_bits(x)


class PrefixTensors:
    """Device tensors of the prefix engine. ``ps`` sizes every per-request buffer and scratch: the largest prefix shape
    of the model's presets (a program reads a prefix of each buffer: the readers index pages, so wider buffers only
    hold unused pages). The weight arenas depend on the op geometry only (np / kt / piece), not on the shape."""

    def __init__(
        self,
        device,
        pp: H.PrefixParams,
        ps: P.PShape,
        n_sig: Optional[int] = None,
        n_vlm: Optional[int] = None,
        embed=None,
        shared: Optional["PrefixTensors"] = None,
    ):
        import ttnn

        from .arena import memory_config, to_device_layout

        self.device, self.ps = device, ps
        dram = ttnn.DRAM_MEMORY_CONFIG

        def up(t, dtype, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=device, memory_config=dram)

        def arena(a, dtype):
            return ttnn.from_torch(
                to_device_layout(a),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=memory_config(a.shape[1]),
            )

        n_sig = len(pp.sig) if n_sig is None else n_sig
        n_vlm = len(pp.vlm) if n_vlm is None else n_vlm
        if shared is not None:  # the arenas depend on the op geometry only (np / kt / piece), not on the shape
            self.ws, self.wv, self.wv16 = shared.ws, shared.wv, shared.wv16
        else:
            self.ws = [arena(H.sig_layer_arena(pp.sig[i], i, ps), ttnn.bfloat8_b) for i in range(n_sig)]
            self.wv = [arena(H.vlm_layer_arena(pp.vlm[i], i, ps), ttnn.bfloat8_b) for i in range(n_vlm)]
            self.wv16 = [arena(H.vlm_qkv_arena(pp.vlm[i], i, ps), ttnn.bfloat16) for i in range(n_vlm)]
        if shared is not None:
            self.wpatch, self.wproj = shared.wpatch, shared.wproj
            self.svec, self.gvec, self.vvec, self.consts, self.pos = (
                shared.svec,
                shared.gvec,
                shared.vvec,
                shared.consts,
                shared.pos,
            )
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
        # SigLIP rows: the largest vision group's images x 256 patches (pe_geometry.vision_groups)
        srows = max(n for _, n in P.vision_groups(ps.img)) * P.S_IMG * 32
        self.x_s = scratch([srows, 1152], f32)
        self.xn_s = scratch([srows, 1152], b16)
        self.qkv_s = scratch([srows, 4608], b16)
        self.ctx_s = scratch([srows, 1536], b16)
        self.h_s = scratch([srows, 4352], b16)
        self.x_v = scratch([rows, 2048], f32)
        self.xn_v = scratch([rows, 2048], b16)
        self.q_v = scratch([8 * rows, 256], b16)
        self.ctx_v = scratch([rows, 2048], b16)
        self.h_v = scratch([rows, 16384], b8)
        self.diag = ttnn.from_torch(
            torch.zeros(P.NCORES, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram,
        )
        self.embed = embed
        # request inputs (fixed addresses; the host rewrites their contents)
        self.im2col = scratch([ps.s_m * 32, P.S_KP * 32], b16)  # every camera
        self.tokens = ttnn.from_torch(
            torch.zeros(1, ps.ntok, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=dram,
        )
        self.vmask = up(torch.zeros(32, ps.ptv * 32), ttnn.bfloat16)

    def all(self) -> List:
        return (
            list(self.ws)
            + list(self.wv)
            + list(self.wv16)
            + [self.wpatch, self.wproj, self.svec, self.gvec, self.vvec, self.consts, self.pos]
            + list(self.rope)
            + [
                self.x_s,
                self.xn_s,
                self.qkv_s,
                self.ctx_s,
                self.h_s,
                self.x_v,
                self.xn_v,
                self.q_v,
                self.ctx_v,
                self.h_v,
                self.diag,
                self.embed,
                self.im2col,
                self.tokens,
                self.vmask,
            ]
        )


class PrefixEngineProgram:
    """One launch of the prefix engine over the ops ``ops = (first, stop)`` of the prefix shape ``ps``; ``expert`` is
    the model's ExpertMegakernel (core coordinates and the expert-side compile-time args the shared binary reads)."""

    def __init__(self, device, expert, tensors: PrefixTensors, ps: P.PShape, ops, extra_defines=(), group=None):
        import dataclasses

        import ttnn

        self.extra_defines = list(extra_defines)  # test-only A / B builds (override by name); empty in production
        self.first, self.stop = int(ops[0]), int(ops[1])
        if not 0 <= self.first < self.stop <= P.N_OPS:
            raise ValueError(f"op range {ops} outside [0, {P.N_OPS})")
        P.check_ops(ps)
        # a VISION group (first image, images): SigLIP over those cameras, im2col read and x_v written at their rows
        # (the DRAM tensors are bank-interleaved by tile page, so a row offset whose page count is a multiple of the 8
        # banks is a plain address offset: pe_program.common_args)
        self.group = (0, ps.img) if group is None else tuple(group)
        if self.group != (0, ps.img):
            if (self.first, self.stop) != VISION_OPS or self.group not in P.vision_groups(ps.img):
                raise ValueError(f"group {group} of {ps.img} cameras: not a VISION group")
            ps = dataclasses.replace(ps, img=self.group[1])
            P.check_ops(ps, group=True)
        elif ps.img > 2 and self.first < P.OP_EMBED:
            raise ValueError(f"{ps.img} cameras: SigLIP runs per vision group (pe_geometry.vision_groups)")
        self.device, self.mk, self.t, self.ps = device, expert, tensors, ps
        self.shape = expert.shape
        if ps.mt > tensors.ps.mt or ps.ptv > tensors.ps.ptv or ps.ntok > tensors.ps.ntok:
            raise ValueError(f"prefix shape {ps} exceeds the allocated buffers {tensors.ps}")
        # the arena holds the largest op layout of the range; the expert ids 0..30 sit at its start
        self.need = P.arena_need(ps, (self.first, self.stop))
        self.p1_pages = [(c.cb_id, c.fmt, c.page_bytes) for c in G.cb_table(self.shape)]
        p1 = sum(pb for i, _, pb in self.p1_pages if i != G.CB_SYNC)
        self.tail = max(64, self.need - p1)
        self.arena_bytes = p1 + self.tail
        # hub stamps of this program (wall clock at every op end, record k % 4096 with k counted from the launch)
        self.times = ttnn.from_torch(
            torch.zeros(4096, 16, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

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

    def common_args(self, p1_common: List[int], kv, reps: int = 1) -> List[int]:
        t = self.t
        c = list(p1_common) + [0] * (P.PA0 - G.N_COMMON_ARGS)
        assert len(c) == P.PA0
        c += [0] * (P.PA_N - P.PA0)
        addr = lambda x: int(x.buffer_address())
        for i, ten in (
            (P.PA_X_S, t.x_s),
            (P.PA_XN_S, t.xn_s),
            (P.PA_QKV_S, t.qkv_s),
            (P.PA_CTX_S, t.ctx_s),
            (P.PA_H_S, t.h_s),
            (P.PA_X_V, t.x_v),
            (P.PA_XN_V, t.xn_v),
            (P.PA_Q_V, t.q_v),
            (P.PA_CTX_V, t.ctx_v),
            (P.PA_H_V, t.h_v),
            (P.PA_IM2COL, t.im2col),
            (P.PA_TOK, t.tokens),
            (P.PA_EMB, t.embed),
            (P.PA_VMASK, t.vmask),
            (P.PA_POS, t.pos),
            (P.PA_SVEC, t.svec),
            (P.PA_VVEC, t.vvec),
            (P.PA_GVEC, t.gvec),
            (P.PA_CONST, t.consts),
            (P.PA_WPATCH, t.wpatch),
            (P.PA_WPROJ, t.wproj),
            (P.PA_DIAG, t.diag),
            (P.PA_TIMES, self.times),
        ):
            c[i] = addr(ten)
        r0 = self.group[0] * P.S_IMG  # the group's first row tile
        for i, width, page in ((P.PA_IM2COL, P.S_KP, 2048), (P.PA_X_V, P.V_D, 4096)):  # widths in tiles
            assert (r0 * width) % P.N_BANKS == 0, (r0, width)
            c[i] += (r0 * width // P.N_BANKS) * page
        for i, ten in enumerate(t.rope):
            c[P.PA_COSQ + i] = addr(ten)
        for i, ten in enumerate(t.ws):
            c[P.PA_WS + i] = addr(ten)
        for i, ten in enumerate(t.wv16):
            c[P.PA_WV16 + i] = addr(ten)
        for i, ten in enumerate(t.wv):
            c[P.PA_WV + i] = addr(ten)
        for l in range(G.N_LAYERS):  # the prefix writes the caches the expert loop reads (C_K_ADDR / C_V_ADDR)
            assert c[G.C_K_ADDR + l] == addr(kv[l][0]) and c[G.C_V_ADDR + l] == addr(kv[l][1])
        for x in range(P.NCOL):  # NoC x of logical column x (the norm statistics exchange addresses cores by index)
            c[P.PA_NOCX0 + x] = int(self.mk.noc((x, 0))[0])
        assert all(int(self.mk.noc((x, y))[0]) == c[P.PA_NOCX0 + x] for x in range(P.NCOL) for y in range(P.GRID_Y))
        assert self.ps.mt * self.ps.ncg <= self.ps.norm_rounds * self.ps.norm_cores  # norm items: <= 2 rounds
        c[P.PA_EXPERT] = 0  # the expert loop is its own program
        c[P.PA_OPFIRST] = self.first
        c[P.PA_DBGSTOP] = self.stop
        c[P.PA_REPS] = int(reps)
        return c

    # ------------------------------------------------------------------ program
    def cb_descriptors(self) -> List:
        import ttnn

        cores = core_range_set()
        fmt = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32, "raw": ttnn.bfloat16}
        pbytes = {"bf16": P.T16, "bfp8": P.T8, "fp32": P.T32, "raw": 64}

        def cbd(total, ids_fmts):
            return ttnn.CBDescriptor(
                total_size=total,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt[f], page_size=pb) for i, f, pb in ids_fmts
                ],
            )

        cbs = [
            cbd(P.PSYNC_BYTES, [(P.P_SYNC, "raw", P.PSYNC_BYTES)]),
            cbd(2 * P.OPD_BYTES, [(P.P_OPD, "raw", P.OPD_BYTES)]),
        ]
        for f in ("bf16", "bfp8", "fp32", "raw"):
            ids = [i for i, ff in P.PFMT.items() if ff == f]
            cbs.append(cbd(pbytes[f], [(i, f, pbytes[f]) for i in ids]))
        # CB_SYNC, then CB 0 (arena_lo() = the base of CB 0) .. 30: the arena starts at the address it has in the
        # expert's CB order
        for i, f, pb in sorted(self.p1_pages, key=lambda c: c[0] != G.CB_SYNC):
            cbs.append(cbd(pb, [(i, f, pb)]))
        cbs.append(cbd(self.tail, [(P.P_TAIL, "raw", self.tail)]))
        return cbs

    def program(self, kv, mask, tables, noise, out):
        import ttnn

        mk, sh, ps = self.mk, self.shape, self.ps
        mk.check_kv(kv)
        cores = core_range_set()
        # compile-time args: the expert loop's list (the shared binary compiles the expert code), then the prefix
        # engine's
        ct = mk.compile_time_args(kv, mask, tables, noise, out)
        pe_ct0 = len(ct)
        ct += [
            ps.mt,
            ps.ptv,
            ps.rv,
            ps.lt,
            ps.ntok,
            _f32_bits(1e-6),
            _f32_bits(1e-6),
            _f32_bits(2048**0.5),
            self.arena_bytes,
            ps.img,
            ps.ncg,
            ps.norm_cores,
        ]
        assert len(ct) - pe_ct0 == P.PT_ACC
        ct.extend(ttnn.TensorAccessorArgs(self.t.x_s).get_compile_time_args())
        ct.extend(ttnn.TensorAccessorArgs(kv[0][0]).get_compile_time_args())
        common = self.common_args(mk.common_args(kv, mask, tables, noise, out, mk.ngen), kv)
        rt = ttnn.RuntimeArgs()
        rt_t = ttnn.RuntimeArgs()  # the TRISC reads the expert loop's per-core args only
        for x in range(G.GRID[0]):
            for y in range(G.GRID[1]):
                a_ = self.core_args((x, y))
                rt[x][y] = a_
                rt_t[x][y] = a_[: G.N_RT_ARGS]
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for c in G.cb_table(sh):
            if c.fp32_unpack:
                modes[c.cb_id] = ttnn.UnpackToDestMode.UnpackToDestFp32
        for i in P.P_FP32_UNPACK:
            modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
        defines = self.defines(pe_ct0)
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
                common_runtime_args=common[: P.PA_BRISC_N],
                defines=defines,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNELS["trisc"],
                source_type=fp,
                core_ranges=cores,
                compile_time_args=ct,
                runtime_args=rt_t,
                common_runtime_args=common[: P.PA_TRISC_N],
                defines=defines,
                config=compute_config(modes),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores(), cbs=self.cb_descriptors())

    def defines(self, pe_ct0: int) -> List:
        # MK_FID8_HIFI2: the expert code compiled (never run) into these programs keeps the single-program build's
        # fidelity, so the prefix engine compiles to the measured machine code
        fid = FIDELITY_DEFINES["vision" if (self.first, self.stop) == VISION_OPS else "prefix"]
        spill = [("PE_MERGE_SPILL", "1")] if self.ps.force_spill else []  # A / B: the spill merge at any part count
        return merge_defines(
            [("PE_CT0", str(pe_ct0)), ("MK_FID8_HIFI2", "1")] + CODEGEN_DEFINES + fid + spill, self.extra_defines
        )

    def io_tensors(self, kv) -> List:
        return [t for t in self.t.all() if t is not None] + [self.times] + [t for pair in kv for t in pair]

    def run(self, kv, mask, tables, noise, out) -> None:
        """Enqueue this launch. ``kv`` = the 18 (K, V) bfp8 caches (the PREFIX program writes them), ``mask`` /
        ``tables`` / ``noise`` / ``out`` = the expert's buffers (their accessor args are compiled into the binary)."""
        import ttnn

        ttnn.generic_op(self.io_tensors(kv), self.program(kv, mask, tables, noise, out))
