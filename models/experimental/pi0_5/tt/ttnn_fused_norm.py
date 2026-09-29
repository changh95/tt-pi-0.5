# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""adaRMS folding for the expert: the norm's scale/shift move into the following matmul, the rest of the norm is a
per-row rsqrt applied by the consumer.

    rms_norm(x) * scale + shift = r * x * scale + shift,   r = rsqrt(mean(x^2) + eps)  (per row)
    (rms_norm(x) * scale + shift) @ W = r * (x @ (diag(scale) W)) + shift @ W

so per (layer, denoising step) the folded weight ``W' = diag(scale) W`` (bf8, DRAM) and the folded bias
``c = shift @ W`` (bf16 tile row) replace the ``ttnn.rms_norm`` launch. The consumers apply ``r`` and ``c``:
``FusedExpertAttention`` on its q/k/v tiles, ``GegluRC`` on the up|gate tiles. ``r`` itself is one small program
(``RowRsqrt``: one core per tile-row, ~5 us).
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
import ttnn

TILE = 32
KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels")
_CONST_CACHE: dict = {}  # constant tiles shared by every block of a device (see RowRsqrt)


def _is_mesh(device) -> bool:
    try:
        return len(device.get_device_ids()) > 1
    except Exception:
        return False


def to_torch_any(t: ttnn.Tensor) -> torch.Tensor:
    try:
        shards = ttnn.get_device_tensors(t)
        if len(shards) > 1:
            t = shards[0]
    except Exception:
        pass
    return ttnn.to_torch(t)


def device_tensor(device, t: torch.Tensor, dtype, memory_config=ttnn.L1_MEMORY_CONFIG) -> ttnn.Tensor:
    kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(device)} if _is_mesh(device) else {}
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config, **kw)


def _tile_bytes(dtype) -> int:
    return 1088 if dtype == ttnn.bfloat8_b else 2048


def _cb(idx, n, fmt, cores):
    return ttnn.CBDescriptor(
        total_size=n * _tile_bytes(fmt), core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=_tile_bytes(fmt))])


def fold_adarms(w: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``w`` [D, N] (input-major, as ``ttnn.linear`` consumes it), ``scale``/``shift`` [D] (scale includes the +1)
    -> (``diag(scale) @ w`` [D, N], ``shift @ w`` [N]) in fp32."""
    w32 = w.float()
    return w32 * scale.float().reshape(-1, 1), shift.float().reshape(1, -1) @ w32


def bias_row_tensor(device, c: torch.Tensor) -> ttnn.Tensor:
    """[N] -> [1, 1, 32, N] bf16 with the bias in row 0 (the kernels broadcast row 0 of each tile), DRAM."""
    t = torch.zeros(1, 1, TILE, c.numel(), dtype=torch.float32)
    t[0, 0, 0] = c.float()
    return device_tensor(device, t, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG)


class RowRsqrt:
    """r = rsqrt(mean_k(x[row, k]^2) + eps) per row: one core per tile-row, output [1, 1, rows, 32] bf16 (L1) with
    the value in column 0 of each row (what ``mul_tiles_bcast_cols`` reads)."""

    def __init__(self, device, width: int, eps: float):
        if width % TILE:
            raise ValueError("width must be a tile multiple")
        self.device, self.width, self.eps = device, width, eps
        self.kt = width // TILE
        # shared per device: each small interleaved L1 tensor costs a page on every bank (18 blocks x 2 would be ~72 KB per core)
        key = (id(device), "row_rsqrt", width, float(eps))
        if key not in _CONST_CACHE:
            _CONST_CACHE[key] = (
                device_tensor(device, torch.full((1, 1, TILE, TILE), 1.0 / width), ttnn.bfloat16),
                device_tensor(device, torch.full((1, 1, TILE, TILE), float(eps)), ttnn.bfloat16),
            )
        self.scaler, self.eps_t = _CONST_CACHE[key]

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        rows = 1
        for d in tuple(x.padded_shape)[:-1]:
            rows *= int(d)
        if int(x.padded_shape[-1]) != self.width or rows % TILE:
            raise ValueError(f"RowRsqrt: got {tuple(x.padded_shape)}, expected [.., rows%32==0, {self.width}]")
        nrt = rows // TILE
        grid = self.device.compute_with_storage_grid_size()
        if nrt > grid.x * grid.y:
            raise ValueError("too many rows")
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x)) for i in range(nrt)])
        r = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, rows, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG)
        x_dt = x.dtype
        cbs = [_cb(0, self.kt, x_dt, cores), _cb(1, 1, ttnn.bfloat16, cores), _cb(2, 1, ttnn.bfloat16, cores),
               _cb(3, self.kt, ttnn.bfloat16, cores), _cb(4, 1, ttnn.bfloat16, cores), _cb(16, 1, ttnn.bfloat16, cores)]
        r_ct = []
        for t in (x, self.scaler, self.eps_t):
            r_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        w_ct = list(ttnn.TensorAccessorArgs(r).get_compile_time_args())
        r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i in range(nrt):
            cx, cy = i % grid.x, i // grid.x
            r_rt[cx][cy] = [x.buffer_address(), self.scaler.buffer_address(), self.eps_t.buffer_address(), i, self.kt]
            w_rt[cx][cy] = [r.buffer_address(), i]
        fp = ttnn.KernelDescriptor.SourceType.FILE_PATH
        kd = os.path.join(KDIR, "row_rsqrt")
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(kd, "reader.cpp"), source_type=fp, core_ranges=cores, compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(kd, "writer.cpp"), source_type=fp, core_ranges=cores, compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(kd, "compute.cpp"), source_type=fp, core_ranges=cores, compile_time_args=[self.kt], runtime_args=[],
                                  config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)),
        ]
        ttnn.generic_op([x, self.scaler, self.eps_t, r], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return r


class GegluRC:
    """h = (r*u + c_u) * gelu(r*g + c_g) for ug = [u | g] [B(,1), rows, 2*mlp] (unnormalised x times the
    scale-folded up|gate weights), r from ``RowRsqrt``, c the folded biases [1, 1, 32, 2*mlp]. Output
    [B, 1, rows, mlp] bf16 (L1). Cores: tile-rows x column blocks (64 cores)."""

    def __init__(self, device, mlp_dim: int):
        self.device, self.mlp_dim = device, mlp_dim
        self.half_t = mlp_dim // TILE

    def __call__(self, ug: ttnn.Tensor, r: ttnn.Tensor, c: ttnn.Tensor) -> ttnn.Tensor:
        shp = tuple(ug.padded_shape)
        batch = int(shp[0]) if len(shp) == 4 else 1
        rows = int(shp[-2]) * (1 if len(shp) == 4 else int(shp[0]))
        if int(shp[-1]) != 2 * self.mlp_dim or rows % TILE:
            raise ValueError(f"GegluRC: got {shp}")
        nrt = rows // TILE
        nb = max(1, 64 // nrt)
        while self.half_t % nb:
            nb //= 2
        kb = self.half_t // nb
        grid = self.device.compute_with_storage_grid_size()
        n_cores = nrt * nb
        if n_cores > grid.x * grid.y:
            raise ValueError("too many cores")
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(i % grid.x, i // grid.x), ttnn.CoreCoord(i % grid.x, i // grid.x)) for i in range(n_cores)])
        out_shape = [batch, 1, rows // batch, self.mlp_dim] if len(shp) == 4 else [int(shp[0]), int(shp[1]), self.mlp_dim]
        h = ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG)
        bf16, ug_dt = ttnn.bfloat16, ug.dtype
        cbs = [_cb(0, kb, ug_dt, cores), _cb(1, kb, ug_dt, cores), _cb(2, 1, bf16, cores), _cb(3, kb, bf16, cores), _cb(4, kb, bf16, cores),
               _cb(5, kb, bf16, cores), _cb(6, kb, bf16, cores), _cb(7, kb, bf16, cores), _cb(16, kb, bf16, cores)]
        r_ct = []
        for t in (ug, r, c):
            r_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        w_ct = list(ttnn.TensorAccessorArgs(h).get_compile_time_args())
        r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i in range(n_cores):
            rt, blk = divmod(i, nb)
            cx, cy = i % grid.x, i // grid.x
            r_rt[cx][cy] = [ug.buffer_address(), r.buffer_address(), c.buffer_address(), rt, blk, kb, 2 * self.half_t]
            w_rt[cx][cy] = [h.buffer_address(), rt, blk, kb, self.half_t]
        fp = ttnn.KernelDescriptor.SourceType.FILE_PATH
        kd = os.path.join(KDIR, "geglu_rc")
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(kd, "reader.cpp"), source_type=fp, core_ranges=cores, compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(kd, "writer.cpp"), source_type=fp, core_ranges=cores, compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(kd, "compute.cpp"), source_type=fp, core_ranges=cores, compile_time_args=[kb], runtime_args=[],
                                  config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)),
        ]
        ttnn.generic_op([ug, r, c, h], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        return h


class FoldedExpertNorms:
    """Per-step folded weights of one expert block: ``wqkv[s]``/``cqkv[s]`` (norm_in folded into the qkv matmul) and
    ``wug[s]``/``cug[s]`` (norm_post folded into the up|gate matmul)."""

    def __init__(self, device, wqkv_torch: Optional[torch.Tensor], wug_torch: torch.Tensor, mods: List[Tuple],
                 weight_dtype=ttnn.bfloat8_b, fold_attn: bool = False):
        # fold_attn: also fold norm_in into the qkv weights (the fused attention applies r / c). Measured 2026-09-17:
        # numerically fine (PCC 0.999 vs the unfolded path) but no time gain (row_rsqrt 7.5 us + the prologue vs the
        # 14.7 us rms_norm), so off by default; the MLP side saves 14 us (B=1) / 20 us (B=4) per layer.
        self.fold_attn = fold_attn and wqkv_torch is not None
        self.wqkv: List[ttnn.Tensor] = []
        self.cqkv: List[ttnn.Tensor] = []
        self.wug: List[ttnn.Tensor] = []
        self.cug: List[ttnn.Tensor] = []
        for scale_in, shift_in, _g1, scale_post, shift_post, _g2 in mods:
            if self.fold_attn:
                si, hi = to_torch_any(scale_in).float().flatten(), to_torch_any(shift_in).float().flatten()
                w, c = fold_adarms(wqkv_torch, si, hi)
                self.wqkv.append(device_tensor(device, w, weight_dtype, ttnn.DRAM_MEMORY_CONFIG))
                self.cqkv.append(bias_row_tensor(device, c))
            sp, hp = to_torch_any(scale_post).float().flatten(), to_torch_any(shift_post).float().flatten()
            w, c = fold_adarms(wug_torch, sp, hp)
            self.wug.append(device_tensor(device, w, weight_dtype, ttnn.DRAM_MEMORY_CONFIG))
            self.cug.append(bias_row_tensor(device, c))
