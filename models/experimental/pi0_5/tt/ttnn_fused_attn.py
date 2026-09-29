# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused expert attention as ONE ``ttnn.generic_op`` program (kernels in ``tt/kernels/fused_attn``).

Replaces, per expert layer and denoising step, ``nlp_create_qkv_heads`` + ``rotary_embedding`` (q, k) + the two
cache fills + ``scaled_dot_product_attention`` + ``nlp_concat_heads`` (128.8 us in-trace on one Blackhole chip) by a
single launch on ``num_heads x suffix_tile_rows`` cores (40 us; PCC 0.9996 vs the ttnn path, 2026-09-17).

Per core (head ``h``, query tile-row ``r``): RoPE of q (the 1/sqrt(dh) scale is folded into the q tables, the
rotate-half sign into the sin tables), RoPE of the KV head's suffix k rows, S = q K^T over the cache PREFIX rows plus
the local suffix rows (the suffix is never written to the cache: nothing else reads it), key mask on the last key
tile, row softmax, P V, 1/rowsum. Output: the head-concatenated context ``[1, 1, S, H*dh]`` in the cache dtype.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import ttnn

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "fused_attn")
TILE = 32


def _tile_bytes(dtype) -> int:
    return 1088 if dtype == ttnn.bfloat8_b else 2048


def _to_torch_any(t: ttnn.Tensor) -> torch.Tensor:
    """Host copy of a device tensor; the chip-0 shard of a replicated mesh tensor."""
    try:
        shards = ttnn.get_device_tensors(t)
        if len(shards) > 1:
            t = shards[0]
    except Exception:
        pass
    return ttnn.to_torch(t)


def _is_mesh(device) -> bool:
    try:
        return len(device.get_device_ids()) > 1
    except Exception:
        return False


# Constant tiles shared by every block of a device: an interleaved L1 tensor takes a page on EVERY bank, so 18 blocks x
# 6 small constants would cost ~216 KB of L1 per core (the single-chip VLM matmuls then clash with them).
_CONST_CACHE: dict = {}


class FusedExpertAttention:
    """Builds the constant operands once (RoPE tables with folded scale/sign, key mask, reduce scaler) and runs the
    fused program on ``batch x num_heads x suffix_tile_rows`` cores (one launch for the whole batch)."""

    def __init__(self, device, num_heads: int, num_kv_heads: int, head_dim: int, seq_len: int):
        if num_kv_heads != 1:
            raise ValueError("fused expert attention implements one KV head (GQA over all q heads)")
        if head_dim % TILE:
            raise ValueError("head_dim must be a tile multiple")
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = ((seq_len + TILE - 1) // TILE) * TILE  # the expert runs on the tile-padded suffix rows
        self.dht = head_dim // TILE
        self.st = self.seq_len // TILE  # query tile-rows = core rows (bug 2026-09-17: seq_len // TILE gave 1 for 50)
        self.scale = 1.0 / math.sqrt(head_dim)
        self._const: Optional[Tuple[ttnn.Tensor, ...]] = None  # cosq, sinq, cosk, sink, scaler
        self._mask_cache = {}
        self._compile_cache = {}

    # ---- constants ----
    def _device_tensor(self, t: torch.Tensor, dtype) -> ttnn.Tensor:
        kw = {}
        if _is_mesh(self.device):
            kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(self.device)
        return ttnn.from_torch(
            t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG, **kw
        )

    def _constants(self, cos: ttnn.Tensor, sin: ttnn.Tensor):
        if self._const is None:
            key = (id(self.device), "rope", self.seq_len, self.head_dim, round(self.scale, 8))
            if key not in _CONST_CACHE:
                c = _to_torch_any(cos).float().reshape(1, 1, self.seq_len, self.head_dim)
                s = _to_torch_any(sin).float().reshape(1, 1, self.seq_len, self.head_dim)
                half = self.head_dim // 2
                s_signed = torch.cat([-s[..., :half], s[..., half:]], dim=-1)  # rotate_half: (-x2, x1)
                _CONST_CACHE[key] = (
                    self._device_tensor(c * self.scale, ttnn.bfloat16),
                    self._device_tensor(s_signed * self.scale, ttnn.bfloat16),
                    self._device_tensor(c, ttnn.bfloat16),
                    self._device_tensor(s_signed, ttnn.bfloat16),
                    self._device_tensor(torch.ones(1, 1, TILE, TILE), ttnn.bfloat16),
                )
            self._const = _CONST_CACHE[key]
        return self._const

    def _mask(self, prefix_len: int, valid_len: int) -> ttnn.Tensor:
        """Additive mask for the LAST key tile: keys >= valid_len get -30000 (exp -> 0)."""
        key = (id(self.device), "mask", prefix_len, valid_len, self.st)
        if key not in _CONST_CACHE:
            nkt = prefix_len // TILE + self.st
            first_masked = valid_len - (nkt - 1) * TILE
            if first_masked < 0:
                raise ValueError("fused expert attention: more than one key tile would need masking")
            m = torch.zeros(1, 1, TILE, TILE)
            if first_masked < TILE:
                m[..., first_masked:] = -30000.0
            _CONST_CACHE[key] = self._device_tensor(m, ttnn.bfloat16)
        return _CONST_CACHE[key]

    # ---- program ----
    def __call__(
        self,
        xqkv: ttnn.Tensor,
        cache_k: ttnn.Tensor,
        cache_v: ttnn.Tensor,
        prefix_len: int,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        r: Optional[ttnn.Tensor] = None,
        c: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """``xqkv`` [B, 1, S, (H+2)*dh] (q heads | k | v), caches [B, 1, logical_len, dh] whose rows
        ``0..prefix_len-1`` hold each request's prefix K/V; returns ctx [B, 1, S, H*dh] in the cache dtype (new L1
        tensor). Core grid: x = head, y = request * St + query tile-row."""
        if prefix_len % TILE:
            raise ValueError("prefix_len must be a tile multiple")
        rows = int(xqkv.padded_shape[-2]) if hasattr(xqkv, "padded_shape") else int(xqkv.shape[-2])
        if rows != self.seq_len or int(cos.padded_shape[-2] if hasattr(cos, "padded_shape") else cos.shape[-2]) != self.seq_len:
            raise ValueError(f"fused expert attention built for {self.seq_len} suffix rows, got xqkv rows {rows}")
        batch = int(xqkv.shape[0])
        if int(cache_k.shape[0]) != batch:
            raise ValueError("xqkv batch and cache batch differ")
        grid = self.device.compute_with_storage_grid_size()
        if batch * self.st > grid.y or self.num_heads > grid.x:
            raise ValueError(f"batch {batch}: {self.num_heads} x {batch * self.st} cores exceed the {grid.x} x {grid.y} grid")
        cosq, sinq, cosk, sink, scaler = self._constants(cos, sin)
        valid_len = int(cache_k.shape[-2])
        padded_rows = int(cache_k.padded_shape[-2]) if hasattr(cache_k, "padded_shape") else valid_len
        if prefix_len + self.seq_len > padded_rows:
            raise ValueError("prefix + suffix rows exceed the padded cache")
        # the reader streams the prefix in chunks of 4 tile rows (compute.cpp CH) and may read up to the chunk end
        if -(-(prefix_len // TILE) // 4) * 4 * TILE > padded_rows:
            raise ValueError("the last 4-row prefix chunk would read past the padded cache")
        mask = self._mask(prefix_len, valid_len)
        pt = prefix_len // TILE
        nqkv_t = int(xqkv.shape[-1]) // TILE
        nh, dht, st = self.num_heads, self.dht, self.st
        q_dt, kv_dt = xqkv.dtype, cache_k.dtype
        ctx = ttnn.allocate_tensor_on_device(
            ttnn.Shape([batch, 1, self.seq_len, nh * self.head_dim]), kv_dt, ttnn.TILE_LAYOUT, self.device, ttnn.L1_MEMORY_CONFIG
        )
        xqkv_bstride = st * nqkv_t
        kv_bstride = (padded_rows // TILE) * dht
        ctx_bstride = st * nh * dht
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nh - 1, batch * st - 1))])
        bf16 = ttnn.bfloat16

        def cb(idx, n, fmt):
            return ttnn.CBDescriptor(
                total_size=n * _tile_bytes(fmt),
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=_tile_bytes(fmt))],
            )

        nkt = pt + st
        # L1 footprint ~440 KB per core: the K/V prefix streams through rings of 3 chunks x 4 rows (compute.cpp
        # CH = 4), the k RoPE tables come one row at a time, the RoPE temporaries hold one row, and P is written in
        # place of S (CB 17; no CB 19). The whole-prefix version (773 KB) clashed with the L1 buffers of the
        # disaggregated expert chips (socket FIFOs); the 2 x 8-row ring version (620 KB) still did with 512 KB FIFOs.
        kv_ring = 3 * 4 * dht
        cbs = [
            cb(0, dht, q_dt), cb(1, st * dht, q_dt), cb(2, st * dht, q_dt),
            cb(3, dht, bf16), cb(4, dht, bf16), cb(5, dht, bf16), cb(6, dht, bf16),
            cb(7, 1, bf16), cb(8, 1, bf16),
            cb(9, kv_ring, kv_dt), cb(10, st * dht, kv_dt), cb(11, kv_ring, kv_dt),
            cb(13, dht, bf16), cb(14, dht, bf16), cb(15, dht, bf16), cb(16, dht, kv_dt),
            cb(17, nkt, bf16), cb(18, 1, bf16), cb(20, 1, bf16), cb(21, dht, bf16),
            cb(22, 1, bf16), cb(23, 1, bf16),
        ]
        rc = r is not None
        if rc:
            if c is None:
                raise ValueError("r and c go together")
            # norm_in folded into the qkv weights: r [rows, 32] row rsqrt tiles, c [1, 1, 32, (H+2) dh] folded bias
            cbs += [cb(19, st, bf16), cb(24, dht, bf16), cb(25, dht, bf16), cb(26, dht, bf16),
                    cb(27, dht, bf16), cb(28, st * dht, bf16), cb(29, st * dht, bf16)]
        ins = [xqkv, cosq, sinq, cosk, sink, mask, scaler, cache_k, cache_v] + ([r, c] if rc else [])
        r_ct = []
        for t in ins:
            r_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        w_ct = list(ttnn.TensorAccessorArgs(ctx).get_compile_time_args())
        r_rt, w_rt, c_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addrs = [t.buffer_address() for t in (xqkv, cosq, sinq, cosk, sink, mask, scaler, cache_k, cache_v)]
        for h in range(nh):
            for y in range(batch * st):
                b, rr = divmod(y, st)
                r_rt[h][y] = addrs + [h, rr, pt, nqkv_t, dht, nh, st, b, xqkv_bstride, kv_bstride] + ([r.buffer_address(), c.buffer_address()] if rc else [])
                w_rt[h][y] = [ctx.buffer_address(), h, rr, nh * dht, dht, b, ctx_bstride]
                c_rt[h][y] = [rr]
        defines = [("RC_PROLOGUE", "1")] if rc else []
        fp = ttnn.KernelDescriptor.SourceType.FILE_PATH
        kernels = [
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "reader.cpp"), source_type=fp, core_ranges=cores,
                                  compile_time_args=r_ct, runtime_args=r_rt, defines=defines, config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "writer.cpp"), source_type=fp, core_ranges=cores,
                                  compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=os.path.join(KDIR, "compute.cpp"), source_type=fp, core_ranges=cores,
                                  compile_time_args=[dht, st, pt], runtime_args=c_rt, defines=defines,
                                  config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False)),
        ]
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op(ins + [ctx], prog)
        return ctx
