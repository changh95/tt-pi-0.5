# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-chip helpers for the fused pi0.5 graph on a Blackhole MeshDevice (2x p300 = 1x4 ring).

Tensor parallelism (``FusedConfig.tp``) shards the SigLIP tower and the VLM prefill over the mesh:
attention heads and MLP columns live on different chips, every row-parallel projection (attention
out / MLP down) produces a partial sum that ``tp_all_reduce`` turns into the full activation on
every chip. The action expert is replicated: all chips run it on identical inputs and produce
identical outputs, so the host reads chip 0.

Measured on the 1x4 ring (ttnn.all_reduce inside a Metal trace, FABRIC_1D_RING, 2026-09-17):
[736,2048] bf16 94.6 us, [512,1152] 50.3 us, [64,1024] 17.0 us (FABRIC_1D / Linear: 139 / 61 / 24.5 us).
"""

from __future__ import annotations

from typing import Optional

import torch
import ttnn


def num_devices(device) -> int:
    try:
        return int(device.get_num_devices())
    except AttributeError:
        return 1


def is_mesh(device) -> bool:
    return num_devices(device) > 1


def topology(cfg) -> "ttnn.Topology":
    return ttnn.Topology.Ring if cfg.ccl_topology == "ring" else ttnn.Topology.Linear


def fabric_config(cfg) -> "ttnn.FabricConfig":
    return ttnn.FabricConfig.FABRIC_1D_RING if cfg.ccl_topology == "ring" else ttnn.FabricConfig.FABRIC_1D


def open_mesh(cfg, mesh_shape=(1, 4), l1_small_size: int = 24576, trace_region_size: Optional[int] = None, num_command_queues: int = 1):
    """Enable the fabric and open the mesh. ``trace_region_size`` defaults to the FusedConfig's."""
    rows, cols = mesh_shape
    if rows * cols > 1:
        ttnn.set_fabric_config(fabric_config(cfg))
    kwargs = dict(l1_small_size=l1_small_size, num_command_queues=num_command_queues)
    if cfg.enabled and cfg.trace:
        kwargs["trace_region_size"] = cfg.trace_region_size if trace_region_size is None else trace_region_size
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(rows, cols), **kwargs)
    mesh.enable_program_cache()
    return mesh


def close_mesh(mesh, cfg=None):
    n = num_devices(mesh)
    ttnn.close_mesh_device(mesh)
    if n > 1:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def tp_all_reduce(x: ttnn.Tensor, cfg, memory_config=None) -> ttnn.Tensor:
    """Sum the per-chip partial ``x`` over the mesh (identical result on every chip). Frees ``x``."""
    out = ttnn.all_reduce(x, topology=topology(cfg), memory_config=memory_config or x.memory_config())
    ttnn.deallocate(x)
    return out


# ----------------------------------------------------------------------------- weight sharding


def shard_cols(device, w: torch.Tensor, tp: int, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    """``[K, N]`` column-parallel weight -> chip i holds ``w[:, i*N/tp:(i+1)*N/tp]`` (replicated if tp == 1)."""
    if tp == 1:
        return ttnn.from_torch(
            w.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
    assert w.shape[-1] % tp == 0, (tuple(w.shape), tp)
    return ttnn.from_torch(
        w.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
    )


def shard_rows(device, w: torch.Tensor, tp: int, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    """``[K, N]`` row-parallel weight -> chip i holds ``w[i*K/tp:(i+1)*K/tp, :]``."""
    if tp == 1:
        return ttnn.from_torch(
            w.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
    assert w.shape[-2] % tp == 0, (tuple(w.shape), tp)
    return ttnn.from_torch(
        w.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-2),
    )


def per_chip_cols(device, chunks: list, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
    """``chunks[i]`` (``[K, N_i]``, all the same width) -> chip i holds ``chunks[i]`` (tp == len(chunks))."""
    if len(chunks) == 1:
        return ttnn.from_torch(
            chunks[0].contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
    return ttnn.from_torch(
        torch.cat(chunks, dim=-1).contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
    )


def bias_cols(device, b: torch.Tensor, tp: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """1-D bias of a column-parallel projection -> ``[1, N/tp]`` per chip."""
    b2 = b.reshape(1, -1)
    if tp == 1:
        return ttnn.from_torch(
            b2.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    return ttnn.from_torch(
        b2.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=-1),
    )


def bias_once(device, b: torch.Tensor, tp: int, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """1-D bias of a row-parallel projection whose partials are all-reduced: chip 0 holds the bias,
    the other chips zeros, so the reduced sum carries it exactly once (bit-exact, no /tp rounding)."""
    if tp == 1:
        return ttnn.from_torch(
            b.reshape(1, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    stacked = torch.zeros(tp, b.numel(), dtype=b.dtype)
    stacked[0] = b
    return ttnn.from_torch(
        stacked.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )


def replicate_host(t: ttnn.Tensor, device) -> ttnn.Tensor:
    """Single host tensor -> multi-device host tensor replicated over ``device`` (no-op for one chip)."""
    if not is_mesh(device):
        return t
    return ttnn.from_torch(
        ttnn.to_torch(t), dtype=t.dtype, layout=t.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(device)
    )


def chip0(t: ttnn.Tensor) -> ttnn.Tensor:
    """The chip-0 shard of a (replicated) mesh tensor, or ``t`` itself on one chip."""
    try:
        shards = ttnn.get_device_tensors(t)
    except Exception:
        return t
    return shards[0] if len(shards) > 1 else t


__all__ = [
    "num_devices",
    "is_mesh",
    "topology",
    "fabric_config",
    "open_mesh",
    "close_mesh",
    "tp_all_reduce",
    "shard_cols",
    "shard_rows",
    "per_chip_cols",
    "bias_cols",
    "bias_once",
    "replicate_host",
    "chip0",
]
