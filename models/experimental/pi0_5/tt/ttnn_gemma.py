# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma transformer blocks - TTNN Implementation (Optimized).

This module implements Gemma 2B style transformer layers using TTNN operations:
    - RMSNorm (using native ttnn.rms_norm)
    - Multi-Query Attention (MQA) with fused QKV and native RoPE
    - GeGLU MLP (gated GELU activation)
    - Native head operations (nlp_create_qkv_heads, nlp_concat_heads)

Architecture configurations:
    - Gemma 2B (VLM): width=2048, depth=18, mlp_dim=16384, heads=8, kv_heads=1
    - Gemma 300M (Expert): width=1024, depth=18, mlp_dim=4096, heads=8, kv_heads=1

Optimizations over baseline:
    1. Fused QKV projection (1 linear instead of 3)
    2. Native ttnn.experimental.nlp_create_qkv_heads
    3. Native ttnn.experimental.rotary_embedding (split-half pattern)
    4. Native ttnn.experimental.nlp_concat_heads for output
    5. Native ttnn.rms_norm (single fused kernel)
    6. Pure TTNN RoPE precomputation

Fused graph (the only inference path; every ``forward_fused_*`` method below):
    - VLM attention writes its rotated K / V straight into the backbone-owned KV caches
      (``fill_cache`` at row 0) so the expert never refills the prefix; cos/sin slices are cached
      per seq_len (no per-layer slices).
    - Expert attention runs on the tile-padded 64-row suffix (no slice(q) after RoPE), writes
      K/V rows prefix_len..prefix_len+63 with the fork's ``rotary_embedding_to_cache`` +
      ``fill_cache`` and attends over the cache (logical rows = prefix_len + action_horizon, so the
      SDPA kernel masks the padded suffix rows exactly as before).
    - Expert gated residuals: ONE ``dit_minimal_matmul_addcmul_fused`` per residual
      (``hidden + 1.0 * (x @ W) * gate``) with a bf16 hidden stream, bf16 o_proj / down_proj and
      bf16 gates (``PI05_FUSED_RESIDUAL=bf16``): replaces linear + mac (2 launches) and the
      3 typecasts around a bf8 down-proj fusion (``PI05_FUSED_RESIDUAL=legacy``); the residual is not
      rounded to bf8.
    - GeGLU: gate linear with ``activation="gelu"`` (same non-approximate SFPU GELU as ``ttnn.gelu``)
      + up linear + multiply; the VLM MLP is chunked by 256 rows by default (``PI05_MLP_CHUNK``).
    - SDPA program configs are knobs (``PI05_SDPA_{VLM,EXPERT}_CHUNKS``), default = the op's program.
"""

import math
import os

from typing import Dict, Optional, Tuple

import torch
import ttnn
from .ttnn_ccl import tp_all_reduce as _tp_all_reduce


def _fill_cache_batched(cache, src, update_idx):
    """``fill_cache`` of ``src [B, KVH, S, dh]`` into rows ``update_idx..`` of every batch entry of
    ``cache [B, KVH, L, dh]`` (fill_cache writes one batch entry per launch: B launches, B-1 slices)."""
    b = int(src.shape[0])
    if b == 1:
        ttnn.fill_cache(cache, src, 0, update_idx=update_idx)
        return
    kvh, s, dh = int(src.shape[1]), int(src.shape[2]), int(src.shape[3])
    for i in range(b):
        piece = ttnn.slice(src, [i, 0, 0, 0], [i + 1, kvh, s, dh])
        ttnn.fill_cache(cache, piece, i, update_idx=update_idx)
        ttnn.deallocate(piece)


def rotary_embedding_to_cache(k, cos, sin, cache, update_idx):
    """Rotate ``k`` and write it into ``cache`` rows ``update_idx..``: the fork's fused op when the
    running tt-metal has it, else rotary_embedding + fill_cache (2 launches, identical values)."""
    fused = getattr(ttnn.experimental, "rotary_embedding_to_cache", None)
    if os.environ.get("PI05_NO_FUSED_ROTARY", "").strip() in ("1", "true", "yes"):
        fused = None  # measure / serve exactly what an unmodified tt-metal runs
    if fused is not None:
        return fused(k, cos, sin, cache, update_idx)
    k_rope = ttnn.experimental.rotary_embedding(k, cos, sin)
    ttnn.fill_cache(cache, k_rope, 0, update_idx=update_idx)
    ttnn.deallocate(k_rope)
    return cache


from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig


# ============================================================================
# Fused-graph helpers
# ============================================================================

_TTNN_DTYPE_BY_NAME = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}


def ttnn_dtype_from_name(name: str) -> ttnn.DataType:
    return _TTNN_DTYPE_BY_NAME[name]


def sdpa_program_config_from_chunks(device: ttnn.Device, chunks: Optional[Tuple[int, int]]):
    """``(q_chunk, k_chunk)`` -> SDPAProgramConfig on the FULL compute grid; None -> None (the op's default program)."""
    if chunks is None:
        return None
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=chunks[0],
        k_chunk_size=chunks[1],
        exp_approx_mode=False,
    )


def with_fp32_acc(ckc):
    """Same math fidelity / approx mode as ``ckc`` with fp32 destination accumulation (the explicit
    matmul programs otherwise accumulate their K-block partial sums in bf16)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ckc.math_fidelity,
        math_approx_mode=ckc.math_approx_mode,
        fp32_dest_acc_en=True,
        packer_l1_acc=ckc.packer_l1_acc,
    )


def mcast1d_program_config(device: ttnn.Device, n: int, per_core_n: int = 2, in0_block_w: int = 8, per_core_m: int = 2):
    """``ttnn.linear`` program config for the expert's M = 64-row projections: 1D multicast of the
    activation (in0) over the full grid, N split ``per_core_n`` tiles per core (probe 2026-09-13:
    qkv [64,1024]x[1024,2560] 23.5 -> 10.0 us, up [64,1024]x[1024,4096] 22.1 -> 14.2 us in-trace)."""
    grid = device.compute_with_storage_grid_size()
    n_tiles = n // 32
    cores = grid.x * grid.y
    while per_core_n * cores < n_tiles:
        per_core_n *= 2
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=min(per_core_n, 4),
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


def mcast2d_program_config(device: ttnn.Device, m: int, k: int, n: int, in0_block_w: int = 8, fused_activation=None):
    """``ttnn.linear`` 2D-multicast program config on the full compute grid for the big VLM matmuls
    (M rows over grid.y, N columns over grid.x). Probe 2026-09-13 (`probe_vlm_mlp.py`): the unchunked
    down projection [736,16384]x[16384,2048] took 2.90 ms with the auto program and 0.198 ms with this
    config (in0_block_w 8, per_core_M 3, per_core_N 6, subblocks 1x2), PCC unchanged."""
    grid = device.compute_with_storage_grid_size()
    m_tiles, k_tiles, n_tiles = -(-m // 32), -(-k // 32), -(-n // 32)
    per_core_m = -(-m_tiles // grid.y)
    per_core_n = -(-n_tiles // grid.x)
    while in0_block_w > 1 and k_tiles % in0_block_w:
        in0_block_w //= 2
    out_subblock_w = 4 if per_core_n % 4 == 0 else (2 if per_core_n % 2 == 0 else 1)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid.x, grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=fused_activation,
    )


def dit_config_from_blocks(device: ttnn.Device, blocks: Optional[Tuple[int, ...]]):
    """``(M, K, N, subblock_h, subblock_w[, grid_x, grid_y])`` tiles -> MinimalMatmulConfig; a grid size of
    0 (or a 5-tuple) means the device's compute grid extent on that axis; None -> the op default."""
    if blocks is None:
        return None
    m, k, n, sh, sw = blocks[:5]
    gx, gy = (blocks[5], blocks[6]) if len(blocks) >= 7 else (0, 0)
    dev_grid = device.compute_with_storage_grid_size()
    grid = ttnn.CoreCoord(min(gx, dev_grid.x) if gx else dev_grid.x, min(gy, dev_grid.y) if gy else dev_grid.y)
    return ttnn.MinimalMatmulConfig(
        M_block_size=m,
        K_block_size=k,
        N_block_size=n,
        subblock_h=sh,
        subblock_w=sw,
        compute_with_storage_grid_size=grid,
    )


# ============================================================================
# RMSNorm (TTNN - Optimized)
# ============================================================================


def rms_norm_ttnn(
    x: ttnn.Tensor,
    weight: ttnn.Tensor,
    eps: float = 1e-6,
) -> ttnn.Tensor:
    """
    OPTIMIZED: RMSNorm using ttnn.rms_norm fused operation.

    NOTE: The weight tensor should already have the Gemma-style +1 offset
    pre-applied during initialization (not computed here every forward pass).

    Args:
        x: TTNN tensor (batch_size, seq_len, hidden_dim)
        weight: TTNN weight tensor with +1 offset already applied (1, hidden_dim)
        eps: Epsilon for numerical stability

    Returns:
        Normalized TTNN tensor (bfloat16)
    """
    return ttnn.rms_norm(
        x,
        weight=weight,
        epsilon=eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def adarms_norm_precomputed(
    x: ttnn.Tensor,
    scale: ttnn.Tensor,
    shift: ttnn.Tensor,
    eps: float,
) -> ttnn.Tensor:
    """
    Apply adaRMS when (scale, shift, gate) have been precomputed from cond via the
    dense projection outside the critical path. `scale` already has +1 baked into
    the dense bias, so this reduces to rms_norm with runtime weight/bias.
    """
    return ttnn.rms_norm(
        x,
        weight=scale,
        bias=shift,
        epsilon=eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


# ============================================================================
# Rotary Position Embeddings (TTNN Meta Format)
# ============================================================================


def precompute_freqs_cis_meta_format(
    head_dim: int,
    max_seq_len: int,
    device: ttnn.Device,
    base: float = 10000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Precompute cos and sin for rotary embeddings using pure TTNN operations.

    ttnn.experimental.rotary_embedding uses the split-half pattern (same as Gemma):
    - rotate_half(x) = cat(-x[..., dim/2:], x[..., :dim/2])
    - result = x * cos + rotate_half(x) * sin

    For this to work correctly, cos/sin must have shape [1, 1, max_seq_len, head_dim]
    where the values are repeated: [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}]

    This matches how the rotation pairs x[i] with x[i+dim/2] for i < dim/2.

    Args:
        head_dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        device: TTNN device
        base: Base for frequency computation

    Returns:
        Tuple of (cos, sin) each of shape (1, 1, max_seq_len, head_dim) as TTNN tensors
    """
    half_dim = head_dim // 2

    # Compute inverse frequencies using ttnn.arange
    # indices: [0, 2, 4, ..., head_dim-2]
    indices = ttnn.arange(0, head_dim, 2, device=device, dtype=ttnn.float32)
    # Convert to TILE_LAYOUT early (required for unary ops like pow, reciprocal, cos, sin)
    indices = ttnn.to_layout(indices, ttnn.TILE_LAYOUT)

    # freqs = 1.0 / (base ** (indices / head_dim))
    exponents = ttnn.multiply(indices, 1.0 / head_dim)
    ttnn.deallocate(indices)
    base_powers = ttnn.pow(base, exponents)
    ttnn.deallocate(exponents)
    freqs = ttnn.reciprocal(base_powers)  # Shape: [half_dim]
    ttnn.deallocate(base_powers)

    # Compute positions: [0, 1, 2, ..., max_seq_len-1]
    t = ttnn.arange(0, max_seq_len, 1, device=device, dtype=ttnn.float32)  # Shape: [max_seq_len]
    t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)

    # Outer product: t[i] * freqs[j] -> [max_seq_len, half_dim]
    # Reshape for broadcasting: t -> [max_seq_len, 1], freqs -> [1, half_dim]
    t_col = ttnn.reshape(t, (max_seq_len, 1))
    ttnn.deallocate(t)
    freqs_row = ttnn.reshape(freqs, (1, half_dim))
    ttnn.deallocate(freqs)
    freqs_outer = ttnn.multiply(t_col, freqs_row)  # Shape: [max_seq_len, half_dim]
    ttnn.deallocate(t_col)
    ttnn.deallocate(freqs_row)

    # Compute cos/sin: [max_seq_len, half_dim]
    cos_half = ttnn.cos(freqs_outer)
    sin_half = ttnn.sin(freqs_outer)
    ttnn.deallocate(freqs_outer)

    # Repeat for full head_dim: [c0, c1, ..., c_{n/2-1}, c0, c1, ..., c_{n/2-1}]
    # This matches the split-half rotation where x[i] pairs with x[i+dim/2]
    cos_2d = ttnn.concat([cos_half, cos_half], dim=-1)  # [seq, head_dim]
    sin_2d = ttnn.concat([sin_half, sin_half], dim=-1)  # [seq, head_dim]
    ttnn.deallocate(cos_half)
    ttnn.deallocate(sin_half)

    # Reshape to add batch and head dimensions: [1, 1, seq, head_dim]
    cos = ttnn.reshape(cos_2d, (1, 1, max_seq_len, head_dim))
    sin = ttnn.reshape(sin_2d, (1, 1, max_seq_len, head_dim))
    ttnn.deallocate(cos_2d)
    ttnn.deallocate(sin_2d)

    # Convert to bfloat16 for use with rotary_embedding
    cos = ttnn.typecast(cos, ttnn.bfloat16)
    sin = ttnn.typecast(sin, ttnn.bfloat16)

    return cos, sin


# ============================================================================
# Multi-Query Attention (TTNN - Optimized)
# ============================================================================


class GemmaAttentionTTNN:
    """
    Gemma Multi-Query Attention using TTNN operations.

    OPTIMIZED:
    1. Fused QKV projection (1 linear instead of 3)
    2. Native ttnn.experimental.nlp_create_qkv_heads
    3. Native ttnn.experimental.rotary_embedding (split-half pattern)
    4. Native ttnn.experimental.nlp_concat_heads for output
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: ttnn.Device,
        cos_meta: Optional[ttnn.Tensor] = None,
        sin_meta: Optional[ttnn.Tensor] = None,
        expected_seq_len: Optional[int] = None,
        *,
        fused_cfg: FusedConfig,
        role: str = "vlm",
    ):
        """
        Initialize attention layer with TTNN weights.

        Args:
            config: Gemma configuration
            weights: TTNN weight tensors (including fused wqkv)
            layer_idx: Layer index
            device: TTNN device
            cos_meta: Precomputed cos for native TTNN RoPE [1, 1, max_seq, head_dim]
            sin_meta: Precomputed sin for native TTNN RoPE [1, 1, max_seq, head_dim]
            expected_seq_len: expert role: the suffix length the fused expert attention is built for
            fused_cfg: fused-graph knobs
            role: "vlm" or "expert" (selects the SDPA program-config knob of the fused path)
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.fused_cfg = fused_cfg
        self.role = role
        self._fused_attn = None  # PI05_EXPERT_ATTN=fused (expert role): tt/ttnn_fused_attn.FusedExpertAttention
        # Fused path: cos/sin slices cached per seq_len (built on the compile pass, outside the trace)
        self._rope_cache: Dict[int, Tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._sdpa_config_fused = None
        self._mm_config_fused = None  # PI05_EXPERT_MM=minimal: minimal_matmul config for the expert qkv projection
        self._qkv_program_config = None  # PI05_EXPERT_MM=mcast1d: explicit ttnn.linear program config
        self._qkv_pc_by_rows: Dict[int, object] = {}  # rows (batch * seq) -> mcast1d program config
        chunks = fused_cfg.sdpa_expert if role == "expert" else fused_cfg.sdpa_vlm
        self._sdpa_config_fused = sdpa_program_config_from_chunks(device, chunks)
        if role == "expert" and fused_cfg.expert_mm == "minimal":
            self._mm_config_fused = dit_config_from_blocks(device, fused_cfg.expert_mm_blocks)
        self._want_qkv_mcast1d = (
            role == "expert" and fused_cfg.expert_mm in ("mcast1d", "mcast1d_fp32")
        )  # the config needs self.wqkv (assigned below): built lazily in _qkv_heads
        # PI05_VLM_ATTN_PC=mcast2d[_fp32]: explicit 2D multicast configs for the VLM qkv / o_proj linears
        # (probe 2026-09-13: [736,2048]x[2048,2560] 0.387 -> 0.035 ms, [736,2048]x[2048,2048] 0.405 -> 0.029 ms)
        self._want_vlm_mcast2d = role == "vlm" and fused_cfg.vlm_attn_pc in ("mcast2d", "mcast2d_fp32")
        self._vlm_pc_cache: Dict[Tuple[str, int], object] = {}
        # fp32 destination accumulation for the explicit-program linears (mcast1d_fp32 / mcast2d_fp32)
        self._pc_fp32 = (role == "expert" and fused_cfg.expert_mm == "mcast1d_fp32") or (
            role == "vlm" and fused_cfg.vlm_attn_pc == "mcast2d_fp32"
        )

        # OPTIMIZATION: Use fused QKV weight (single linear instead of 3)
        self.wqkv = weights["self_attn.wqkv"]
        self.o_proj = weights["self_attn.o_proj.weight"]

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.width
        self.scale = 1.0 / math.sqrt(self.head_dim)
        if role == "expert" and getattr(fused_cfg, "expert_attn", "ttnn") == "fused" and expected_seq_len is not None:
            from .ttnn_fused_attn import FusedExpertAttention

            self._fused_attn = FusedExpertAttention(
                device, self.num_heads, self.num_kv_heads, self.head_dim, int(expected_seq_len)
            )

        # Store meta format cos/sin for native TTNN RoPE (split-half pattern)
        self.cos_meta = cos_meta
        self.sin_meta = sin_meta

        # WormholeComputeKernelConfig for Blackhole — LoFi for large matmuls (faster)
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------ fused graph

    def _rope(self, seq_len: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos/sin ``[1, 1, seq_len, head_dim]`` slices of the RoPE tables cached per seq_len (the first call
        happens on the compile pass, before capture)."""
        if seq_len not in self._rope_cache:
            cos = ttnn.slice(self.cos_meta, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
            sin = ttnn.slice(self.sin_meta, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
            self._rope_cache[seq_len] = (cos, sin)
        return self._rope_cache[seq_len]

    def _vlm_pc(self, which: str, seq_len: int, weight: ttnn.Tensor):
        key = (which, seq_len)
        if key not in self._vlm_pc_cache:
            self._vlm_pc_cache[key] = mcast2d_program_config(
                self.device, seq_len, int(weight.shape[-2]), int(weight.shape[-1]), in0_block_w=8
            )
        return self._vlm_pc_cache[key]

    def _qkv_proj(self, hidden_states: ttnn.Tensor, weight: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """normed [B, S, D] -> xqkv [B, 1, S, (H + 2 KVH) dh] (q heads | k | v) in L1, the qkv dtype.
        ``weight`` overrides ``self.wqkv`` (the per-step scale-folded copy of PI05_EXPERT_NORM_FOLD)."""
        wqkv = self.wqkv if weight is None else weight
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        x4 = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))
        # PI05_KV_DTYPE=bf16: bf16 q/k/v (K/V go into bf16 caches, the SDPA output is bf16)
        qkv_dtype = ttnn.bfloat16 if self.fused_cfg.kv_dtype == "bf16" else ttnn.bfloat8_b
        if self._mm_config_fused is not None:
            xqkv = ttnn.experimental.minimal_matmul(
                x4,
                wqkv,
                config=self._mm_config_fused,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=qkv_dtype,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )
        else:
            pc = None
            if self._want_qkv_mcast1d:
                # 1D multicast: every core computes ALL rows (batch * seq) for its N slice -> per_core_M = rows / 32
                rows = int(batch_size) * int(seq_len)
                pc = self._qkv_pc_by_rows.get(rows)
                if pc is None:
                    pc = mcast1d_program_config(self.device, int(wqkv.shape[-1]), per_core_m=max(1, rows // 32))
                    self._qkv_pc_by_rows[rows] = pc
                self._qkv_program_config = pc
            if self._want_vlm_mcast2d:
                pc = self._vlm_pc("qkv", int(batch_size) * int(seq_len), wqkv)  # 2D mcast folds the batch into M
            ckc = self.compute_kernel_config_hifi2
            if pc is not None and self._pc_fp32:
                ckc = with_fp32_acc(ckc)
            xqkv = ttnn.linear(
                x4,
                wqkv,
                dtype=qkv_dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=ckc,
                program_config=pc,
            )
        return xqkv

    def _qkv_heads(self, hidden_states: ttnn.Tensor):
        """normed [B, S, D] -> (q [B, H, S, dh], k [B, KVH, S, dh], v [B, KVH, S, dh]) bf8 in L1 (ttnn ops)."""
        xqkv = self._qkv_proj(hidden_states)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)
        return q, k, v

    def forward_fused_vlm(
        self,
        hidden_states: ttnn.Tensor,
        cache_k: ttnn.Tensor,
        cache_v: ttnn.Tensor,
        need_output: bool = True,
        attn_mask: Optional[ttnn.Tensor] = None,
    ) -> Optional[ttnn.Tensor]:
        """VLM (prefill) attention that also fills the backbone-owned KV cache.

        ``attn_mask`` [B, 1, S, S] bf16 additive (``fused_host.attention_inputs``): hides the prompt's pad keys from
        every query (openpi's prefix padding mask). The RoPE positions of the prefix are ``0..S-1``: for a
        right-padded prompt they equal openpi's ``cumsum(valid) - 1`` on every valid token.

        Rotated K and V of the ``S`` prefix rows are written to rows ``0..S-1`` of ``cache_k`` /
        ``cache_v`` (``fill_cache`` at update_idx 0: same dtype bf8, S % 32 == 0). The SDPA runs on the
        S-row K/V (NOT on the cache: its rows S.. hold the expert's suffix from the previous step).
        ``need_output=False`` (last layer with PI05_SKIP_VLM_TAIL) stops after the cache write.
        """
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        q, k, v = self._qkv_heads(hidden_states)
        cos_sliced, sin_sliced = self._rope(seq_len)

        k_rope = ttnn.experimental.rotary_embedding(k, cos_sliced, sin_sliced)
        ttnn.deallocate(k)
        if k_rope.shape[2] != seq_len:  # only when seq_len % 32 != 0 (never for the served shapes)
            k_rope = ttnn.slice(k_rope, [0, 0, 0, 0], [batch_size, self.num_kv_heads, seq_len, self.head_dim])
        _fill_cache_batched(cache_k, k_rope, 0)
        _fill_cache_batched(cache_v, v, 0)

        if not need_output:
            ttnn.deallocate(q)
            ttnn.deallocate(k_rope)
            ttnn.deallocate(v)
            return None

        q_rope = ttnn.experimental.rotary_embedding(q, cos_sliced, sin_sliced)
        ttnn.deallocate(q)
        if q_rope.shape[2] != seq_len:
            q_rope = ttnn.slice(q_rope, [0, 0, 0, 0], [batch_size, self.num_heads, seq_len, self.head_dim])

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_rope,
            k_rope,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.scale,
            program_config=self._sdpa_config_fused,
        )
        ttnn.deallocate(q_rope)
        ttnn.deallocate(k_rope)
        ttnn.deallocate(v)

        attn_concat = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)
        # TP: the row-parallel partial is emitted in bf16 (its all-reduce sums bf16 and the residual add
        # consumes it), so the sum is rounded once instead of once per chip's bf8 partial.
        tp = self.fused_cfg.tp
        output = ttnn.linear(
            attn_concat,
            self.o_proj,
            dtype=ttnn.bfloat16 if tp > 1 else ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=with_fp32_acc(self.compute_kernel_config_hifi4)
            if (self._want_vlm_mcast2d and self._pc_fp32)
            else self.compute_kernel_config_hifi4,
            program_config=self._vlm_pc("o", int(batch_size) * int(seq_len), self.o_proj) if self._want_vlm_mcast2d else None,
        )
        ttnn.deallocate(attn_concat)
        return ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))

    def forward_fused_expert(
        self,
        hidden_states: ttnn.Tensor,
        cache_k: ttnn.Tensor,
        cache_v: ttnn.Tensor,
        prefix_len: int,
        attn_in: Dict[str, ttnn.Tensor],
    ) -> ttnn.Tensor:
        """Expert attention on the tile-padded suffix (``S`` = round_up(action_horizon) rows).

        openpi semantics: the action tokens attend every valid prefix key and every real action token (``attn_in``
        masks hide the prompt's pad keys and the tile-padding suffix rows) and are rotated at positions
        ``n_valid + [0, S)`` (``attn_in`` RoPE rows). ``attn_in`` holds the graph's persistent attention inputs
        (``PI0ModelTTNN._fused_attn_inputs``).

        Fused attention (``PI05_EXPERT_ATTN=fused``, default): one generic_op program (RoPE + masked attention over
        the cache prefix + local suffix + head concat); the suffix K/V are never written to the cache.
        ttnn ops (``PI05_EXPERT_ATTN=ttnn``): K is rotated and written to ``cache_k`` rows ``prefix_len..`` by
        ``rotary_embedding_to_cache`` (update_idx % 32 == 0), V by ``fill_cache``; the SDPA attends the whole cache
        under ``attn_in["sdpa_mask"]``. Returns the head-concatenated context ``[B, 1, S, H*dh]`` (L1); the caller
        owns the o_proj.
        """
        if self._fused_attn is not None:
            xqkv = self._qkv_proj(hidden_states)
            ctx = self._fused_attn(xqkv, cache_k, cache_v, prefix_len, attn_in["tables"], attn_in["exp_mask"])
            ttnn.deallocate(xqkv)
            return ctx
        cos, sin = attn_in.get("cos"), attn_in.get("sin")
        if cos is None:
            raise ValueError("PI05_EXPERT_ATTN=ttnn rotates every request with one RoPE table: the batch needs one n_valid")
        q, k, v = self._qkv_heads(hidden_states)

        q_rope = ttnn.experimental.rotary_embedding(q, cos, sin)
        ttnn.deallocate(q)
        if int(k.shape[0]) == 1:
            rotary_embedding_to_cache(k, cos, sin, cache_k, prefix_len)
        else:  # B requests: rotate once, then one fill per request (fill_cache writes one batch entry)
            k_rope = ttnn.experimental.rotary_embedding(k, cos, sin)
            _fill_cache_batched(cache_k, k_rope, prefix_len)
            ttnn.deallocate(k_rope)
        _fill_cache_batched(cache_v, v, prefix_len)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_rope,
            cache_k,
            cache_v,
            attn_mask=attn_in["sdpa_mask"],
            is_causal=False,
            scale=self.scale,
            program_config=self._sdpa_config_fused,
        )
        ttnn.deallocate(q_rope)
        attn_concat = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)
        return attn_concat


# ============================================================================
# GeGLU MLP (TTNN)
# ============================================================================


class GemmaMLPTTNN:
    """
    Gemma MLP with GeGLU activation using TTNN.

    Uses chunking along sequence dimension combined with auto L1 sharding
    to fit large intermediate tensors (mlp_dim=16384) in L1 memory.

    Strategy:
    - Chunk input along sequence dimension (e.g., 544 → 3 chunks of 256)
    - Let matmul auto-compute optimal sharding for L1
    - Subsequent ops inherit the sharding from matmul output
    - Accumulate results in L1, concatenate at end
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, torch.Tensor],
        device: ttnn.Device,
        *,
        fused_cfg: FusedConfig,
        role: str = "vlm",
    ):
        """
        Initialize MLP with weights.

        Args:
            config: Gemma configuration
            weights: TTNN (or PyTorch, converted here) weight tensors
            device: TTNN device
            fused_cfg: fused-graph knobs
        """
        self.config = config
        self.device = device
        self.fused_cfg = fused_cfg
        self.role = role
        self._mm_config_fused = None  # PI05_EXPERT_MM=minimal: minimal_matmul config for the expert gate / up
        self._up_program_config = None  # PI05_EXPERT_MM=mcast1d: explicit ttnn.linear program config for up_proj
        if role == "expert" and fused_cfg.expert_mm == "minimal":
            self._mm_config_fused = dit_config_from_blocks(device, fused_cfg.expert_mm_blocks)
        self._want_mcast1d = role == "expert" and fused_cfg.expert_mm in ("mcast1d", "mcast1d_fp32")
        self._mcast1d_fp32 = role == "expert" and fused_cfg.expert_mm == "mcast1d_fp32"
        self._down_pc_cache: Dict[
            int, object
        ] = {}  # PI05_VLM_DOWN_PC=mcast2d: unchunked down-proj program config per seq_len
        self._gateup_pc_cache: Dict[int, Tuple[object, object]] = {}  # PI05_VLM_GATEUP_PC=mcast2d

        # WormholeComputeKernelConfig for MLP — LoFi is 18% faster than HiFi2 for large matmuls
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        # Convert weights to TTNN if they're PyTorch tensors
        # Use bfloat8_b for all MLP weights — reduces bandwidth for VLM too
        mlp_dtype = ttnn.bfloat8_b

        def to_ttnn(w):
            if isinstance(w, torch.Tensor):
                return ttnn.from_torch(
                    w.T.contiguous(),  # Transpose for linear
                    dtype=mlp_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
            return w

        self.gate_proj = to_ttnn(weights["mlp.gate_proj.weight"])
        self.up_proj = to_ttnn(weights["mlp.up_proj.weight"])
        self.down_proj = to_ttnn(weights["mlp.down_proj.weight"])
        self.mlp_dim = config.mlp_dim
        # PI05_EXPERT_GEGLU: the backbone ships a prebuilt [width, 2*mlp] = [up | gate] weight
        self._use_geglu = fused_cfg.expert_geglu and role == "expert" and "mlp.fused_gate_up" in weights
        self.fused_up_gate = weights.get("mlp.fused_gate_up") if self._use_geglu else None
        self._up_gate_program_config = None
        self._upgate_pc_by_rows: Dict[int, object] = {}
        self._up_pc_by_rows: Dict[int, object] = {}

        # VLM MLP sequence chunk (PI05_MLP_CHUNK, tile-aligned; default 256: smaller chunks have lower
        # per-op time but the slice/concat overhead from more chunks dominates; 0 = unchunked)
        self.chunk_size_fused = fused_cfg.mlp_chunk

    # ------------------------------------------------------------------ fused graph

    def up_gate_linear(self, x: ttnn.Tensor, act_dtype: ttnn.DataType, weight: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """one matmul [S, width] x [width, 2*mlp] (1D multicast program as the up-proj); ``weight`` overrides the
        fused up|gate weight (the per-step scale-folded copy of PI05_EXPERT_NORM_FOLD)."""
        w = self.fused_up_gate if weight is None else weight
        mlp_ckc = self.compute_kernel_config_hifi2
        if self._want_mcast1d:
            self._up_gate_program_config = self._mcast1d_pc(self._upgate_pc_by_rows, x, int(w.shape[-1]))
        return ttnn.linear(
            x,
            w,
            dtype=act_dtype,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=with_fp32_acc(mlp_ckc)
            if (self._up_gate_program_config is not None and self._mcast1d_fp32)
            else mlp_ckc,
            program_config=self._up_gate_program_config,
        )

    def forward_fused_pre_down(self, x: ttnn.Tensor, act_dtype: ttnn.DataType) -> ttnn.Tensor:
        """GeGLU without the separate gelu launch: ``gelu(x @ Wg) * (x @ Wu)`` as gate linear with
        ``activation="gelu"`` (UnaryOpType.GELU, approx=False == ``ttnn.gelu`` default) + up linear +
        multiply. ``act_dtype`` is the dtype of the intermediates (bf16 for the fused bf16 residual,
        bf8 for ``PI05_FUSED_RESIDUAL`` mixed / legacy). Output ``[.., mlp_dim]`` in L1."""
        mlp_ckc = self.compute_kernel_config_hifi2
        if self._use_geglu:
            ug = self.up_gate_linear(x, act_dtype)
            # ttnn.geglu indexes the split dim as 3 -> rank-4 view (free reshapes of a tile-aligned tensor)
            b, s_rows, two_mlp = ug.shape[0], ug.shape[1], ug.shape[2]
            ug4 = ttnn.reshape(ug, (b, 1, s_rows, two_mlp))
            hidden_out = ttnn.geglu(ug4, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(ug)
            return ttnn.reshape(hidden_out, (b, s_rows, two_mlp // 2))
        if self._mm_config_fused is not None:
            gate = ttnn.experimental.minimal_matmul(
                x,
                self.gate_proj,
                fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False),
                config=self._mm_config_fused,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=act_dtype,
                compute_kernel_config=mlp_ckc,
            )
            up = ttnn.experimental.minimal_matmul(
                x,
                self.up_proj,
                config=self._mm_config_fused,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=act_dtype,
                compute_kernel_config=mlp_ckc,
            )
        else:
            gate = ttnn.linear(
                x,
                self.gate_proj,
                dtype=act_dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                activation="gelu",
                compute_kernel_config=mlp_ckc,
            )
            if self._want_mcast1d:
                self._up_program_config = self._mcast1d_pc(self._up_pc_by_rows, x, int(self.up_proj.shape[-1]))
            up = ttnn.linear(
                x,
                self.up_proj,
                dtype=act_dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=with_fp32_acc(mlp_ckc)
                if (self._up_program_config is not None and self._mcast1d_fp32)
                else mlp_ckc,
                program_config=self._up_program_config,
            )
        hidden_out = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return hidden_out

    def _mcast1d_pc(self, cache: Dict[int, object], x: ttnn.Tensor, n: int):
        """1D-multicast program config for ``x [B, S, K] @ [K, n]`` keyed by the row count (per_core_M = rows / 32)."""
        rows = 1
        for d in tuple(x.shape)[:-1]:
            rows *= int(d)
        pc = cache.get(rows)
        if pc is None:
            pc = mcast1d_program_config(self.device, n, per_core_m=max(1, rows // 32))
            cache[rows] = pc
        return pc

    def _down_out_dtype(self):
        """VLM down-proj output: bf8 on one chip (as shipped); bf16 for the TP partial that is all-reduced
        (one rounding of the full sum instead of one per chip)."""
        return ttnn.bfloat16 if self.fused_cfg.tp > 1 else ttnn.bfloat8_b

    def _gateup_program_configs(self, seq_len: int):
        """PI05_VLM_GATEUP_PC=mcast2d: explicit 2D multicast configs for the unchunked gate (+GELU) / up
        projections [S,2048]x[2048,16384] (probe 2026-09-13: 0.56 ms each with in0_block_w 4; the auto
        program was 0.40 ms in isolation but its choice depends on the free L1 of the process); auto -> None."""
        if self.fused_cfg.vlm_gateup_pc != "mcast2d":
            return None, None
        if seq_len not in self._gateup_pc_cache:
            k, n = int(self.gate_proj.shape[-2]), int(self.gate_proj.shape[-1])
            gelu = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, False)
            self._gateup_pc_cache[seq_len] = (
                mcast2d_program_config(self.device, seq_len, k, n, in0_block_w=4, fused_activation=gelu),
                mcast2d_program_config(self.device, seq_len, k, n, in0_block_w=4),
            )
        return self._gateup_pc_cache[seq_len]

    def _down_program_config(self, seq_len: int):
        """2D multicast config for the unchunked down projection (cached per seq_len); None = auto."""
        if self.fused_cfg.vlm_down_pc != "mcast2d":
            return None
        if seq_len not in self._down_pc_cache:
            self._down_pc_cache[seq_len] = mcast2d_program_config(
                self.device, seq_len, int(self.down_proj.shape[-2]), int(self.down_proj.shape[-1])
            )
        return self._down_pc_cache[seq_len]

    def forward_fused_vlm(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """VLM MLP ``[B, S, D] -> [B, S, D]`` (bf8, L1) with the fused GeGLU per chunk.

        ``chunk_size_fused`` (PI05_MLP_CHUNK, default 256; non-tile-aligned tails are padded per
        chunk). ``0`` = unchunked: gate / up / product in
        DRAM (fit is device-gated: the author chunked to keep the intermediates in L1) and one
        down-proj reading the weights once instead of once per chunk.
        """
        mlp_ckc = self.compute_kernel_config_hifi2
        batch_size, seq_len, hidden = x.shape[0], x.shape[1], x.shape[2]
        chunk = self.chunk_size_fused

        if chunk == 0 and int(batch_size) > 1:
            # B requests: the MLP is row-independent, so fold the batch into rows (free view) when the
            # per-chip gate/up intermediates still fit L1, else run the unchunked path per request.
            rows = int(batch_size) * int(seq_len)
            if rows <= 1536:
                x_rows = ttnn.reshape(x, (1, rows, hidden))
                out_rows = self.forward_fused_vlm(x_rows)
                return ttnn.reshape(out_rows, (batch_size, seq_len, hidden))
            outs = []
            for b in range(int(batch_size)):
                xb = ttnn.slice(x, [b, 0, 0], [b + 1, seq_len, hidden])
                outs.append(self.forward_fused_vlm(xb))
                ttnn.deallocate(xb)
            out = ttnn.concat(outs, dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
            for o in outs:
                ttnn.deallocate(o)
            return out

        if chunk == 0 or seq_len <= chunk:
            # Unchunked (PI05_MLP_CHUNK=0): the weights are read once. gate / up through the auto
            # program (0.40 ms each for 736 rows, L1 out); the down projection needs an explicit 2D
            # multicast program config -- the auto program took 2.90 ms for [736,16384]x[16384,2048],
            # the config 0.20 ms (probe_vlm_mlp.py, 2026-09-13). PI05_VLM_DOWN_PC=auto restores the
            # auto program (A/B).
            mem = ttnn.L1_MEMORY_CONFIG
            gate_pc, up_pc = self._gateup_program_configs(seq_len)
            gate = ttnn.linear(
                x,
                self.gate_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=mem,
                activation=None if gate_pc is not None else "gelu",  # with a program config the GELU is inside it
                compute_kernel_config=mlp_ckc,
                program_config=gate_pc,
            )
            up = ttnn.linear(
                x,
                self.up_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=mem,
                compute_kernel_config=mlp_ckc,
                program_config=up_pc,
            )
            hidden_out = ttnn.multiply(gate, up, memory_config=mem)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            output = ttnn.linear(
                hidden_out,
                self.down_proj,
                dtype=self._down_out_dtype(),
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
                program_config=self._down_program_config(seq_len),
            )
            ttnn.deallocate(hidden_out)
            return output

        x4 = ttnn.reshape(x, [batch_size, 1, seq_len, hidden])
        num_chunks = (seq_len + chunk - 1) // chunk
        output_chunks = []
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk
            chunk_end = min(chunk_start + chunk, seq_len)
            actual = chunk_end - chunk_start
            padded = ((actual + 31) // 32) * 32
            needs_pad = actual != padded

            x_chunk = ttnn.slice(x4, [0, 0, chunk_start, 0], [batch_size, 1, chunk_end, hidden])
            if needs_pad:
                x_chunk = ttnn.to_memory_config(x_chunk, ttnn.DRAM_MEMORY_CONFIG)
                x_chunk = ttnn.pad(x_chunk, padding=((0, 0), (0, 0), (0, padded - actual), (0, 0)), value=0.0)

            gate = ttnn.linear(
                x_chunk,
                self.gate_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                activation="gelu",
                compute_kernel_config=mlp_ckc,
            )
            up = ttnn.linear(
                x_chunk,
                self.up_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
            )
            ttnn.deallocate(x_chunk)
            hidden_out = ttnn.multiply(gate, up)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            out_chunk = ttnn.linear(
                hidden_out,
                self.down_proj,
                dtype=self._down_out_dtype(),
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
            )
            ttnn.deallocate(hidden_out)
            if needs_pad:
                out_chunk = ttnn.slice(out_chunk, [0, 0, 0, 0], [batch_size, 1, actual, hidden])
            output_chunks.append(out_chunk)

        output = output_chunks[0]
        for i in range(1, len(output_chunks)):
            output = ttnn.concat([output, output_chunks[i]], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(output_chunks[i])
        return ttnn.reshape(output, [batch_size, seq_len, hidden])


# ============================================================================
# Full Transformer Block (TTNN)
# ============================================================================


class GemmaBlockTTNN:
    """
    Complete Gemma transformer block using TTNN.

    Architecture: Pre-LN with residual connections
        x -> RMSNorm -> Attention -> + -> RMSNorm -> MLP -> +
        |______________________________|___________________|
    """

    def __init__(
        self,
        config: GemmaConfig,
        weights: Dict[str, ttnn.Tensor],
        layer_idx: int,
        device: ttnn.Device,
        cos_meta: Optional[ttnn.Tensor] = None,
        sin_meta: Optional[ttnn.Tensor] = None,
        expected_seq_len: Optional[int] = None,
        *,
        fused_cfg: FusedConfig,
        role: str = "vlm",
    ):
        """
        Initialize transformer block with TTNN weights.

        Args:
            config: Gemma configuration
            weights: TTNN weight tensors
            layer_idx: Layer index
            device: TTNN device
            cos_meta: Precomputed cos for native TTNN RoPE [1, 1, max_seq, head_dim]
            sin_meta: Precomputed sin for native TTNN RoPE [1, 1, max_seq, head_dim]
            expected_seq_len: expert role: the suffix length of the fused expert attention
            fused_cfg: fused-graph knobs
            role: "vlm" or "expert"
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.use_adarms = config.use_adarms
        self.fused_cfg = fused_cfg
        self.role = role
        self._dit_config = None
        self._dit_config_by_rows: Dict[int, object] = {}
        self._dit_config = dit_config_from_blocks(device, fused_cfg.dit_blocks)

        if self.use_adarms:
            # Pi0.5: adaRMS dense projection weights
            self.input_ln_dense_weight = weights["input_layernorm.dense.weight"]
            self.input_ln_dense_bias = weights["input_layernorm.dense.bias"]
            self.post_attn_ln_dense_weight = weights["post_attention_layernorm.dense.weight"]
            self.post_attn_ln_dense_bias = weights["post_attention_layernorm.dense.bias"]
        else:
            self.input_layernorm_weight = weights["input_layernorm.weight"]
            self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]

        # PI05_EXPERT_NORM_FOLD: torch originals of the qkv / up|gate weights (set by the backbone loader) and the
        # per-step folded weights (set by backbone.fold_expert_norms after the adaRMS mods exist)
        self._torch_wqkv = weights.get("_torch_wqkv")
        self._torch_fused_ug = weights.get("_torch_fused_ug")
        self._folded = None
        self._row_rsqrt = None
        self._geglu_rc = None
        self.attention = GemmaAttentionTTNN(
            config, weights, layer_idx, device, cos_meta, sin_meta, expected_seq_len, fused_cfg=fused_cfg, role=role
        )
        self.mlp = GemmaMLPTTNN(config, weights, device, fused_cfg=fused_cfg, role=role)

    # ------------------------------------------------------------------ fused graph

    def forward_fused_vlm(
        self,
        hidden_states: ttnn.Tensor,
        cache_k: ttnn.Tensor,
        cache_v: ttnn.Tensor,
        kv_only: bool = False,
        attn_mask: Optional[ttnn.Tensor] = None,
    ) -> Optional[ttnn.Tensor]:
        """VLM block (plain RMSNorm, ungated residuals) that fills the KV cache of this layer.

        Plain Gemma block (cos/sin slices cached per seq_len) plus the two ``fill_cache`` writes;
        ``kv_only=True`` (last layer, PI05_SKIP_VLM_TAIL) computes only what the expert consumes. The
        caller owns / frees ``hidden_states``.
        """
        tp = self.fused_cfg.tp
        normed = rms_norm_ttnn(hidden_states, self.input_layernorm_weight, self.config.rms_norm_eps)
        attn_output = self.attention.forward_fused_vlm(normed, cache_k, cache_v, need_output=not kv_only, attn_mask=attn_mask)
        ttnn.deallocate(normed)
        if kv_only:
            return None
        if tp > 1:  # o_proj is row-parallel: sum the per-chip partials (full [B, S, D] on every chip)
            attn_output = _tp_all_reduce(attn_output, self.fused_cfg)
        hidden_mid = ttnn.add(hidden_states, attn_output)  # ungated residual
        ttnn.deallocate(attn_output)

        normed = rms_norm_ttnn(hidden_mid, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        mlp_output = self.mlp.forward_fused_vlm(normed)
        ttnn.deallocate(normed)
        if tp > 1:  # down_proj is row-parallel
            mlp_output = _tp_all_reduce(mlp_output, self.fused_cfg)
        out = ttnn.add(hidden_mid, mlp_output)
        ttnn.deallocate(mlp_output)
        ttnn.deallocate(hidden_mid)
        return out

    def forward_fused_expert(
        self,
        hidden_states: ttnn.Tensor,
        precomputed_mod: Tuple,
        cache_k: ttnn.Tensor,
        cache_v: ttnn.Tensor,
        prefix_len: int,
        attn_in: Dict[str, ttnn.Tensor],
        step: Optional[int] = None,
    ) -> ttnn.Tensor:
        """Expert block on the 64-row suffix with precomputed adaRMS modulations (owned by the model).

        ``PI05_FUSED_RESIDUAL``:
          bf16   -> ``hidden = dit(ctx bf16, Wo bf16, 1.0, hidden bf16, gate bf16)`` (ctx typecast bf8->bf16
                    first) and ``hidden = dit(gelu(g)*u bf16, Wdown bf16, 1.0, hidden bf16, gate bf16)``:
                    act == weight == residual format, bf16 gate broadcast; residual + gates in DRAM (one
                    buffer type for the two addcmul inputs); output dtype bf16 explicit.
          mixed  -> as bf16 but the bf8 activations feed the fused op directly (no typecast).
          legacy -> linear(o) + mac, and typecast x3 + bf8 dit for the down-proj: the numerics of the
                    port as first shipped.
        Consumes (frees) ``hidden_states``; returns the new hidden ``[B, S, D]`` bf16.
        """
        if self._folded is not None and step is not None and self.attention._fused_attn is not None and self.fused_cfg.residual != "legacy":
            return self._forward_fused_expert_folded(hidden_states, precomputed_mod, cache_k, cache_v, prefix_len, attn_in, step)
        scale_in, shift_in, attn_gate, scale_post, shift_post, mlp_gate = precomputed_mod
        eps = self.config.rms_norm_eps
        mode = self.fused_cfg.residual
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        # ---- attention + gated residual ----
        normed = adarms_norm_precomputed(hidden_states, scale_in, shift_in, eps)
        attn_concat = self.attention.forward_fused_expert(normed, cache_k, cache_v, prefix_len, attn_in)
        ttnn.deallocate(normed)
        attn_concat = ttnn.reshape(attn_concat, (batch_size, seq_len, attn_concat.shape[-1]))
        if mode == "legacy":
            attn_output = ttnn.linear(
                attn_concat,
                self.attention.o_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.attention.compute_kernel_config_hifi4,
            )
            ttnn.deallocate(attn_concat)
            new_hidden = ttnn.mac(attn_gate, attn_output, hidden_states)
            ttnn.deallocate(attn_output)
        else:
            if mode == "bf16" and attn_concat.dtype != ttnn.bfloat16:
                ctx = ttnn.typecast(attn_concat, ttnn.bfloat16)
                ttnn.deallocate(attn_concat)
                attn_concat = ctx
            new_hidden = self._dit_residual(attn_concat, self.attention.o_proj, hidden_states, attn_gate)
            ttnn.deallocate(attn_concat)
        ttnn.deallocate(hidden_states)
        hidden_states = new_hidden

        # ---- MLP + gated residual ----
        normed = adarms_norm_precomputed(hidden_states, scale_post, shift_post, eps)
        act_dtype = ttnn_dtype_from_name(self.fused_cfg.expert_mlp_act_dtype())
        hidden_out = self.mlp.forward_fused_pre_down(normed, act_dtype)
        ttnn.deallocate(normed)
        if mode == "legacy":
            hs_8b = ttnn.typecast(hidden_states, ttnn.bfloat8_b)
            gate_8b = ttnn.typecast(mlp_gate, ttnn.bfloat8_b)
            fused = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                matmul_input_tensor=hidden_out,
                matmul_weight_tensor=self.mlp.down_proj,
                scalar=1.0,
                addcmul_input_tensor1=hs_8b,
                addcmul_input_tensor2=gate_8b,
            )
            ttnn.deallocate(hidden_out)
            ttnn.deallocate(hs_8b)
            ttnn.deallocate(gate_8b)
            ttnn.deallocate(hidden_states)
            new_hidden = ttnn.typecast(fused, ttnn.bfloat16)
            ttnn.deallocate(fused)
        else:
            new_hidden = self._dit_residual(hidden_out, self.mlp.down_proj, hidden_states, mlp_gate)
            ttnn.deallocate(hidden_out)
            ttnn.deallocate(hidden_states)
        return new_hidden

    def set_folded(self, folded) -> None:
        """Install the per-step folded qkv / up|gate weights (``ttnn_fused_norm.FoldedExpertNorms``)."""
        from .ttnn_fused_norm import GegluRC, RowRsqrt

        self._folded = folded
        self._row_rsqrt = RowRsqrt(self.device, self.config.width, self.config.rms_norm_eps)
        self._geglu_rc = GegluRC(self.device, self.config.mlp_dim)

    def _forward_fused_expert_folded(
        self, hidden_states: ttnn.Tensor, precomputed_mod: Tuple, cache_k: ttnn.Tensor, cache_v: ttnn.Tensor, prefix_len: int,
        attn_in: Dict[str, ttnn.Tensor], step: int
    ) -> ttnn.Tensor:
        """Expert block with both adaRMS norms folded into the step's qkv / up|gate weights:
        [adaRMS_in -> qkv linear -> fused attention] -> o_proj gated residual ->
        row_rsqrt -> up|gate linear (W') -> fused GeGLU (applies r, c) -> down gated residual: 7 launches
        (the attention-side fold, PI05_EXPERT_NORM_FOLD_ATTN=1, replaces the first bracket by row_rsqrt -> qkv linear (W')
        -> fused attention with r/c prologue: 6 launches, measured no faster)."""
        scale_in, shift_in, attn_gate, _s_post, _h_post, mlp_gate = precomputed_mod
        mode = self.fused_cfg.residual
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        attn = self.attention

        if self._folded.fold_attn:
            r_in = self._row_rsqrt(hidden_states)
            xqkv = attn._qkv_proj(hidden_states, weight=self._folded.wqkv[step])
            ctx = attn._fused_attn(xqkv, cache_k, cache_v, prefix_len, attn_in["tables"], attn_in["exp_mask"], r=r_in,
                                   c=self._folded.cqkv[step])
            ttnn.deallocate(xqkv)
            ttnn.deallocate(r_in)
        else:
            normed = adarms_norm_precomputed(hidden_states, scale_in, shift_in, self.config.rms_norm_eps)
            ctx = attn.forward_fused_expert(normed, cache_k, cache_v, prefix_len, attn_in)
            ttnn.deallocate(normed)
        ctx = ttnn.reshape(ctx, (batch_size, seq_len, ctx.shape[-1]))
        if mode == "bf16" and ctx.dtype != ttnn.bfloat16:
            ctx16 = ttnn.typecast(ctx, ttnn.bfloat16)
            ttnn.deallocate(ctx)
            ctx = ctx16
        hidden_mid = self._dit_residual(ctx, attn.o_proj, hidden_states, attn_gate)
        ttnn.deallocate(ctx)
        ttnn.deallocate(hidden_states)

        r_post = self._row_rsqrt(hidden_mid)
        ug = self.mlp.up_gate_linear(hidden_mid, ttnn.bfloat16, weight=self._folded.wug[step])
        h = self._geglu_rc(ug, r_post, self._folded.cug[step])
        ttnn.deallocate(ug)
        ttnn.deallocate(r_post)
        h = ttnn.reshape(h, (batch_size, seq_len, h.shape[-1]))
        new_hidden = self._dit_residual(h, self.mlp.down_proj, hidden_mid, mlp_gate)
        ttnn.deallocate(h)
        ttnn.deallocate(hidden_mid)
        return new_hidden

    def _dit_residual(self, x: ttnn.Tensor, weight: ttnn.Tensor, hidden: ttnn.Tensor, gate: ttnn.Tensor) -> ttnn.Tensor:
        """``hidden + (x @ weight) * gate`` in one launch. One request (64 rows): the tuned PI05_DIT_BLOCKS
        config on ``[1, 64, K]``. B requests: the batch is folded into rows (free views) and the M block
        covers all rows (probe 2026-09-17: o_proj at B=4 76 us with the B=1 blocks vs 43.5 us with M_block=2B)."""
        b, s = int(x.shape[0]), int(x.shape[1])
        if b == 1:
            return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                x, weight, 1.0, hidden, gate, config=self._dit_config, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
            )
        rows = b * s
        cfg = self._dit_config_by_rows.get(rows)
        if cfg is None:
            blocks = self.fused_cfg.dit_blocks
            if blocks is not None:
                blocks = (max(1, rows // 32),) + tuple(blocks[1:])
            cfg = dit_config_from_blocks(self.device, blocks)
            self._dit_config_by_rows[rows] = cfg
        x_rows = ttnn.reshape(x, (1, rows, int(x.shape[-1])))
        h_rows = ttnn.reshape(hidden, (1, rows, int(hidden.shape[-1])))
        out = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x_rows, weight, 1.0, h_rows, gate, config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16
        )
        return ttnn.reshape(out, (b, s, int(hidden.shape[-1])))


# Default exports
GemmaAttention = GemmaAttentionTTNN
GemmaMLP = GemmaMLPTTNN
GemmaBlock = GemmaBlockTTNN
