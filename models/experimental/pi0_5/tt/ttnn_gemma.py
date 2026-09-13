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

Fused graph (``TT_FUSED=1``, ``FusedConfig.enabled``; every ``forward_fused_*`` method below, the
legacy ``forward`` methods are untouched):
    - VLM attention writes its rotated K / V straight into the backbone-owned KV caches
      (``fill_cache`` at row 0) so the expert never refills the prefix (was 2 fill_cache per
      layer per denoising step); cos/sin slices are cached per seq_len (no per-layer slices).
    - Expert attention runs on the tile-padded 64-row suffix (no slice(q) after RoPE), writes
      K/V rows prefix_len..prefix_len+63 with the fork's ``rotary_embedding_to_cache`` +
      ``fill_cache`` and attends over the cache (logical rows = prefix_len + action_horizon, so the
      SDPA kernel masks the padded suffix rows exactly as before).
    - Expert gated residuals: ONE ``dit_minimal_matmul_addcmul_fused`` per residual
      (``hidden + 1.0 * (x @ W) * gate``) with a bf16 hidden stream, bf16 o_proj / down_proj and
      bf16 gates (``PI05_FUSED_RESIDUAL=bf16``): replaces linear + mac (2 launches) and the
      3 typecasts around the legacy bf8 down-proj fusion; the residual is no longer rounded to bf8.
    - GeGLU: gate linear with ``activation="gelu"`` (same non-approximate SFPU GELU as ``ttnn.gelu``)
      + up linear + multiply; the VLM MLP keeps the legacy 256-row chunking (``PI05_MLP_CHUNK``).
    - SDPA program configs are knobs (``PI05_SDPA_{VLM,EXPERT}_CHUNKS``), default = legacy.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig


# ============================================================================
# Fused-graph helpers (only used when FusedConfig.enabled)
# ============================================================================

_TTNN_DTYPE_BY_NAME = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b}


def ttnn_dtype_from_name(name: str) -> ttnn.DataType:
    return _TTNN_DTYPE_BY_NAME[name]


def sdpa_program_config_from_chunks(device: ttnn.Device, chunks: Optional[Tuple[int, int]]):
    """``(q_chunk, k_chunk)`` -> SDPAProgramConfig on the FULL compute grid; None -> None (legacy)."""
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


def adarms_norm_ttnn(
    x: ttnn.Tensor,
    dense_weight: ttnn.Tensor,
    dense_bias: ttnn.Tensor,
    cond: ttnn.Tensor,
    eps: float = 1e-6,
    device: ttnn.Device = None,
    ones_weight: Optional[ttnn.Tensor] = None,  # deprecated, kept for API compat
    use_fused: bool = True,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    # Use fused C++ op if available (source build custom op)
    if use_fused and hasattr(ttnn.experimental, "fused_adaptive_rms"):
        # The fused op slices the modulation with rank-3 start/end/step vectors, so it requires a
        # rank-3 cond. Match the fallback path's reshape for 2D [batch, cond_dim] conditioning.
        if len(cond.shape) == 2:
            cond = ttnn.reshape(cond, (cond.shape[0], 1, -1))
        return ttnn.experimental.fused_adaptive_rms(x, dense_weight, dense_bias, cond, eps)
    """
    Adaptive RMSNorm (Pi0.5) using TTNN operations.

    Projects conditioning vector to (scale, shift, gate) and applies:
        normed = x / RMS(x) * (1 + scale) + shift
    Returns (normed, gate) where gate is used for gated residual.

    Args:
        x: TTNN tensor (batch_size, seq_len, hidden_dim)
        dense_weight: Projection weight (cond_dim, hidden_dim * 3) — transposed for TTNN
        dense_bias: Projection bias (1, hidden_dim * 3)
        cond: Conditioning vector (batch_size, cond_dim)
        eps: Epsilon for numerical stability
        device: TTNN device

    Returns:
        Tuple of (normed output, gate tensor)
    """
    batch_size = x.shape[0]

    # Project conditioning: cond (B, cond_dim) -> modulation (B, hidden_dim * 3)
    cond_2d = ttnn.reshape(cond, (batch_size, 1, -1)) if len(cond.shape) == 2 else cond
    modulation = ttnn.linear(
        cond_2d,
        dense_weight,
        bias=dense_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Split into scale, shift, gate — each (B, 1, hidden_dim)
    # Single chunk op replaces 3 separate slices
    scale, shift, gate = ttnn.chunk(modulation, 3, dim=-1)
    ttnn.deallocate(modulation)

    # FUSED: rms_norm with dynamic weight=scale and bias=shift
    # The +1 offset is pre-baked into the dense bias during initialization
    # so scale already contains (1 + learned_scale)
    normed = ttnn.rms_norm(x, weight=scale, bias=shift, epsilon=eps, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(scale)
    ttnn.deallocate(shift)

    return normed, gate


def adarms_norm_precomputed(
    x: ttnn.Tensor,
    scale: ttnn.Tensor,
    shift: ttnn.Tensor,
    eps: float,
) -> ttnn.Tensor:
    """
    Apply adaRMS when (scale, shift, gate) have been precomputed from cond via the
    dense projection outside the critical path. `scale` already has +1 baked into
    the bias (same as in adarms_norm_ttnn), so this reduces to rms_norm with
    runtime weight/bias.
    """
    return ttnn.rms_norm(
        x,
        weight=scale,
        bias=shift,
        epsilon=eps,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def gated_residual_ttnn(
    x: ttnn.Tensor,
    y: ttnn.Tensor,
    gate: Optional[ttnn.Tensor],
) -> ttnn.Tensor:
    """Gated residual: x + y * gate (Pi0.5) or x + y (Pi0)."""
    if gate is None:
        return ttnn.add(x, y)
    # FUSED: mac(gate, y, x) = gate * y + x — single op instead of mul + add
    return ttnn.mac(gate, y, x)


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
        fused_cfg: Optional[FusedConfig] = None,
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
            expected_seq_len: If set, pre-slice RoPE cos/sin for this seq_len (saves 2 slice ops per forward)
            fused_cfg: TT_FUSED knobs (None / disabled -> the legacy ``forward`` only)
            role: "vlm" or "expert" (selects the SDPA program-config knob of the fused path)
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.fused_cfg = fused_cfg
        self.role = role
        # Fused path: cos/sin slices cached per seq_len (built on the compile pass, outside the trace)
        self._rope_cache: Dict[int, Tuple[ttnn.Tensor, ttnn.Tensor]] = {}
        self._sdpa_config_fused = None
        self._mm_config_fused = None  # PI05_EXPERT_MM=minimal: minimal_matmul config for the expert qkv projection
        self._qkv_program_config = None  # PI05_EXPERT_MM=mcast1d: explicit ttnn.linear program config
        if fused_cfg is not None and fused_cfg.enabled:
            chunks = fused_cfg.sdpa_expert if role == "expert" else fused_cfg.sdpa_vlm
            self._sdpa_config_fused = sdpa_program_config_from_chunks(device, chunks)
            if role == "expert" and fused_cfg.expert_mm == "minimal":
                self._mm_config_fused = dit_config_from_blocks(device, fused_cfg.expert_mm_blocks)
        self._want_qkv_mcast1d = (
            fused_cfg is not None and fused_cfg.enabled and role == "expert" and fused_cfg.expert_mm in ("mcast1d", "mcast1d_fp32")
        )  # the config needs self.wqkv (assigned below): built lazily in _qkv_heads
        # PI05_VLM_ATTN_PC=mcast2d[_fp32]: explicit 2D multicast configs for the VLM qkv / o_proj linears
        # (probe 2026-09-13: [736,2048]x[2048,2560] 0.387 -> 0.035 ms, [736,2048]x[2048,2048] 0.405 -> 0.029 ms)
        self._want_vlm_mcast2d = (
            fused_cfg is not None and fused_cfg.enabled and role == "vlm" and fused_cfg.vlm_attn_pc in ("mcast2d", "mcast2d_fp32")
        )
        self._vlm_pc_cache: Dict[Tuple[str, int], object] = {}
        # fp32 destination accumulation for the explicit-program linears (mcast1d_fp32 / mcast2d_fp32)
        self._pc_fp32 = fused_cfg is not None and fused_cfg.enabled and (
            (role == "expert" and fused_cfg.expert_mm == "mcast1d_fp32") or (role == "vlm" and fused_cfg.vlm_attn_pc == "mcast2d_fp32")
        )

        # OPTIMIZATION: Use fused QKV weight (single linear instead of 3)
        self.wqkv = weights["self_attn.wqkv"]
        self.o_proj = weights["self_attn.o_proj.weight"]

        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.width
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Store meta format cos/sin for native TTNN RoPE (split-half pattern)
        self.cos_meta = cos_meta
        self.sin_meta = sin_meta

        self._sdpa_config = None

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

        # OPTIMIZATION: Pre-slice RoPE for known sequence length (saves 2 slice ops per forward)
        self._presliced_seq_len = expected_seq_len
        if expected_seq_len is not None and cos_meta is not None:
            self._cos_presliced = ttnn.slice(cos_meta, [0, 0, 0, 0], [1, 1, expected_seq_len, self.head_dim])
            self._sin_presliced = ttnn.slice(sin_meta, [0, 0, 0, 0], [1, 1, expected_seq_len, self.head_dim])
        else:
            self._cos_presliced = None
            self._sin_presliced = None

        # OPTIMIZATION: Lazy-allocated output buffers for KV concat (source build supports output_tensor)
        self._kv_concat_k = None
        self._kv_concat_v = None

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        OPTIMIZED forward pass using fused QKV and native TTNN operations.

        Key optimizations:
        1. Single fused QKV linear (3x fewer linear ops)
        2. Native ttnn.experimental.nlp_create_qkv_heads
        3. Native ttnn.experimental.rotary_embedding (split-half pattern)
        4. Native ttnn.experimental.nlp_concat_heads for output

        Args:
            hidden_states: TTNN tensor (batch, seq_len, hidden_dim)
            cos, sin: Unused (kept for API compatibility, native RoPE uses self.cos_meta/sin_meta)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache

        Returns:
            Tuple of (output, optional_cache)
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Reshape to 4D for nlp_create_qkv_heads: [batch, 1, seq, hidden]
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))

        # OPTIMIZATION 1: Single fused QKV linear (instead of 3 separate)
        # Output: [batch, 1, seq, Q_dim + K_dim + V_dim]

        # QKV linear — L1 interleaved directly (avoids to_memory_config op)
        xqkv = ttnn.linear(
            hidden_states,
            self.wqkv,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )

        # OPTIMIZATION 2: Native TTNN head splitting (no PyTorch transfers!)
        # This splits the fused QKV into separate Q, K, V with proper head layout
        # Output shapes: q=[batch, num_heads, seq, head_dim], k/v=[batch, num_kv_heads, seq, head_dim]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # OPTIMIZATION 3: Apply RoPE using native TTNN (split-half pattern)
        # Use pre-sliced cos/sin when available (saves 2 slice ops per forward)
        if self._cos_presliced is not None and seq_len == self._presliced_seq_len:
            cos_sliced = self._cos_presliced
            sin_sliced = self._sin_presliced
            needs_dealloc = False
        else:
            cos_sliced = ttnn.slice(self.cos_meta, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
            sin_sliced = ttnn.slice(self.sin_meta, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
            needs_dealloc = True

        # Q: normal rotary_embedding (output used immediately for SDPA)
        q_rope_padded = ttnn.experimental.rotary_embedding(q, cos_sliced, sin_sliced)
        # rotary_embedding pads output to tile boundary, slice back to original seq_len
        q_rope = ttnn.slice(q_rope_padded, [0, 0, 0, 0], [batch_size, self.num_heads, seq_len, self.head_dim])

        # Handle KV cache via FUSED rotary_embedding_to_cache + V fill_cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            prefix_len = past_k.shape[2]
            suffix_len = k.shape[2]
            full_seq = prefix_len + suffix_len
            # (Re)allocate the concat KV buffer when missing or when the prefix length
            # changes (e.g. a different language-token count between runs).
            if self._kv_concat_k is None or self._kv_concat_k.shape[2] != full_seq:
                cache_dtype = past_k.dtype
                self._kv_concat_k = ttnn.zeros(
                    [1, self.num_kv_heads, full_seq, self.head_dim],
                    dtype=cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                self._kv_concat_v = ttnn.zeros(
                    [1, self.num_kv_heads, full_seq, self.head_dim],
                    dtype=cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            # Refill the prefix KV on EVERY call. The prefix (VLM) KV changes each replan
            # because the observation changes, so this MUST NOT be cached across
            # sample_actions() calls — doing so makes a multi-replan rollout attend to the
            # FIRST frame's prefix forever. This is silently wrong and invisible to the
            # single-call PCC / determinism / per-step tests, but breaks closed-loop control.
            ttnn.fill_cache(self._kv_concat_k, past_k, 0, update_idx=0)
            ttnn.fill_cache(self._kv_concat_v, past_v, 0, update_idx=0)
            # Ensure dtype match for cache write
            if k.dtype != self._kv_concat_k.dtype:
                k = ttnn.typecast(k, self._kv_concat_k.dtype)
            if v.dtype != self._kv_concat_v.dtype:
                v = ttnn.typecast(v, self._kv_concat_v.dtype)
            # FUSED: rotate K and write directly to cache at prefix_len offset
            ttnn.experimental.rotary_embedding_to_cache(k, cos_sliced, sin_sliced, self._kv_concat_k, prefix_len)
            # V: no rotation, regular fill_cache
            ttnn.fill_cache(self._kv_concat_v, v, 0, update_idx=prefix_len)
            k_rope = self._kv_concat_k
            v = self._kv_concat_v
            if needs_dealloc:
                ttnn.deallocate(cos_sliced)
                ttnn.deallocate(sin_sliced)
        else:
            # VLM path: rotate K normally (no cache update)
            k_rope_padded = ttnn.experimental.rotary_embedding(k, cos_sliced, sin_sliced)
            k_rope = ttnn.slice(k_rope_padded, [0, 0, 0, 0], [batch_size, self.num_kv_heads, seq_len, self.head_dim])
            if needs_dealloc:
                ttnn.deallocate(cos_sliced)
                ttnn.deallocate(sin_sliced)

        new_cache = (k_rope, v) if use_cache else None

        # Use TTNN scaled dot product attention with tuned program config
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_rope,
            k_rope,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scale,
            program_config=self._sdpa_config,
        )

        # OPTIMIZATION 4: Native TTNN head concatenation (no PyTorch transfers!)
        # attn_output: [batch, num_heads, seq, head_dim] -> [batch, 1, seq, num_heads * head_dim]
        attn_concat = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Output projection with HiFi4 for precision-critical residual path
        output = ttnn.linear(
            attn_concat,
            self.o_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
        )

        # Reshape back to 3D: [batch, 1, seq, hidden] -> [batch, seq, hidden]
        output = ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))

        return output, new_cache

    # ------------------------------------------------------------------ fused graph

    def _rope(self, seq_len: int) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos/sin ``[1, 1, seq_len, head_dim]`` slices cached per seq_len (exact: the same slice the
        legacy forward takes every call; the first call happens on the compile pass, before capture)."""
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

    def _qkv_heads(self, hidden_states: ttnn.Tensor):
        """normed [B, S, D] -> (q [B, H, S, dh], k [B, KVH, S, dh], v [B, KVH, S, dh]) bf8 in L1 (legacy ops)."""
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        x4 = ttnn.reshape(hidden_states, (batch_size, 1, seq_len, -1))
        if self._mm_config_fused is not None:
            xqkv = ttnn.experimental.minimal_matmul(
                x4,
                self.wqkv,
                config=self._mm_config_fused,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )
        else:
            if self._want_qkv_mcast1d and self._qkv_program_config is None:
                self._qkv_program_config = mcast1d_program_config(self.device, int(self.wqkv.shape[-1]))
            pc = self._qkv_program_config
            if self._want_vlm_mcast2d:
                pc = self._vlm_pc("qkv", seq_len, self.wqkv)
            ckc = self.compute_kernel_config_hifi2
            if pc is not None and self._pc_fp32:
                ckc = with_fp32_acc(ckc)
            xqkv = ttnn.linear(
                x4,
                self.wqkv,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=ckc,
                program_config=pc,
            )
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
    ) -> Optional[ttnn.Tensor]:
        """VLM (prefill) attention that also fills the backbone-owned KV cache.

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
        ttnn.fill_cache(cache_k, k_rope, 0, update_idx=0)
        ttnn.fill_cache(cache_v, v, 0, update_idx=0)

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
            attn_mask=None,
            is_causal=False,
            scale=self.scale,
            program_config=self._sdpa_config_fused,
        )
        ttnn.deallocate(q_rope)
        ttnn.deallocate(k_rope)
        ttnn.deallocate(v)

        attn_concat = ttnn.experimental.nlp_concat_heads(attn_output, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_output)
        output = ttnn.linear(
            attn_concat,
            self.o_proj,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=with_fp32_acc(self.compute_kernel_config_hifi4) if (self._want_vlm_mcast2d and self._pc_fp32) else self.compute_kernel_config_hifi4,
            program_config=self._vlm_pc("o", seq_len, self.o_proj) if self._want_vlm_mcast2d else None,
        )
        ttnn.deallocate(attn_concat)
        return ttnn.reshape(output, (batch_size, seq_len, self.hidden_size))

    def forward_fused_expert(
        self,
        hidden_states: ttnn.Tensor,
        cache_k: ttnn.Tensor,
        cache_v: ttnn.Tensor,
        prefix_len: int,
    ) -> ttnn.Tensor:
        """Expert attention on the tile-padded suffix (``S`` = 64 rows for action_horizon 50).

        K is rotated and written to ``cache_k`` rows ``prefix_len..prefix_len+S-1`` by the fork's
        ``rotary_embedding_to_cache`` (update_idx % 32 == 0, same dtype bf8, prefix_len + S <= padded
        cache rows), V by ``fill_cache``; the SDPA attends over the whole cache whose LOGICAL row count
        is prefix_len + action_horizon, so rows beyond it (the zero-padded suffix rows) are masked by the
        kernel exactly as the legacy 786-row concat buffer was. No slice(q): rotary_embedding returns the
        tile-padded rows and every downstream op is row-independent (rows 0..49 are bit-identical).
        Returns the head-concatenated context ``[B, 1, S, H*dh]`` bf8 (L1); the caller owns the o_proj.
        """
        seq_len = hidden_states.shape[1]
        q, k, v = self._qkv_heads(hidden_states)
        cos_sliced, sin_sliced = self._rope(seq_len)

        q_rope = ttnn.experimental.rotary_embedding(q, cos_sliced, sin_sliced)
        ttnn.deallocate(q)
        ttnn.experimental.rotary_embedding_to_cache(k, cos_sliced, sin_sliced, cache_k, prefix_len)
        ttnn.fill_cache(cache_v, v, 0, update_idx=prefix_len)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_output = ttnn.transformer.scaled_dot_product_attention(
            q_rope,
            cache_k,
            cache_v,
            attn_mask=None,
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
        fused_cfg: Optional[FusedConfig] = None,
        role: str = "vlm",
    ):
        """
        Initialize MLP with weights.

        Args:
            config: Gemma configuration
            weights: PyTorch weight tensors (will be converted to TTNN)
            device: TTNN device
            fused_cfg: TT_FUSED knobs (None / disabled -> the legacy ``forward`` only)
        """
        self.config = config
        self.device = device
        self.fused_cfg = fused_cfg
        self.role = role
        self._mm_config_fused = None  # PI05_EXPERT_MM=minimal: minimal_matmul config for the expert gate / up
        self._up_program_config = None  # PI05_EXPERT_MM=mcast1d: explicit ttnn.linear program config for up_proj
        if fused_cfg is not None and fused_cfg.enabled and role == "expert" and fused_cfg.expert_mm == "minimal":
            self._mm_config_fused = dit_config_from_blocks(device, fused_cfg.expert_mm_blocks)
        self._want_mcast1d = fused_cfg is not None and fused_cfg.enabled and role == "expert" and fused_cfg.expert_mm in ("mcast1d", "mcast1d_fp32")
        self._mcast1d_fp32 = fused_cfg is not None and fused_cfg.enabled and role == "expert" and fused_cfg.expert_mm == "mcast1d_fp32"
        self._down_pc_cache: Dict[int, object] = {}  # PI05_VLM_DOWN_PC=mcast2d: unchunked down-proj program config per seq_len
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

        # Fuse gate+up weights for single matmul: [hidden, 2*mlp_dim]
        gate_w = weights["mlp.gate_proj.weight"]
        up_w = weights["mlp.up_proj.weight"]
        if isinstance(gate_w, torch.Tensor) and isinstance(up_w, torch.Tensor):
            fused = torch.cat([gate_w, up_w], dim=0)  # [2*mlp_dim, hidden]
            self.fused_gate_up = ttnn.from_torch(
                fused.T.contiguous(),  # [hidden, 2*mlp_dim]
                dtype=mlp_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        else:
            self.fused_gate_up = None
        self.gate_proj = to_ttnn(gate_w)
        self.up_proj = to_ttnn(up_w)
        self.down_proj = to_ttnn(weights["mlp.down_proj.weight"])
        self.mlp_dim = config.mlp_dim

        # Chunk size must be tile-aligned (multiple of 32)
        # 256 is optimal: smaller chunks have lower per-op time but the slice/concat
        # overhead from more chunks dominates
        self.chunk_size = 256

        # Fused graph: no fused gate_up copy is kept (the separate gate/up weights are what the
        # fused GeGLU uses; in the served model the weights already arrive as ttnn tensors, so the
        # legacy path never had one either) and the VLM chunk size is the PI05_MLP_CHUNK knob.
        self.chunk_size_fused = self.chunk_size
        if fused_cfg is not None and fused_cfg.enabled:
            if not fused_cfg.keep_fused_gate_up_copy() and self.fused_gate_up is not None:
                ttnn.deallocate(self.fused_gate_up)
                self.fused_gate_up = None
            self.chunk_size_fused = fused_cfg.mlp_chunk

    # ------------------------------------------------------------------ fused graph

    def forward_fused_pre_down(self, x: ttnn.Tensor, act_dtype: ttnn.DataType) -> ttnn.Tensor:
        """GeGLU without the separate gelu launch: ``gelu(x @ Wg) * (x @ Wu)`` as gate linear with
        ``activation="gelu"`` (UnaryOpType.GELU, approx=False == ``ttnn.gelu`` default) + up linear +
        multiply. ``act_dtype`` is the dtype of the intermediates (bf16 for the fused bf16 residual,
        bf8 for the legacy-numerics residual). Output ``[.., mlp_dim]`` in L1."""
        mlp_ckc = self.compute_kernel_config_hifi2
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
            if self._want_mcast1d and self._up_program_config is None:
                self._up_program_config = mcast1d_program_config(self.device, int(self.up_proj.shape[-1]))
            up = ttnn.linear(
                x,
                self.up_proj,
                dtype=act_dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=with_fp32_acc(mlp_ckc) if (self._up_program_config is not None and self._mcast1d_fp32) else mlp_ckc,
                program_config=self._up_program_config,
            )
        hidden_out = ttnn.multiply(gate, up)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return hidden_out

    def _gateup_program_configs(self, seq_len: int):
        """PI05_VLM_GATEUP_PC=mcast2d: explicit 2D multicast configs for the unchunked gate (+GELU) / up
        projections [S,2048]x[2048,16384] (probe 2026-09-13: 0.56 ms each with in0_block_w 4; the auto
        program was 0.40 ms in isolation but its choice depends on the free L1 of the process); auto -> None."""
        if self.fused_cfg is None or self.fused_cfg.vlm_gateup_pc != "mcast2d":
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
        if self.fused_cfg is None or self.fused_cfg.vlm_down_pc != "mcast2d":
            return None
        if seq_len not in self._down_pc_cache:
            self._down_pc_cache[seq_len] = mcast2d_program_config(
                self.device, seq_len, int(self.down_proj.shape[-2]), int(self.down_proj.shape[-1])
            )
        return self._down_pc_cache[seq_len]

    def forward_fused_vlm(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """VLM MLP ``[B, S, D] -> [B, S, D]`` (bf8, L1) with the fused GeGLU per chunk.

        ``chunk_size_fused`` (PI05_MLP_CHUNK, default 256 = legacy chunking; the legacy per-chunk
        padding logic is kept for non-tile-aligned tails). ``0`` = unchunked: gate / up / product in
        DRAM (fit is device-gated: the author chunked to keep the intermediates in L1) and one
        down-proj reading the weights once instead of once per chunk.
        """
        mlp_ckc = self.compute_kernel_config_hifi2
        batch_size, seq_len, hidden = x.shape[0], x.shape[1], x.shape[2]
        chunk = self.chunk_size_fused

        if chunk == 0 or seq_len <= chunk:
            # Unchunked (PI05_MLP_CHUNK=0): the weights are read once. gate / up through the auto
            # program (0.40 ms each for 736 rows, L1 out); the down projection needs an explicit 2D
            # multicast program config -- the auto program took 2.90 ms for [736,16384]x[16384,2048],
            # the config 0.20 ms (probe_vlm_mlp.py, 2026-09-13). PI05_VLM_DOWN_PC=auto restores the
            # auto program (A/B).
            mem = ttnn.L1_MEMORY_CONFIG
            gate_pc, up_pc = self._gateup_program_configs(seq_len)
            gate = ttnn.linear(
                x, self.gate_proj, dtype=ttnn.bfloat8_b, memory_config=mem,
                activation=None if gate_pc is not None else "gelu",  # with a program config the GELU is inside it
                compute_kernel_config=mlp_ckc, program_config=gate_pc,
            )
            up = ttnn.linear(x, self.up_proj, dtype=ttnn.bfloat8_b, memory_config=mem, compute_kernel_config=mlp_ckc,
                             program_config=up_pc)
            hidden_out = ttnn.multiply(gate, up, memory_config=mem)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            output = ttnn.linear(
                hidden_out, self.down_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc, program_config=self._down_program_config(seq_len),
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
                x_chunk, self.gate_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG,
                activation="gelu", compute_kernel_config=mlp_ckc,
            )
            up = ttnn.linear(
                x_chunk, self.up_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
            )
            ttnn.deallocate(x_chunk)
            hidden_out = ttnn.multiply(gate, up)
            ttnn.deallocate(gate)
            ttnn.deallocate(up)
            out_chunk = ttnn.linear(
                hidden_out, self.down_proj, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG,
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

    def forward_pre_down(self, x) -> ttnn.Tensor:
        """Fast path only: return hidden_out (pre-down-projection) for external fusion."""
        mlp_ckc = self.compute_kernel_config_hifi2
        if self.fused_gate_up is not None:
            gate_up = ttnn.linear(
                x,
                self.fused_gate_up,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
            )
            gate = ttnn.slice(gate_up, [0, 0, 0], [gate_up.shape[0], gate_up.shape[1], self.mlp_dim])
            up = ttnn.slice(gate_up, [0, 0, self.mlp_dim], [gate_up.shape[0], gate_up.shape[1], self.mlp_dim * 2])
            ttnn.deallocate(gate_up)
        else:
            gate = ttnn.linear(
                x,
                self.gate_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
            )
            up = ttnn.linear(
                x,
                self.up_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
            )
        gate_activated = ttnn.gelu(gate)
        ttnn.deallocate(gate)
        hidden_out = ttnn.multiply(gate_activated, up)
        ttnn.deallocate(gate_activated)
        ttnn.deallocate(up)
        return hidden_out

    def forward(self, x) -> ttnn.Tensor:
        """
        Forward pass with fast path for small sequences and chunked for large.

        Args:
            x: Input tensor [batch, seq, hidden] or [batch, 1, seq, hidden] (PyTorch or TTNN)

        Returns:
            TTNN output tensor [batch, seq, hidden] or [batch, 1, seq, hidden]
        """
        # Convert PyTorch to TTNN if needed
        was_torch = isinstance(x, torch.Tensor)
        if was_torch:
            x = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        batch_size = x.shape[0]
        was_3d = len(x.shape) == 3

        # Fast path: small sequences (e.g., expert with ~50 tokens) — no chunking
        seq_dim = x.shape[1] if was_3d else x.shape[2]
        if seq_dim <= self.chunk_size:
            mlp_ckc = self.compute_kernel_config_hifi2
            if self.fused_gate_up is not None:
                gate_up = ttnn.linear(
                    x,
                    self.fused_gate_up,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=mlp_ckc,
                )
                gate = ttnn.slice(gate_up, [0, 0, 0], [gate_up.shape[0], gate_up.shape[1], self.mlp_dim])
                up = ttnn.slice(gate_up, [0, 0, self.mlp_dim], [gate_up.shape[0], gate_up.shape[1], self.mlp_dim * 2])
                ttnn.deallocate(gate_up)
            else:
                gate = ttnn.linear(
                    x,
                    self.gate_proj,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=mlp_ckc,
                )
                up = ttnn.linear(
                    x,
                    self.up_proj,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=mlp_ckc,
                )
            gate_activated = ttnn.gelu(gate)
            ttnn.deallocate(gate)
            hidden_out = ttnn.multiply(gate_activated, up)
            ttnn.deallocate(gate_activated)
            ttnn.deallocate(up)
            output = ttnn.linear(
                hidden_out,
                self.down_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
            )
            ttnn.deallocate(hidden_out)
            if was_torch:
                output = ttnn.to_torch(output)
            return output

        # Chunked path for large sequences (VLM with 544 tokens)
        # Always work with 4D tensors (ttnn.slice requires 4D coordinates)
        if was_3d:
            x = ttnn.reshape(x, [batch_size, 1, x.shape[1], x.shape[2]])

        seq_len = x.shape[2]
        hidden = x.shape[3]

        # Calculate number of chunks (tile-aligned)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        output_chunks = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            actual_chunk_size = chunk_end - chunk_start

            # Pad last chunk to tile alignment (32) if needed — NOT to full chunk_size
            tile_aligned_size = ((actual_chunk_size + 31) // 32) * 32
            needs_chunk_padding = actual_chunk_size != tile_aligned_size
            padded_chunk_size = tile_aligned_size

            # Slice input chunk (always 4D)
            x_chunk = ttnn.slice(x, [0, 0, chunk_start, 0], [batch_size, 1, chunk_end, hidden])

            # Pad chunk if needed for tile alignment
            # Move to DRAM for multicore pad support (avoids L1 fallback warning)
            if needs_chunk_padding:
                pad_amount = padded_chunk_size - actual_chunk_size
                x_chunk = ttnn.to_memory_config(x_chunk, ttnn.DRAM_MEMORY_CONFIG)
                x_chunk = ttnn.pad(x_chunk, padding=((0, 0), (0, 0), (0, pad_amount), (0, 0)), value=0.0)

            # Fused gate+up projection — single matmul instead of 2 separate
            mlp_ckc = self.compute_kernel_config_hifi2
            if self.fused_gate_up is not None:
                gate_up = ttnn.linear(
                    x_chunk,
                    self.fused_gate_up,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    compute_kernel_config=mlp_ckc,
                )
                ttnn.deallocate(x_chunk)
                gate = ttnn.slice(
                    gate_up, [0, 0, 0, 0], [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], self.mlp_dim]
                )
                up = ttnn.slice(
                    gate_up,
                    [0, 0, 0, self.mlp_dim],
                    [gate_up.shape[0], gate_up.shape[1], gate_up.shape[2], self.mlp_dim * 2],
                )
                ttnn.deallocate(gate_up)
            else:
                gate = ttnn.linear(
                    x_chunk,
                    self.gate_proj,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
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

            # GELU activation
            gate_activated = ttnn.gelu(gate)
            ttnn.deallocate(gate)

            # Element-wise multiply
            hidden_out = ttnn.multiply(gate_activated, up)
            ttnn.deallocate(gate_activated)
            ttnn.deallocate(up)

            # Down projection
            output_chunk = ttnn.linear(
                hidden_out,
                self.down_proj,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=mlp_ckc,
            )
            ttnn.deallocate(hidden_out)

            # Slice back to actual size if padded
            if needs_chunk_padding:
                output_chunk = ttnn.slice(output_chunk, [0, 0, 0, 0], [batch_size, 1, actual_chunk_size, hidden])

            output_chunks.append(output_chunk)

        # Concatenate all chunks along sequence dimension (always 4D now)
        if len(output_chunks) == 1:
            output = output_chunks[0]
        else:
            output = output_chunks[0]
            for i in range(1, len(output_chunks)):
                output = ttnn.concat([output, output_chunks[i]], dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
                ttnn.deallocate(output_chunks[i])

        # Move final output to L1
        output = ttnn.to_memory_config(output, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Reshape back to 3D if input was 3D
        if was_3d:
            output = ttnn.reshape(output, [batch_size, seq_len, hidden])

        # Convert back to PyTorch if input was PyTorch
        if was_torch:
            output = ttnn.to_torch(output)

        return output


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
        fused_cfg: Optional[FusedConfig] = None,
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
            expected_seq_len: If set, pre-slice RoPE for this seq_len in attention
            fused_cfg: TT_FUSED knobs (None / disabled -> the legacy ``forward`` only)
            role: "vlm" or "expert" (fused path)
        """
        self.config = config
        self.layer_idx = layer_idx
        self.device = device
        self.use_adarms = config.use_adarms
        self.fused_cfg = fused_cfg
        self.role = role
        self._dit_config = None
        if fused_cfg is not None and fused_cfg.enabled:
            self._dit_config = dit_config_from_blocks(device, fused_cfg.dit_blocks)

        if self.use_adarms:
            # Pi0.5: adaRMS dense projection weights
            self.input_ln_dense_weight = weights["input_layernorm.dense.weight"]
            self.input_ln_dense_bias = weights["input_layernorm.dense.bias"]
            self.post_attn_ln_dense_weight = weights["post_attention_layernorm.dense.weight"]
            self.post_attn_ln_dense_bias = weights["post_attention_layernorm.dense.bias"]
            # Pre-allocate ones weight for adaRMS (avoids allocation per call)
            self._adarms_ones = ttnn.ones((1, config.width), device=device, dtype=ttnn.bfloat16)
            self._adarms_ones = ttnn.to_layout(self._adarms_ones, ttnn.TILE_LAYOUT)
        else:
            self.input_layernorm_weight = weights["input_layernorm.weight"]
            self.post_attention_layernorm_weight = weights["post_attention_layernorm.weight"]

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
    ) -> Optional[ttnn.Tensor]:
        """VLM block (plain RMSNorm, ungated residuals) that fills the KV cache of this layer.

        Same launches as the legacy ``forward`` minus the per-layer cos/sin slices, plus the two
        ``fill_cache`` writes; ``kv_only=True`` (last layer, PI05_SKIP_VLM_TAIL) computes only what the
        expert consumes. The caller owns / frees ``hidden_states``.
        """
        normed = rms_norm_ttnn(hidden_states, self.input_layernorm_weight, self.config.rms_norm_eps)
        attn_output = self.attention.forward_fused_vlm(normed, cache_k, cache_v, need_output=not kv_only)
        ttnn.deallocate(normed)
        if kv_only:
            return None
        hidden_mid = ttnn.add(hidden_states, attn_output)  # legacy gated_residual_ttnn(gate=None)
        ttnn.deallocate(attn_output)

        normed = rms_norm_ttnn(hidden_mid, self.post_attention_layernorm_weight, self.config.rms_norm_eps)
        mlp_output = self.mlp.forward_fused_vlm(normed)
        ttnn.deallocate(normed)
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
    ) -> ttnn.Tensor:
        """Expert block on the 64-row suffix with precomputed adaRMS modulations (owned by the model).

        ``PI05_FUSED_RESIDUAL``:
          bf16   -> ``hidden = dit(ctx bf16, Wo bf16, 1.0, hidden bf16, gate bf16)`` (ctx typecast bf8->bf16
                    first) and ``hidden = dit(gelu(g)*u bf16, Wdown bf16, 1.0, hidden bf16, gate bf16)``:
                    act == weight == residual format, bf16 gate broadcast; residual + gates in DRAM (the
                    legacy path's buffer types for the two addcmul inputs); output dtype bf16 explicit.
          mixed  -> as bf16 but the bf8 activations feed the fused op directly (no typecast).
          legacy -> linear(o) + mac, and typecast x3 + bf8 dit for the down-proj: the shipped numerics.
        Consumes (frees) ``hidden_states``; returns the new hidden ``[B, S, D]`` bf16.
        """
        scale_in, shift_in, attn_gate, scale_post, shift_post, mlp_gate = precomputed_mod
        eps = self.config.rms_norm_eps
        mode = self.fused_cfg.residual
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        # ---- attention + gated residual ----
        normed = adarms_norm_precomputed(hidden_states, scale_in, shift_in, eps)
        attn_concat = self.attention.forward_fused_expert(normed, cache_k, cache_v, prefix_len)
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
            if mode == "bf16":
                ctx = ttnn.typecast(attn_concat, ttnn.bfloat16)
                ttnn.deallocate(attn_concat)
                attn_concat = ctx
            new_hidden = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                attn_concat,
                self.attention.o_proj,
                1.0,
                hidden_states,
                attn_gate,
                config=self._dit_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
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
            new_hidden = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                hidden_out,
                self.mlp.down_proj,
                1.0,
                hidden_states,
                mlp_gate,
                config=self._dit_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            ttnn.deallocate(hidden_out)
            ttnn.deallocate(hidden_states)
        return new_hidden

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor, ttnn.Tensor]] = None,
        use_cache: bool = False,
        adarms_cond: Optional[ttnn.Tensor] = None,
        precomputed_mod: Optional[Tuple] = None,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Forward pass using TTNN operations.

        Args:
            hidden_states: TTNN input tensor
            cos, sin: Unused (kept for API compatibility, passed through to attention)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached KV
            use_cache: Whether to return cache
            adarms_cond: Pi0.5 time conditioning vector (TTNN tensor)
            precomputed_mod: Optional precomputed adaRMS modulations for this step/layer,
                a 6-tuple (scale_in, shift_in, gate_in, scale_post, shift_post, gate_post)
                computed at init. When provided, the per-layer dense projection + chunk is
                skipped (owned externally — do NOT deallocate its tensors here).

        Returns:
            Tuple of (output, optional_cache)
        """
        # Pre-attention norm
        if precomputed_mod is not None:
            scale_in, shift_in, attn_gate = precomputed_mod[0], precomputed_mod[1], precomputed_mod[2]
            normed = adarms_norm_precomputed(hidden_states, scale_in, shift_in, self.config.rms_norm_eps)
        elif self.use_adarms and adarms_cond is not None:
            normed, attn_gate = adarms_norm_ttnn(
                hidden_states,
                self.input_ln_dense_weight,
                self.input_ln_dense_bias,
                adarms_cond,
                self.config.rms_norm_eps,
                self.device,
                ones_weight=self._adarms_ones,
            )
        else:
            normed = rms_norm_ttnn(
                hidden_states,
                self.input_layernorm_weight,
                self.config.rms_norm_eps,
            )
            attn_gate = None

        # Attention with gated residual
        attn_output, new_cache = self.attention.forward(
            normed,
            cos,
            sin,
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )
        hidden_states = gated_residual_ttnn(hidden_states, attn_output, attn_gate)
        ttnn.deallocate(attn_output)
        # Only deallocate attn_gate if we own it (freshly produced by adarms_norm_ttnn).
        # When precomputed_mod is provided, the gate is a cached tensor owned by PI0ModelTTNN.
        if attn_gate is not None and precomputed_mod is None:
            ttnn.deallocate(attn_gate)

        # Pre-MLP norm
        if precomputed_mod is not None:
            scale_post, shift_post, mlp_gate = precomputed_mod[3], precomputed_mod[4], precomputed_mod[5]
            normed = adarms_norm_precomputed(hidden_states, scale_post, shift_post, self.config.rms_norm_eps)
        elif self.use_adarms and adarms_cond is not None:
            normed, mlp_gate = adarms_norm_ttnn(
                hidden_states,
                self.post_attn_ln_dense_weight,
                self.post_attn_ln_dense_bias,
                adarms_cond,
                self.config.rms_norm_eps,
                self.device,
                ones_weight=self._adarms_ones,
            )
        else:
            normed = rms_norm_ttnn(
                hidden_states,
                self.post_attention_layernorm_weight,
                self.config.rms_norm_eps,
            )
            mlp_gate = None

        # MLP with gated residual — fuse down_proj + gated residual when gate is present
        if mlp_gate is not None:
            # Fused: hidden_states = hidden_states + 1.0 * linear(hidden_out, down_proj) * gate
            # All tensors must match weight dtype (bfloat8_b) for the fused kernel
            hidden_out = self.mlp.forward_pre_down(normed)
            ttnn.deallocate(normed)
            hs_8b = ttnn.typecast(hidden_states, ttnn.bfloat8_b)
            gate_8b = ttnn.typecast(mlp_gate, ttnn.bfloat8_b)
            hidden_states = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                matmul_input_tensor=hidden_out,
                matmul_weight_tensor=self.mlp.down_proj,
                scalar=1.0,
                addcmul_input_tensor1=hs_8b,
                addcmul_input_tensor2=gate_8b,
            )
            ttnn.deallocate(hidden_out)
            ttnn.deallocate(hs_8b)
            ttnn.deallocate(gate_8b)
            if precomputed_mod is None:
                ttnn.deallocate(mlp_gate)
            # Cast back to bfloat16 for subsequent layers
            hidden_states = ttnn.typecast(hidden_states, ttnn.bfloat16)
        else:
            mlp_output = self.mlp.forward(normed)
            ttnn.deallocate(normed)
            hidden_states = ttnn.add(hidden_states, mlp_output)
            ttnn.deallocate(mlp_output)
        # ReadDeviceProfiler removed for performance

        return hidden_states, new_cache


# Default exports
GemmaAttention = GemmaAttentionTTNN
GemmaMLP = GemmaMLPTTNN
GemmaBlock = GemmaBlockTTNN
