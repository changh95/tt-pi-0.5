# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Suffix Embedding module - TTNN Implementation

This module embeds the noisy actions for the expert transformer, computes the pi0.5 time
conditioning, and performs the Euler step.

Components:
    - action_in_proj: Projects actions from action_dim to expert width
    - action_out_proj: Projects expert output back to action_dim (inside the fused Euler step)
    - time_mlp_in / time_mlp_out: pi0.5 adaRMS conditioning from the sinusoidal time embedding
      (``embed_timestep`` + ``compute_adarms_cond_from_time``, precomputed per step at model build)

Fused graph: the suffix is carried at 64 rows (action_horizon 50 zero-padded to the tile boundary;
the host keeps rows [:50]), ``embed_actions_fused`` emits the hidden stream in DRAM (the buffer type
of the fused residual op's two addcmul inputs), and ``euler_step_fused`` folds
``velocity = h @ W_out + b; x = x_t + dt * velocity`` into ONE ``dit_minimal_matmul_addcmul_fused``
(``x_t + dt * (h @ W_out + b) * ones``; dt is the op's fp32 scalar attribute); needs W_out in bf16
(x_t is bf16: residual format == weight format).
"""

from typing import Dict, Optional

import torch
import ttnn

from models.experimental.pi0_5.common.configs import SuffixConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from .ttnn_common import (
    create_sinusoidal_pos_embedding_ttnn,
    precompute_sinusoidal_scaling_factor,
    tensor_1d_to_2d_ttnn,
)


class SuffixEmbeddingTTNN:
    """
    TTNN implementation of suffix embedding.

    Uses TTNN operations for efficient execution on Tenstorrent hardware.
    """

    def __init__(
        self,
        config: SuffixConfig,
        weights: Dict[str, ttnn.Tensor],
        device: ttnn.Device,
        fused_cfg: FusedConfig,
    ):
        """
        Initialize suffix embedding with TTNN weights.

        Args:
            config: Suffix configuration
            weights: Dictionary with TTNN weight tensors
            device: TTNN device
            fused_cfg: fused-graph knobs
        """
        self.config = config
        self.device = device
        self.weights = weights
        self.fused_cfg = fused_cfg
        from .ttnn_gemma import dit_config_from_blocks

        # ternary_b of the Euler fused op: ones[1, action_dim] in the SAME buffer type (L1) as x_t
        self._ones_action = ttnn.from_torch(
            torch.ones(1, config.action_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        self._dit_config = dit_config_from_blocks(device, fused_cfg.euler_dit_blocks)

        self.indices = ttnn.arange(0, 512, 1, device=device, dtype=ttnn.float32)
        self.indices = ttnn.to_layout(self.indices, ttnn.TILE_LAYOUT)  # Pre-convert for trace compatibility

        # Pre-compute sinusoidal scaling factor (constant across timesteps, saves ~8 ops per call)
        self._sin_scaling_factor = precompute_sinusoidal_scaling_factor(
            config.expert_width, min_period=4e-3, max_period=4.0, indices=self.indices
        )

    def compute_adarms_cond_from_time(self, time_emb: ttnn.Tensor) -> ttnn.Tensor:
        """
        Compute Pi0.5 adaRMS conditioning from a time embedding only (no action_emb dependency).
        Used to pre-compute per-step adarms_cond at model init, since it is constant per step index.
        """
        time_2d = ttnn.reshape(time_emb, (time_emb.shape[0], 1, -1)) if len(time_emb.shape) == 2 else time_emb
        adarms_cond = ttnn.linear(
            time_2d,
            self.weights["time_mlp_in.weight"],
            bias=self.weights["time_mlp_in.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        adarms_cond = ttnn.silu(adarms_cond)
        adarms_cond = ttnn.linear(
            adarms_cond,
            self.weights["time_mlp_out.weight"],
            bias=self.weights["time_mlp_out.bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        adarms_cond = ttnn.silu(adarms_cond)
        return adarms_cond

    def embed_timestep(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """
        Create timestep embedding.

        Args:
            timestep: TTNN tensor (batch_size,)

        Returns:
            TTNN tensor (batch_size, expert_width)
        """
        return create_sinusoidal_pos_embedding_ttnn(
            timestep,
            self.config.expert_width,
            min_period=4e-3,
            max_period=4.0,
            device=self.device,
            indices=self.indices,
            precomputed_scaling_factor=self._sin_scaling_factor,
        )

    # ------------------------------------------------------------------ fused graph

    def embed_actions_fused(self, noisy_actions: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 64, action_dim]`` (tile-padded noise) -> ``[1, 64, expert_width]`` bf16 in DRAM: the
        action_in_proj linear; DRAM so the first expert layer's fused residual sees its residual
        and gate in the same buffer type."""
        return ttnn.linear(
            noisy_actions,
            self.weights["action_in_proj.weight"],
            bias=self.weights["action_in_proj.bias"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def euler_step_fused(self, expert_output: ttnn.Tensor, x_t: ttnn.Tensor, dt: float) -> ttnn.Tensor:
        """``x_t + dt * (h @ W_out + b_out) * ones`` in one launch (replaces linear + mul + add).

        h ``[1, 64, width]`` bf16 (L1), W_out bf16 ``[width, action_dim]``, x_t ``[1, 64, action_dim]``
        bf16 (L1), ones ``[1, action_dim]`` bf16 (L1): act == weight == residual format, both addcmul
        inputs in L1, ``dt`` the fp32 scalar attribute. Output bf16 in L1 (the next step's x_t).
        """
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            expert_output,
            self.weights["action_out_proj.weight"],
            float(dt),
            x_t,
            self._ones_action,
            bias_tensor=self.weights["action_out_proj.bias"],
            config=self._dit_config,  # PI05_EULER_DIT_BLOCKS
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )


def convert_suffix_weights_to_ttnn(
    torch_weights: Dict[str, torch.Tensor],
    device: ttnn.Device,
    dtype: Optional[ttnn.DataType] = None,
    weight_dtype_overrides: Optional[Dict[str, ttnn.DataType]] = None,
) -> Dict[str, ttnn.Tensor]:
    """
    Convert PyTorch suffix weights to TTNN format.

    Args:
        torch_weights: Dictionary of PyTorch weight tensors
        device: TTNN device
        dtype: TTNN data type (default: bfloat8_b for weights, bfloat16 for bias)
        weight_dtype_overrides: per-key weight dtype (``PI0ModelTTNN`` passes ``action_out_proj.weight`` in
            bf16 because the Euler fused op needs weight format == x_t format); None = the defaults above

    Returns:
        Dictionary of TTNN weight tensors
    """
    if dtype is None:
        weight_dtype = ttnn.bfloat8_b
        bias_dtype = ttnn.bfloat16
    else:
        weight_dtype = dtype
        bias_dtype = dtype
    overrides = weight_dtype_overrides or {}

    ttnn_weights = {}

    for key, value in torch_weights.items():
        if "bias" in key:
            # Bias: expand to [1, out_features] using TTNN (no torch.unsqueeze)
            ttnn_weights[key] = tensor_1d_to_2d_ttnn(value, device, dtype=bias_dtype)
        else:
            # Weight: transpose for TTNN [in, out] format
            ttnn_weights[key] = ttnn.from_torch(
                value.T.contiguous(),
                dtype=overrides.get(key, weight_dtype),
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    return ttnn_weights


# Default export
SuffixEmbedding = SuffixEmbeddingTTNN
