# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PaliGemma backbone wrapper - TTNN Implementation

This module combines vision, language, and action expert components:
    - SigLIP Vision Tower: Processes images to embeddings
    - Gemma 2B Language Model: VLM backbone for prefix (images + language)
    - Gemma 300M Action Expert: Processes suffix (state + actions)

The dual-expert architecture shares attention layers:
    - VLM and Expert compute separate Q, K, V
    - K, V are concatenated for shared attention
    - Outputs are split and processed through separate MLPs

Optimizations:
    1. Fused QKV weights (single linear instead of 3)
    2. Native TTNN RoPE (ttnn.experimental.rotary_embedding)
    3. Pre-added RMSNorm weights (Gemma-style +1 offset)

Fused graph (``TT_FUSED=1``): the backbone OWNS the 18 expert KV caches
``[1, 1, prefix_len + action_horizon, head_dim]`` bf8 (L1); ``forward_vlm_fused`` writes the prefix
rows through every VLM layer on EVERY call (the multi-replan invariant: a new observation always
refreshes the prefix) and ``forward_expert_fused`` only writes the suffix rows -- the legacy
per-layer-per-step prefix refill (2 x fill_cache x 18 x 10) is gone. The last VLM layer stops after
its K/V write and the unused final VLM norm is skipped (``PI05_SKIP_VLM_TAIL``, exact: nothing in
``sample_actions`` reads the VLM hidden output). SigLIP runs batched over the camera images from a
host im2col input and the language embedding is emitted in TILE layout.
"""

from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.fused_host import kv_cache_plan
from .ttnn_common import tensor_1d_to_2d_ttnn
from .ttnn_gemma import (
    GemmaBlockTTNN,
    rms_norm_ttnn,
    adarms_norm_ttnn,
    adarms_norm_precomputed,
    precompute_freqs_cis_meta_format,
    ttnn_dtype_from_name,
)
from .ttnn_siglip import (
    SigLIPVisionTowerTTNN,
    MultiModalProjectorTTNN,
)


class PaliGemmaBackboneTTNN:
    """
    PaliGemma backbone using TTNN operations.
    """

    def __init__(
        self,
        config: PaliGemmaConfig,
        weights: Dict[str, Dict[str, torch.Tensor]],
        device: ttnn.Device,
        fused_cfg: Optional[FusedConfig] = None,
    ):
        """
        Initialize PaliGemma backbone with TTNN.

        Args:
            config: PaliGemma configuration
            weights: Categorized PyTorch weights
            device: TTNN device
            fused_cfg: TT_FUSED knobs (None / disabled -> legacy weights, dtypes and forwards)
        """
        self.config = config
        self.device = device
        self.fused_cfg = fused_cfg
        self._fused = fused_cfg is not None and fused_cfg.enabled
        # Fused residual op: residual (bf16 hidden) format must equal the weight format -> expert
        # o_proj / down_proj in bf16 (PI05_FUSED_RESIDUAL != legacy). Legacy: None -> bf8 as shipped.
        self._expert_residual_weight_dtype = (
            ttnn_dtype_from_name(fused_cfg.expert_residual_weight_dtype()) if self._fused else None
        )
        # Backbone-owned expert KV caches (fused graph), allocated by allocate_kv_caches()
        self.kv_caches: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None
        self.kv_cache_plan: Optional[Dict[str, int]] = None

        # Convert embedding to TTNN (use lm_head if embed_tokens not available - tied embeddings)
        embed_weight = weights["vlm_language"].get("model.embed_tokens.weight")
        if embed_weight is None:
            embed_weight = weights["vlm_language"].get("lm_head.weight")
        if embed_weight is not None:
            self.vlm_embed_tokens = ttnn.from_torch(
                embed_weight,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,  # Embeddings use row major
                device=device,
            )
        else:
            self.vlm_embed_tokens = None

        # Convert norms - OPTIMIZATION: Pre-add Gemma-style +1 offset
        # Note: +1.0 is done on host (torch), unsqueeze done on device via tensor_1d_to_2d_ttnn
        self.vlm_norm = tensor_1d_to_2d_ttnn(
            weights["vlm_language"]["model.norm.weight"] + 1.0, device, dtype=ttnn.bfloat16
        )

        self.use_expert_adarms = config.expert_config.use_adarms
        if self.use_expert_adarms:
            # Pi0.5: adaRMS for final expert norm - dense projection
            norm_dense_w = weights["action_expert"]["model.norm.dense.weight"]
            norm_dense_b = weights["action_expert"]["model.norm.dense.bias"].clone()
            # Pre-add +1 to scale portion of bias (first expert_width elements)
            expert_width = config.expert_config.width
            norm_dense_b[:expert_width] += 1.0
            self.expert_norm_dense_weight = ttnn.from_torch(
                norm_dense_w.T.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.expert_norm_dense_bias = tensor_1d_to_2d_ttnn(norm_dense_b, device, dtype=ttnn.bfloat16)
            # Pre-allocate ones weight for adaRMS (needed for trace compatibility)
            self._expert_norm_ones = ttnn.ones((1, config.expert_config.width), device=device, dtype=ttnn.bfloat16)
            self._expert_norm_ones = ttnn.to_layout(self._expert_norm_ones, ttnn.TILE_LAYOUT)
            self.expert_norm = None
        else:
            self.expert_norm = tensor_1d_to_2d_ttnn(
                weights["action_expert"]["model.norm.weight"] + 1.0, device, dtype=ttnn.bfloat16
            )

        # Initialize vision tower
        self.vision_tower = SigLIPVisionTowerTTNN(
            config.siglip_config,
            weights["vlm_vision"],
            device,
            fused_cfg=fused_cfg,
        )

        # Initialize projector
        self.mm_projector = MultiModalProjectorTTNN(weights["vlm_projector"], device)

        # Store torch weights for blocks (converted on demand)
        self.torch_weights = weights

        # Precompute RoPE using pure TTNN for native ttnn.experimental.rotary_embedding
        # This format is required by ttnn.experimental.rotary_embedding (split-half pattern)
        self.cos_meta, self.sin_meta = precompute_freqs_cis_meta_format(
            config.vlm_config.head_dim,
            config.max_seq_len,
            device,
        )

        # Expert RoPE in Meta format (pure TTNN, no torch)
        self.expert_cos_meta, self.expert_sin_meta = precompute_freqs_cis_meta_format(
            config.expert_config.head_dim,
            config.max_seq_len,
            device,
        )

        # Initialize VLM transformer blocks (18 layers for Gemma 2B)
        # Pre-slice RoPE for known prefix_len: 2 images × 256 patches + 32 lang = 544 tokens
        vlm_seq_len = 2 * config.siglip_config.num_patches + 32  # 544
        self.vlm_blocks = []
        for i in range(config.vlm_config.depth):
            block_weights = self._get_vlm_block_weights_ttnn(weights["vlm_language"], i)
            self.vlm_blocks.append(
                GemmaBlockTTNN(
                    config.vlm_config,
                    block_weights,
                    i,
                    device,
                    self.cos_meta,
                    self.sin_meta,
                    expected_seq_len=vlm_seq_len,
                    fused_cfg=fused_cfg,
                    role="vlm",
                )
            )

        # Initialize Expert transformer blocks (18 layers for Gemma 300M)
        # Expert always processes suffix_len = action_horizon (50 for Pi0.5)
        # Pre-slice RoPE for this known length to save 2 slice ops per layer per step
        expert_seq_len = 50  # action_horizon for Pi0.5
        self.expert_blocks = []
        for i in range(config.expert_config.depth):
            block_weights = self._get_expert_block_weights_ttnn(weights["action_expert"], i)
            self.expert_blocks.append(
                GemmaBlockTTNN(
                    config.expert_config,
                    block_weights,
                    i,
                    device,
                    self.expert_cos_meta,
                    self.expert_sin_meta,
                    expected_seq_len=expert_seq_len,
                    fused_cfg=fused_cfg,
                    role="expert",
                )
            )

    def _get_vlm_block_weights_ttnn(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, ttnn.Tensor]:
        """Extract VLM block weights and convert to TTNN with fused QKV optimization."""
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}

        # OPTIMIZATION: Create fused QKV weight for single linear call
        q_key = f"{prefix}self_attn.q_proj.weight"
        k_key = f"{prefix}self_attn.k_proj.weight"
        v_key = f"{prefix}self_attn.v_proj.weight"

        if q_key in weights and k_key in weights and v_key in weights:
            # Get Q, K, V weights, transpose for TTNN linear, and convert to TTNN
            # Use bfloat8_b for VLM weights too — reduces bandwidth
            vlm_weight_dtype = ttnn.bfloat8_b
            wq_ttnn = ttnn.from_torch(
                weights[q_key].T.contiguous(),
                dtype=vlm_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            wk_ttnn = ttnn.from_torch(
                weights[k_key].T.contiguous(),
                dtype=vlm_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            wv_ttnn = ttnn.from_torch(
                weights[v_key].T.contiguous(),
                dtype=vlm_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

            # Concatenate using TTNN: [hidden, Q_dim + K_dim + V_dim]
            block_weights["self_attn.wqkv"] = ttnn.concat(
                [wq_ttnn, wk_ttnn, wv_ttnn],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(wq_ttnn)
            ttnn.deallocate(wk_ttnn)
            ttnn.deallocate(wv_ttnn)

        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]

                # Skip individual Q, K, V weights (now fused)
                if new_key in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]:
                    continue

                # Transpose weight matrices for TTNN linear
                if "weight" in new_key and "layernorm" not in new_key and "norm" not in new_key:
                    value = value.T
                    layout = ttnn.TILE_LAYOUT
                elif "layernorm" in new_key or "norm" in new_key:
                    # OPTIMIZATION: Pre-add Gemma-style +1 offset to norm weights
                    value = value + 1.0
                    layout = ttnn.TILE_LAYOUT
                else:
                    layout = ttnn.TILE_LAYOUT

                # Handle 1D tensors (biases, norms) using tensor_1d_to_2d_ttnn (no torch.unsqueeze)
                # Use bfloat8_b for weight matrices, bfloat16 for norms/biases
                if len(value.shape) == 1:
                    block_weights[new_key] = tensor_1d_to_2d_ttnn(value, self.device, dtype=ttnn.bfloat16)
                else:
                    w_dtype = (
                        vlm_weight_dtype
                        if ("weight" in new_key and "norm" not in new_key and "layernorm" not in new_key)
                        else ttnn.bfloat16
                    )
                    block_weights[new_key] = ttnn.from_torch(
                        value,
                        dtype=w_dtype,
                        layout=layout,
                        device=self.device,
                    )
        return block_weights

    def _get_expert_block_weights_ttnn(
        self,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ) -> Dict[str, ttnn.Tensor]:
        """Extract expert block weights and convert to TTNN with fused QKV optimization."""
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}

        # Use bfloat8_b for expert weights to reduce memory bandwidth
        expert_weight_dtype = ttnn.bfloat8_b

        # OPTIMIZATION: Create fused QKV weight for single linear call
        q_key = f"{prefix}self_attn.q_proj.weight"
        k_key = f"{prefix}self_attn.k_proj.weight"
        v_key = f"{prefix}self_attn.v_proj.weight"

        if q_key in weights and k_key in weights and v_key in weights:
            # Get Q, K, V weights, transpose for TTNN linear, and convert to TTNN
            wq_ttnn = ttnn.from_torch(
                weights[q_key].T.contiguous(),
                dtype=expert_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            wk_ttnn = ttnn.from_torch(
                weights[k_key].T.contiguous(),
                dtype=expert_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            wv_ttnn = ttnn.from_torch(
                weights[v_key].T.contiguous(),
                dtype=expert_weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

            # Concatenate using TTNN: [hidden, Q_dim + K_dim + V_dim]
            block_weights["self_attn.wqkv"] = ttnn.concat(
                [wq_ttnn, wk_ttnn, wv_ttnn],
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(wq_ttnn)
            ttnn.deallocate(wk_ttnn)
            ttnn.deallocate(wv_ttnn)

        for key, value in weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]

                # Skip individual Q, K, V weights (now fused)
                if new_key in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"]:
                    continue

                is_adarms_dense = "layernorm.dense" in new_key
                is_norm_scalar = ("layernorm" in new_key or "norm" in new_key) and not is_adarms_dense

                if is_adarms_dense:
                    # Pi0.5: adaRMS dense projection weights
                    if "weight" in new_key:
                        # Transpose for TTNN linear: [out, in] -> [in, out]
                        block_weights[new_key] = ttnn.from_torch(
                            value.T.contiguous(),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=self.device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                    else:
                        # Bias: 1D -> 2D
                        # Pre-add +1 to scale portion of bias (first hidden_dim elements)
                        # This avoids a runtime add(scale, 1.0) op in adarms_norm_ttnn
                        hidden_dim = self.config.expert_config.width
                        value = value.clone()
                        value[:hidden_dim] += 1.0
                        block_weights[new_key] = tensor_1d_to_2d_ttnn(value, self.device, dtype=ttnn.bfloat16)
                elif "weight" in new_key and not is_norm_scalar:
                    # Regular weight matrices: transpose for TTNN linear
                    # Use bfloat8_b for expert weights (output projection, etc.)
                    w_dtype = expert_weight_dtype
                    if self._expert_residual_weight_dtype is not None and new_key in (
                        "self_attn.o_proj.weight",
                        "mlp.down_proj.weight",
                    ):
                        # Fused gated residual: weight format == residual (bf16 hidden) format
                        w_dtype = self._expert_residual_weight_dtype
                    block_weights[new_key] = ttnn.from_torch(
                        value.T.contiguous(),
                        dtype=w_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.device,
                    )
                elif is_norm_scalar:
                    # Standard RMSNorm: Pre-add Gemma-style +1 offset
                    if len(value.shape) == 1:
                        block_weights[new_key] = tensor_1d_to_2d_ttnn(value + 1.0, self.device, dtype=ttnn.bfloat16)
                    else:
                        block_weights[new_key] = ttnn.from_torch(
                            value + 1.0,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=self.device,
                        )
                else:
                    # Other tensors
                    if len(value.shape) == 1:
                        block_weights[new_key] = tensor_1d_to_2d_ttnn(value, self.device, dtype=ttnn.bfloat16)
                    else:
                        block_weights[new_key] = ttnn.from_torch(
                            value,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=self.device,
                        )
        return block_weights

    def embed_image(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        """
        Embed images through vision tower and projector (TTNN).

        Args:
            pixel_values: PyTorch tensor (batch_size, channels, height, width)

        Returns:
            TTNN tensor (batch_size, num_patches, vlm_width)
        """
        vision_features = self.vision_tower.forward(pixel_values)
        return self.mm_projector.forward(vision_features)

    def embed_language_tokens(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        Embed language tokens using TTNN.

        Args:
            token_ids: TTNN tensor of token IDs

        Returns:
            TTNN tensor of embeddings
        """
        return ttnn.embedding(token_ids, self.vlm_embed_tokens)

    # ------------------------------------------------------------------ fused graph

    def embed_language_tokens_fused(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Token gather emitted in TILE layout (fused tilize inside the embedding op: token_len % 32 == 0
        and 2048 % 32 == 0) so the following scalar multiply and concat run on tiles directly, instead of
        the legacy ROW_MAJOR output that is tilized/untilized inside ``mul`` and re-tilized by ``concat``.
        Same bf16 values as the legacy row-major gather (exact)."""
        return ttnn.embedding(token_ids, self.vlm_embed_tokens, layout=ttnn.TILE_LAYOUT)

    def embed_images_fused(self, im2col_dev: ttnn.Tensor) -> ttnn.Tensor:
        """Host im2col ``[B, 256, 608]`` bf16 ROW_MAJOR (persistent trace input) -> ``[1, B*256, 2048]``
        image tokens in the legacy concat order (camera 0 rows, then camera 1 rows; tile-aligned reshape)."""
        vision_features = self.vision_tower.forward_fused(im2col_dev)
        proj = self.mm_projector.forward(vision_features)
        ttnn.deallocate(vision_features)
        b, p, d = proj.shape[0], proj.shape[1], proj.shape[2]
        if b == 1:
            return proj
        return ttnn.reshape(proj, (1, b * p, d))

    def allocate_kv_caches(self, prefix_len: int, action_horizon: int) -> Dict[str, int]:
        """(Re)allocate the 18 backbone-owned expert KV caches for this serving shape. Called on the
        compile pass (before trace capture); a no-op while the shape is unchanged. Geometry and the
        rotary_embedding_to_cache / fill_cache tile constraints come from ``fused_host.kv_cache_plan``."""
        plan = kv_cache_plan(prefix_len, action_horizon)
        if self.kv_caches is not None and self.kv_cache_plan == plan:
            return plan
        if self.kv_caches is not None:
            for k, v in self.kv_caches:
                ttnn.deallocate(k)
                ttnn.deallocate(v)
        cfg = self.config.expert_config
        self.kv_caches = []
        for _ in self.expert_blocks:
            shape = [1, cfg.num_kv_heads, plan["logical_len"], cfg.head_dim]
            k = ttnn.zeros(
                shape,
                dtype=ttnn.bfloat8_b,  # == the qkv linear output dtype of both VLM and expert
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            v = ttnn.zeros(
                shape,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.kv_caches.append((k, v))
        self.kv_cache_plan = plan
        return plan

    def forward_vlm_fused(self, prefix_embs: ttnn.Tensor) -> None:
        """VLM prefill whose only product is the prefix K/V written into ``self.kv_caches`` (rows 0..S-1
        of every layer). Consumes ``prefix_embs``. With ``skip_vlm_tail`` the last layer stops after
        its K/V write and the final norm is skipped (their outputs are never read by sample_actions)."""
        if self.kv_caches is None:
            raise RuntimeError("allocate_kv_caches() must run before forward_vlm_fused()")
        skip_tail = self.fused_cfg.skip_vlm_tail
        hidden_states = prefix_embs
        last = len(self.vlm_blocks) - 1
        for i, block in enumerate(self.vlm_blocks):
            cache_k, cache_v = self.kv_caches[i]
            if skip_tail and i == last:
                block.forward_fused_vlm(hidden_states, cache_k, cache_v, kv_only=True)
                ttnn.deallocate(hidden_states)
                return None
            new_hidden = block.forward_fused_vlm(hidden_states, cache_k, cache_v)
            ttnn.deallocate(hidden_states)
            hidden_states = new_hidden
        # PI05_SKIP_VLM_TAIL=0: compute the legacy final norm too (A/B parity), result unused
        final = rms_norm_ttnn(hidden_states, self.vlm_norm, self.config.vlm_config.rms_norm_eps)
        ttnn.deallocate(hidden_states)
        ttnn.deallocate(final)
        return None

    def forward_expert_fused(
        self,
        hidden_states: ttnn.Tensor,
        precomputed_block_mods: List[Tuple],
        precomputed_final_mod: Tuple,
    ) -> ttnn.Tensor:
        """Expert on the 64-row suffix reading the backbone-owned caches; consumes ``hidden_states`` and
        returns the final adaRMS-normed hidden ``[1, 64, width]`` bf16 (L1)."""
        prefix_len = self.kv_cache_plan["prefix_len"]
        for i, block in enumerate(self.expert_blocks):
            cache_k, cache_v = self.kv_caches[i]
            hidden_states = block.forward_fused_expert(
                hidden_states, precomputed_block_mods[i], cache_k, cache_v, prefix_len
            )
        scale_f, shift_f = precomputed_final_mod[0], precomputed_final_mod[1]
        out = adarms_norm_precomputed(hidden_states, scale_f, shift_f, self.config.expert_config.rms_norm_eps)
        ttnn.deallocate(hidden_states)
        return out

    def forward_vlm(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through VLM backbone using TTNN.

        Args:
            hidden_states: Prefix embeddings (TTNN tensor)
            attention_mask: Attention mask (TTNN tensor)
            position_ids: Position indices (TTNN tensor)
            past_key_values: Cached KV from previous forward
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, optional_new_cache)
        """
        new_cache = [] if use_cache else None

        for i, block in enumerate(self.vlm_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                None,  # cos - unused, native TTNN RoPE uses cos_meta stored in block
                None,  # sin - unused, native TTNN RoPE uses sin_meta stored in block
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                new_cache.append(new_kv)

        # Final norm using TTNN
        hidden_states = rms_norm_ttnn(
            hidden_states,
            self.vlm_norm,
            self.config.vlm_config.rms_norm_eps,
        )

        return hidden_states, new_cache

    def forward_expert(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]] = None,
        use_cache: bool = False,
        adarms_cond: Optional[ttnn.Tensor] = None,
        precomputed_block_mods: Optional[List] = None,
        precomputed_final_mod: Optional[Tuple] = None,
    ) -> Tuple[ttnn.Tensor, Optional[List[Tuple[ttnn.Tensor, ttnn.Tensor]]]]:
        """
        Forward pass through action expert using TTNN.

        Args:
            hidden_states: Suffix embeddings (TTNN tensor)
            attention_mask: Attention mask (TTNN tensor)
            position_ids: Position indices (TTNN tensor)
            past_key_values: Cached KV from VLM prefix (for cross-attention)
            use_cache: Whether to return updated cache
            adarms_cond: Pi0.5 time conditioning vector (TTNN tensor)

        Returns:
            Tuple of (output, optional_new_cache)
        """
        new_cache = [] if use_cache else None

        # Pre-reshape adaRMS conditioning to 3D once (avoids 37 reshapes inside adarms_norm_ttnn)
        if adarms_cond is not None and len(adarms_cond.shape) == 2:
            adarms_cond = ttnn.reshape(adarms_cond, (adarms_cond.shape[0], 1, -1))

        for i, block in enumerate(self.expert_blocks):
            past_kv = past_key_values[i] if past_key_values else None
            block_mod = precomputed_block_mods[i] if precomputed_block_mods is not None else None
            hidden_states, new_kv = block.forward(
                hidden_states,
                None,  # cos - unused, native TTNN RoPE uses cos_meta stored in block
                None,  # sin - unused, native TTNN RoPE uses sin_meta stored in block
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
                adarms_cond=adarms_cond,
                precomputed_mod=block_mod,
            )
            if use_cache:
                new_cache.append(new_kv)

        # Final norm using TTNN
        if precomputed_final_mod is not None:
            scale_f, shift_f = precomputed_final_mod[0], precomputed_final_mod[1]
            hidden_states = adarms_norm_precomputed(
                hidden_states,
                scale_f,
                shift_f,
                self.config.expert_config.rms_norm_eps,
            )
        elif self.use_expert_adarms and adarms_cond is not None:
            hidden_states, _ = adarms_norm_ttnn(
                hidden_states,
                self.expert_norm_dense_weight,
                self.expert_norm_dense_bias,
                adarms_cond,
                self.config.expert_config.rms_norm_eps,
                self.device,
                ones_weight=self._expert_norm_ones,
            )
        else:
            hidden_states = rms_norm_ttnn(
                hidden_states,
                self.expert_norm,
                self.config.expert_config.rms_norm_eps,
            )

        return hidden_states, new_cache

    def forward_shared_attention(
        self,
        prefix_embs: ttnn.Tensor,
        suffix_embs: ttnn.Tensor,
        prefix_mask: Optional[ttnn.Tensor] = None,
        suffix_mask: Optional[ttnn.Tensor] = None,
        prefix_position_ids: Optional[ttnn.Tensor] = None,
        suffix_position_ids: Optional[ttnn.Tensor] = None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Forward pass with shared attention between VLM and Expert (TTNN).

        Args:
            prefix_embs: VLM prefix embeddings (TTNN tensor)
            suffix_embs: Expert suffix embeddings (TTNN tensor)
            prefix_mask: Prefix attention mask
            suffix_mask: Suffix attention mask
            prefix_position_ids: Prefix positions
            suffix_position_ids: Suffix positions

        Returns:
            Tuple of (vlm_output, expert_output)
        """
        # Process prefix through VLM
        vlm_output, vlm_cache = self.forward_vlm(
            prefix_embs,
            prefix_mask,
            prefix_position_ids,
            use_cache=True,
        )

        # Process suffix through expert
        expert_output, _ = self.forward_expert(
            suffix_embs,
            suffix_mask,
            suffix_position_ids,
            past_key_values=None,
            use_cache=False,
        )

        return vlm_output, expert_output


# Default export
PaliGemmaBackbone = PaliGemmaBackboneTTNN
