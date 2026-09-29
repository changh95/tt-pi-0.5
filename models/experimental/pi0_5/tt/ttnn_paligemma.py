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

Fused graph (the only inference path): the backbone OWNS the 18 expert KV caches
``[1, 1, prefix_len + action_horizon, head_dim]`` bf8 (L1); ``forward_vlm_fused`` writes the prefix
rows through every VLM layer on EVERY call (the multi-replan invariant: a new observation always
refreshes the prefix) and ``forward_expert_fused`` only writes the suffix rows (no per-layer-per-step
prefix refill). The last VLM layer stops after its K/V write and the unused final VLM norm is skipped
(``PI05_SKIP_VLM_TAIL``, exact: nothing in ``sample_actions_fused`` reads the VLM hidden output).
SigLIP runs batched over the camera images from a host im2col input and the language embedding is
emitted in TILE layout.
"""

from typing import Dict, List, Optional, Tuple

import os
import torch
import ttnn
from dataclasses import replace as _dc_replace
from .ttnn_ccl import shard_cols as _shard_cols, shard_rows as _shard_rows, per_chip_cols as _per_chip_cols


from models.experimental.pi0_5.common.configs import PaliGemmaConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.fused_host import kv_cache_plan
from .ttnn_common import tensor_1d_to_2d_ttnn
from .ttnn_gemma import (
    GemmaBlockTTNN,
    rms_norm_ttnn,
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
        fused_cfg: FusedConfig,
        action_horizon: int = 50,
    ):
        """
        Initialize PaliGemma backbone with TTNN.

        Args:
            config: PaliGemma configuration
            weights: Categorized PyTorch weights
            device: TTNN device
            fused_cfg: fused-graph knobs (resolved, see ``PI0ModelTTNN``)
            action_horizon: action tokens per request (the expert's suffix; tile-padded on the device)
        """
        self.config = config
        self.device = device
        self.fused_cfg = fused_cfg
        # Tensor-parallel degree of the SigLIP tower / VLM prefill (1 = full weights on every chip)
        self.tp = fused_cfg.tp
        # Fused residual op: residual (bf16 hidden) format must equal the weight format -> expert
        # o_proj / down_proj in bf16 (PI05_FUSED_RESIDUAL != legacy); bf8 with PI05_FUSED_RESIDUAL=legacy.
        self._expert_residual_weight_dtype = ttnn_dtype_from_name(fused_cfg.expert_residual_weight_dtype())
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
        self.vlm_blocks = []
        # TP: each chip runs num_heads / tp query heads (the single MQA K/V head is replicated), so the
        # block sees a config with the LOCAL head count; width / head_dim / kv heads are unchanged.
        vlm_block_config = config.vlm_config
        if self.tp > 1:
            assert config.vlm_config.num_heads % self.tp == 0, (config.vlm_config.num_heads, self.tp)
            assert config.vlm_config.num_kv_heads == 1, "TP sharding assumes the MQA VLM (1 KV head)"
            vlm_block_config = _dc_replace(config.vlm_config, num_heads=config.vlm_config.num_heads // self.tp)
        for i in range(config.vlm_config.depth):
            block_weights = self._get_vlm_block_weights_ttnn(weights["vlm_language"], i)
            self.vlm_blocks.append(
                GemmaBlockTTNN(
                    vlm_block_config,
                    block_weights,
                    i,
                    device,
                    self.cos_meta,
                    self.sin_meta,
                    fused_cfg=fused_cfg,
                    role="vlm",
                )
            )

        # Initialize Expert transformer blocks (18 layers for Gemma 300M)
        # Expert always processes suffix_len = action_horizon (50 for Pi0.5): the fused expert attention
        # (PI05_EXPERT_ATTN=fused) is built for this length
        expert_seq_len = int(action_horizon)
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
        """Extract VLM block weights and convert to TTNN with fused QKV optimization.

        TP (``self.tp`` > 1): chip i holds ``[wq heads i*H/tp..(i+1)*H/tp-1 | wk | wv]`` (K/V of the single
        MQA head replicated, so every chip fills the full KV cache), the row block of o_proj matching its
        heads, the column block of gate / up and the row block of down; the block all-reduces the two
        row-parallel partials. Norm weights are replicated.
        """
        prefix = f"model.layers.{layer_idx}."
        block_weights = {}
        tp = self.tp

        # OPTIMIZATION: Create fused QKV weight for single linear call
        q_key = f"{prefix}self_attn.q_proj.weight"
        k_key = f"{prefix}self_attn.k_proj.weight"
        v_key = f"{prefix}self_attn.v_proj.weight"

        if tp > 1 and q_key in weights and k_key in weights and v_key in weights:
            vlm_weight_dtype = ttnn.bfloat8_b
            cfg = self.config.vlm_config
            hl = cfg.num_heads // tp * cfg.head_dim  # local q columns per chip
            wq_t, wk_t, wv_t = weights[q_key].T, weights[k_key].T, weights[v_key].T  # [D, H*dh], [D, dh], [D, dh]
            chunks = [torch.cat([wq_t[:, i * hl : (i + 1) * hl], wk_t, wv_t], dim=-1) for i in range(tp)]
            block_weights["self_attn.wqkv"] = _per_chip_cols(self.device, chunks, vlm_weight_dtype)
        elif q_key in weights and k_key in weights and v_key in weights:
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
                    is_proj = "weight" in new_key and "norm" not in new_key and "layernorm" not in new_key
                    w_dtype = vlm_weight_dtype if is_proj else ttnn.bfloat16
                    if tp > 1 and new_key in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                        block_weights[new_key] = _shard_cols(self.device, value, tp, w_dtype)  # [D, mlp/tp]
                    elif tp > 1 and new_key in ("mlp.down_proj.weight", "self_attn.o_proj.weight"):
                        block_weights[new_key] = _shard_rows(self.device, value, tp, w_dtype)  # [mlp/tp | H*dh/tp, D]
                    else:
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

        # PI05_EXPERT_GEGLU: one [width, 2*mlp] matmul + ttnn.geglu (= first half * gelu(second half)),
        # so the fused weight is [up | gate] along the output dim.
        if self.fused_cfg.expert_geglu:
            g_key, u_key = f"{prefix}mlp.gate_proj.weight", f"{prefix}mlp.up_proj.weight"
            if g_key in weights and u_key in weights:
                fused_ug = torch.cat([weights[u_key], weights[g_key]], dim=0).T.contiguous()  # [width, 2*mlp]
                block_weights["mlp.fused_gate_up"] = ttnn.from_torch(
                    fused_ug,
                    dtype=expert_weight_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

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
            if getattr(self.fused_cfg, "expert_norm_fold", False):
                # torch originals for the per-step adaRMS folding (tt/ttnn_fused_norm.py); freed after folding
                if os.environ.get("PI05_EXPERT_NORM_FOLD_ATTN", "0") == "1":
                    block_weights["_torch_wqkv"] = torch.cat(
                        [weights[q_key].T, weights[k_key].T, weights[v_key].T], dim=-1
                    ).contiguous()
                gk, uk = f"{prefix}mlp.gate_proj.weight", f"{prefix}mlp.up_proj.weight"
                if gk in weights and uk in weights:
                    block_weights["_torch_fused_ug"] = torch.cat([weights[uk], weights[gk]], dim=0).T.contiguous()

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
                    if new_key in (
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

    # ------------------------------------------------------------------ fused graph

    def embed_language_tokens_fused(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """Token gather emitted in TILE layout (fused tilize inside the embedding op: token_len % 32 == 0
        and 2048 % 32 == 0) so the following scalar multiply and concat run on tiles directly, instead of
        a ROW_MAJOR output that is tilized/untilized inside ``mul`` and re-tilized by ``concat``.
        Same bf16 values as a row-major gather (exact)."""
        return ttnn.embedding(token_ids, self.vlm_embed_tokens, layout=ttnn.TILE_LAYOUT)

    def embed_images_fused(self, im2col_dev: ttnn.Tensor, batch: int = 1) -> ttnn.Tensor:
        """Host im2col ``[B*N, 256, 608]`` bf16 ROW_MAJOR (persistent trace input; request-major camera
        order) -> ``[B, N*256, 2048]`` image tokens per request in the legacy concat order (camera 0 rows,
        then camera 1 rows; tile-aligned reshape = a free view)."""
        vision_features = self.vision_tower.forward_fused(im2col_dev)
        proj = self.mm_projector.forward(vision_features)
        ttnn.deallocate(vision_features)
        bn, p, d = proj.shape[0], proj.shape[1], proj.shape[2]
        if bn == batch:
            return proj
        assert bn % batch == 0, (bn, batch)
        return ttnn.reshape(proj, (batch, (bn // batch) * p, d))

    def allocate_kv_caches(self, prefix_len: int, action_horizon: int, batch: int = 1) -> Dict[str, int]:
        """(Re)allocate the 18 backbone-owned expert KV caches for this serving shape (``batch`` requests,
        one cache entry each). Called on the compile pass (before trace capture); a no-op while the shape is
        unchanged. Geometry and the rotary_embedding_to_cache / fill_cache tile constraints come from
        ``fused_host.kv_cache_plan``."""
        plan = kv_cache_plan(prefix_len, action_horizon, batch=batch)
        if self.kv_caches is not None and self.kv_cache_plan == plan:
            return plan
        # Cache SETS are kept per batch size so a server can hold one trace per batch size and switch between
        # them (the trace is bound to its caches' addresses); a set is only rebuilt when its plan changes.
        sets = self.__dict__.setdefault("_kv_sets", {})
        prev = sets.get(plan["batch"])
        if prev is not None and prev[1] == plan:
            self.kv_caches, self.kv_cache_plan = prev
            return plan
        if prev is not None:
            for k, v in prev[0]:
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            del sets[plan["batch"]]
        cfg = self.config.expert_config
        self.kv_caches = []
        # == the qkv linear output dtype of both VLM and expert (PI05_KV_DTYPE; fill_cache needs equal dtypes)
        kv_dtype = ttnn.bfloat16 if self.fused_cfg.kv_dtype == "bf16" else ttnn.bfloat8_b
        # L1 for one or two requests (the reference's SDPA 75 -> 53 us win); DRAM for larger batches: at
        # batch 4 the 36 caches are ~30 MB of L1 and the VLM prefill's static circular buffers clashed with
        # the L1 buffers (dataflow_buffer.cpp "clash with L1 buffers", 2026-09-17). With several sets alive
        # (multi-shape serving) only the batch-1 set stays in L1.
        kv_mem = ttnn.L1_MEMORY_CONFIG if (plan["batch"] == 1 or (plan["batch"] <= 2 and not sets)) else ttnn.DRAM_MEMORY_CONFIG
        for _ in self.expert_blocks:
            shape = [plan["batch"], cfg.num_kv_heads, plan["cache_len"], cfg.head_dim]
            k = ttnn.zeros(
                shape,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=kv_mem,
            )
            v = ttnn.zeros(
                shape,
                dtype=kv_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=kv_mem,
            )
            self.kv_caches.append((k, v))
        self.kv_cache_plan = plan
        sets[plan["batch"]] = (self.kv_caches, plan)
        return plan

    def forward_vlm_fused(self, prefix_embs: ttnn.Tensor, attn_mask: Optional[ttnn.Tensor] = None) -> None:
        """VLM prefill whose only product is the prefix K/V written into ``self.kv_caches`` (rows 0..S-1
        of every layer). Consumes ``prefix_embs``. ``attn_mask`` [B, 1, S, S] hides the prompt's pad keys
        (openpi's prefix padding mask). With ``skip_vlm_tail`` the last layer stops after
        its K/V write and the final norm is skipped (their outputs are never read by sample_actions)."""
        if self.kv_caches is None:
            raise RuntimeError("allocate_kv_caches() must run before forward_vlm_fused()")
        skip_tail = self.fused_cfg.skip_vlm_tail
        hidden_states = prefix_embs
        last = len(self.vlm_blocks) - 1
        for i, block in enumerate(self.vlm_blocks):
            cache_k, cache_v = self.kv_caches[i]
            if skip_tail and i == last:
                block.forward_fused_vlm(hidden_states, cache_k, cache_v, kv_only=True, attn_mask=attn_mask)
                ttnn.deallocate(hidden_states)
                return None
            new_hidden = block.forward_fused_vlm(hidden_states, cache_k, cache_v, attn_mask=attn_mask)
            ttnn.deallocate(hidden_states)
            hidden_states = new_hidden
        # PI05_SKIP_VLM_TAIL=0: compute the final norm too (A/B parity), result unused
        final = rms_norm_ttnn(hidden_states, self.vlm_norm, self.config.vlm_config.rms_norm_eps)
        ttnn.deallocate(hidden_states)
        ttnn.deallocate(final)
        return None

    def fold_expert_norms(self, block_mods_per_step: List[List[Tuple]]) -> None:
        """Build every expert block's per-step folded qkv / up|gate weights from the precomputed adaRMS mods
        (``block_mods_per_step[step][layer]``) and drop the torch originals."""
        from .ttnn_fused_norm import FoldedExpertNorms

        for i, block in enumerate(self.expert_blocks):
            if block._torch_fused_ug is None:
                raise RuntimeError("expert_norm_fold needs the torch up|gate weights (expert_geglu must be on)")
            mods = [block_mods_per_step[s][i] for s in range(len(block_mods_per_step))]
            block.set_folded(FoldedExpertNorms(self.device, block._torch_wqkv, block._torch_fused_ug, mods,
                                               fold_attn=block._torch_wqkv is not None))
            block._torch_wqkv = None
            block._torch_fused_ug = None

    def forward_expert_fused(
        self,
        hidden_states: ttnn.Tensor,
        precomputed_block_mods: List[Tuple],
        precomputed_final_mod: Tuple,
        attn_in: Dict[str, ttnn.Tensor],
        step: Optional[int] = None,
    ) -> ttnn.Tensor:
        """Expert on the tile-padded suffix reading the backbone-owned caches; consumes ``hidden_states`` and
        returns the final adaRMS-normed hidden ``[B, S, width]`` bf16 (L1). ``attn_in``: the graph's persistent
        attention inputs (key masks, RoPE rows at n_valid + [0, S)), see ``GemmaAttentionTTNN.forward_fused_expert``."""
        prefix_len = self.kv_cache_plan["prefix_len"]
        for i, block in enumerate(self.expert_blocks):
            cache_k, cache_v = self.kv_caches[i]
            hidden_states = block.forward_fused_expert(
                hidden_states, precomputed_block_mods[i], cache_k, cache_v, prefix_len, attn_in, step=step
            )
        scale_f, shift_f = precomputed_final_mod[0], precomputed_final_mod[1]
        out = adarms_norm_precomputed(hidden_states, scale_f, shift_f, self.config.expert_config.rms_norm_eps)
        ttnn.deallocate(hidden_states)
        return out


# Default export
PaliGemmaBackbone = PaliGemmaBackboneTTNN
