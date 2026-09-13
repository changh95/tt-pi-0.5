# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Main PI0 model - TTNN Implementation (Inference Only)

This module assembles all PI0 components into a complete model:
    - PrefixEmbedding: Images + language → embeddings
    - SuffixEmbedding: State + actions + timestep → embeddings
    - PaliGemmaBackbone: VLM + Action Expert transformers

Architecture:
    1. Process images through SigLIP vision tower
    2. Embed language tokens through Gemma embeddings
    3. Concatenate to form prefix embeddings
    4. Prefill prefix, cache KV, denoise actions iteratively

Optimizations:
    1. Pre-computed timesteps
    2. Denoising loop stays entirely on device
    3. Single transfer at the end for final actions

Fused / traced graph (the DEFAULT since the device pass of 2026-09-13; ``TT_FUSED=0`` = the legacy
``sample_actions`` untouched; read ONCE at build time into ``self.fused_cfg``; with the fused path the legacy entry points stay callable
ONLY under ``PI05_FUSED_RESIDUAL=legacy`` -- ``bf16`` / ``mixed`` store the expert o_proj / down_proj
in bf16, which the legacy block's bf8 residual cannot feed into ``dit_minimal_matmul_addcmul_fused``;
``sample_actions`` / ``sample_actions_traced`` then raise a RuntimeError up front):
    ``sample_actions_fused(images, lang_tokens, noise)`` runs the WHOLE device graph
    (host im2col -> SigLIP x cameras -> language embedding -> VLM prefill writing the backbone-owned
    KV caches -> 10 x (expert on the 64-row suffix + fused Euler step)) from three persistent device
    inputs (im2col [B,256,608] bf16 ROW_MAJOR, tokens [1,L] uint32 ROW_MAJOR, noise [1,64,32] bf16 TILE
    L1) and, with ``PI05_TRACE=1``, replays ONE Metal trace per request:
    ``copy_host_to_device_tensor`` x3 -> ``execute_trace`` -> one [1,64,32] readback -> [:50].
    The first call (the server's warm-up, before READY) allocates the inputs and caches, runs the
    graph eagerly once (program cache, lazily built constant slices) and captures the trace. No
    host->device write happens inside the graph. ``PI05_TRACE=0`` runs the same graph eagerly.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import ttnn

from models.experimental.pi0_5.common.configs import (
    PI0ModelConfig,
    PrefixConfig,
    SuffixConfig,
    PaliGemmaConfig,
    DenoiseConfig,
)
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.fused_host import (
    check_fused_shape_contract,
    euler_dts,
    im2col_patches,
    pad_rows,
    round_up,
    unpad_rows,
)
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from .ttnn_prefix import PrefixEmbeddingTTNN
from .ttnn_suffix import SuffixEmbeddingTTNN, convert_suffix_weights_to_ttnn
from .ttnn_paligemma import PaliGemmaBackboneTTNN


class PI0ModelTTNN:
    """
    Complete PI0 model implementation using TTNN.

    Maximizes execution on Tenstorrent hardware while keeping
    control flow and preprocessing on host.
    """

    def __init__(
        self,
        config: PI0ModelConfig,
        weight_loader: PI0WeightLoader,
        device: ttnn.Device,
        fused: Optional[FusedConfig] = None,
    ):
        """
        Initialize PI0 model with TTNN.

        Args:
            config: Model configuration
            weight_loader: Loaded weights
            device: TTNN device
            fused: TT_FUSED knobs; None -> ``FusedConfig.from_env()`` (read once, here). Disabled ->
                every component is built and runs exactly as before.
        """
        self.config = config
        self.weight_loader = weight_loader
        self.device = device
        self.fused_cfg = FusedConfig.from_env() if fused is None else fused
        self.fused = self.fused_cfg.enabled
        if self.fused and not config.pi05:
            raise RuntimeError("TT_FUSED=1 supports the pi0.5 (adaRMS) expert only")
        # Fused graph state (persistent device inputs, trace, output)
        self._suffix_rows = round_up(config.action_horizon)  # 50 -> 64
        self._fused_shape_key = None
        self._fused_in_im2col: List[ttnn.Tensor] = []
        self._fused_in_tokens = None
        self._fused_in_noise = None
        self._fused_trace_id = None
        self._fused_out = None

        # Initialize denoising config
        self.denoise_config = DenoiseConfig(
            num_steps=config.num_denoising_steps,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

        pad_steps = ((self.denoise_config.num_steps + 31) // 32) * 32

        # Create timestep indices on device using ttnn.arange
        self.timestep_indices = ttnn.arange(0, pad_steps, 1, device=self.device, dtype=ttnn.bfloat16)

        # Pre-compute all timestep tensors for the denoising loop
        num_steps = self.denoise_config.num_steps
        self._precomputed_timesteps = []
        for i in range(num_steps):
            t_val = 1.0 - i / num_steps
            t_torch = torch.tensor([t_val], dtype=torch.float32)
            t_ttnn = ttnn.from_torch(t_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            t_ttnn = ttnn.reshape(t_ttnn, (1,))
            self._precomputed_timesteps.append(t_ttnn)

        # Default initial flow-matching noise buffer (seeded). sample_actions() reuses it for a
        # deterministic, reproducible policy and reallocates it when the input batch size differs;
        # the traced denoising path also reads it. Pass an explicit `noise=` to override.
        x_t_torch = torch.randn(1, self.config.action_horizon, self.config.action_dim)
        self._default_noise_torch = x_t_torch.clone()  # fused path: same seeded noise, host copy
        self.x_t_ttnn = ttnn.from_torch(
            x_t_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all model components."""
        # Suffix embedding with TTNN weights
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=self.config.pi05,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()

        # Convert weights to TTNN (fused: action_out_proj in bf16 for the Euler fused op, x_t is bf16)
        overrides = {"action_out_proj.weight": ttnn.bfloat16} if self.fused else None
        ttnn_weights = convert_suffix_weights_to_ttnn(pi0_weights, self.device, weight_dtype_overrides=overrides)
        self.suffix_embedding = SuffixEmbeddingTTNN(suffix_config, ttnn_weights, self.device, fused_cfg=self.fused_cfg)

        # Pre-compute per-step adaRMS conditioning for Pi0.5.
        # adarms_cond depends only on timestep value, which is fixed per step index, so the
        # time MLP (reshape + linear + silu + linear + silu) and sinusoidal embedding can be
        # hoisted out of the denoising loop, saving ~10 ops/step on the critical path.
        self._precomputed_adarms_cond = None
        if self.config.pi05:
            self._precomputed_adarms_cond = []
            for t_tensor in self._precomputed_timesteps:
                time_emb = self.suffix_embedding.embed_timestep(t_tensor)
                adarms_cond = self.suffix_embedding.compute_adarms_cond_from_time(time_emb)
                self._precomputed_adarms_cond.append(adarms_cond)
                ttnn.deallocate(time_emb)

        # Backbone
        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
            max_seq_len=self.config.max_seq_len,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = PaliGemmaBackboneTTNN(paligemma_config, weights, self.device, fused_cfg=self.fused_cfg)

        # Pre-compute per-(step, layer) adaRMS modulations for Pi0.5.
        # For each denoising step, for each expert layer, the modulation tensor
        #     modulation = linear(adarms_cond[step], layer.input_ln_dense_{w,b})
        # is constant (cond is constant per step, weights are frozen). Splitting it into
        # (scale, shift, gate) triples once at init lets the per-layer critical path skip
        # the dense projection and chunk entirely — each adaRMS call collapses to a single
        # ttnn.rms_norm that reads cached scale/shift tensors.
        self._precomputed_block_mods = None
        self._precomputed_final_mod = None
        if self.config.pi05 and self._precomputed_adarms_cond is not None and self.backbone.use_expert_adarms:
            expert_blocks = self.backbone.expert_blocks
            self._precomputed_block_mods = []
            self._precomputed_final_mod = []
            for step_idx, cond in enumerate(self._precomputed_adarms_cond):
                per_layer = []
                for block in expert_blocks:
                    mod_in = ttnn.linear(
                        cond,
                        block.input_ln_dense_weight,
                        bias=block.input_ln_dense_bias,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    scale_in, shift_in, gate_in = ttnn.chunk(mod_in, 3, dim=-1)
                    ttnn.deallocate(mod_in)
                    mod_post = ttnn.linear(
                        cond,
                        block.post_attn_ln_dense_weight,
                        bias=block.post_attn_ln_dense_bias,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    scale_post, shift_post, gate_post = ttnn.chunk(mod_post, 3, dim=-1)
                    ttnn.deallocate(mod_post)
                    per_layer.append((scale_in, shift_in, gate_in, scale_post, shift_post, gate_post))
                self._precomputed_block_mods.append(per_layer)

                mod_final = ttnn.linear(
                    cond,
                    self.backbone.expert_norm_dense_weight,
                    bias=self.backbone.expert_norm_dense_bias,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                scale_f, shift_f, gate_f_unused = ttnn.chunk(mod_final, 3, dim=-1)
                ttnn.deallocate(mod_final)
                ttnn.deallocate(gate_f_unused)
                self._precomputed_final_mod.append((scale_f, shift_f))

        # Prefix embedding with backbone functions
        prefix_config = PrefixConfig(
            vlm_hidden_size=self.config.vlm_config.width,
            num_image_tokens=self.config.siglip_config.num_patches,
        )
        self.prefix_embedding = PrefixEmbeddingTTNN(
            prefix_config,
            self.device,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
            embed_images_fused_fn=self.backbone.embed_images_fused,
            embed_language_fused_fn=self.backbone.embed_language_tokens_fused,
        )

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Embed prefix (images + language) using TTNN.

        Args:
            images: List of input images (PyTorch)
            img_masks: Image validity masks (PyTorch)
            lang_tokens: Language token IDs (TTNN)
            lang_masks: Language masks (TTNN)

        Returns:
            Tuple of (embeddings, padding_mask, attention_mask) as TTNN tensors
        """
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(
        self,
        state: ttnn.Tensor,
        noisy_actions: ttnn.Tensor,
        timestep: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[ttnn.Tensor]]:
        """
        Embed suffix (state + noisy actions + timestep) using TTNN.

        Args:
            state: Robot state (TTNN)
            noisy_actions: Noisy actions (TTNN)
            timestep: Diffusion timestep (TTNN)

        Returns:
            Tuple of (embeddings, padding_mask, attention_mask, adarms_cond)
        """
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def _check_legacy_path_available(self, entry: str) -> None:
        """The legacy entry points reuse the shipped expert block, which typecasts its residual / gate
        to bf8 before ``dit_minimal_matmul_addcmul_fused(..., self.mlp.down_proj, ...)``. A model built
        with ``TT_FUSED=1`` and ``PI05_FUSED_RESIDUAL`` in (``bf16``, ``mixed``) stores o_proj /
        down_proj in bf16 for the fused residual op, and the device op then fails hard
        (``TT_FATAL(ternary_a_data_format == in1_data_format)``). Fail here, in Python, instead."""
        if self.fused and not self.fused_cfg.legacy_sample_actions_available:
            raise RuntimeError(
                f"PI0ModelTTNN.{entry} is not available on a model built with TT_FUSED=1 and "
                f"PI05_FUSED_RESIDUAL={self.fused_cfg.residual!r}: the expert o_proj / down_proj are "
                "bfloat16 for the fused residual op while the legacy expert block feeds a bfloat8_b "
                "residual into dit_minimal_matmul_addcmul_fused (residual format must equal the weight "
                "format). Use sample_actions_fused, or build the model with PI05_FUSED_RESIDUAL=legacy "
                "(or TT_FUSED unset) for the legacy path."
            )

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample actions via denoising (TTNN inference).

        This runs the full denoising loop:
        1. Compute prefix embeddings (images + language) once
        2. Forward prefix through VLM and cache KV
        3. For each denoising step: compute suffix, forward through expert with cached KV

        Args:
            images: Input images (PyTorch)
            img_masks: Image masks (PyTorch)
            lang_tokens: Language tokens (PyTorch)
            lang_masks: Language masks (PyTorch)
            state: Robot state (PyTorch)
            noise: Optional initial flow-matching noise (batch_size, action_horizon, action_dim).
                If None, a cached deterministic noise buffer is used (reallocated to match state's
                batch dimension) for a reproducible policy; pass a tensor for matched-noise
                comparisons or seeded sampling.

        Returns:
            Sampled actions (PyTorch)
        """
        self._check_legacy_path_available("sample_actions")
        # Convert inputs to TTNN
        lang_tokens_ttnn = lang_tokens
        lang_masks_ttnn = lang_masks
        state_ttnn = state

        # Step 1: Embed prefix (images + language) using TTNN
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens_ttnn, lang_masks_ttnn)

        # Step 2: Forward prefix through VLM and cache KV
        _, prefix_kv_cache = self.backbone.forward_vlm(prefix_embs, use_cache=True)

        # Get timesteps using pure Python list (for control flow on host)
        num_steps = self.denoise_config.num_steps
        # Create timesteps as Python list: [1.0, 0.9, 0.8, ..., 0.0]
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        # Step 3: Initial flow-matching noise. By default we reuse a cached, seeded noise buffer so
        # the policy is deterministic and reproducible (the right choice for closed-loop robot eval);
        # the buffer is (re)allocated whenever the input batch size changes, so batch_size > 1 now
        # produces a correctly-sized action tensor. Pass `noise` to inject a specific tensor (e.g.
        # for matched-noise ttnn-vs-torch comparisons or seeded sampling).
        batch_size = state.shape[0]
        if noise is not None:
            x_t_ttnn = (
                noise
                if isinstance(noise, ttnn.Tensor)
                else ttnn.from_torch(
                    noise,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            )
        else:
            if self.x_t_ttnn is None or self.x_t_ttnn.shape[0] != batch_size:
                self.x_t_ttnn = ttnn.from_torch(
                    torch.randn(batch_size, self.config.action_horizon, self.config.action_dim),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            x_t_ttnn = self.x_t_ttnn

        # Step 4: Denoising loop (stays on device!)
        for i in range(num_steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t

            # Use pre-computed timestep tensor (no slice/reshape per step)
            t_tensor = self._precomputed_timesteps[i]

            # Embed suffix (x_t_ttnn already on device - no transfer!)
            if self.config.pi05 and self._precomputed_adarms_cond is not None:
                suffix_embs, suffix_pad, suffix_att, adarms_cond = self.suffix_embedding.embed_suffix_pi05_cached(
                    x_t_ttnn, self._precomputed_adarms_cond[i]
                )
            else:
                suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state_ttnn, x_t_ttnn, t_tensor)

            # Forward through expert with cached prefix KV (pass adarms_cond for Pi0.5)
            expert_output, _ = self.backbone.forward_expert(
                suffix_embs,
                past_key_values=prefix_kv_cache,
                adarms_cond=adarms_cond,
                precomputed_block_mods=(
                    self._precomputed_block_mods[i] if self._precomputed_block_mods is not None else None
                ),
                precomputed_final_mod=(
                    self._precomputed_final_mod[i] if self._precomputed_final_mod is not None else None
                ),
            )

            # Extract action output (skip state token in PI0 mode)
            if not self.config.pi05:
                action_output = ttnn.slice(
                    expert_output, [0, 1, 0], [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
                )
            else:
                action_output = expert_output

            # Project to velocity
            velocity = self.suffix_embedding.project_output(action_output)

            # Euler step ON DEVICE (no transfer per step!)
            velocity_scaled = ttnn.mul(velocity, dt)
            x_t_ttnn = ttnn.add(x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

            # Clear profiler buffer after each denoising step (~500 ops)
            # ReadDeviceProfiler removed for performance

        # Convert back to PyTorch only at the very end (1 transfer instead of 10!)
        return x_t_ttnn

    def _run_denoising_loop(
        self,
        state_ttnn: ttnn.Tensor,
        x_t_ttnn: ttnn.Tensor,
        prefix_kv_cache,
    ) -> ttnn.Tensor:
        """Run the 10-step denoising loop. Factored out for trace capture."""
        num_steps = self.denoise_config.num_steps
        timesteps = [1.0 - i / num_steps for i in range(num_steps + 1)]

        for i in range(num_steps):
            dt = timesteps[i + 1] - timesteps[i]
            t_tensor = self._precomputed_timesteps[i]

            if self.config.pi05 and self._precomputed_adarms_cond is not None:
                suffix_embs, suffix_pad, suffix_att, adarms_cond = self.suffix_embedding.embed_suffix_pi05_cached(
                    x_t_ttnn, self._precomputed_adarms_cond[i]
                )
            else:
                suffix_embs, suffix_pad, suffix_att, adarms_cond = self.embed_suffix(state_ttnn, x_t_ttnn, t_tensor)

            expert_output, _ = self.backbone.forward_expert(
                suffix_embs,
                past_key_values=prefix_kv_cache,
                adarms_cond=adarms_cond,
                precomputed_block_mods=(
                    self._precomputed_block_mods[i] if self._precomputed_block_mods is not None else None
                ),
                precomputed_final_mod=(
                    self._precomputed_final_mod[i] if self._precomputed_final_mod is not None else None
                ),
            )

            if not self.config.pi05:
                action_output = ttnn.slice(
                    expert_output, [0, 1, 0], [expert_output.shape[0], expert_output.shape[1], expert_output.shape[2]]
                )
            else:
                action_output = expert_output

            velocity = self.suffix_embedding.project_output(action_output)
            velocity_scaled = ttnn.mul(velocity, dt)
            x_t_ttnn = ttnn.add(x_t_ttnn, velocity_scaled, memory_config=ttnn.L1_MEMORY_CONFIG)

        return x_t_ttnn

    def setup_trace(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ):
        """
        Set up 2CQ + Trace for the denoising loop.

        Call this once to compile and capture. Then call execute_trace() for fast inference.
        """
        # Step 1: Run prefix (not traced — runs once per new observation)
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        _, self._trace_prefix_kv_cache = self.backbone.forward_vlm(prefix_embs, use_cache=True)

        # Step 2: Compile pass — run denoising loop to JIT compile all kernels
        x_t_compile = self.x_t_ttnn
        self._run_denoising_loop(state, x_t_compile, self._trace_prefix_kv_cache)

        # Step 3: Capture trace — re-run denoising loop under trace capture
        # The x_t tensor at self.x_t_ttnn address will be the input
        self._trace_x_t = self.x_t_ttnn
        self._trace_state = state

        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._trace_output = self._run_denoising_loop(self._trace_state, self._trace_x_t, self._trace_prefix_kv_cache)
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)

        self._trace_id = trace_id

    def execute_trace(self) -> ttnn.Tensor:
        """Execute the captured denoising trace. Call setup_trace() first."""
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
        return self._trace_output

    def sample_actions_traced(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions using 2CQ + Trace for the denoising loop.

        First call sets up the trace. Subsequent calls execute it.
        The prefix (SigLIP + VLM) runs normally each time.
        """
        self._check_legacy_path_available("sample_actions_traced")
        # Run prefix (new observation each time)
        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        _, prefix_kv_cache = self.backbone.forward_vlm(prefix_embs, use_cache=True)

        if not hasattr(self, "_trace_id"):
            # First call: compile + capture trace
            # Compile pass
            x_t_compile = self.x_t_ttnn
            self._run_denoising_loop(state, x_t_compile, prefix_kv_cache)

            # Store references for trace
            self._trace_prefix_kv_cache = prefix_kv_cache
            self._trace_x_t = self.x_t_ttnn
            self._trace_state = state

            # Capture trace
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            self._trace_output = self._run_denoising_loop(
                self._trace_state, self._trace_x_t, self._trace_prefix_kv_cache
            )
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            self._trace_id = trace_id
        else:
            # Update prefix KV cache in-place at the same tensor addresses
            # Copy new KV cache values to the trace's pre-allocated KV tensors
            for i in range(len(prefix_kv_cache)):
                new_k, new_v = prefix_kv_cache[i]
                old_k, old_v = self._trace_prefix_kv_cache[i]
                ttnn.copy(new_k, old_k)
                ttnn.copy(new_v, old_v)

        # Execute traced denoising loop
        ttnn.execute_trace(self.device, self._trace_id, cq_id=0, blocking=True)
        return self._trace_output

    def release_trace(self):
        """Release trace resources (the legacy loop trace and the fused whole-graph trace)."""
        if hasattr(self, "_trace_id"):
            ttnn.release_trace(self.device, self._trace_id)
            del self._trace_id
        if self._fused_trace_id is not None:
            ttnn.release_trace(self.device, self._fused_trace_id)
            self._fused_trace_id = None
            self._fused_out = None

    # ======================================================================
    # Fused / traced whole-graph path (TT_FUSED=1)
    # ======================================================================

    def _fused_host_inputs(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        noise: Optional[torch.Tensor],
    ) -> Tuple[List[ttnn.Tensor], ttnn.Tensor, ttnn.Tensor]:
        """torch request data -> HOST ttnn tensors with the persistent inputs' shape / dtype / layout.

        images: N x [1, 3, 224, 224] float in [-1, 1] (the server's preprocessing); the host im2col
        is the exact permutation the legacy device unfold performed, rounded to bf16 by the upload
        (the legacy path rounded the same pixels to bf16 before its unfold). One im2col tensor
        [N, 256, 608] when the cameras are batched (PI05_SIGLIP_BATCHED=1), else one per camera.
        lang_tokens: [1, L] int ids -> uint32 ROW_MAJOR (same conversion as the legacy upload).
        noise: [1, 50, 32] float or None (-> the model's seeded default) -> zero-padded to 64 rows.
        """
        if not images:
            raise ValueError("at least one image is required")
        if any(isinstance(img, ttnn.Tensor) for img in images) or isinstance(lang_tokens, ttnn.Tensor):
            raise TypeError("sample_actions_fused takes torch inputs (the host builds the trace inputs)")
        pixels = torch.cat([img.reshape(1, *img.shape[-3:]).float() for img in images], dim=0)
        patch = self.config.siglip_config.patch_size
        pad_to = self.backbone.vision_tower.patch_embed.in_features_padded
        im2col = im2col_patches(pixels, patch, pad_to=pad_to)  # [N, 256, 608] fp32
        if self.fused_cfg.siglip_batched:
            im2col_hosts = [ttnn.from_torch(im2col, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)]
        else:
            im2col_hosts = [
                ttnn.from_torch(im2col[i : i + 1].contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
                for i in range(im2col.shape[0])
            ]
        tokens_host = ttnn.from_torch(lang_tokens.reshape(1, -1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        noise_t = self._default_noise_torch if noise is None else noise
        if isinstance(noise_t, ttnn.Tensor):
            raise TypeError("sample_actions_fused needs the noise as a torch tensor")
        noise_t = noise_t.reshape(1, self.config.action_horizon, self.config.action_dim).float()
        noise_host = ttnn.from_torch(pad_rows(noise_t, self._suffix_rows), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return im2col_hosts, tokens_host, noise_host

    @staticmethod
    def _shape_key(im2col_hosts, tokens_host, noise_host):
        return (
            tuple(tuple(int(d) for d in t.shape) for t in im2col_hosts),
            tuple(int(d) for d in tokens_host.shape),
            tuple(int(d) for d in noise_host.shape),
        )

    def _fused_release_inputs(self):
        for t in self._fused_in_im2col:
            ttnn.deallocate(t)
        self._fused_in_im2col = []
        for t in (self._fused_in_tokens, self._fused_in_noise):
            if t is not None:
                ttnn.deallocate(t)
        self._fused_in_tokens = None
        self._fused_in_noise = None
        self._fused_shape_key = None

    def _fused_prepare(self, im2col_hosts, tokens_host, noise_host) -> bool:
        """First call for a shape: allocate the persistent inputs (uploading these host tensors) and
        the KV caches, run the graph once eagerly (kernel compile / program cache, constant slices and
        tables built outside the trace), then capture the trace (PI05_TRACE=1). Returns True when it
        prepared (inputs already hold this call's data), False when nothing had to be done."""
        key = self._shape_key(im2col_hosts, tokens_host, noise_host)
        if self._fused_shape_key is not None:
            if key == self._fused_shape_key:
                return False
            if self._fused_trace_id is not None:
                raise RuntimeError(
                    f"the fused trace was captured for inputs {self._fused_shape_key} but got {key}: the traced "
                    "graph is shape-bound (PI05_NUM_IMAGES / PI05_TOKEN_LEN are fixed at startup)"
                )
            self._fused_release_inputs()

        num_images = sum(k[0] for k in key[0])
        token_len = key[1][-1]
        plan = check_fused_shape_contract(num_images, token_len, self.config.action_horizon)

        device = self.device
        self._fused_in_im2col = [
            ttnn.to_device(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in im2col_hosts
        ]
        self._fused_in_tokens = ttnn.to_device(tokens_host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._fused_in_noise = ttnn.to_device(noise_host, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.backbone.allocate_kv_caches(plan["prefix_len"], self.config.action_horizon)
        self._fused_shape_key = key

        # Compile pass (eager): program cache, cos/sin slices, SigLIP pos table -- all outside the trace
        out = self._fused_device_graph()
        ttnn.synchronize_device(device)
        ttnn.deallocate(out)

        if self.fused_cfg.trace:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            try:
                self._fused_out = self._fused_device_graph()
            except BaseException:
                # A capture left open hangs the device close: end it before propagating.
                try:
                    ttnn.end_trace_capture(device, trace_id, cq_id=0)
                    ttnn.release_trace(device, trace_id)
                finally:
                    self._fused_out = None
                raise
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            ttnn.synchronize_device(device)
            self._fused_trace_id = trace_id
        return True

    def _fused_write_inputs(self, im2col_hosts, tokens_host, noise_host) -> None:
        """Per request: host -> the persistent device buffers (the only host->device traffic)."""
        for host, dev in zip(im2col_hosts, self._fused_in_im2col):
            ttnn.copy_host_to_device_tensor(host, dev, cq_id=0)
        ttnn.copy_host_to_device_tensor(tokens_host, self._fused_in_tokens, cq_id=0)
        ttnn.copy_host_to_device_tensor(noise_host, self._fused_in_noise, cq_id=0)

    def _fused_device_graph(self) -> ttnn.Tensor:
        """The whole device graph, reading only the persistent inputs; returns x_T [1, 64, 32] bf16 (L1).
        Captured as-is into the trace, so nothing in here may touch the host."""
        prefix_embs = self.prefix_embedding.embed_prefix_fused(self._fused_in_im2col, self._fused_in_tokens)
        self.backbone.forward_vlm_fused(prefix_embs)  # consumes prefix_embs, fills the KV caches

        num_steps = self.denoise_config.num_steps
        dts = euler_dts(num_steps)
        x_t = self._fused_in_noise
        for i in range(num_steps):
            suffix_embs = self.suffix_embedding.embed_actions_fused(x_t)  # [1, 64, width] bf16 DRAM
            expert_out = self.backbone.forward_expert_fused(
                suffix_embs, self._precomputed_block_mods[i], self._precomputed_final_mod[i]
            )  # consumes suffix_embs
            x_next = self.suffix_embedding.euler_step_fused(expert_out, x_t, dts[i])
            ttnn.deallocate(expert_out)
            if x_t is not self._fused_in_noise:
                ttnn.deallocate(x_t)
            x_t = x_next
        return x_t

    def sample_actions_fused(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Whole-graph fused (and traced) inference. torch in -> torch ``[1, action_horizon, action_dim]``
        float32 out. First call for a shape: allocate + compile + capture (the server's warm-up)."""
        if not self.fused:
            raise RuntimeError("sample_actions_fused needs TT_FUSED=1 (FusedConfig.enabled)")
        if self._precomputed_block_mods is None or self._precomputed_final_mod is None:
            raise RuntimeError("fused path needs the precomputed pi0.5 adaRMS modulations")
        im2col_hosts, tokens_host, noise_host = self._fused_host_inputs(images, lang_tokens, noise)
        prepared = self._fused_prepare(im2col_hosts, tokens_host, noise_host)
        if not prepared:
            self._fused_write_inputs(im2col_hosts, tokens_host, noise_host)

        if self._fused_trace_id is not None:
            ttnn.execute_trace(self.device, self._fused_trace_id, cq_id=0, blocking=True)
            actions = ttnn.to_torch(self._fused_out)
        else:
            out = self._fused_device_graph()
            actions = ttnn.to_torch(out)
            ttnn.deallocate(out)
        return unpad_rows(actions.float(), self.config.action_horizon)

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: ttnn.Device,
        config: Optional[PI0ModelConfig] = None,
    ) -> "PI0ModelTTNN":
        """
        Load pretrained PI0 model to TTNN device.

        Args:
            model_path: Path to model or HuggingFace model ID
            device: TTNN device
            config: Optional configuration override

        Returns:
            Loaded PI0 model
        """
        weight_loader = PI0WeightLoader(model_path)

        if config is None:
            config = PI0ModelConfig(
                action_dim=weight_loader.config.action_dim,
                action_horizon=weight_loader.config.action_horizon,
            )

        return cls(config, weight_loader, device)


# Default export
PI0Model = PI0ModelTTNN
