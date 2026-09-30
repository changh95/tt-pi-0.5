# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Main pi0.5 model - TTNN Implementation (Inference Only)

This module assembles all components into the complete model:
    - PrefixEmbedding: Images + language -> embeddings
    - SuffixEmbedding: noisy actions -> embeddings; Euler step
    - PaliGemmaBackbone: VLM + Action Expert transformers

Architecture:
    1. Process images through SigLIP vision tower
    2. Embed language tokens through Gemma embeddings
    3. Concatenate to form prefix embeddings
    4. Prefill prefix, cache KV, denoise actions iteratively

Build-time precomputation: the timesteps, the per-step adaRMS conditioning and the per-(step, layer)
adaRMS modulations are constants and are computed once in ``__init__``.

Fused / traced graph (the only inference path; knobs read ONCE at build time into ``self.fused_cfg``,
see ``common/fused_config.py``):
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
from typing import Dict, List, Optional, Tuple, Union

import os
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
    attention_inputs,
    check_fused_shape_contract,
    prefix_valid_mask,
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
from .ttnn_ccl import num_devices as _mesh_num_devices, is_mesh as _is_mesh, chip0 as _chip0


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
            fused: fused-graph knobs; None -> ``FusedConfig.from_env()`` (read once, here).
        """
        self.config = config
        self.weight_loader = weight_loader
        self.device = device
        self.fused_cfg = FusedConfig.from_env() if fused is None else fused
        # Multi-chip: PI05_TP=0 (auto) -> the mesh size; single chip -> 1 (see tt/ttnn_ccl.py)
        self.fused_cfg = self.fused_cfg.resolved(_mesh_num_devices(device))
        if not config.pi05:
            raise RuntimeError("the fused graph supports the pi0.5 (adaRMS) expert only")
        # Fused graph state (persistent device inputs, trace, output)
        self._suffix_rows = round_up(config.action_horizon)  # 50 -> 64
        self._fused_shape_key = None
        self._fused_in_im2col: List[ttnn.Tensor] = []
        self._fused_in_tokens = None
        self._fused_in_noise = None
        self._fused_attn_dev: Dict[str, ttnn.Tensor] = {}  # persistent attention inputs (masks, RoPE rows)
        self._fused_attn_key = None  # prefix validity the attention inputs were last written for
        self._fused_trace_id = None
        self._fused_out = None
        self._expert_rope_host = None  # (cos, sin) host copies of the expert RoPE tables

        # Initialize denoising config
        self.denoise_config = DenoiseConfig(
            num_steps=config.num_denoising_steps,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

        # Pre-compute all timestep tensors for the denoising loop
        num_steps = self.denoise_config.num_steps
        self._precomputed_timesteps = []
        for i in range(num_steps):
            t_val = 1.0 - i / num_steps
            t_torch = torch.tensor([t_val], dtype=torch.float32)
            t_ttnn = ttnn.from_torch(t_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
            t_ttnn = ttnn.reshape(t_ttnn, (1,))
            self._precomputed_timesteps.append(t_ttnn)

        # Default initial flow-matching noise (seeded, host copy): sample_actions_fused uses it when no
        # `noise=` is passed, for a deterministic, reproducible policy.
        self._default_noise_torch = torch.randn(1, self.config.action_horizon, self.config.action_dim)

        # Initialize components
        self._init_components()
        self._init_megakernel()

    def _init_megakernel(self):
        """PI05_MEGAKERNEL (docs/megakernel/DESIGN.md §4.12): ``expert`` runs the whole 10 x 18 expert loop, the action
        in / out projections and the Euler steps as ONE generic_op (tt/megakernel/) after the ttnn prefix. The stamp
        (``megakernel_backend`` / ``megakernel_program``) is what tests assert; refusals raise, never fall back."""
        self.megakernel_backend = self.fused_cfg.megakernel
        self.megakernel_program = None
        self._mk: Dict[str, object] = {}
        self._mk_params = None
        self._mk_l1: Dict[object, Tuple] = {}  # per prepared shape: L1 allocator signature at its trace capture
        self.megakernel_l1 = None
        if self.megakernel_backend == "off":
            return
        if self.megakernel_backend == "whole":
            raise RuntimeError("PI05_MEGAKERNEL=whole (phase 2: the whole sample_actions as one program) is not built yet")
        from .megakernel.geometry import megakernel_refusal

        why = megakernel_refusal(self.fused_cfg.kv_dtype, self.denoise_config.num_steps)
        if why is not None:
            raise RuntimeError(f"PI05_MEGAKERNEL=expert refused: {why}")
        if _is_mesh(self.device):
            raise RuntimeError("PI05_MEGAKERNEL=expert is single-chip: TT_MESH_SHAPE must be 1x1")
        why = self.megakernel_device_refusal(self.device)
        if why is not None:
            raise RuntimeError(f"PI05_MEGAKERNEL=expert refused: {why}")
        from .megakernel.host_model import expert_params
        from .megakernel.program import KERNEL_SOURCES, kernel_digest

        cw = self.weight_loader.categorized_weights
        self._mk_params = expert_params(cw["action_expert"], cw["pi0_projections"],
                                        eps=self.config.expert_config.rms_norm_eps,
                                        num_steps=self.denoise_config.num_steps)
        self.megakernel_program = {"kernel_digest": kernel_digest(),
                                   "sources": [os.path.basename(p) for p in KERNEL_SOURCES]}

    @staticmethod
    def megakernel_device_refusal(device) -> Optional[str]:
        """DESIGN.md §4.12 refusal (d): the megakernel program needs the 136,192 B kernel-config ring that only the
        64 KiB worker-L1 cut leaves (common/device_open.py). Without the cut, tt-metal fails at the first launch with
        "Program size ... too large for kernel config buffer" by a margin of ~100 B (verify_p1_r1), so name it here,
        before any weight upload. Worker L1 = the L1 + L1_SMALL allocator regions per bank."""
        from models.experimental.pi0_5.common.device_open import MEGAKERNEL_WORKER_L1_SIZE

        worker = sum(int(ttnn.get_memory_view(device, bt).total_bytes_per_bank)
                     for bt in (ttnn.BufferType.L1, ttnn.BufferType.L1_SMALL))
        if worker > MEGAKERNEL_WORKER_L1_SIZE:
            return (f"the device was opened without the 64 KiB worker-L1 cut (worker L1 {worker} B per bank > "
                    f"{MEGAKERNEL_WORKER_L1_SIZE}); open it with common/device_open.py (open_pi05_device / "
                    "device_kwargs), or set PI05_MEGAKERNEL=off for the stock-op comparator path")
        return None

    def l1_signature(self) -> Tuple:
        """L1 allocator state (DESIGN.md §4.12 replay guard): an execute_trace replay does not re-validate the
        megakernel's static circular buffers against L1 buffers allocated after the capture, so the model refuses to
        replay when this changed."""
        mv = ttnn.get_memory_view(self.device, ttnn.BufferType.L1)
        return (int(mv.total_bytes_allocated_per_bank), int(mv.largest_contiguous_bytes_free_per_bank), len(mv.block_table))

    def _check_l1_guard(self) -> None:
        ref = getattr(self, "_mk_l1", {}).get(self._fused_shape_key)
        if ref is None or os.environ.get("PI05_MK_L1_GUARD", "1") == "0":
            return
        now = self.l1_signature()
        if now != ref:
            raise RuntimeError(
                f"PI05_MEGAKERNEL: L1 allocation state changed since the trace capture ({ref} -> {now}: "
                "allocated bytes / largest free block / blocks per bank); an L1 buffer allocated after the capture "
                "can overlap the megakernel's circular buffers during a replay. Free it before sample_actions_fused.")

    def _megakernel_for(self, prefix_len: int, batch: int):
        """The ExpertMegakernel of this serving shape (built once; the weight arenas are shared by the shapes)."""
        from .megakernel import geometry as MG
        from .megakernel.program import ExpertMegakernel

        if batch != 1:
            raise RuntimeError(f"PI05_MEGAKERNEL=expert serves batch 1 only (got batch {batch})")
        shape = MG.shape_for(prefix_len, self._suffix_rows)
        mk = self._mk.get(shape.name)
        if mk is None:
            arenas = None
            for other in self._mk.values():
                if other.plan.bank == MG.plan_banks(shape).bank:
                    arenas = (other.w8, other.w16)
            mk = ExpertMegakernel(self.device, self._mk_params, shape, arenas=arenas)
            self._mk[shape.name] = mk
        return mk

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

        # Convert weights to TTNN (action_out_proj in bf16 for the Euler fused op, x_t is bf16)
        overrides = {"action_out_proj.weight": ttnn.bfloat16}
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
        self.backbone = PaliGemmaBackboneTTNN(
            paligemma_config, weights, self.device, fused_cfg=self.fused_cfg, action_horizon=self.config.action_horizon
        )

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
            if getattr(self.fused_cfg, "expert_norm_fold", False):
                import time as _time

                t_fold = _time.perf_counter()
                self.backbone.fold_expert_norms(self._precomputed_block_mods)
                print(f"[pi0.5] expert adaRMS norms folded into per-step weights in {_time.perf_counter() - t_fold:.1f}s", flush=True)

        # Prefix embedding with backbone functions
        prefix_config = PrefixConfig(
            vlm_hidden_size=self.config.vlm_config.width,
            num_image_tokens=self.config.siglip_config.num_patches,
        )
        self.prefix_embedding = PrefixEmbeddingTTNN(
            prefix_config,
            self.device,
            embed_images_fused_fn=self.backbone.embed_images_fused,
            embed_language_fused_fn=self.backbone.embed_language_tokens_fused,
        )

    def release_trace(self):
        """Release the fused whole-graph trace."""
        if self._fused_trace_id is not None:
            ttnn.release_trace(self.device, self._fused_trace_id)
            self._fused_trace_id = None
            self._fused_out = None

    # ======================================================================
    # Fused / traced whole-graph path
    # ======================================================================

    def _fused_host_inputs(
        self,
        images: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        noise: Optional[torch.Tensor],
        lang_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[List[ttnn.Tensor], ttnn.Tensor, ttnn.Tensor, torch.Tensor]:
        """torch request data -> HOST ttnn tensors with the persistent inputs' shape / dtype / layout.

        images: B*N x [1, 3, 224, 224] float in [-1, 1] (the server's preprocessing), request-major
        (request 0's N cameras, then request 1's, ...; a list of per-request lists is accepted too).
        B = lang_tokens.shape[0] requests share the trace. The host im2col is the exact permutation the
        legacy device unfold performed, rounded to bf16 by the upload. One im2col tensor [B*N, 256, 608]
        when the cameras are batched (PI05_SIGLIP_BATCHED=1), else one per camera.
        lang_tokens: [B, L] int ids, right-padded -> uint32 ROW_MAJOR (same conversion as the legacy upload).
        lang_masks: [B, L] bool, True on the real tokens (a right-padded prefix of each row). None -> ``tokens != 0``
        (PaliGemma ``<pad>`` = 0, never a prompt token). The padded tokens are hidden from every query.
        noise: [B, H, 32] float (a [1, H, 32] noise is repeated over B) or None (-> the model's seeded
        default) -> zero-padded to round_up(H) rows.
        Returns the host tensors and the prefix validity ``[B, N*256 + L]`` (bool; every camera is valid).
        """
        if not images:
            raise ValueError("at least one image is required")
        if isinstance(images[0], (list, tuple)):
            images = [img for req in images for img in req]
        if any(isinstance(img, ttnn.Tensor) for img in images) or isinstance(lang_tokens, ttnn.Tensor):
            raise TypeError("sample_actions_fused takes torch inputs (the host builds the trace inputs)")
        batch = int(lang_tokens.shape[0]) if lang_tokens.dim() > 1 else 1
        if len(images) % batch != 0:
            raise ValueError(f"{len(images)} images do not split over {batch} requests")
        tokens = lang_tokens.reshape(batch, -1)
        lmask = (tokens != 0) if lang_masks is None else torch.as_tensor(lang_masks).reshape(batch, -1).bool()
        if lmask.shape != tokens.shape:
            raise ValueError(f"lang_masks {tuple(lmask.shape)} != lang_tokens {tuple(tokens.shape)}")
        n_lang = lmask.long().sum(dim=1)
        right_padded = torch.arange(tokens.shape[1])[None, :] < n_lang[:, None]
        if not torch.equal(lmask, right_padded):
            # the prefix RoPE positions are 0..P-1, which equal openpi's cumsum(valid) - 1 only for right padding
            raise ValueError("lang_masks must mark a right-padded prompt (real tokens first, then padding)")
        valid = prefix_valid_mask(len(images) // batch, lmask)
        pixels = torch.cat([img.reshape(1, *img.shape[-3:]).float() for img in images], dim=0)
        patch = self.config.siglip_config.patch_size
        pad_to = self.backbone.vision_tower.patch_embed.in_features_padded
        im2col = im2col_patches(pixels, patch, pad_to=pad_to)  # [N, 256, 608] fp32
        # On a MeshDevice every persistent input is replicated (identical on all chips); the
        # multi-device HOST tensors are built here so copy_host_to_device_tensor / to_device can
        # write all shards. Single chip: plain host tensors (mesh_mapper=None), as before.
        mapper = self._host_mapper()
        if self.fused_cfg.siglip_batched:
            im2col_hosts = [
                ttnn.from_torch(im2col, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)
            ]
        else:
            im2col_hosts = [
                ttnn.from_torch(
                    im2col[i : i + 1].contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mapper,
                )
                for i in range(im2col.shape[0])
            ]
        tokens_host = ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mapper)
        noise_t = self._default_noise_torch if noise is None else noise
        if isinstance(noise_t, ttnn.Tensor):
            raise TypeError("sample_actions_fused needs the noise as a torch tensor")
        noise_t = noise_t.reshape(-1, self.config.action_horizon, self.config.action_dim).float()
        if noise_t.shape[0] == 1 and batch > 1:
            noise_t = noise_t.expand(batch, -1, -1).contiguous()
        if noise_t.shape[0] != batch:
            raise ValueError(f"noise batch {noise_t.shape[0]} != {batch} requests")
        noise_host = ttnn.from_torch(
            pad_rows(noise_t, self._suffix_rows), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
        )
        return im2col_hosts, tokens_host, noise_host, valid

    def _host_mapper(self):
        return ttnn.ReplicateTensorToMesh(self.device) if _is_mesh(self.device) else None

    # Persistent attention inputs: name -> memory config. The masks and the expert RoPE rows depend on the
    # prompt length, so they are trace inputs written before a replay (only when the prefix validity changed).
    _ATTN_INPUTS_FUSED = {"vlm_mask": "dram", "exp_mask": "dram", "cosq": "l1", "sinq": "l1", "cosk": "l1", "sink": "l1"}
    _ATTN_INPUTS_TTNN = {"vlm_mask": "dram", "sdpa_mask": "dram", "cos": "l1", "sin": "l1"}

    def _attn_input_names(self) -> Dict[str, str]:
        if getattr(self, "megakernel_backend", "off") == "expert":
            return self._ATTN_INPUTS_FUSED  # the megakernel reads exp_mask and the four q / k RoPE tables
        fused_attn = self.backbone.expert_blocks[0].attention._fused_attn is not None
        return self._ATTN_INPUTS_FUSED if fused_attn else self._ATTN_INPUTS_TTNN

    def _fused_attn_hosts(self, valid: torch.Tensor) -> Dict[str, ttnn.Tensor]:
        """Prefix validity [B, P] -> HOST ttnn tensors of the attention inputs (``fused_host.attention_inputs``)."""
        if self._expert_rope_host is None:
            bb = self.backbone
            self._expert_rope_host = (ttnn.to_torch(_chip0(bb.expert_cos_meta)).float(),
                                      ttnn.to_torch(_chip0(bb.expert_sin_meta)).float())
        cos, sin = self._expert_rope_host
        att = attention_inputs(valid, self.backbone.kv_cache_plan, cos, sin,
                               1.0 / float(self.config.expert_config.head_dim) ** 0.5)
        names = self._attn_input_names()
        if "cos" in names:  # ttnn rotary_embedding takes one [1, 1, S, dh] table for the whole batch
            if int(att["n_valid"].min()) != int(att["n_valid"].max()):
                raise ValueError(
                    "PI05_EXPERT_ATTN=ttnn needs one prompt length per batch (n_valid %s); use the fused attention"
                    % att["n_valid"].tolist())
            att["cos"], att["sin"] = att["cos"][:1], att["sin"][:1]
        mapper = self._host_mapper()
        return {k: ttnn.from_torch(att[k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for k in names}

    def _fused_attn_in(self) -> Dict[str, ttnn.Tensor]:
        """The dict the backbone reads (``forward_expert_fused`` / ``GemmaAttentionTTNN.forward_fused_expert``)."""
        d = dict(self._fused_attn_dev)
        if "cosq" in d:
            d["tables"] = (d["cosq"], d["sinq"], d["cosk"], d["sink"])
        return d

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
        for t in [self._fused_in_tokens, self._fused_in_noise] + list(self._fused_attn_dev.values()):
            if t is not None:
                ttnn.deallocate(t)
        self._fused_in_tokens = None
        self._fused_in_noise = None
        self._fused_attn_dev = {}
        self._fused_attn_key = None
        self._fused_shape_key = None

    def _fused_prepare(self, im2col_hosts, tokens_host, noise_host, valid) -> bool:
        """First call for a shape: allocate the persistent inputs (uploading these host tensors) and
        the KV caches, run the graph once eagerly (kernel compile / program cache, constant slices and
        tables built outside the trace), then capture the trace (PI05_TRACE=1). Returns True when it
        prepared (inputs already hold this call's data), False when nothing had to be done."""
        key = self._shape_key(im2col_hosts, tokens_host, noise_host)
        if self._fused_shape_key is not None and key == self._fused_shape_key:
            return False
        shapes = self.__dict__.setdefault("_fused_shapes", {})
        if self._fused_shape_key is not None:
            # park the active shape (inputs, caches, trace, output) and switch: one prepared entry per shape,
            # e.g. one trace per batch size for a server that batches requests
            shapes[self._fused_shape_key] = dict(
                im2col=self._fused_in_im2col, tokens=self._fused_in_tokens, noise=self._fused_in_noise,
                attn=self._fused_attn_dev, attn_key=self._fused_attn_key,
                trace_id=self._fused_trace_id, out=self._fused_out, plan=self.backbone.kv_cache_plan)
            entry = shapes.get(key)
            if entry is not None:
                self._fused_in_im2col, self._fused_in_tokens, self._fused_in_noise = entry["im2col"], entry["tokens"], entry["noise"]
                self._fused_attn_dev, self._fused_attn_key = entry["attn"], entry["attn_key"]
                self._fused_trace_id, self._fused_out, self._fused_shape_key = entry["trace_id"], entry["out"], key
                plan = entry["plan"]
                self.backbone.allocate_kv_caches(plan["prefix_len"], self.config.action_horizon, batch=plan["batch"])
                return False
            self._fused_in_im2col, self._fused_in_tokens, self._fused_in_noise = [], None, None
            self._fused_attn_dev, self._fused_attn_key = {}, None
            self._fused_trace_id, self._fused_out, self._fused_shape_key = None, None, None

        batch = key[1][0]
        num_images = sum(k[0] for k in key[0]) // batch  # cameras per request
        token_len = key[1][-1]
        plan = check_fused_shape_contract(num_images, token_len, self.config.action_horizon, batch=batch)
        if self.megakernel_backend == "expert":
            self._megakernel_for(plan["prefix_len"], batch)  # refusals (shape, batch) before any allocation

        device = self.device
        self._fused_in_im2col = [ttnn.to_device(t, device, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in im2col_hosts]
        self._fused_in_tokens = ttnn.to_device(tokens_host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._fused_in_noise = ttnn.to_device(noise_host, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.backbone.allocate_kv_caches(plan["prefix_len"], self.config.action_horizon, batch=batch)
        mem = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}
        names = self._attn_input_names()
        self._fused_attn_dev = {
            k: ttnn.to_device(t, device, memory_config=mem[names[k]]) for k, t in self._fused_attn_hosts(valid).items()
        }
        self._fused_attn_key = valid.numpy().tobytes()
        self._fused_shape_key = key

        # Compile pass (eager): program cache, cos/sin slices, SigLIP pos table -- all outside the trace
        # Two eager passes with the folded expert: with one pass the capture found a VLM matmul program missing from
        # the program cache ("Cannot load new binaries during trace capture", 1x4 mesh, batch 1, 2026-09-17).
        default_passes = "2" if getattr(self.fused_cfg, "expert_norm_fold", False) else "1"
        for _ in range(max(1, int(os.environ.get("PI05_FUSED_COMPILE_PASSES", default_passes)))):
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
            if self.megakernel_backend == "expert":
                self._mk_l1[key] = self.l1_signature()
                mv = ttnn.get_memory_view(device, ttnn.BufferType.L1)
                self.megakernel_l1 = {"total_per_bank": int(mv.total_bytes_per_bank),
                                      "allocated_per_bank": int(mv.total_bytes_allocated_per_bank),
                                      "largest_free_per_bank": int(mv.largest_contiguous_bytes_free_per_bank),
                                      "cb_union": self._megakernel_for(plan["prefix_len"], batch).cb_union}
        return True

    def _fused_write_inputs(self, im2col_hosts, tokens_host, noise_host, valid) -> None:
        """Per request: host -> the persistent device buffers (the only host->device traffic). The attention inputs
        are rewritten only when the prefix validity (the prompt length) differs from the last call's."""
        for host, dev in zip(im2col_hosts, self._fused_in_im2col):
            ttnn.copy_host_to_device_tensor(host, dev, cq_id=0)
        ttnn.copy_host_to_device_tensor(tokens_host, self._fused_in_tokens, cq_id=0)
        ttnn.copy_host_to_device_tensor(noise_host, self._fused_in_noise, cq_id=0)
        attn_key = valid.numpy().tobytes()
        if attn_key != self._fused_attn_key:
            for k, host in self._fused_attn_hosts(valid).items():
                ttnn.copy_host_to_device_tensor(host, self._fused_attn_dev[k], cq_id=0)
            self._fused_attn_key = attn_key

    def _fused_device_graph(self) -> ttnn.Tensor:
        """The whole device graph, reading only the persistent inputs; returns x_T [B, round_up(H), 32] bf16 (L1).
        Captured as-is into the trace, so nothing in here may touch the host."""
        attn_in = self._fused_attn_in()
        prefix_embs = self.prefix_embedding.embed_prefix_fused(self._fused_in_im2col, self._fused_in_tokens)
        self.backbone.forward_vlm_fused(prefix_embs, attn_in["vlm_mask"])  # consumes prefix_embs, fills the KV caches

        if self.megakernel_backend == "expert":
            mk = self._megakernel_for(self.backbone.kv_cache_plan["prefix_len"], self.backbone.kv_cache_plan["batch"])
            return mk.run(self.backbone.kv_caches, attn_in["exp_mask"], attn_in["tables"], self._fused_in_noise)

        num_steps = self.denoise_config.num_steps
        dts = euler_dts(num_steps)
        x_t = self._fused_in_noise
        for i in range(num_steps):
            suffix_embs = self.suffix_embedding.embed_actions_fused(x_t)  # [B, S, width] bf16 DRAM
            expert_out = self.backbone.forward_expert_fused(
                suffix_embs, self._precomputed_block_mods[i], self._precomputed_final_mod[i], attn_in, step=i
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
        lang_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Whole-graph fused (and traced) inference. torch in -> torch ``[B, action_horizon, action_dim]``
        float32 out. ``lang_masks`` [B, L] marks the real (right-padded) prompt tokens; None -> ``tokens != 0``.
        First call for a shape: allocate + compile + capture (the server's warm-up)."""
        if self._precomputed_block_mods is None or self._precomputed_final_mod is None:
            raise RuntimeError("fused path needs the precomputed pi0.5 adaRMS modulations")
        im2col_hosts, tokens_host, noise_host, valid = self._fused_host_inputs(images, lang_tokens, noise, lang_masks)
        prepared = self._fused_prepare(im2col_hosts, tokens_host, noise_host, valid)
        if not prepared:
            self._fused_write_inputs(im2col_hosts, tokens_host, noise_host, valid)

        if self._fused_trace_id is not None:
            self._check_l1_guard()
            ttnn.execute_trace(self.device, self._fused_trace_id, cq_id=0, blocking=True)
            actions = ttnn.to_torch(_chip0(self._fused_out))  # expert replicated: every chip holds x_T
        else:
            out = self._fused_device_graph()
            actions = ttnn.to_torch(_chip0(out))
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
