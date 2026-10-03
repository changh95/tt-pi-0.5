# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Model - PyTorch Reference Implementation.

This module provides the complete PI0 model that orchestrates all components:
    - PrefixEmbedding: Images + language → embeddings
    - SuffixEmbedding: State + actions + timestep → embeddings
    - PaliGemmaBackbone: VLM + Action Expert transformers
    - DenoisingModule: Flow matching for action generation
"""

from typing import List, Optional, Tuple

import torch

from models.experimental.pi0.common.configs import (
    PI0ModelConfig,
    SuffixConfig,
    PrefixConfig,
    PaliGemmaConfig,
    DenoiseConfig,
)
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.reference.torch_suffix import SuffixEmbedding
from models.experimental.pi0.reference.torch_prefix import PrefixEmbedding
from models.experimental.pi0.reference.torch_paligemma import PaliGemmaBackbone
from models.experimental.pi0.reference.torch_denoise import DenoisingModule, KVCacheManager

# Additive bias of a masked key (openpi masks with a large negative logit before the softmax)
MASK_NEG = -1.0e9


def prefix_attention_inputs(prefix_pad_masks: torch.Tensor, action_horizon: int):
    """openpi pi0.5 attention inputs from the prefix validity mask ``[B, P]`` (bool).

    The prefix is bidirectional (every prefix ar_mask is 0), so a prefix query attends every VALID prefix key;
    pad keys (padded language tokens, masked cameras) are hidden. Prefix positions are ``cumsum(valid) - 1``.
    The action tokens (pi0.5: no state token; first action ar_mask 1, the rest 0) attend every valid prefix key and
    every action token, and sit at positions ``n_valid + [0, H)``.

    Returns ``(vlm_mask [B,1,P,P], vlm_pos [B,P], expert_mask [B,1,H,P+H], expert_pos [B,H])``, masks additive fp32.
    """
    valid = prefix_pad_masks.bool()
    b, p = valid.shape
    key_bias = torch.where(valid, 0.0, MASK_NEG).to(torch.float32)  # [B, P]
    vlm_mask = key_bias[:, None, None, :].expand(b, 1, p, p)
    vlm_pos = torch.cumsum(valid.long(), dim=1) - 1
    exp_bias = torch.cat([key_bias, torch.zeros(b, action_horizon, device=valid.device)], dim=1)
    expert_mask = exp_bias[:, None, None, :].expand(b, 1, action_horizon, p + action_horizon)
    n_valid = valid.long().sum(dim=1, keepdim=True)
    expert_pos = n_valid + torch.arange(action_horizon, device=valid.device)[None, :]
    return vlm_mask, vlm_pos.clamp(min=0), expert_mask, expert_pos


class PI0Model:
    """
    Complete PI0 model implementation (PyTorch).

    This class orchestrates all components for inference.
    """

    def __init__(
        self,
        config: PI0ModelConfig,
        weight_loader: PI0WeightLoader,
    ):
        """
        Initialize PI0 model.

        Args:
            config: Model configuration
            weight_loader: Loaded weights
        """
        self.config = config
        self.weight_loader = weight_loader

        # Initialize components (order matters: backbone before prefix)
        self._init_suffix_embedding()
        self._init_backbone()
        self._init_prefix_embedding()
        self._init_denoising()

    def _init_suffix_embedding(self):
        """Initialize suffix embedding module."""
        suffix_config = SuffixConfig(
            action_dim=self.config.action_dim,
            action_horizon=self.config.action_horizon,
            expert_width=self.config.expert_config.width,
            pi05=self.config.pi05,
        )
        pi0_weights = self.weight_loader.get_pi0_projections()
        self.suffix_embedding = SuffixEmbedding(suffix_config, pi0_weights)

    def _init_prefix_embedding(self):
        """Initialize prefix embedding module (after backbone)."""
        prefix_config = PrefixConfig(
            vlm_hidden_size=self.config.vlm_config.width,
        )
        # Create prefix embedding with backbone's embedding functions
        self.prefix_embedding = PrefixEmbedding(
            prefix_config,
            embed_image_fn=self.backbone.embed_image,
            embed_language_fn=self.backbone.embed_language_tokens,
        )

    def _init_backbone(self):
        """Initialize PaliGemma backbone."""
        paligemma_config = PaliGemmaConfig(
            vlm_config=self.config.vlm_config,
            expert_config=self.config.expert_config,
            siglip_config=self.config.siglip_config,
        )
        weights = self.weight_loader.categorized_weights
        self.backbone = PaliGemmaBackbone(paligemma_config, weights)

    def _init_denoising(self):
        """Initialize denoising module."""
        denoise_config = DenoiseConfig(
            num_steps=self.config.num_denoising_steps,
        )
        self.denoising = DenoisingModule(denoise_config, self._denoise_forward)

        # KV cache for inference
        self.kv_cache = KVCacheManager(
            num_layers=self.config.expert_config.depth,
            max_seq_len=self.config.max_seq_len,
            num_kv_heads=self.config.expert_config.num_kv_heads,
            head_dim=self.config.expert_config.head_dim,
        )

    def _denoise_forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        kv_cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        state: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for denoising (predicts velocity)."""
        # Embed suffix
        suffix_embs, _, _, adarms_cond = self.suffix_embedding.embed_suffix(
            state,
            noisy_actions,
            timestep,
        )

        # Forward through expert (pass adarms_cond for Pi0.5)
        expert_output, _ = self.backbone.forward_expert(
            suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=kv_cache,
            adarms_cond=adarms_cond,
        )

        # Project back to action dimension (skip state token if PI0)
        if not self.config.pi05:
            action_output = expert_output[:, 1:, :]
        else:
            action_output = expert_output

        velocity = self.suffix_embedding.project_output(action_output)

        return velocity

    def embed_prefix(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images and language to form prefix."""
        return self.prefix_embedding.embed_prefix(images, img_masks, lang_tokens, lang_masks)

    def embed_suffix(
        self,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed state and actions to form suffix."""
        return self.suffix_embedding.embed_suffix(state, noisy_actions, timestep)

    def forward_inference(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for inference (generates actions via denoising).

        Returns:
            Generated actions (batch_size, action_horizon, action_dim)
        """
        batch_size = state.shape[0]
        device = state.device

        prefix_embs, prefix_pad_masks, _ = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        vlm_kwargs, expert_kwargs = {}, {}
        if self.config.pi05:
            # pi0.5 (openpi): pad keys are masked and the action tokens sit at positions n_valid + [0, H)
            vlm_mask, vlm_pos, expert_mask, expert_pos = prefix_attention_inputs(
                prefix_pad_masks.reshape(batch_size, -1), self.config.action_horizon
            )
            vlm_kwargs = {"attention_mask": vlm_mask.to(prefix_embs.dtype), "position_ids": vlm_pos}
            expert_kwargs = {"attention_mask": expert_mask.to(prefix_embs.dtype), "position_ids": expert_pos}

        # Prefill: process prefix and cache KV
        _, vlm_cache = self.backbone.forward_vlm(prefix_embs, use_cache=True, **vlm_kwargs)

        # Denoise to generate actions
        actions = self.denoising.sample_actions(
            batch_size,
            prefix_kv_cache=vlm_cache,
            device=device,
            state=state,
            **expert_kwargs,
        )

        return actions

    # Alias for compatibility with ttnn_pi0_reference API
    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for forward_inference (sample_actions API)."""
        return self.forward_inference(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
        )
