# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Embedding module - TTNN Implementation

This module handles embedding of images and language tokens to create the
prefix part of the sequence for transformer processing.

Components:
    - Image embedding via SigLIP vision tower
    - Language token embedding via Gemma embeddings
    - Concatenation of image and language embeddings with proper masking

Attention Pattern:
    - All prefix tokens can attend to each other (bidirectional)
    - Suffix tokens can attend to prefix (cross-attention)

Fused graph (``TT_FUSED=1``, ``embed_prefix_fused``): only ``prefix_embs`` is produced. The legacy
image / language pad masks and the zero attention mask are never consumed downstream (every SDPA
runs with ``attn_mask=None``), and the per-image ``ttnn.from_torch(mask, device=...)`` uploads they
needed are host->device writes INSIDE the device graph -- not allowed in a captured trace. Dropping
them is exact. The language embedding arrives in TILE layout, so the sqrt(width) scale and the
concat run on tiles (no ROW_MAJOR round trips).
"""

import math
from typing import List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PrefixConfig


class PrefixEmbeddingTTNN:
    """
    TTNN implementation of prefix embedding.

    Uses TTNN operations for efficient execution on Tenstorrent hardware.
    """

    def __init__(
        self,
        config: PrefixConfig,
        device: ttnn.Device,
        embed_image_fn=None,
        embed_language_fn=None,
        embed_images_fused_fn=None,
        embed_language_fused_fn=None,
    ):
        """
        Initialize prefix embedding with TTNN.

        Args:
            config: Prefix configuration
            device: TTNN device
            embed_image_fn: Function to embed images
            embed_language_fn: Function to embed language tokens
            embed_images_fused_fn: fused graph: host-im2col device tensor [B, 256, 608] -> [1, B*256, D]
            embed_language_fused_fn: fused graph: token ids -> [1, L, D] TILE embeddings
        """
        self.config = config
        self.device = device
        self.embed_image_fn = embed_image_fn
        self.embed_language_fn = embed_language_fn
        self.embed_images_fused_fn = embed_images_fused_fn
        self.embed_language_fused_fn = embed_language_fused_fn

        # Zero attention mask, built lazily to match the actual concatenated prefix shape
        # (batch, seq_len) instead of a hardcoded 2-image/32-token/batch-1 contract. Cached and
        # reused while the shape is stable (the LIBERO contract yields (1, 544) = 2*256 + 32).
        self.prefix_att_masks = None

    def embed_images(
        self,
        images: List[torch.Tensor],
        img_masks: List[torch.Tensor],
    ) -> Tuple[List[ttnn.Tensor], List[ttnn.Tensor]]:
        """
        Embed multiple images using TTNN.

        Args:
            images: List of PyTorch image tensors (vision tower handles TTNN conversion)
            img_masks: List of PyTorch mask tensors

        Returns:
            Tuple of (image_embeddings, expanded_masks) as TTNN tensors
        """
        if self.embed_image_fn is None:
            raise RuntimeError("embed_image_fn not set")

        image_embs = []
        expanded_masks = []

        for img, mask in zip(images, img_masks):
            # embed_image_fn handles PyTorch->TTNN conversion internally
            img_emb = self.embed_image_fn(img)
            image_embs.append(img_emb)

            # Expand mask - convert from PyTorch if needed
            shape = img_emb.shape
            batch_size, num_tokens = shape[0], shape[1]

            if isinstance(mask, torch.Tensor):
                # Convert PyTorch mask to TTNN, reshape on device (no torch.unsqueeze)
                mask_ttnn = ttnn.from_torch(
                    mask.float(),  # (batch_size,)
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                # Reshape to 2D and convert to TILE on device
                mask_ttnn = ttnn.reshape(mask_ttnn, (batch_size, 1))
                mask_ttnn = ttnn.to_layout(mask_ttnn, ttnn.TILE_LAYOUT)
                # Expand on device using ttnn.repeat (no round-trip!)
                expanded_mask = ttnn.repeat(mask_ttnn, (1, num_tokens), memory_config=ttnn.L1_MEMORY_CONFIG)
            else:
                # Already TTNN - expand directly on device (no round-trip!)
                mask_reshaped = ttnn.reshape(mask, (batch_size, 1))
                expanded_mask = ttnn.repeat(mask_reshaped, (1, num_tokens), memory_config=ttnn.L1_MEMORY_CONFIG)

            expanded_masks.append(expanded_mask)

        return image_embs, expanded_masks

    def embed_language(
        self,
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Embed language tokens using TTNN.

        Args:
            lang_tokens: TTNN tensor of token IDs
            lang_masks: TTNN tensor of validity masks

        Returns:
            TTNN tensor of scaled embeddings
        """
        if self.embed_language_fn is None:
            raise RuntimeError("embed_language_fn not set")

        lang_emb = self.embed_language_fn(lang_tokens)

        # Scale by sqrt(hidden_dim) - use scalar multiply
        hidden_dim = lang_emb.shape[-1]
        scale = math.sqrt(hidden_dim)

        return ttnn.mul(lang_emb, scale)

    def embed_prefix(
        self,
        images: List[ttnn.Tensor],
        img_masks: List[ttnn.Tensor],
        lang_tokens: ttnn.Tensor,
        lang_masks: ttnn.Tensor,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """
        Main embedding function for prefix (TTNN version).

        Args:
            images: List of TTNN image tensors
            img_masks: List of TTNN mask tensors
            lang_tokens: TTNN tensor of language tokens
            lang_masks: TTNN tensor of language masks

        Returns:
            Tuple of (prefix_embs, prefix_pad_masks, prefix_att_masks)
        """
        embs = []
        pad_masks = []
        num_tokens_list = []

        # Process images
        if images and self.embed_image_fn is not None:
            image_embs, img_pad_masks = self.embed_images(images, img_masks)
            for img_emb, img_pad_mask in zip(image_embs, img_pad_masks):
                embs.append(img_emb)
                pad_masks.append(img_pad_mask)
                num_tokens_list.append(img_emb.shape[1])

        # Process language
        if self.embed_language_fn is not None:
            lang_emb = self.embed_language(lang_tokens, lang_masks)
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            num_tokens_list.append(lang_emb.shape[1])

        # Concatenate using TTNN
        prefix_embs = ttnn.concat(embs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        prefix_pad_masks = ttnn.concat(pad_masks, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Zero attention mask sized to the actual prefix (batch, seq_len); cached and reused while
        # the shape is stable, so callers with a different image count / language length / batch get
        # a correctly-shaped mask instead of the fixed (1, 544) tensor.
        att_shape = (prefix_embs.shape[0], prefix_embs.shape[1])
        if self.prefix_att_masks is None or tuple(self.prefix_att_masks.shape) != att_shape:
            self.prefix_att_masks = ttnn.zeros(
                att_shape,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        prefix_att_masks = self.prefix_att_masks

        return prefix_embs, prefix_pad_masks, prefix_att_masks

    # ------------------------------------------------------------------ fused graph

    def embed_prefix_fused(self, im2col_inputs: List[ttnn.Tensor], lang_tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``prefix_embs [1, sum(B_i)*256 + L, D]`` (bf16, L1) from the persistent device inputs: one
        im2col tensor per SigLIP pass (one ``[B, 256, 608]`` when the cameras are batched, else one
        ``[1, 256, 608]`` per camera in legacy order) and the ``[1, L]`` uint32 token ids. No masks."""
        if self.embed_images_fused_fn is None or self.embed_language_fused_fn is None:
            raise RuntimeError("fused embedding functions not set")
        embs = [self.embed_images_fused_fn(x) for x in im2col_inputs]

        lang_emb = self.embed_language_fused_fn(lang_tokens)  # [1, L, D] TILE
        scale = math.sqrt(lang_emb.shape[-1])
        lang_scaled = ttnn.mul(lang_emb, scale, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(lang_emb)
        embs.append(lang_scaled)

        prefix_embs = ttnn.concat(embs, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        for t in embs:
            ttnn.deallocate(t)
        return prefix_embs


# Default export
PrefixEmbedding = PrefixEmbeddingTTNN
