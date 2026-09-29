# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Prefix Embedding module - TTNN Implementation

This module embeds the camera images and language tokens into the prefix part of the sequence
for transformer processing.

Components:
    - Image embedding via the SigLIP vision tower (host im2col input) + projector
    - Language token embedding via Gemma embeddings (scaled by sqrt(width))
    - Concatenation of image and language embeddings

Attention Pattern:
    - All prefix tokens can attend to each other (bidirectional)
    - Suffix tokens can attend to prefix (cross-attention)

Fused graph (``embed_prefix_fused``): only ``prefix_embs`` is produced. The attention masks are
persistent trace inputs built on the host (``fused_host.attention_inputs``); no host->device write
happens inside the device graph (not allowed in a captured trace). The language embedding arrives in
TILE layout, so the sqrt(width) scale and the concat run on tiles (no ROW_MAJOR round trips).
"""

import math
from typing import List

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
        embed_images_fused_fn=None,
        embed_language_fused_fn=None,
    ):
        """
        Initialize prefix embedding with TTNN.

        Args:
            config: Prefix configuration
            device: TTNN device
            embed_images_fused_fn: host-im2col device tensor [B, 256, 608] -> [1, B*256, D]
            embed_language_fused_fn: token ids -> [1, L, D] TILE embeddings
        """
        self.config = config
        self.device = device
        self.embed_images_fused_fn = embed_images_fused_fn
        self.embed_language_fused_fn = embed_language_fused_fn

    # ------------------------------------------------------------------ fused graph

    def embed_prefix_fused(self, im2col_inputs: List[ttnn.Tensor], lang_tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``prefix_embs [1, sum(B_i)*256 + L, D]`` (bf16, L1) from the persistent device inputs: one
        im2col tensor per SigLIP pass (one ``[B, 256, 608]`` when the cameras are batched, else one
        ``[1, 256, 608]`` per camera in camera order) and the ``[1, L]`` uint32 token ids. No masks."""
        if self.embed_images_fused_fn is None or self.embed_language_fused_fn is None:
            raise RuntimeError("fused embedding functions not set")
        batch = int(lang_tokens.shape[0])  # requests sharing the trace
        embs = [self.embed_images_fused_fn(x, batch) for x in im2col_inputs]  # each [B, n*256, D]

        lang_emb = self.embed_language_fused_fn(lang_tokens)  # [B, L, D] TILE
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
