# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side (torch only) inputs of the pi0.5 megakernel (``tt/ttnn_pi05_model.py``).

    im2col_patches        camera images -> SigLIP patch rows (the patch-embedding matmul's input)
    pad_rows / unpad_rows tile-padded action rows (H -> round_up(H)) and back
    kv_cache_plan         geometry of the 18 K / V caches the prefix engine fills
    prefix_valid_mask     [B, P] validity of the prefix tokens (cameras, then the right-padded prompt)
    attention_inputs      openpi's pi0.5 attention for the action tokens: the additive key row over
                          [prefix | suffix] keys and the RoPE rows at positions n_valid + [0, S)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

TILE = 32

# Additive bias of a masked expert key. -30000 is exact in bf16 and exp(-30000 - rowmax) is 0; every row keeps a
# valid key.
MASK_NEG = -30000.0


def round_up(x: int, m: int = TILE) -> int:
    return ((x + m - 1) // m) * m


def im2col_patches(images: torch.Tensor, patch: int, pad_to: Optional[int] = None) -> torch.Tensor:
    """``[B, C, H, W]`` -> ``[B, (H/p)*(W/p), C*p*p]``, optionally zero-padded to ``pad_to`` features.

    Patch order: raster over (ph, pw). Feature order: (kh, kw, c), the order of the patch-embedding weight
    ``conv.permute(0, 2, 3, 1).reshape(O, -1)``. A pure permutation of the pixel values, so it commutes with the bf16
    rounding of the upload.
    """
    if images.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(images.shape)}")
    b, c, h, w = images.shape
    if h % patch or w % patch:
        raise ValueError(f"image {h}x{w} is not a multiple of the patch size {patch}")
    ph, pw = h // patch, w // patch
    x = images.reshape(b, c, ph, patch, pw, patch)  # b c ph kh pw kw
    x = x.permute(0, 2, 4, 3, 5, 1)  # b ph pw kh kw c
    x = x.reshape(b, ph * pw, patch * patch * c)
    if pad_to is not None:
        if pad_to < x.shape[-1]:
            raise ValueError(f"pad_to={pad_to} < {x.shape[-1]} features")
        if pad_to > x.shape[-1]:
            x = torch.nn.functional.pad(x, (0, pad_to - x.shape[-1]))
    return x.contiguous()


def pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """``[B, R, D]`` -> ``[B, rows, D]`` with zero rows appended."""
    if x.dim() != 3:
        raise ValueError("expected [B, R, D]")
    r = x.shape[1]
    if rows < r:
        raise ValueError(f"rows={rows} < {r}")
    if rows == r:
        return x
    return torch.nn.functional.pad(x, (0, 0, 0, rows - r))


def unpad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    return x[:, :rows]


def kv_cache_plan(prefix_len: int, action_horizon: int, batch: int = 1) -> Dict[str, int]:
    """K / V caches ``[batch, 1, cache_len, head_dim]``: the prefix engine writes rows ``0..prefix_len-1``. Which keys
    an action query sees is decided by the additive key row (``attention_inputs``), not by the cache length.
    ``cache_len`` covers the prefix and the tile-padded suffix, and the prefix rounded up to 4-tile chunks."""
    if prefix_len <= 0 or action_horizon <= 0:
        raise ValueError("prefix_len and action_horizon must be positive")
    if prefix_len % TILE != 0:
        raise ValueError(f"prefix_len={prefix_len} must be a multiple of {TILE}")
    suffix_rows = round_up(action_horizon)
    chunked_prefix = -(-(prefix_len // TILE) // 4) * 4 * TILE
    return {
        "batch": batch,
        "prefix_len": prefix_len,
        "action_horizon": action_horizon,
        "suffix_rows": suffix_rows,
        "cache_len": round_up(max(prefix_len + suffix_rows, chunked_prefix)),
    }


def prefix_valid_mask(
    num_images: int, lang_masks: torch.Tensor, img_masks: Optional[torch.Tensor] = None, tokens_per_image: int = 256
) -> torch.Tensor:
    """``[B, num_images*256 + L]`` bool: camera tokens (``img_masks [B, num_images]``, default all valid), then the
    language tokens (``lang_masks [B, L]``). openpi's ``prefix_pad_masks``."""
    lang = lang_masks.reshape(lang_masks.shape[0], -1).bool()
    b = lang.shape[0]
    img = torch.ones(b, num_images, dtype=torch.bool) if img_masks is None else img_masks.reshape(b, num_images).bool()
    return torch.cat([img.repeat_interleave(tokens_per_image, dim=1), lang], dim=1)


def attention_inputs(
    valid: torch.Tensor,
    plan: Dict[str, int],
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
    prefix_keys: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """openpi pi0.5 attention of the action tokens, as host tensors (fp32; the model rounds them to bf16).

    ``valid [B, P]`` = ``prefix_valid_mask``; ``cos`` / ``sin`` = the expert RoPE tables ``[>= n_valid + S, dh]`` in
    the split-half layout (``tt.ttnn_gemma.precompute_freqs_cis_meta_format``). The action tokens attend the valid
    prefix keys and the ``action_horizon`` real action rows (not the tile-padding rows), and are rotated at positions
    ``n_valid + [0, S)``. Returns:

    ``exp_mask``  [B, 1, 32, P' + S]  additive key row (prefix keys, then suffix keys), 32 identical rows; with
                  ``prefix_keys`` = P' > P the prefix part ends with P' - P masked pad keys (the expert's chunking)
    ``cosq`` / ``sinq`` / ``cosk`` / ``sink``  [B, 1, S, dh]  RoPE rows; the rotate-half sign is folded into the sin
                  tables and ``scale`` = 1/sqrt(dh) into the q tables
    ``n_valid``   [B] long
    """
    valid = valid.bool()
    b, p = valid.shape
    if p != plan["prefix_len"]:
        raise ValueError(f"valid mask has {p} prefix tokens, the plan {plan['prefix_len']}")
    h, s = plan["action_horizon"], plan["suffix_rows"]
    key = torch.where(valid, 0.0, MASK_NEG)  # [B, P]
    if prefix_keys is not None:
        if prefix_keys < p:
            raise ValueError(f"prefix_keys={prefix_keys} < {p} prefix tokens")
        key = torch.cat([key, torch.full((b, prefix_keys - p), MASK_NEG)], dim=1)
    suffix_key = torch.full((b, s), MASK_NEG)
    suffix_key[:, :h] = 0.0
    n_valid = valid.long().sum(dim=1)
    if int(n_valid.min()) < 1:
        raise ValueError("every request needs at least one valid prefix token")
    if int(n_valid.max()) + s > cos.shape[-2]:
        raise ValueError(f"RoPE table has {cos.shape[-2]} rows < n_valid + {s}")
    rows = n_valid[:, None] + torch.arange(s)[None, :]  # [B, S]
    c = cos.reshape(-1, cos.shape[-1]).float()[rows]  # [B, S, dh]
    sn = sin.reshape(-1, sin.shape[-1]).float()[rows]
    half = c.shape[-1] // 2
    s_signed = torch.cat([-sn[..., :half], sn[..., half:]], dim=-1)  # rotate_half(x) = (-x2, x1)
    exp_key = torch.cat([key, suffix_key], dim=1)
    return {
        "exp_mask": exp_key[:, None, None, :].expand(b, 1, TILE, exp_key.shape[1]).contiguous(),
        "cosq": (c * scale)[:, None].contiguous(),
        "sinq": (s_signed * scale)[:, None].contiguous(),
        "cosk": c[:, None].contiguous(),
        "sink": s_signed[:, None].contiguous(),
        "n_valid": n_valid,
    }
