# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side (torch-only) pieces of the fused pi0.5 device graph and the exact reformulations it
relies on. No ttnn import: ``tests/test_fused_host.py`` proves every function here against the
reference math on the CPU, and the ttnn modules call the same functions at build / request time.

Contents
    im2col_patches            host unfold of the camera images in the order the device unfold produced
                              (patch raster (ph, pw), features (kh, kw, c)) -> the trace's persistent input
    device_unfold_reference   torch emulation of the legacy on-device permute/reshape chain (for the test)
    conv_weight_to_patch_linear  the legacy patch-embedding weight prep ([588 -> 608 zero rows, 1152])
    positional_table          SigLIP positional embedding gathered by arange and repeated over the batch
    pad_rows / unpad_rows     50 -> 64 row suffix (zero rows) and the host-side [:50]
    euler_dts                 the dt sequence of the denoising loop (same arithmetic as the legacy loop)
    dit_reference             a1 + scalar * (act @ W + bias) * a2 == ttnn.experimental.dit_minimal_matmul_addcmul_fused
    kv_cache_plan             cache geometry + the rotary_embedding_to_cache / fill_cache tile constraints
    persistent_input_specs    shapes / dtypes / layouts of the trace inputs
    prefix_concat_batched     [B,256,D] image tokens -> [1, B*256, D] view (== legacy per-image concat order)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

TILE = 32


def round_up(x: int, m: int = TILE) -> int:
    return ((x + m - 1) // m) * m


# ----------------------------------------------------------------------------------------
# SigLIP patch embedding
# ----------------------------------------------------------------------------------------


def im2col_patches(images: torch.Tensor, patch: int, pad_to: Optional[int] = None) -> torch.Tensor:
    """``[B, C, H, W]`` -> ``[B, (H/p)*(W/p), C*p*p]`` (optionally zero-padded to ``pad_to`` features).

    Patch order: raster over (ph, pw). Feature order: (kh, kw, c) -- exactly what the legacy device
    path produces (``permute(0,2,3,1)`` -> ``reshape(B,ph,p,pw,p,C)`` -> ``permute(0,1,3,2,4,5)`` ->
    ``reshape(B, P, p*p*C)``), so the same ``[608, 1152]`` linear weight applies. A pure permutation
    of the pixel values: exact, and it commutes with the bf16 rounding of the upload.
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


def device_unfold_reference(images: torch.Tensor, patch: int) -> torch.Tensor:
    """Torch transcript of ``PatchEmbeddingTTNN.forward`` + ``_unfold_conv2d`` (legacy device chain)."""
    b, c, h, w = images.shape
    ph, pw = h // patch, w // patch
    x = images.permute(0, 2, 3, 1)  # NHWC
    x = x.reshape(b, ph, patch, pw, patch, c)
    x = x.permute(0, 1, 3, 2, 4, 5)
    return x.reshape(b, ph * pw, patch * patch * c)


def conv_weight_to_patch_linear(conv_weight: torch.Tensor, pad_to: int) -> torch.Tensor:
    """Conv ``[O, C, p, p]`` -> linear ``[pad_to, O]`` in (kh, kw, c) feature order with zero rows
    588..pad_to-1 (the legacy ``PatchEmbeddingTTNN.__init__`` prep, done in torch)."""
    o = conv_weight.shape[0]
    w = conv_weight.permute(0, 2, 3, 1).reshape(o, -1).T.contiguous()  # [C*p*p, O]
    if pad_to < w.shape[0]:
        raise ValueError("pad_to smaller than the number of features")
    if pad_to > w.shape[0]:
        w = torch.nn.functional.pad(w, (0, 0, 0, pad_to - w.shape[0]))
    return w.contiguous()


def positional_table(pos_emb: torch.Tensor, batch: int) -> torch.Tensor:
    """``embedding(arange(N), pos_emb)`` is the identity gather -> the table itself, repeated over the
    batch so the add is same-shape on device (no batch broadcast needed). Exact."""
    if pos_emb.dim() != 2:
        raise ValueError("pos_emb must be [num_patches, hidden]")
    return pos_emb.unsqueeze(0).expand(batch, -1, -1).contiguous()


def prefix_concat_batched(image_tokens: torch.Tensor) -> torch.Tensor:
    """``[B, P, D]`` (one row-block per camera) -> ``[1, B*P, D]``: the same token order as the legacy
    ``concat([img0, img1, ...], dim=1)``; on device this is a tile-aligned reshape (a view)."""
    b, p, d = image_tokens.shape
    return image_tokens.reshape(1, b * p, d)


# ----------------------------------------------------------------------------------------
# 64-row suffix / Euler step
# ----------------------------------------------------------------------------------------


def pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """``[B, R, D]`` -> ``[B, rows, D]`` with zero rows appended (rows >= R)."""
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


def euler_dts(num_steps: int) -> List[float]:
    """The legacy loop's ``dt`` values: ``timesteps[i+1] - timesteps[i]`` with
    ``timesteps = [1.0 - i / num_steps]`` (python floats -> the same fp32 scalar attribute)."""
    ts = [1.0 - i / num_steps for i in range(num_steps + 1)]
    return [ts[i + 1] - ts[i] for i in range(num_steps)]


def dit_reference(
    act: torch.Tensor,
    weight_in_out: torch.Tensor,
    scalar: float,
    a1: torch.Tensor,
    a2: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Math of ``ttnn.experimental.dit_minimal_matmul_addcmul_fused(act, W, scalar, a1, a2, bias)``:
    ``a1 + scalar * (act @ W + bias) * a2`` with ``W`` in ``[K, N]`` (ttnn linear layout) and
    ``a2`` broadcast over rows (``[1, N]``) or full ``[M, N]``."""
    y = act @ weight_in_out
    if bias is not None:
        y = y + bias
    return a1 + scalar * y * a2


def euler_step_reference(h: torch.Tensor, w_out: torch.Tensor, b_out: torch.Tensor, x_t: torch.Tensor, dt: float):
    """Legacy: ``velocity = h @ W + b; x = x_t + dt * velocity``."""
    return x_t + dt * (h @ w_out + b_out)


# ----------------------------------------------------------------------------------------
# KV cache geometry (backbone-owned caches filled by the VLM, expert writes rows prefix_len..)
# ----------------------------------------------------------------------------------------


def kv_cache_plan(prefix_len: int, action_horizon: int, tile: int = TILE) -> Dict[str, int]:
    """Cache ``[1, 1, logical_len, head_dim]`` with ``logical_len = prefix_len + action_horizon``
    (SDPA masks keys >= logical_len exactly as the legacy concat buffer did), padded to a tile
    multiple. The expert writes ``suffix_rows = round_up(action_horizon)`` rows at
    ``update_idx = prefix_len``. Raises when the fused ops' tile constraints do not hold."""
    if prefix_len <= 0 or action_horizon <= 0:
        raise ValueError("prefix_len and action_horizon must be positive")
    logical_len = prefix_len + action_horizon
    padded_len = round_up(logical_len, tile)
    suffix_rows = round_up(action_horizon, tile)
    if prefix_len % tile != 0:
        raise ValueError(
            f"prefix_len={prefix_len} must be a multiple of {tile}: rotary_embedding_to_cache / fill_cache need "
            "update_idx % 32 == 0 (num_images*256 + PI05_TOKEN_LEN with PI05_TOKEN_LEN % 32 == 0)"
        )
    if prefix_len + suffix_rows > padded_len:
        raise ValueError(
            f"prefix_len + suffix_rows = {prefix_len + suffix_rows} > padded cache {padded_len}: the expert's tile-padded "
            "suffix would not fit (fill_cache: update_idx + input height <= cache height)"
        )
    return {
        "prefix_len": prefix_len,
        "action_horizon": action_horizon,
        "logical_len": logical_len,
        "padded_len": padded_len,
        "suffix_rows": suffix_rows,
        "expert_update_idx": prefix_len,
        "vlm_update_idx": 0,
    }


def build_kv_cache_reference(
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    suffix_k: torch.Tensor,
    suffix_v: torch.Tensor,
    plan: Dict[str, int],
) -> tuple:
    """Torch emulation of the two writes (VLM at row 0, expert at row prefix_len) into zero-initialised
    padded caches; returns the caches. ``suffix_k/v`` are the tile-padded (``suffix_rows``) tensors."""
    b, h, _, d = prefix_k.shape
    ck = torch.zeros(b, h, plan["padded_len"], d, dtype=prefix_k.dtype)
    cv = torch.zeros_like(ck)
    p = plan["prefix_len"]
    ck[:, :, :p] = prefix_k
    cv[:, :, :p] = prefix_v
    ck[:, :, p : p + plan["suffix_rows"]] = suffix_k
    cv[:, :, p : p + plan["suffix_rows"]] = suffix_v
    return ck, cv


# ----------------------------------------------------------------------------------------
# Trace inputs
# ----------------------------------------------------------------------------------------


def persistent_input_specs(
    num_images: int,
    token_len: int,
    action_horizon: int,
    action_dim: int,
    image_size: int = 224,
    patch: int = 14,
    channels: int = 3,
) -> Dict[str, dict]:
    """Shapes / dtypes / layouts of the persistent device inputs of the fused graph. The host tensor
    written per request must match these exactly (``copy_host_to_device_tensor`` rule)."""
    num_patches = (image_size // patch) ** 2
    feats = channels * patch * patch
    return {
        "im2col": {
            "shape": (num_images, num_patches, round_up(feats)),
            "dtype": "bfloat16",
            "layout": "ROW_MAJOR",
            "memory": "DRAM",
        },
        "tokens": {"shape": (1, token_len), "dtype": "uint32", "layout": "ROW_MAJOR", "memory": "DRAM"},
        "noise": {
            "shape": (1, round_up(action_horizon), action_dim),
            "dtype": "bfloat16",
            "layout": "TILE",
            "memory": "L1",
        },
        "prefix_len": num_images * num_patches + token_len,
    }


def check_fused_shape_contract(num_images: int, token_len: int, action_horizon: int) -> Dict[str, int]:
    """Everything the fused graph assumes about the serving shape, as plain arithmetic."""
    if token_len <= 0 or token_len % TILE != 0:
        raise ValueError(f"PI05_TOKEN_LEN={token_len} must be a positive multiple of 32 for the fused graph")
    if num_images < 1:
        raise ValueError("at least one image")
    prefix_len = num_images * 256 + token_len
    return kv_cache_plan(prefix_len, action_horizon)


__all__ = [
    "TILE",
    "round_up",
    "im2col_patches",
    "device_unfold_reference",
    "conv_weight_to_patch_linear",
    "positional_table",
    "prefix_concat_batched",
    "pad_rows",
    "unpad_rows",
    "euler_dts",
    "dit_reference",
    "euler_step_reference",
    "kv_cache_plan",
    "build_kv_cache_reference",
    "persistent_input_specs",
    "check_fused_shape_contract",
]
