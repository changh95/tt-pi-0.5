# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the phase-2 prefix engine: parameters from the checkpoint, weight arenas, vector / table tensors, and a
torch model of the kernel's own decomposition (``host_prefix``) for CPU checks.

Torch only (``upload_*`` need ttnn). Page / tile orders mirror ``kernels_p2/pe_ncrisc.hpp`` (weight feeder: page ``g`` of
an op lives in bank ``g % 8`` at ``op_off + (g // 8) * page_bytes``) and ``pe_trisc.hpp`` (a mode-R page is
``[k in piece][t in 2]``; a mode-S page is one K block of one pair, same shape).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from . import pe_geometry as P

T = 32


# ======================================================================================================== parameters
@dataclass
class SigLayer:
    wqkv: torch.Tensor  # [1152, 4608]  [q | k | v], heads padded 72 -> 96, q (and bq) x 1/sqrt(72)
    bqkv: torch.Tensor  # [4608]
    wo: torch.Tensor  # [1536, 1152]  (input rows padded like the heads)
    bo: torch.Tensor
    ln1w: torch.Tensor
    ln1b: torch.Tensor
    ln2w: torch.Tensor
    ln2b: torch.Tensor
    wfc1: torch.Tensor  # [1152, 4352]
    bfc1: torch.Tensor  # [4352]
    wfc2: torch.Tensor  # [4352, 1152]
    bfc2: torch.Tensor


@dataclass
class VlmLayer:
    wqkv: torch.Tensor  # [2048, 2560]  [q | k | v]
    wo: torch.Tensor  # [2048, 2048]
    wug: torch.Tensor  # [2048, 32768] [up | gate]
    wd: torch.Tensor  # [16384, 2048]
    g1: torch.Tensor  # 1 + input_layernorm.weight
    g2: torch.Tensor  # 1 + post_attention_layernorm.weight


@dataclass
class PrefixParams:
    patch_w: torch.Tensor  # [608, 1152]  im2col (h, w, c) rows, zero-padded 588 -> 608
    pos_b: torch.Tensor  # [256, 1152]  position table + patch bias
    sig: List[SigLayer]
    post_w: torch.Tensor
    post_b: torch.Tensor
    proj_w: torch.Tensor  # [1152, 2048]
    proj_b: torch.Tensor
    vlm: List[VlmLayer]
    eps_s: float = 1e-6
    eps_v: float = 1e-6


def _pad_heads_out(w: torch.Tensor, heads: int, dh: int, dhp: int) -> torch.Tensor:
    """[heads * dh, K] (torch Linear weight, out x in) -> [heads * dhp, K] with zero rows at the pad dims."""
    k = w.shape[1]
    out = torch.zeros(heads, dhp, k, dtype=w.dtype)
    out[:, :dh] = w.reshape(heads, dh, k)
    return out.reshape(heads * dhp, k)


def _pad_heads_vec(b: torch.Tensor, heads: int, dh: int, dhp: int) -> torch.Tensor:
    out = torch.zeros(heads, dhp, dtype=b.dtype)
    out[:, :dh] = b.reshape(heads, dh)
    return out.reshape(-1)


def prefix_params(cat: Dict[str, Dict[str, torch.Tensor]], n_sig: int = 27, n_vlm: int = 18,
                  fold: bool = True) -> PrefixParams:
    vis, proj, lang = cat["vlm_vision"], cat["vlm_projector"], cat["vlm_language"]
    f = lambda t: t.detach().float()
    heads, dh, dhp = 16, 72, 96
    scale = 1.0 / math.sqrt(dh)
    conv = f(vis["vision_model.embeddings.patch_embedding.weight"])  # [1152, 3, 14, 14]
    pw = conv.permute(0, 2, 3, 1).reshape(conv.shape[0], -1).T  # [588, 1152] (h, w, c) rows
    patch_w = torch.zeros(P.S_KP * T, pw.shape[1])
    patch_w[: pw.shape[0]] = pw
    pos_b = f(vis["vision_model.embeddings.position_embedding.weight"]) + f(vis["vision_model.embeddings.patch_embedding.bias"])
    sig = []
    for i in range(n_sig):
        g = lambda n: f(vis[f"vision_model.encoder.layers.{i}.{n}"])
        wq = _pad_heads_out(g("self_attn.q_proj.weight"), heads, dh, dhp) * scale
        wk = _pad_heads_out(g("self_attn.k_proj.weight"), heads, dh, dhp)
        wv = _pad_heads_out(g("self_attn.v_proj.weight"), heads, dh, dhp)
        bq = _pad_heads_vec(g("self_attn.q_proj.bias"), heads, dh, dhp) * scale
        bk = _pad_heads_vec(g("self_attn.k_proj.bias"), heads, dh, dhp)
        bv = _pad_heads_vec(g("self_attn.v_proj.bias"), heads, dh, dhp)
        wo = g("self_attn.out_proj.weight")  # [1152 out, 1152 in]
        wo_p = _pad_heads_out(wo.T.contiguous(), heads, dh, dhp)  # in rows padded -> [1536, 1152]
        fc1 = g("mlp.fc1.weight").T  # [1152, 4304]
        wfc1 = torch.zeros(fc1.shape[0], P.S_I * T)
        wfc1[:, : fc1.shape[1]] = fc1
        bfc1 = torch.zeros(P.S_I * T)
        bfc1[: fc1.shape[1]] = g("mlp.fc1.bias")
        fc2 = g("mlp.fc2.weight").T  # [4304, 1152]
        wfc2 = torch.zeros(P.S_I * T, fc2.shape[1])
        wfc2[: fc2.shape[0]] = fc2
        sig.append(SigLayer(
            wqkv=torch.cat([wq, wk, wv], 0).T.contiguous(), bqkv=torch.cat([bq, bk, bv]), wo=wo_p,
            bo=g("self_attn.out_proj.bias"), ln1w=g("layer_norm1.weight"), ln1b=g("layer_norm1.bias"),
            ln2w=g("layer_norm2.weight"), ln2b=g("layer_norm2.bias"), wfc1=wfc1, bfc1=bfc1, wfc2=wfc2,
            bfc2=g("mlp.fc2.bias")))
    vlm = []
    for i in range(n_vlm):
        g = lambda n: f(lang[f"model.layers.{i}.{n}"])
        vlm.append(VlmLayer(
            wqkv=torch.cat([g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")],
                           0).T.contiguous(),
            wo=g("self_attn.o_proj.weight").T.contiguous(),
            wug=torch.cat([g("mlp.up_proj.weight"), g("mlp.gate_proj.weight")], 0).T.contiguous(),
            wd=g("mlp.down_proj.weight").T.contiguous(),
            g1=1.0 + g("input_layernorm.weight"), g2=1.0 + g("post_attention_layernorm.weight")))
    pp = PrefixParams(
        patch_w=patch_w, pos_b=pos_b, sig=sig, post_w=f(vis["vision_model.post_layernorm.weight"]),
        post_b=f(vis["vision_model.post_layernorm.bias"]), proj_w=f(proj["linear.weight"]).T.contiguous(),
        proj_b=f(proj["linear.bias"]), vlm=vlm)
    return fold_norms(pp) if fold else pp


def fold_norms(pp: PrefixParams) -> PrefixParams:
    """Every norm's affine part into the matmul that consumes it (fp32, before quantisation): LN(x) W + b =
    ((x - mu) rstd) (diag(g) W) + (b + beta W), RMS the same with g = 1 + w and no beta. Scaling K row k of W by g_k
    scales whole bfp8 exponent blocks (16 elements along N), so the weight quantisation error is unchanged; the norms
    then leave g = 1, beta = 0 (the kernels apply only (x - mu) rstd / x r)."""
    one = lambda v: torch.ones_like(v)
    zero = lambda v: torch.zeros_like(v)
    sig = []
    for s in pp.sig:
        sig.append(SigLayer(
            wqkv=s.ln1w[:, None] * s.wqkv, bqkv=s.bqkv + s.ln1b @ s.wqkv, wo=s.wo, bo=s.bo,
            ln1w=one(s.ln1w), ln1b=zero(s.ln1b), ln2w=one(s.ln2w), ln2b=zero(s.ln2b),
            wfc1=s.ln2w[:, None] * s.wfc1, bfc1=s.bfc1 + s.ln2b @ s.wfc1, wfc2=s.wfc2, bfc2=s.bfc2))
    vlm = [VlmLayer(wqkv=v.g1[:, None] * v.wqkv, wo=v.wo, wug=v.g2[:, None] * v.wug, wd=v.wd, g1=one(v.g1), g2=one(v.g2))
           for v in pp.vlm]
    return PrefixParams(patch_w=pp.patch_w, pos_b=pp.pos_b, sig=sig, post_w=one(pp.post_w), post_b=zero(pp.post_b),
                        proj_w=pp.post_w[:, None] * pp.proj_w, proj_b=pp.proj_b + pp.post_b @ pp.proj_w, vlm=vlm)


# ======================================================================================================== arenas
def w_tile(w: torch.Tensor, kt: int, nt: int) -> torch.Tensor:
    return w[kt * T:(kt + 1) * T, nt * T:(nt + 1) * T]


def op_pages(o: P.Op, w: torch.Tensor) -> List[torch.Tensor]:
    """The op's weight pages in op-global order (column 0's pages, then column 1's, ...); page = [2 * piece, 32, 32]."""
    nq, pc = P.mm_nq(o), o.piece
    wt = w.reshape(w.shape[0] // T, T, w.shape[1] // T, T).permute(0, 2, 1, 3)  # [Kt, Nt, 32, 32]
    pages = []
    for x in range(P.NCOL):
        p0, npr = P.mm_pair0(o, x), P.mm_pairs(o, x)
        cols = [(P.pair_col(o, p, 0), P.pair_col(o, p, 1)) for p in range(p0, p0 + npr)]
        if o.mode == P.MM_R:
            order = [(c, q) for c in cols for q in range(nq)]
        else:
            order = [(c, q) for q in range(nq) for c in cols]
        for (c0, c1), q in order:
            ks = slice(q * pc, (q + 1) * pc)
            pg = torch.stack([wt[ks, c0], wt[ks, c1]], dim=1)  # [piece, 2, 32, 32]
            pages.append(pg.reshape(2 * pc, T, T))
    assert len(pages) == P.op_pages_total(o)
    return pages


def stripe(ops_w: Sequence[Tuple[P.Op, torch.Tensor]], ps: P.PShape, tile_bytes: int) -> torch.Tensor:
    """Pages of the given ops -> [8 banks, tiles per bank, 32, 32] (the WIDTH_SHARDED arena's per-bank streams)."""
    o_last = ops_w[-1][0]
    bank_bytes = P.mm_arena_off(o_last, ps) + P.op_bank_bytes(o_last)
    assert bank_bytes % tile_bytes == 0
    arena = torch.zeros(P.N_BANKS, bank_bytes // tile_bytes, T, T)
    for o, w in ops_w:
        off = P.mm_arena_off(o, ps)
        pb = P.mm_page_bytes(o)
        assert off % tile_bytes == 0 and pb % tile_bytes == 0 and P.mm_wtile(o) == tile_bytes
        for g, pg in enumerate(op_pages(o, w)):
            t0 = (off + (g // P.N_BANKS) * pb) // tile_bytes
            arena[g % P.N_BANKS, t0:t0 + pg.shape[0]] = pg
    return arena


def sig_layer_arena(sp: SigLayer, layer: int, ps: P.PShape) -> torch.Tensor:
    base = P.OP_S0 + layer * P.OPS_PER_LAYER
    ops = [(P.describe(base + 1, ps), sp.wqkv), (P.describe(base + 3, ps), sp.wo), (P.describe(base + 5, ps), sp.wfc1),
           (P.describe(base + 6, ps), sp.wfc2)]
    return stripe(ops, ps, P.T8)


def vlm_layer_arena(vp: VlmLayer, layer: int, ps: P.PShape) -> torch.Tensor:
    base = P.OP_V0 + layer * P.OPS_PER_LAYER
    ops = [(P.describe(base + 1, ps), vp.wqkv), (P.describe(base + 3, ps), vp.wo), (P.describe(base + 5, ps), vp.wug),
           (P.describe(base + 6, ps), vp.wd)]
    return stripe(ops, ps, P.T8)


def patch_arena(pp: PrefixParams, ps: P.PShape) -> torch.Tensor:
    return stripe([(P.describe(0, ps), pp.patch_w)], ps, P.T16)


def proj_arena(pp: PrefixParams, ps: P.PShape) -> torch.Tensor:
    return stripe([(P.describe(P.OP_PROJ, ps), pp.proj_w)], ps, P.T8)


# ======================================================================================================== vectors / tables
def rowb(v: torch.Tensor) -> torch.Tensor:
    """[N] -> [32, N]: row-broadcast tiles (tile n = 32 copies of v[32n:32n+32])."""
    return v.reshape(1, -1).expand(T, -1).contiguous()


def svec(pp: PrefixParams) -> torch.Tensor:
    rows = []
    for s in pp.sig:
        v = torch.cat([s.ln1w, s.ln1b, s.bqkv, s.bo, s.ln2w, s.ln2b, s.bfc1, s.bfc2])
        assert v.numel() == P.SV_N * T
        rows.append(v)
    return rowb(torch.cat(rows))


def vvec(pp: PrefixParams) -> torch.Tensor:
    return rowb(torch.cat([torch.cat([v.g1, v.g2]) for v in pp.vlm]))


def gvec(pp: PrefixParams) -> torch.Tensor:
    v = torch.cat([pp.post_w, pp.post_b, pp.proj_b])
    assert v.numel() == P.GV_N * T
    return rowb(v)


def consts() -> torch.Tensor:
    ones = torch.ones(T, T)
    return torch.cat([ones, ones / 2048.0, ones / 1152.0, torch.eye(T), torch.zeros(T, T)], dim=1)


def rope_tables(ps: P.PShape, head_dim: int = 256, base: float = 10000.0) -> Dict[str, torch.Tensor]:
    """cos / signed-sin [MT*32, 256] at positions 0..MT*32-1 (split-half rotate; q tables x 1/sqrt(256) = 1/16)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    ang = torch.outer(torch.arange(ps.mt * T, dtype=torch.float32), inv)
    c, s = torch.cos(ang), torch.sin(ang)
    cos = torch.cat([c, c], 1)
    sin = torch.cat([-s, s], 1)
    return {"cosq": cos / 16.0, "sinq": sin / 16.0, "cosk": cos, "sink": sin}


def vlm_key_mask(valid: torch.Tensor, ps: P.PShape) -> torch.Tensor:
    """prefix validity [P] (bool) -> [32, P] row-broadcast additive key bias (0 valid, -1e9 pad)."""
    from models.experimental.pi0_5.reference.torch_pi0_model import MASK_NEG

    v = valid.reshape(-1).bool()
    assert v.numel() == ps.ptv * T
    return rowb(torch.where(v, 0.0, MASK_NEG).float())


# ======================================================================================================== host model
def _ln(x, w, b, eps):
    mu = x.mean(-1, keepdim=True)
    d = x - mu
    return d * torch.rsqrt((d * d).mean(-1, keepdim=True) + eps) * w + b


def _rms(x, g, eps):
    return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * g


def host_siglip(pp: PrefixParams, im2col: torch.Tensor, layers: Optional[int] = None) -> torch.Tensor:
    """[512, 608] -> the post-LN SigLIP output [512, 1152] with the kernel's decomposition (fp32)."""
    x = im2col.float() @ pp.patch_w + pp.pos_b.repeat(2, 1)
    for s in pp.sig[: layers if layers is not None else len(pp.sig)]:
        xn = _ln(x, s.ln1w, s.ln1b, pp.eps_s)
        qkv = xn @ s.wqkv + s.bqkv
        q, k, v = qkv.split(1536, dim=1)
        ctx = torch.zeros(512, 1536)
        for img in range(2):
            rs = slice(img * 256, (img + 1) * 256)
            qh = q[rs].reshape(256, 16, 96).transpose(0, 1)
            kh = k[rs].reshape(256, 16, 96).transpose(0, 1)
            vh = v[rs].reshape(256, 16, 96).transpose(0, 1)
            a = torch.softmax(qh @ kh.transpose(1, 2), dim=-1) @ vh
            ctx[rs] = a.transpose(0, 1).reshape(256, 1536)
        x = x + ctx @ s.wo + s.bo
        xn = _ln(x, s.ln2w, s.ln2b, pp.eps_s)
        x = x + F.gelu(xn @ s.wfc1 + s.bfc1, approximate="tanh") @ s.wfc2 + s.bfc2
    return _ln(x, pp.post_w, pp.post_b, pp.eps_s)


def host_vlm(pp: PrefixParams, x0: torch.Tensor, valid: torch.Tensor, ps: P.PShape,
             layers: Optional[int] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """VLM prefill on x0 [MT*32, 2048] (rows >= P are pad rows) -> (x, [(K, V) [P, 256] per layer]) (fp32)."""
    tab = rope_tables(ps)
    rows = ps.mt * T
    bias = torch.where(valid.reshape(-1).bool(), 0.0, -1.0e9)[None, :]  # [1, P]
    x = x0.float().clone()
    kv = []
    n = layers if layers is not None else len(pp.vlm)

    def rope(t, c, s):
        h = t.shape[-1] // 2
        return t * c + torch.cat([t[..., h:], t[..., :h]], -1) * s

    for li, lp in enumerate(pp.vlm[:n]):
        xn = _rms(x, lp.g1, pp.eps_v)
        qkv = xn @ lp.wqkv
        q = qkv[:, :2048].reshape(rows, 8, 256).transpose(0, 1)
        k = qkv[:, 2048:2304]
        v = qkv[:, 2304:2560]
        q = rope(q, tab["cosq"], tab["sinq"])
        k = rope(k, tab["cosk"], tab["sink"])
        kp, vp = k[: ps.ptv * T], v[: ps.ptv * T]
        kv.append((kp, vp))
        if li == len(pp.vlm) - 1:
            break
        a = torch.softmax(q @ kp.T + bias, dim=-1) @ vp  # [8, rows, 256]
        x = x + a.transpose(0, 1).reshape(rows, 2048) @ lp.wo
        xn = _rms(x, lp.g2, pp.eps_v)
        ug = xn @ lp.wug
        x = x + (ug[:, :16384] * F.gelu(ug[:, 16384:], approximate="tanh")) @ lp.wd
    return x, kv
