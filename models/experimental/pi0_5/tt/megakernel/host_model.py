# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host (torch) model of the megakernel's expert loop: parameters, adaRMS folds, and two reference loops.

* ``expert_params(weights)`` builds every constant the megakernel needs from the checkpoint's torch weights: the
  per-step adaRMS conditioning (sinusoidal time embedding -> time MLP, fp64 / fp32 on the host), the per-(step, layer)
  modulations, the per-layer projections and the action in / out projections.
* ``loop_reference`` is the plain formula (normed = rms(x) * (1 + scale) + shift, q/k/v, RoPE at n_valid + [0, S),
  masked softmax over [prefix | suffix] keys, gated residuals, GeGLU, final adaRMS, Euler).
* ``loop_decomposed`` computes the same thing the way the kernel does (folded weights, r / c epilogues, per-chunk
  flash parts merged with diag weights, 8-way K-split down reduce) so the decomposition itself is checked on the CPU.

Both take the attention inputs the fused graph already builds (``fused_host.attention_inputs``: the key mask row and
the q / k RoPE tables with the rotate-half sign and 1/sqrt(dh) folded in) and the prefix K / V per layer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

WIDTH = 1024
N_LAYERS = 18
N_STEPS = 10
DH = 256
NH = 8
MLP = 4096


def sinusoid(t: float, dim: int = WIDTH, min_period: float = 4e-3, max_period: float = 4.0) -> torch.Tensor:
    """reference/torch_suffix.create_sinusoidal_pos_embedding for one timestep (fp64 on the host, as there)."""
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64)
    period = min_period * (max_period / min_period) ** fraction
    x = (1.0 / period) * 2 * math.pi * torch.tensor([t], dtype=torch.float64)[:, None]
    return torch.cat([torch.sin(x), torch.cos(x)], dim=1).float()[0]


@dataclass
class ExpertParams:
    wqkv: List[torch.Tensor]  # [1024, 2560] input-major (q heads | k | v)
    wo: List[torch.Tensor]  # [2048, 1024]
    wug: List[torch.Tensor]  # [1024, 8192] ([up | gate])
    wd: List[torch.Tensor]  # [4096, 1024]
    mods: List[List[Tuple[torch.Tensor, ...]]]  # [step][layer] (scale_in, shift_in, g_attn, scale_post, shift_post, g_mlp)
    final: List[Tuple[torch.Tensor, torch.Tensor]]  # [step] (scale_f, shift_f)
    w_in: torch.Tensor  # [32, 1024]
    b_in: torch.Tensor  # [1024]
    w_out: torch.Tensor  # [1024, 32]
    b_out: torch.Tensor  # [32]
    eps: float = 1e-6
    dts: Tuple[float, ...] = ()


def expert_params(expert: Dict[str, torch.Tensor], proj: Dict[str, torch.Tensor], eps: float = 1e-6,
                  num_steps: int = N_STEPS) -> ExpertParams:
    """``expert`` = weight_loader.categorized_weights["action_expert"], ``proj`` = ["pi0_projections"]."""
    f = lambda k: expert[k].float()
    wqkv, wo, wug, wd = [], [], [], []
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        wqkv.append(torch.cat([f(p + "self_attn.q_proj.weight").T, f(p + "self_attn.k_proj.weight").T,
                               f(p + "self_attn.v_proj.weight").T], dim=1).contiguous())
        wo.append(f(p + "self_attn.o_proj.weight").T.contiguous())
        wug.append(torch.cat([f(p + "mlp.up_proj.weight"), f(p + "mlp.gate_proj.weight")], dim=0).T.contiguous())
        wd.append(f(p + "mlp.down_proj.weight").T.contiguous())
    ts = [1.0 - i / num_steps for i in range(num_steps + 1)]
    dts = tuple(ts[i + 1] - ts[i] for i in range(num_steps))
    mods, final = [], []
    for s in range(num_steps):
        te = sinusoid(ts[s])
        c = F.silu(F.linear(te, proj["time_mlp_in.weight"].float(), proj["time_mlp_in.bias"].float()))
        cond = F.silu(F.linear(c, proj["time_mlp_out.weight"].float(), proj["time_mlp_out.bias"].float()))
        per = []
        for i in range(N_LAYERS):
            p = f"model.layers.{i}."
            mi = F.linear(cond, f(p + "input_layernorm.dense.weight"), f(p + "input_layernorm.dense.bias"))
            mp = F.linear(cond, f(p + "post_attention_layernorm.dense.weight"), f(p + "post_attention_layernorm.dense.bias"))
            si, hi, gi = mi.chunk(3)
            sp, hp, gp = mp.chunk(3)
            per.append((si, hi, gi, sp, hp, gp))
        mods.append(per)
        mf = F.linear(cond, f("model.norm.dense.weight"), f("model.norm.dense.bias"))
        sf, hf, _ = mf.chunk(3)
        final.append((sf, hf))
    return ExpertParams(
        wqkv=wqkv, wo=wo, wug=wug, wd=wd, mods=mods, final=final,
        w_in=proj["action_in_proj.weight"].float().T.contiguous(), b_in=proj["action_in_proj.bias"].float(),
        w_out=proj["action_out_proj.weight"].float().T.contiguous(), b_out=proj["action_out_proj.bias"].float(),
        eps=eps, dts=dts)


def fold(w: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """norm(x) * (1 + scale) + shift @ w == r * (x @ w') + c with w' = diag(1 + scale) w, c = shift @ w."""
    return w * (1.0 + scale).reshape(-1, 1), shift.reshape(1, -1) @ w


def rope(x: torch.Tensor, cos: torch.Tensor, sin_signed: torch.Tensor) -> torch.Tensor:
    """x [S, 256]: rotate-half RoPE with the sign folded into sin (fused_host.attention_inputs tables)."""
    half = x.shape[-1] // 2
    return x * cos + torch.cat([x[..., half:], x[..., :half]], dim=-1) * sin_signed


def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


@dataclass
class AttnInputs:
    mask: torch.Tensor  # [P + S] additive key bias
    cosq: torch.Tensor  # [S, 256] (1/sqrt(dh) folded)
    sinq: torch.Tensor
    cosk: torch.Tensor
    sink: torch.Tensor


def attn_inputs_from(att: Dict[str, torch.Tensor], b: int = 0) -> AttnInputs:
    """From ``fused_host.attention_inputs(...)`` (request ``b``)."""
    return AttnInputs(mask=att["exp_mask"][b, 0, 0].float(), cosq=att["cosq"][b, 0].float(), sinq=att["sinq"][b, 0].float(),
                      cosk=att["cosk"][b, 0].float(), sink=att["sink"][b, 0].float())


def loop_reference(p: ExpertParams, kv: Sequence[Tuple[torch.Tensor, torch.Tensor]], a: AttnInputs,
                   noise: torch.Tensor, taps: Optional[Dict] = None) -> torch.Tensor:
    """Plain fp32 formulas. ``kv[l] = (K [P, 256] roped, V [P, 256])``; ``noise`` [S, 32]. Returns x_0 [S, 32]."""
    x_t = noise.float().clone()
    for s in range(N_STEPS):
        x = x_t @ p.w_in + p.b_in
        for l in range(N_LAYERS):
            si, hi, gi, sp, hp, gp = p.mods[s][l]
            n = x * rms(x, p.eps) * (1 + si) + hi
            qkv = n @ p.wqkv[l]
            q = qkv[:, : NH * DH].reshape(-1, NH, DH)
            k = qkv[:, NH * DH: NH * DH + DH]
            v = qkv[:, NH * DH + DH:]
            q = torch.stack([rope(q[:, h], a.cosq, a.sinq) for h in range(NH)], dim=1)
            k = rope(k, a.cosk, a.sink)
            kk = torch.cat([kv[l][0].float(), k], dim=0)
            vv = torch.cat([kv[l][1].float(), v], dim=0)
            sc = torch.einsum("shd,kd->hsk", q, kk) + a.mask[None, None, :]
            pr = torch.softmax(sc, dim=-1)
            ctx = torch.einsum("hsk,kd->shd", pr, vv).reshape(-1, NH * DH)
            x = x + gi * (ctx @ p.wo[l])
            n = x * rms(x, p.eps) * (1 + sp) + hp
            ug = n @ p.wug[l]
            h = ug[:, :MLP] * F.gelu(ug[:, MLP:], approximate="tanh")
            x = x + gp * (h @ p.wd[l])
            if taps is not None and (s, l) in taps:
                taps[(s, l)] = x.clone()
        sf, hf = p.final[s]
        n = x * rms(x, p.eps) * (1 + sf) + hf
        v = n @ p.w_out + p.b_out
        x_t = x_t + p.dts[s] * v
    return x_t


def loop_decomposed(p: ExpertParams, kv: Sequence[Tuple[torch.Tensor, torch.Tensor]], a: AttnInputs,
                    noise: torch.Tensor, chunk_tiles: int, taps: Optional[Dict] = None) -> torch.Tensor:
    """The kernel's decomposition in fp32: folds + r / c epilogues, per-chunk flash parts over [prefix | suffix] key
    tiles merged with diag(s_i) weights, the 2-D MLP (h columns kg*16 + 2 ng + {0, 1}, down K-split over 8 row
    groups reduced per owner column), fp32 residual and Euler."""
    S = noise.shape[0]
    P = kv[0][0].shape[0]
    T = 32
    nkt = (P + S) // T
    nch = nkt // chunk_tiles
    x_t = noise.float().clone()
    for s in range(N_STEPS):
        x = x_t @ p.w_in + p.b_in
        for l in range(N_LAYERS):
            si, hi, gi, sp, hp, gp = p.mods[s][l]
            wq, cq = fold(p.wqkv[l], si, hi)
            r_in = rms(x, p.eps)
            qkv = r_in * (x @ wq) + cq
            q = qkv[:, : NH * DH].reshape(-1, NH, DH)
            k = rope(qkv[:, NH * DH: NH * DH + DH], a.cosk, a.sink)
            v = qkv[:, NH * DH + DH:]
            q = torch.stack([rope(q[:, h], a.cosq, a.sinq) for h in range(NH)], dim=1)
            kk = torch.cat([kv[l][0].float(), k], dim=0)
            vv = torch.cat([kv[l][1].float(), v], dim=0)
            ctx = torch.zeros(S, NH, DH)
            for h in range(NH):
                parts = []
                for kc in range(nch):
                    sl = slice(kc * chunk_tiles * T, (kc + 1) * chunk_tiles * T)
                    sc = q[:, h] @ kk[sl].T + a.mask[None, sl]
                    m = sc.max(dim=-1, keepdim=True).values
                    pr = torch.exp(sc - m)
                    parts.append((m, pr.sum(dim=-1, keepdim=True), pr @ vv[sl]))
                M = torch.stack([pt[0] for pt in parts]).max(dim=0).values
                w = [torch.exp(pt[0] - M) for pt in parts]
                L = sum(wi * pt[1] for wi, pt in zip(w, parts))
                ctx[:, h] = sum((wi / L) * pt[2] for wi, pt in zip(w, parts))
            x = x + gi * (ctx.reshape(S, -1) @ p.wo[l])
            wu, cu = fold(p.wug[l], sp, hp)
            r_post = rms(x, p.eps)
            ug = r_post * (x @ wu) + cu
            h = ug[:, :MLP] * F.gelu(ug[:, MLP:], approximate="tanh")
            down = torch.zeros(S, WIDTH)
            for kg in range(8):  # K-split partials summed per owner column (fixed kg order)
                down = down + h[:, kg * 512:(kg + 1) * 512] @ p.wd[l][kg * 512:(kg + 1) * 512]
            x = x + gp * down
            if taps is not None and (s, l) in taps:
                taps[(s, l)] = x.clone()
        sf, hf = p.final[s]
        wout, cout = fold(p.w_out, sf, hf)
        v = rms(x, p.eps) * (x @ wout) + cout + p.b_out
        x_t = x_t + p.dts[s] * v
    return x_t


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    d = (a.norm() * b.norm()).item()
    return float((a @ b).item() / d) if d > 0 else float("nan")
