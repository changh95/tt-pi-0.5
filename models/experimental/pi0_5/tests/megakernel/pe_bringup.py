# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device bring-up of the phase-2 prefix engine, op by op (the prefix-only program, PE_WHOLE=0).

Every check writes the op's inputs into the engine's DRAM scratch from the host, runs ONE op (or a short range) of
the fixed op sequence, reads the outputs back and compares them with the host decomposition (pe_host) on the device's
exact inputs (PCC and rel-L2 against fp32).

    python models/experimental/pi0_5/tests/megakernel/pe_bringup.py --checks patch,sig0 --out r.json [--reps 20]

ALWAYS through bin/with-device.sh; first runs of a changed kernel with WITH_DEVICE_RESET_AFTER=1 and a timeout.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tt.megakernel import geometry as G
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P
from models.experimental.pi0_5.tt.megakernel import pe_host as H
from models.experimental.pi0_5.tt.megakernel.host_model import ExpertParams
from models.experimental.pi0_5.tt.megakernel.pe_program import PrefixTensors, WholeMegakernel, kernel_digest2
from models.experimental.pi0_5.tt.megakernel.program import ExpertMegakernel


AICLK_MHZ = float(os.environ.get("PI05_AICLK_MHZ", "1350"))


def stamp(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def cmp(name, dev, ref, res):
    r = {"pcc": round(pcc(dev, ref), 7), "rel_l2": round(rel(dev, ref), 6),
         "max_abs": float((dev - ref).abs().max()), "finite": bool(torch.isfinite(dev).all()),
         "ref_absmax": float(ref.abs().max())}
    res[name] = r
    stamp(f"{name}: {r}")
    return r


class Rig:
    def __init__(self, dev, shape_name, pp, embed):
        self.dev, self.sh = dev, G.SHAPES[shape_name]
        self.ps = P.pshape_for(self.sh)
        self.pp = pp
        z = torch.zeros(1)
        params = ExpertParams(wqkv=[], wo=[], wug=[], wd=[], mods=[], final=[], w_in=z, b_in=z, w_out=z, b_out=z,
                              eps=1e-6, dts=tuple([-0.1] * G.N_STEPS))
        tiny = lambda dt: ttnn.from_torch(torch.zeros(32, 256), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.mk = ExpertMegakernel(dev, params, self.sh, arenas=([tiny(ttnn.bfloat8_b)] * G.N_STEPS,
                                                                 [tiny(ttnn.bfloat16)] * G.N_STEPS))
        cache_len = 800 if shape_name == "base" else 640
        l1 = lambda s, dt=ttnn.bfloat16: ttnn.from_torch(torch.zeros(s), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev,
                                                           memory_config=ttnn.L1_MEMORY_CONFIG)
        self.kv = [(l1((1, 1, cache_len, 256), ttnn.bfloat8_b), l1((1, 1, cache_len, 256), ttnn.bfloat8_b))
                   for _ in range(G.N_LAYERS)]
        self.mask = ttnn.from_torch(torch.zeros(1, 1, 32, self.sh.prefix_len + self.sh.suffix_rows), dtype=ttnn.bfloat16,
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.tables = [l1((1, 1, self.sh.suffix_rows, 256)) for _ in range(4)]
        self.noise = l1((1, self.sh.suffix_rows, 32))
        stamp("uploading prefix tensors (layer-0 arenas)")
        self.t = PrefixTensors(dev, pp, self.ps, n_sig=min(len(pp.sig), 27), n_vlm=min(len(pp.vlm), 18), embed=embed)
        self.wm = WholeMegakernel(dev, self.mk, self.t)

    def put(self, dev_t, host, dtype):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(host.contiguous().reshape(tuple(dev_t.shape)), dtype=dtype,
                                                        layout=dev_t.layout), dev_t)

    def get(self, dev_t):
        return ttnn.to_torch(dev_t).float()

    def run(self, first, stop, reps=1):
        t0 = time.time()
        self.wm.run(self.kv, self.mask, self.tables, self.noise, whole=False, first=first, stop=stop, reps=reps)
        ttnn.synchronize_device(self.dev)
        return time.time() - t0

    def diag(self):
        d = ttnn.to_torch(self.t.diag).to(torch.int64) & 0xFFFFFFFF
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="base")
    ap.add_argument("--checks", default="patch")
    ap.add_argument("--reps", type=int, default=0, help="also time each check's op(s) with this many in-kernel reps")
    ap.add_argument("--n-sig", type=int, default=1)
    ap.add_argument("--n-vlm", type=int, default=1)
    ap.add_argument("--weights", default=os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    checks = a.checks.split(",")
    res = {"shape": a.shape, "checks": checks, "digest": kernel_digest2(), "time": time.strftime("%F %T")}
    torch.manual_seed(0)
    stamp("loading weights")
    wl = PI0WeightLoader(a.weights)
    cat = wl.categorized_weights
    pp = H.prefix_params(cat, n_sig=a.n_sig, n_vlm=a.n_vlm)
    torch.set_num_threads(16)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1395712)
    try:
        emb_w = cat["vlm_language"].get("model.embed_tokens.weight")
        if emb_w is None:
            emb_w = cat["vlm_language"]["lm_head.weight"]
        if "embed" not in checks:
            emb_w = emb_w[:256]
        embed = ttnn.from_torch(emb_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rig = Rig(dev, a.shape, pp, embed)
        ps, t = rig.ps, rig.t
        bf = lambda x: x.to(torch.bfloat16).float()
        rows_v = ps.mt * 32
        # ---------------- SigLIP
        g = torch.Generator().manual_seed(5)
        pix = torch.rand(2, 3, 224, 224, generator=g) * 2 - 1
        from models.experimental.pi0_5.common.fused_host import im2col_patches

        im = bf(im2col_patches(pix, 14, pad_to=608).reshape(512, 608))
        timings = {}

        def timed(name, first, stop):
            """Device time per op from the hub's wall-clock stamps (median over reps), and the whole range."""
            if not a.reps:
                return
            t0 = time.time()
            rig.run(first, stop, a.reps)
            wall = time.time() - t0
            tt = ttnn.to_torch(t.times).to(torch.int64) & 0xFFFFFFFF
            n = (stop - first) * a.reps
            stamps = [int(tt[4095, 0])] + [int(tt[k, 0]) for k in range(n)]
            d = [((stamps[i + 1] - stamps[i]) & 0xFFFFFFFF) / (AICLK_MHZ * 1e3) for i in range(n)]  # ms
            per_op = {}
            for j in range(stop - first):
                v = sorted(d[j::(stop - first)][1:] or d[j::(stop - first)])
                per_op[first + j] = round(v[len(v) // 2], 4)
            span = ((stamps[-1] - stamps[1 + (stop - first) - 1]) & 0xFFFFFFFF) / (AICLK_MHZ * 1e3) / max(1, a.reps - 1)
            timings[name] = {"per_op_ms": per_op, "per_rep_ms": round(span, 4), "wall_s": round(wall, 3)}
            stamp(f"time {name}: {timings[name]}")

        if "patch" in checks or "sig0" in checks:
            rig.put(t.im2col, im, ttnn.bfloat16)
            stamp("run op 0 (patch)")
            dt = rig.run(0, 1)
            res["wall_patch_s"] = dt
            x = rig.get(t.x_s)
            ref = im @ bf(pp.patch_w) + bf(pp.pos_b).repeat(2, 1)
            cmp("patch", x, ref, res)
            d = rig.diag()
            res["diag_core0"] = [int(v) for v in d[0, :8]]
            stamp(f"diag core 0: {res['diag_core0']}")
            timed("patch", 0, 1)
        if "sig0" in checks:
            sp = pp.sig[0]
            x0 = rig.get(t.x_s)
            base = P.OP_S0
            rig.run(base, base + 1)  # LN1
            xn = rig.get(t.xn_s)
            cmp("sig0_ln1", xn, H._ln(x0, sp.ln1w, sp.ln1b, pp.eps_s), res)
            timed("sig_ln", base, base + 1)
            rig.run(base + 1, base + 2)  # QKV
            qkv = rig.get(t.qkv_s)
            cmp("sig0_qkv", qkv, xn @ sp.wqkv + sp.bqkv, res)
            timed("sig_qkv", base + 1, base + 2)
            rig.run(base + 2, base + 3)  # ATTN
            ctx = rig.get(t.ctx_s)
            q, k, v = qkv.split(1536, dim=1)
            cref = torch.zeros(512, 1536)
            for img in range(2):
                rs = slice(img * 256, (img + 1) * 256)
                qh = q[rs].reshape(256, 16, 96).transpose(0, 1)
                kh = k[rs].reshape(256, 16, 96).transpose(0, 1)
                vh = v[rs].reshape(256, 16, 96).transpose(0, 1)
                cref[rs] = (torch.softmax(qh @ kh.transpose(1, 2), -1) @ vh).transpose(0, 1).reshape(256, 1536)
            cmp("sig0_attn", ctx, cref, res)
            timed("sig_attn", base + 2, base + 3)
            x_before = rig.get(t.x_s)
            rig.run(base + 3, base + 4)  # O (+ residual, in place)
            x1 = rig.get(t.x_s)
            cmp("sig0_o", x1, x_before + ctx @ sp.wo + sp.bo, res)
            rig.run(base + 4, base + 5)  # LN2
            xn2 = rig.get(t.xn_s)
            cmp("sig0_ln2", xn2, H._ln(x1, sp.ln2w, sp.ln2b, pp.eps_s), res)
            rig.run(base + 5, base + 6)  # FC1
            h = rig.get(t.h_s)
            cmp("sig0_fc1", h, F.gelu(xn2 @ sp.wfc1 + sp.bfc1, approximate="tanh"), res)
            timed("sig_fc1", base + 5, base + 6)
            rig.run(base + 6, base + 7)  # FC2
            x2 = rig.get(t.x_s)
            cmp("sig0_fc2", x2, x1 + h @ sp.wfc2 + sp.bfc2, res)
            timed("sig_fc2", base + 6, base + 7)
            # the whole layer from the patch output, against the host layer on the same input
            rig.put(t.x_s, x0, ttnn.float32)
            rig.run(base, base + 7)
            xl = rig.get(t.x_s)
            pp1 = H.PrefixParams(**{**pp.__dict__, "sig": pp.sig[:1]})
            ref_l = x0.clone()
            xn_r = H._ln(ref_l, sp.ln1w, sp.ln1b, pp.eps_s)
            qkv_r = xn_r @ sp.wqkv + sp.bqkv
            q, k, v = qkv_r.split(1536, dim=1)
            for img in range(2):
                rs = slice(img * 256, (img + 1) * 256)
                qh = q[rs].reshape(256, 16, 96).transpose(0, 1)
                kh = k[rs].reshape(256, 16, 96).transpose(0, 1)
                vh = v[rs].reshape(256, 16, 96).transpose(0, 1)
                cref[rs] = (torch.softmax(qh @ kh.transpose(1, 2), -1) @ vh).transpose(0, 1).reshape(256, 1536)
            ref_l = ref_l + cref @ sp.wo + sp.bo
            xn_r = H._ln(ref_l, sp.ln2w, sp.ln2b, pp.eps_s)
            ref_l = ref_l + F.gelu(xn_r @ sp.wfc1 + sp.bfc1, approximate="tanh") @ sp.wfc2 + sp.bfc2
            cmp("sig0_layer", xl, ref_l, res)
            timed("sig_layer", base, base + 7)
            del pp1
        # ---------------- VLM layer 0
        if "vlm0" in checks:
            vp = pp.vlm[0]
            n_lang = 150 if a.shape == "base" else 20
            valid = torch.zeros(ps.ptv * 32, dtype=torch.bool)
            valid[: 512 + n_lang] = True
            rig.put(t.vmask, H.vlm_key_mask(valid, ps), ttnn.bfloat16)
            xv = torch.randn(rows_v, 2048, generator=g) * 2.0
            xv[ps.ptv * 32:] = 0
            rig.put(t.x_v, xv, ttnn.float32)
            base = P.OP_V0
            rig.run(base, base + 1)  # RMS1
            xn = rig.get(t.xn_v)
            cmp("vlm0_rms1", xn, H._rms(xv, vp.g1, pp.eps_v), res)
            timed("vlm_rms", base, base + 1)
            rig.run(base + 1, base + 2)  # QKV -> q scratch, caches
            qd = rig.get(t.q_v).reshape(8, rows_v, 256)
            kd = rig.get(rig.kv[0][0])[0, 0, : ps.ptv * 32]
            vd = rig.get(rig.kv[0][1])[0, 0, : ps.ptv * 32]
            tab = H.rope_tables(ps)
            qkv = xn @ vp.wqkv

            def rope(tt, c, s):
                hh = tt.shape[-1] // 2
                return tt * c + torch.cat([tt[..., hh:], tt[..., :hh]], -1) * s

            qr = rope(qkv[:, :2048].reshape(rows_v, 8, 256).transpose(0, 1), tab["cosq"], tab["sinq"])
            kr = rope(qkv[:, 2048:2304], tab["cosk"], tab["sink"])
            cmp("vlm0_q", qd, qr, res)
            cmp("vlm0_k", kd, kr[: ps.ptv * 32], res)
            cmp("vlm0_v", vd, qkv[: ps.ptv * 32, 2304:2560], res)
            timed("vlm_qkv", base + 1, base + 2)
            rig.run(base + 2, base + 3)  # ATTN
            ctx = rig.get(t.ctx_v)
            bias = torch.where(valid, 0.0, -1.0e9)[None, :]
            cref = (torch.softmax(qd @ kd.T + bias, -1) @ vd).transpose(0, 1).reshape(rows_v, 2048)
            cmp("vlm0_attn", ctx[: ps.ptv * 32], cref[: ps.ptv * 32], res)
            timed("vlm_attn", base + 2, base + 3)
            x_before = rig.get(t.x_v)
            rig.run(base + 3, base + 4)  # O
            x1 = rig.get(t.x_v)
            cmp("vlm0_o", x1, x_before + ctx @ vp.wo, res)
            timed("vlm_o", base + 3, base + 4)
            rig.put(t.x_v, x1, ttnn.float32)
            rig.run(base + 4, base + 5)  # RMS2
            xn2 = rig.get(t.xn_v)
            cmp("vlm0_rms2", xn2, H._rms(x1, vp.g2, pp.eps_v), res)
            rig.run(base + 5, base + 6)  # GU
            hd = rig.get(t.h_v)
            ug = xn2 @ vp.wug
            cmp("vlm0_gu", hd, ug[:, :16384] * F.gelu(ug[:, 16384:], approximate="tanh"), res)
            timed("vlm_gu", base + 5, base + 6)
            rig.run(base + 6, base + 7)  # DOWN
            x2 = rig.get(t.x_v)
            cmp("vlm0_down", x2, x1 + hd @ vp.wd, res)
            timed("vlm_down", base + 6, base + 7)
            rig.put(t.x_v, xv, ttnn.float32)
            rig.run(base, base + 7)
            xl = rig.get(t.x_v)
            ref_x, _ = H.host_vlm(H.PrefixParams(**{**pp.__dict__, "vlm": pp.vlm[:2] if len(pp.vlm) > 1 else pp.vlm + pp.vlm}),
                                  xv, valid, ps, layers=1)
            cmp("vlm0_layer", xl[: ps.ptv * 32], ref_x[: ps.ptv * 32], res)
            timed("vlm_layer", base, base + 7)
        res["timings_ms"] = timings
    finally:
        ttnn.close_device(dev)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    stamp("done")


if __name__ == "__main__":
    main()
