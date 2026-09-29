# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device bring-up of the expert megakernel alone (no VLM): real pi05 expert weights, synthetic prefix K / V (bf8, L1),
the fused graph's real mask / RoPE inputs. Runs ``ngen`` generations and compares the residual x after the last one
(the debug dump) and x_t with the host decomposition on the device's exact inputs.

    python models/experimental/pi0_5/tests/megakernel/mk_bringup.py --shape base --ngen 1 --out r.json [--runs 3]

ALWAYS through bin/with-device.sh; first runs of a changed kernel with WITH_DEVICE_RESET_AFTER=1 and a timeout.
"""
import argparse
import json
import os
import sys
import time

import torch
import ttnn

from models.experimental.pi0_5.common.fused_host import attention_inputs, kv_cache_plan, prefix_valid_mask
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tt.megakernel import geometry as G
from models.experimental.pi0_5.tt.megakernel import host_model as hm
from models.experimental.pi0_5.tt.megakernel.program import ExpertMegakernel, kernel_digest


def stamp(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="base")
    ap.add_argument("--ngen", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--n-lang", type=int, default=None)
    ap.add_argument("--weights", default=os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sh = G.SHAPES[a.shape]
    torch.manual_seed(0)
    stamp("loading weights")
    cw = PI0WeightLoader(a.weights).categorized_weights
    params = hm.expert_params(cw["action_expert"], cw["pi0_projections"])
    plan = kv_cache_plan(sh.prefix_len, sh.horizon)
    n_lang = a.n_lang if a.n_lang is not None else (150 if a.shape == "base" else 20)
    lm = torch.zeros(1, sh.prefix_len - 512, dtype=torch.bool)
    lm[0, :n_lang] = True
    valid = prefix_valid_mask(2, lm)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, 256, 2).float() / 256))
    ang = torch.arange(1024).float()[:, None] * freqs[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)
    sin = torch.cat([ang.sin(), ang.sin()], -1)
    att = attention_inputs(valid, plan, cos, sin, 1.0 / 16)
    noise = torch.zeros(1, sh.suffix_rows, 32)
    noise[0, : sh.horizon] = torch.randn(sh.horizon, 32)
    kv_host = [(torch.randn(1, 1, plan["cache_len"], 256), torch.randn(1, 1, plan["cache_len"], 256)) for _ in range(18)]

    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1395712)
    res = {"shape": a.shape, "ngen": a.ngen, "n_lang": n_lang, "kernel_digest": kernel_digest(), "time": time.strftime("%F %T")}
    try:
        l1 = ttnn.L1_MEMORY_CONFIG
        kv = [(ttnn.from_torch(k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1),
               ttnn.from_torch(v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1))
              for k, v in kv_host]
        mask = ttnn.from_torch(att["exp_mask"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tables = [ttnn.from_torch(att[k], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1)
                  for k in ("cosq", "sinq", "cosk", "sink")]
        noise_d = ttnn.from_torch(noise, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1)
        stamp("uploading arenas")
        t0 = time.time()
        mk = ExpertMegakernel(dev, params, sh)
        res["arena_upload_s"] = round(time.time() - t0, 1)
        stamp(f"arenas uploaded in {res['arena_upload_s']} s")
        outs, xs, walls = [], [], []
        for i in range(a.runs):
            t0 = time.time()
            out = mk.run(kv, mask, tables, noise_d, ngen=a.ngen)
            ttnn.synchronize_device(dev)
            walls.append(time.time() - t0)
            outs.append(ttnn.to_torch(out).float())
            xs.append(ttnn.to_torch(mk.dbg).float())
            stamp(f"run {i}: {walls[-1] * 1e3:.2f} ms")
            ttnn.deallocate(out)
        res["wall_ms"] = [round(w * 1e3, 3) for w in walls]
        res["runs_bit_identical"] = all(torch.equal(outs[0], o) for o in outs) and all(torch.equal(xs[0], x) for x in xs)
        # host oracle on the device's exact inputs
        kvr = [(ttnn.to_torch(k).float()[0, 0, : sh.prefix_len], ttnn.to_torch(v).float()[0, 0, : sh.prefix_len]) for k, v in kv]
        att_dev = dict(att)
        for k, t in zip(("cosq", "sinq", "cosk", "sink"), tables):
            att_dev[k] = ttnn.to_torch(t).float()
        att_dev["exp_mask"] = ttnn.to_torch(mask).float()
        ai = hm.attn_inputs_from(att_dev)
        nz = ttnn.to_torch(noise_d).float()[0]
        last = {}
        xt = hm.loop_decomposed(params, kvr, ai, nz, sh.chunk_tiles, ngen=a.ngen, last=last)
        H = sh.horizon
        x_dev, x_host = xs[0][0, 0], last["x"]
        res["x_pcc_valid_rows"] = hm.pcc(x_dev[:H], x_host[:H])
        res["x_pcc_all_rows"] = hm.pcc(x_dev, x_host)
        res["x_maxabs_valid"] = float((x_dev[:H] - x_host[:H]).abs().max())
        res["x_absmax_host"] = float(x_host[:H].abs().max())
        res["x_nonfinite"] = int((~torch.isfinite(x_dev)).sum())
        res["xt_pcc_valid_rows"] = hm.pcc(outs[0][0, :H], xt[:H])
        res["xt_maxabs_valid"] = float((outs[0][0, :H] - xt[:H]).abs().max())
        # per-column diagnostics of x (which owner columns are off)
        colp = [hm.pcc(x_dev[:H, 32 * n:32 * n + 32], x_host[:H, 32 * n:32 * n + 32]) for n in range(32)]
        res["x_pcc_per_owner_col_min"] = min(colp)
        res["x_pcc_per_owner_col"] = [round(c, 5) for c in colp]
        stamp("RESULT " + json.dumps({k: v for k, v in res.items() if k != "x_pcc_per_owner_col"}))
    finally:
        ttnn.close_device(dev)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
