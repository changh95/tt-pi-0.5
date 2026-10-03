"""Device checks of the multi-config pi0.5 megakernel, run identically on the host (tt-metal-main f856a38a build +
tt-metal-pr fae9cd0 models) and inside the release image (/opt/tt-metal), so the two output sets can be compared
bit for bit. One model alive at a time (each closed before the next).

  golden : serve_pi05_libero.Pi05LiberoPolicy (cameras 2, N 10, H 10, lerobot/pi05_libero) on the 8 call0/call12
           records of openloop_golden.pt -> inputs torch.equal to the golden model inputs, PCC7 vs the GPU output
  bitid  : PI05MegakernelTTNN (lerobot/pi05_base) at one preset per camera count, fixed inputs -> outputs saved
           (.pt) + sha256 of their bytes; two calls each (replay determinism)
  refuse : masked camera, wrong camera count, prompt bucket too small / not compiled, H 65 / N 11 / cameras 5 at
           construction -> each raises with its message
usage: img_check.py --out DIR [--parts golden,bitid,refuse] [--base CKPT] [--libero CKPT] [--golden PT] [--norm JSON]
       [--spm MODEL]
"""
import argparse
import hashlib
import json
import os
import time
import traceback

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--parts", default="golden,bitid,refuse")
ap.add_argument("--base", default="/hf/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba")
ap.add_argument("--libero", default="/hf/hub/models--lerobot--pi05_libero/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd")
ap.add_argument("--golden", default="/chk/openloop_golden.pt")
ap.add_argument("--norm", default="/chk/openpi_pi05_libero_norm_stats.json")
ap.add_argument("--spm", default="/chk/paligemma_tokenizer.model")
ap.add_argument("--configs", default="c1,c2,c3,c4")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
parts = a.parts.split(",")
res = {"time": time.strftime("%F %T"), "parts": parts}

import ttnn  # noqa: E402

from models.experimental.pi0.common.configs import PI0ModelConfig  # noqa: E402
from models.experimental.pi0.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0.tt.megakernel.pe_program import kernel_digest  # noqa: E402
from models.experimental.pi0.tt.ttnn_pi05_model import PI05_DEVICE_PARAMS, PI05MegakernelTTNN  # noqa: E402

res["kernel_digest"] = kernel_digest()
res["ttnn_file"] = ttnn.__file__
import models.experimental.pi0.tt.ttnn_pi05_model as _m  # noqa: E402

res["model_file"] = _m.__file__


def pcc(x, y):
    x, y = x.flatten().double(), y.flatten().double()
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1])


def sha(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def img(seed):
    u8 = np.random.default_rng(seed).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    x = torch.from_numpy(u8).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    return (x * torch.tensor(1.0 / 255.0, dtype=torch.float32) * 2.0 - 1.0).contiguous()


def prompt(n, seed, buf=224):
    g = np.random.default_rng(seed)
    ids = np.zeros(buf, np.int64)
    ids[0] = 2
    ids[1:n] = g.integers(1000, 250000, n - 1)
    return torch.from_numpy(ids)[None], torch.arange(buf)[None] < n


# cameras -> (H, N, real prompt tokens -> bucket)
CFG = {"c1": (1, 10, 10, 20), "c2": (2, 50, 10, 150), "c3": (3, 32, 5, 50), "c4": (4, 64, 10, 100)}

dev = ttnn.open_device(device_id=0, **PI05_DEVICE_PARAMS)
try:
    if "golden" in parts:
        try:
            from models.experimental.pi0_5.server.serve_pi05_libero import Pi05LiberoPolicy
        except ImportError:
            from serve_pi05_libero import Pi05LiberoPolicy
        import inspect

        res["golden_policy_file"] = inspect.getsourcefile(Pi05LiberoPolicy)
        ttnn.close_device(dev)  # the policy opens its own device
        dev = None
        recs = torch.load(a.golden, weights_only=False)["records"]
        pol = Pi05LiberoPolicy(a.libero, a.norm, a.spm, cameras=2, num_steps=10)
        rows = []
        try:
            for r in recs:
                ro, mi = r["raw_obs"], r["model_inputs"]
                obs = {"observation/image": ro["image_224_uint8"], "observation/wrist_image": ro["wrist_image_224_uint8"],
                       "observation/state": ro["state8"], "prompt": ro["prompt"], "__noise_seed__": r["noise_seed"]}
                pol.infer(obs)
                li = pol.last_inputs
                eq = all(torch.equal(li["images"][i], mi["images"][s].float())
                         for i, s in enumerate(["base_0_rgb", "left_wrist_0_rgb"]))
                eq = eq and torch.equal(li["tokens"], mi["tokenized_prompt"].long()) and torch.equal(
                    li["lang_mask"], mi["tokenized_prompt_mask"].bool()) and torch.equal(
                    li["noise"], r["noise"].float().reshape(1, 10, 32))
                y, g = li["actions_model_norm"].reshape(1, 10, 32), r["actions_model_norm"].float().reshape(1, 10, 32)
                rows.append(dict(tag=r["tag"], inputs_equal=bool(eq), preset=str(li["preset"]), pcc7=pcc(y[..., :7], g[..., :7]),
                                 out_sha=sha(li["actions_model_norm"].float())))
                print("golden", rows[-1], flush=True)
        finally:
            stamp = pol.backend_stamp
            pol.close()
        c0 = [x["pcc7"] for x in rows]
        res["golden"] = dict(backend=stamp, rows=rows, n=len(rows), all_inputs_equal=all(x["inputs_equal"] for x in rows),
                             pcc7_mean=float(np.mean(c0)), pcc7_min=float(min(c0)))
        print("GOLDEN", {k: v for k, v in res["golden"].items() if k != "rows"}, flush=True)
        dev = ttnn.open_device(device_id=0, **PI05_DEVICE_PARAMS)

    loader = None
    if "bitid" in parts or "refuse" in parts:
        loader = PI0WeightLoader(a.base)
        _ = loader.categorized_weights

    if "bitid" in parts:
        res["bitid"] = {}
        for c in a.configs.split(","):
            cams, H, N, n = CFG[c]
            cfg = PI0ModelConfig(action_dim=32, action_horizon=H, state_dim=32, num_denoising_steps=N, pi05=True,
                                 num_cameras=cams)
            torch.manual_seed(42)
            t0 = time.time()
            with PI05MegakernelTTNN(cfg, loader, dev) as m:
                build_s = time.time() - t0
                ims = [img(100 + i) for i in range(cams)]
                ids, mask = prompt(n, 7)
                noise = torch.from_numpy(np.random.default_rng(11).standard_normal((1, H, 32)).astype(np.float32))
                preset = m.preset_for(n)
                o1 = m.sample_actions(ims, [torch.ones(1, dtype=torch.bool)] * cams, ids, lang_masks=mask, noise=noise)
                o2 = m.sample_actions(ims, None, ids, lang_masks=mask, noise=noise)
                ts = []
                for _ in range(10):
                    t1 = time.perf_counter()
                    m.sample_actions(ims, None, ids, lang_masks=mask, noise=noise)
                    ts.append((time.perf_counter() - t1) * 1e3)
                ops = len(m.programs[preset.key].visions) + 2
                torch.save(o1, f"{a.out}/bitid_{c}.pt")
                row = dict(cameras=cams, H=H, N=N, n_tokens=n, preset=list(preset.key), kv_dram=bool(m.kv_dram),
                           device_ops_per_call=ops, shape=list(o1.shape), sha=sha(o1), replay_identical=bool(torch.equal(o1, o2)),
                           finite=bool(torch.isfinite(o1).all()), call_ms_median=float(np.median(ts)), build_s=build_s,
                           head=[round(float(v), 5) for v in o1[0, 0, :6]])
            res["bitid"][c] = row
            print("BITID", c, row, flush=True)

    if "refuse" in parts:
        R = {}

        def expect(name, fn, needle):
            try:
                fn()
                R[name] = dict(raised=False, ok=False)
            except Exception as e:  # noqa: BLE001
                R[name] = dict(raised=True, type=type(e).__name__, msg=str(e)[:400], ok=needle in str(e))
            print("REFUSE", name, R[name], flush=True)

        def mk(cams=2, H=10, N=10):
            return PI05MegakernelTTNN(PI0ModelConfig(action_dim=32, action_horizon=H, state_dim=32, num_denoising_steps=N,
                                                     pi05=True, num_cameras=cams), loader, dev)

        expect("H=65 at construction", lambda: mk(H=65), "refused")
        expect("N=11 at construction", lambda: mk(N=11), "refused")
        expect("cameras=5 at construction", lambda: mk(cams=5), "cameras = 5")
        torch.manual_seed(42)
        with mk(2, 10, 10) as m:
            ims = [img(1), img(2)]
            ids, mask = prompt(20, 3)
            expect("masked camera", lambda: m.sample_actions(ims, [torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)], ids, lang_masks=mask), "every camera must be valid")
            expect("1 image on a 2-camera model", lambda: m.sample_actions(ims[:1], None, ids, lang_masks=mask), "serves 2 cameras")
            ids2, mask2 = prompt(40, 3)
            expect("40 tokens in prompt_bucket 32", lambda: m.sample_actions(ims, None, ids2, lang_masks=mask2, prompt_bucket=32), "cannot hold")
            expect("prompt_bucket 48", lambda: m.sample_actions(ims, None, ids, lang_masks=mask, prompt_bucket=48), "compiled")
            expect("batch 2", lambda: m.sample_actions(ims, None, ids.repeat(2, 1), lang_masks=mask.repeat(2, 1)), "batch 1")
            expect("second live model on the device", lambda: mk(2, 10, 10), "still hold device memory")
            ok_after = m.sample_actions(ims, None, ids, lang_masks=mask)
            R["model still serves after the refusals"] = dict(ok=bool(torch.isfinite(ok_after).all()), shape=list(ok_after.shape))
        res["refuse"] = R
        res["refuse_all_ok"] = all(v["ok"] for v in R.values())
finally:
    if dev is not None:
        ttnn.close_device(dev)
    json.dump(res, open(f"{a.out}/img_check.json", "w"), indent=1, default=str)
print("DONE", json.dumps({k: res.get(k) for k in ("kernel_digest", "refuse_all_ok")}), flush=True)
