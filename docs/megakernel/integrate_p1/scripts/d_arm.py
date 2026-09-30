"""verify-p1-r1 device arm, one arm per process; the arm is chosen by PI05_MEGAKERNEL (FusedConfig.from_env), the
device is opened by the repo's own helper (common/device_open.py), exactly as the server does.

  python d_arm.py --shape {base,libero} --out out/X [--nobs N] [--runs 30]

Records: model stamp; per observation the output, the device's own prefix K/V (bf8 caches read back, fp32) and the
device noise (bf16 as uploaded) for the CPU oracle; replay checks (10 calls after other prompts, 10 raw replays);
an output-poisoning check (zero the trace's output buffer, replay, the output must come back -> the trace writes it);
latency with clock witnesses (perf_counter per call; time.time_ns / monotonic_ns around the loop; replay count in a
fixed 10 s window; tt-smi aiclk before/after)."""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc import BASE_WEIGHTS, SPEC, base_config, now, obs_for, pcc  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.device_open import device_kwargs  # noqa: E402
from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.pcc.golden_openpi import (LIBERO_WEIGHTS, libero_config, load_records,  # noqa: E402
                                                                pcc7, record_inputs)
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

SMI = os.path.expanduser("~/.tenstorrent-venv/bin/tt-smi")


def smi():
    try:
        r = subprocess.run([SMI, "-s", "--snapshot_no_tty"], capture_output=True, text=True, timeout=30)
        t = json.loads(r.stdout[r.stdout.find("{"):])["device_info"][0].get("telemetry", {})
        return {"t": now(), "aiclk": t.get("aiclk"), "power": t.get("power"), "temp": t.get("asic_temperature")}
    except Exception as e:  # noqa: BLE001
        return {"t": now(), "err": repr(e)}


def dig(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=["base", "libero"])
    ap.add_argument("--nobs", type=int, default=len(SPEC))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    env = FusedConfig.from_env()
    kw = device_kwargs(env)
    res = {"t0": now(), "arm_env": env.megakernel, "shape": a.shape, "open_kwargs": kw,
           "env": {k: v for k, v in os.environ.items() if k.startswith(("PI05_", "TT_METAL_CACHE"))},
           "smi_open": smi()}
    dev = ttnn.open_device(**kw)
    dev.enable_program_cache()
    save = {}
    try:
        torch.manual_seed(42)
        if a.shape == "base":
            m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=env)
            obs = [obs_for(s, n) for s, n in SPEC[: a.nobs]]
            tags = [s for s, _ in SPEC[: a.nobs]]
        else:
            m = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), dev, fused=env)
            recs = load_records()
            obs = []
            for r in recs:
                im, tk, mk, nz = record_inputs(r, 32)
                obs.append((im, tk, nz, mk))
            tags = [r["tag"] for r in recs]
        res["stamp"] = {"backend": m.megakernel_backend, "program": m.megakernel_program}
        outs, kvs, noises, tcall = [], [], [], []
        for i, (im, tk, nz, mk) in enumerate(obs):
            t0 = time.perf_counter()
            o = m.sample_actions_fused(im, tk, nz, lang_masks=mk).clone()
            tcall.append((time.perf_counter() - t0) * 1e3)
            outs.append(o)
            if a.shape == "base":
                P = m.backbone.kv_cache_plan["prefix_len"]
                kvs.append([(ttnn.to_torch(k).float()[:, :, :P].clone(), ttnn.to_torch(v).float()[:, :, :P].clone())
                            for k, v in m.backbone.kv_caches])
                noises.append(ttnn.to_torch(m._fused_in_noise).float()[:, :50].clone())
            print(now(), "obs", tags[i], f"{tcall[-1]:.1f} ms", dig(o), flush=True)
        res["traced"] = m._fused_trace_id is not None
        res["prefix_len"] = m.backbone.kv_cache_plan["prefix_len"]
        res["kv_cache_shape"] = list(m.backbone.kv_caches[0][0].shape)
        res["kv_cache_dtype"] = str(m.backbone.kv_caches[0][0].dtype)
        res["digests"] = {str(t): dig(o) for t, o in zip(tags, outs)}
        res["distinct_outputs"] = len(set(res["digests"].values()))
        if a.shape == "libero":
            res["pcc7"] = {r["tag"]: pcc7(o, r["actions_model_norm"]) for r, o in zip(recs, outs)}
            v = list(res["pcc7"].values())
            res["pcc7_mean"], res["pcc7_min"] = sum(v) / len(v), min(v)
        # replays of obs0 after all the other prompts
        im, tk, nz, mk = obs[0]
        res["ten_calls_eq_first"] = [bool(torch.equal(m.sample_actions_fused(im, tk, nz, lang_masks=mk), outs[0]))
                                     for _ in range(10)]
        H = outs[0].shape[1]
        raw = []
        for _ in range(10):
            ttnn.execute_trace(dev, m._fused_trace_id, cq_id=0, blocking=True)
            raw.append(bool(torch.equal(ttnn.to_torch(m._fused_out).float()[:, :H], outs[0])))
        res["ten_raw_replays_eq_first"] = raw
        # output poisoning: the traced program (not the host) must write the output buffer
        j = 1 if len(obs) > 1 else 0
        zshape = list(m._fused_out.shape)
        zeros = ttnn.from_torch(torch.full(zshape, 7.0), dtype=m._fused_out.dtype, layout=m._fused_out.layout)
        ttnn.copy_host_to_device_tensor(zeros, m._fused_out, cq_id=0)
        ttnn.synchronize_device(dev)
        poisoned = ttnn.to_torch(m._fused_out).float()
        res["poison_readback_all_7"] = bool((poisoned == 7.0).all())
        imj, tkj, nzj, mkj = obs[j]
        yj = m.sample_actions_fused(imj, tkj, nzj, lang_masks=mkj)
        res["after_poison_eq_obs_j"] = bool(torch.equal(yj, outs[j]))
        res["after_poison_has_7"] = bool((yj == 7.0).any())
        # latency (obs0), warm
        for _ in range(3):
            m.sample_actions_fused(im, tk, nz, lang_masks=mk)
        res["smi_bench_before"] = smi()
        call = []
        w0, m0 = time.time_ns(), time.monotonic_ns()
        for _ in range(a.runs):
            t0 = time.perf_counter()
            m.sample_actions_fused(im, tk, nz, lang_masks=mk)
            call.append((time.perf_counter() - t0) * 1e3)
        res["witness_call"] = {"time_ns_s": (time.time_ns() - w0) / 1e9, "monotonic_s": (time.monotonic_ns() - m0) / 1e9,
                               "sum_perf_counter_s": sum(call) / 1e3, "wall_start_epoch": w0 / 1e9}
        rep = []
        w0 = time.time_ns()
        for _ in range(a.runs):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, m._fused_trace_id, cq_id=0, blocking=True)
            rep.append((time.perf_counter() - t0) * 1e3)
        res["witness_replay"] = {"time_ns_s": (time.time_ns() - w0) / 1e9, "sum_perf_counter_s": sum(rep) / 1e3}
        n10, tw0 = 0, time.time()
        while time.time() - tw0 < 10.0:
            ttnn.execute_trace(dev, m._fused_trace_id, cq_id=0, blocking=True)
            n10 += 1
        res["replays_in_10s_time_time"] = n10
        res["smi_bench_after"] = smi()
        res["call_ms"] = {"median": statistics.median(call), "min": min(call), "max": max(call), "all": call}
        res["replay_ms"] = {"median": statistics.median(rep), "min": min(rep), "max": max(rep), "all": rep}
        res["after_bench_eq_first"] = bool(torch.equal(m.sample_actions_fused(im, tk, nz, lang_masks=mk), outs[0]))
        res["megakernel_l1"] = m.megakernel_l1
        m.release_trace()
        save = {"tags": tags, "outs": outs, "kvs": kvs, "noises": noises}
    finally:
        ttnn.close_device(dev)
    res["t1"] = now()
    torch.save(save, a.out + ".pt")
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=str)
    brief = {k: v for k, v in res.items() if k not in ("call_ms", "replay_ms", "digests", "env", "pcc7")}
    brief["call_median"], brief["replay_median"] = res["call_ms"]["median"], res["replay_ms"]["median"]
    print("RESULT", json.dumps(brief, default=str), flush=True)


if __name__ == "__main__":
    main()
