"""Device, one arm per process (verify-p1-r0).

    python v_arm.py --arm {off,expert} --shape {base,libero} [--cut auto|1|0] --out out/X

base  : SPEC observations -> outputs + the device's prefix K/V (bf8 caches read back) + device noise, saved to
        <out>.pt for the CPU oracle; replay checks; latency with a clock witness.
libero: the 8 openpi golden records -> PCC7; replay checks; latency.
The arm is selected with PI05_MEGAKERNEL in the environment (FusedConfig.from_env), as a user would.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import BASE_WEIGHTS, SPEC, base_config, now, obs_for  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.pcc.golden_openpi import (LIBERO_WEIGHTS, libero_config, load_records,  # noqa: E402
                                                                pcc7, record_inputs)
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

TT_SMI = os.path.expanduser("~/.tenstorrent-venv/bin/tt-smi")


def smi():
    try:
        r = subprocess.run([TT_SMI, "-s", "--snapshot_no_tty"], capture_output=True, text=True, timeout=30)
        d = json.loads(r.stdout[r.stdout.find("{"):])
        t = d["device_info"][0].get("telemetry", {})
        return {"t": now(), "aiclk": t.get("aiclk"), "power": t.get("power"), "temp": t.get("asic_temperature")}
    except Exception as e:  # noqa: BLE001
        return {"t": now(), "err": repr(e)}


def dig(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["off", "expert"])
    ap.add_argument("--shape", required=True, choices=["base", "libero"])
    ap.add_argument("--cut", default="auto")
    ap.add_argument("--nobs", type=int, default=len(SPEC))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    env = FusedConfig.from_env()
    assert env.megakernel == a.arm, f"PI05_MEGAKERNEL={env.megakernel} but --arm {a.arm}"
    kw = dict(device_id=0, l1_small_size=24576, trace_region_size=env.trace_region_size)
    cut = (a.arm != "off") if a.cut == "auto" else a.cut == "1"
    if cut:
        kw["worker_l1_size"] = 1_395_712
    res = {"t0": now(), "arm": a.arm, "shape": a.shape, "cut": cut, "open_kwargs": kw,
           "env": {k: v for k, v in os.environ.items() if k.startswith(("PI05_", "TT_METAL_CACHE"))}}
    res["smi_before_open"] = smi()
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
        res["n_mk_objects"] = len(getattr(m, "_mk", {}))
        outs, kvs, noises = [], [], []
        for i, (im, tk, nz, mk) in enumerate(obs):
            t0 = time.perf_counter()
            o = m.sample_actions_fused(im, tk, nz, lang_masks=mk)
            outs.append(o.clone())
            if a.shape == "base":
                P = m.backbone.kv_cache_plan["prefix_len"]
                kvs.append([(ttnn.to_torch(k).float()[:, :1, :P].clone(), ttnn.to_torch(v).float()[:, :1, :P].clone())
                            for k, v in m.backbone.kv_caches])
                noises.append(ttnn.to_torch(m._fused_in_noise).float()[:, :50].clone())
            print(now(), "obs", tags[i], f"{(time.perf_counter()-t0)*1e3:.1f} ms", dig(o), flush=True)
        res["traced"] = m._fused_trace_id is not None
        res["prefix_len"] = m.backbone.kv_cache_plan["prefix_len"]
        res["digests"] = {str(t): dig(o) for t, o in zip(tags, outs)}
        if a.shape == "libero":
            res["pcc7"] = {r["tag"]: pcc7(o, r["actions_model_norm"]) for r, o in zip(recs, outs)}
            v = list(res["pcc7"].values())
            res["pcc7_mean"], res["pcc7_min"] = sum(v) / len(v), min(v)
        # replays: 10 calls re-uploading obs0 after the other observations, then 10 raw replays of the same trace
        im, tk, nz, mk = obs[0]
        again = [m.sample_actions_fused(im, tk, nz, lang_masks=mk) for _ in range(10)]
        res["ten_calls_bit_identical_to_first"] = [bool(torch.equal(x, outs[0])) for x in again]
        raw = []
        for _ in range(10):
            ttnn.execute_trace(dev, m._fused_trace_id, cq_id=0, blocking=True)
            raw.append(ttnn.to_torch(m._fused_out).float()[:, : outs[0].shape[1]].clone())
        res["ten_raw_replays_bit_identical_to_first"] = [bool(torch.equal(x, outs[0])) for x in raw]
        # latency (clock witness: perf_counter per call, time.time + monotonic_ns around the loop, tt-smi aiclk)
        for _ in range(3):
            m.sample_actions_fused(im, tk, nz, lang_masks=mk)
        res["smi_before_bench"] = smi()
        call = []
        w0, m0 = time.time(), time.monotonic_ns()
        for _ in range(a.runs):
            t0 = time.perf_counter()
            m.sample_actions_fused(im, tk, nz, lang_masks=mk)
            call.append((time.perf_counter() - t0) * 1e3)
        res["witness_call_loop"] = {"time_time_s": time.time() - w0, "monotonic_s": (time.monotonic_ns() - m0) / 1e9,
                                    "sum_perf_counter_s": sum(call) / 1e3}
        rep = []
        w0 = time.time()
        for _ in range(a.runs):
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, m._fused_trace_id, cq_id=0, blocking=True)
            rep.append((time.perf_counter() - t0) * 1e3)
        res["witness_replay_loop_time_time_s"] = time.time() - w0
        res["call_ms"] = {"median": statistics.median(call), "min": min(call), "max": max(call), "all": call}
        res["replay_ms"] = {"median": statistics.median(rep), "min": min(rep), "max": max(rep), "all": rep}
        # sustained replays with aiclk snapshots from a thread (10 s)
        snaps, stop = [], threading.Event()

        def loop():
            while not stop.is_set():
                snaps.append(smi())
                stop.wait(1.0)

        th = threading.Thread(target=loop, daemon=True)
        th.start()
        sus, tend = [], time.perf_counter() + 10
        while time.perf_counter() < tend:
            t0 = time.perf_counter()
            ttnn.execute_trace(dev, m._fused_trace_id, cq_id=0, blocking=True)
            sus.append((time.perf_counter() - t0) * 1e3)
        stop.set()
        th.join(timeout=40)
        res["sustain_replay_ms_median"], res["sustain_n"], res["sustain_smi"] = statistics.median(sus), len(sus), snaps
        last = m.sample_actions_fused(im, tk, nz, lang_masks=mk)
        res["after_bench_bit_identical"] = bool(torch.equal(last, outs[0]))
        m.release_trace()
        save = {"tags": tags, "outs": outs, "kvs": kvs, "noises": noises}
    finally:
        ttnn.close_device(dev)
    res["t1"] = now()
    torch.save(save, a.out + ".pt")
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=str)
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("call_ms", "replay_ms", "sustain_smi", "digests", "env")},
                               default=str), flush=True)


if __name__ == "__main__":
    main()
