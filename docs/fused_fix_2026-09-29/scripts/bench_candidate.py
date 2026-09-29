"""Benchmark one fused pi0.5 code tree (PYTHONPATH decides which) on the p150a, served base shape.

Same inputs for every candidate: seeded images in [-1, 1], seeded 224 token ids, seeded noise.
Reports: first call (compile + capture), per-call wall (sample_actions_fused: upload + replay + readback),
and replay-only wall (execute_trace blocking), median of --runs. Saves the actions for cross-checks.
"""
import argparse
import json
import os
import statistics
import time

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--runs", type=int, default=60)
ap.add_argument("--ckpt", default="/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba")
args = ap.parse_args()

cfg = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
cfg.siglip_config = SigLIPConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                                 num_attention_heads=16, image_size=224, patch_size=14)
fc = FusedConfig.from_env()
print("fused cfg:", fc, flush=True)
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=fc.trace_region_size)
dev.enable_program_cache()
res = {"label": args.label, "time": time.strftime("%F %T")}
try:
    torch.manual_seed(42)
    t0 = time.perf_counter()
    model = PI0ModelTTNN(cfg, PI0WeightLoader(args.ckpt), dev, fused=fc)
    res["build_s"] = time.perf_counter() - t0
    try:
        res["fused_cfg_resolved"] = {k: str(v) for k, v in model.fused_cfg.describe().items()}
    except Exception:
        pass
    g = torch.Generator().manual_seed(0)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    tokens = torch.randint(0, 256000, (1, 224), generator=g)
    noise = torch.randn(1, 50, 32, generator=g)
    t0 = time.perf_counter()
    a0 = model.sample_actions_fused(images, tokens, noise)
    res["first_call_ms"] = (time.perf_counter() - t0) * 1e3
    res["traced"] = model._fused_trace_id is not None
    call, replay, outs = [], [], []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        a = model.sample_actions_fused(images, tokens, noise)
        call.append((time.perf_counter() - t0) * 1e3)
        outs.append(a)
    for _ in range(args.runs):
        t0 = time.perf_counter()
        ttnn.execute_trace(dev, model._fused_trace_id, cq_id=0, blocking=True)
        replay.append((time.perf_counter() - t0) * 1e3)
    res["call_ms"] = {"min": min(call), "median": statistics.median(call), "max": max(call)}
    res["replay_ms"] = {"min": min(replay), "median": statistics.median(replay), "max": max(replay)}
    res["bit_identical_repeats"] = all(torch.equal(o, a0) for o in outs)
    torch.save({"images": images, "tokens": tokens, "noise": noise, "actions": a0}, args.out + ".pt")
    model.release_trace()
finally:
    ttnn.close_device(dev)
print("RESULT", json.dumps(res), flush=True)
json.dump(res, open(args.out + ".json", "w"), indent=1)
