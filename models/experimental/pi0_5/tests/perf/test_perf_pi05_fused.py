# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DEVICE benchmark (not run in the host-only pass): the fused graph eager (PI05_TRACE=0) or traced
(PI05_TRACE=1, default), served shape (2 x 224x224, L tokens, 10 steps), batch 1, host wall-clock per
call including the input upload and the readback.

    # from the model repo root ($ROOT/models/pi05-base-p150), tree python, PYTHONPATH=code
    python code/models/experimental/pi0_5/tests/perf/test_perf_pi05_fused.py [--runs 10]

Baseline to beat (reports/publish-p150/pi05-base-p150.json, tt serve): soak x10 143.1 / 145.0 /
147.1 ms (min / median / max), 224 tokens.
"""

import argparse
import os
import statistics
import time

import torch
import ttnn

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

CHECKPOINT_PATH = os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base")
TOKEN_LEN = int(os.environ.get("PI05_TOKEN_LEN", "224"))
NUM_IMAGES = int(os.environ.get("PI05_NUM_IMAGES", "2"))


def create_pi05_config():
    config = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    return config


def print_memory_headroom(device):
    """Per-bank allocator view after the model is built and the trace is captured: what the served
    process holds persistently (weights, KV caches, persistent inputs, the trace) and the free L1 the
    per-op circular buffers must fit into (a static CB region that reaches into an allocated L1 buffer
    is a hard error: program.cpp validate_circular_buffer_region)."""
    for name, bt in (("DRAM", ttnn.BufferType.DRAM), ("L1", ttnn.BufferType.L1), ("TRACE", ttnn.BufferType.TRACE)):
        try:
            v = ttnn.get_memory_view(device, bt)
        except Exception as e:  # noqa: BLE001 - reporting only
            print(f"memory {name}: unavailable ({e})")
            continue
        print(
            "memory %-5s per bank: total %.1f KB, allocated %.1f KB, free %.1f KB, largest free block %.1f KB (%d banks)"
            % (
                name,
                v.total_bytes_per_bank / 1024,
                v.total_bytes_allocated_per_bank / 1024,
                v.total_bytes_free_per_bank / 1024,
                v.largest_contiguous_bytes_free_per_bank / 1024,
                v.num_banks,
            )
        )


def timeit(fn, runs):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times), statistics.median(times), max(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    fused_cfg = FusedConfig.from_env()
    kwargs = dict(device_id=int(os.environ.get("PI0_DEVICE_ID", "0")), l1_small_size=24576)
    if fused_cfg.trace:
        kwargs["trace_region_size"] = fused_cfg.trace_region_size
    device = ttnn.open_device(**kwargs)
    device.enable_program_cache()
    try:
        torch.manual_seed(42)
        model = PI0ModelTTNN(create_pi05_config(), PI0WeightLoader(CHECKPOINT_PATH), device, fused=fused_cfg)
        images = [torch.full((1, 3, 224, 224), -1.0) for _ in range(NUM_IMAGES)]
        tokens = torch.randint(1, 256000, (1, TOKEN_LEN))

        t0 = time.perf_counter()
        model.sample_actions_fused(images, tokens)  # allocate + compile + capture
        print(
            "fused first call (compile%s): %.0f ms"
            % (" + trace capture" if fused_cfg.trace else "", (time.perf_counter() - t0) * 1000)
        )
        print_memory_headroom(device)
        label = "fused traced" if model._fused_trace_id is not None else "fused eager "
        print(
            "%s      min/med/max ms: %.1f / %.1f / %.1f"
            % ((label,) + timeit(lambda: model.sample_actions_fused(images, tokens), args.runs))
        )
        model.release_trace()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
