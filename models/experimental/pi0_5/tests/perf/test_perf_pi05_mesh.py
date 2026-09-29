# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DEVICE perf: the fused / traced pi0.5 graph on a Blackhole MeshDevice (2x p300 = 1x4 ring; PI05_MESH
selects the shape, 1x1 = one chip). SigLIP + VLM prefill tensor-parallel (PI05_TP, default auto =
mesh size), action expert replicated. Served shape: PI05_NUM_IMAGES cameras, PI05_TOKEN_LEN tokens,
10 denoising steps.

    PI05_MESH=1x4 PI05_WEIGHTS_DIR=... python models/experimental/pi0_5/tests/perf/test_perf_pi05_mesh.py
"""

import argparse
import os
import time

import torch

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tt import ttnn_ccl
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

CHECKPOINT_PATH = os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base")
TOKEN_LEN = int(os.environ.get("PI05_TOKEN_LEN", "224"))
NUM_IMAGES = int(os.environ.get("PI05_NUM_IMAGES", "2"))


def parse_mesh(value: str):
    rows, cols = value.lower().replace("x", ",").split(",")
    return int(rows), int(cols)


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


def timeit(fn, runs):
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[0], ts[len(ts) // 2], ts[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--mesh", default=os.environ.get("PI05_MESH", "1x4"))
    args = ap.parse_args()

    fused_cfg = FusedConfig.from_env()
    if not fused_cfg.enabled:
        raise SystemExit("the mesh path needs the fused graph: unset TT_FUSED or set TT_FUSED=1")
    mesh_shape = parse_mesh(args.mesh)
    device = ttnn_ccl.open_mesh(fused_cfg, mesh_shape, l1_small_size=24576)
    try:
        torch.manual_seed(42)
        t0 = time.perf_counter()
        model = PI0ModelTTNN(create_pi05_config(), PI0WeightLoader(CHECKPOINT_PATH), device, fused=fused_cfg)
        print(
            "model built in %.1f s: mesh %dx%d, tp=%d, expert replicated"
            % (time.perf_counter() - t0, *mesh_shape, model.fused_cfg.tp)
        )
        g = torch.Generator().manual_seed(1)
        images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(NUM_IMAGES)]
        tokens = torch.randint(0, 256000, (1, TOKEN_LEN), generator=g)

        t0 = time.perf_counter()
        out = model.sample_actions_fused(images, tokens)
        print("first call (compile + trace capture): %.0f ms" % ((time.perf_counter() - t0) * 1000))
        label = "fused traced" if model._fused_trace_id is not None else "fused eager "
        mn, md, mx = timeit(lambda: model.sample_actions_fused(images, tokens), args.runs)
        print(
            "%s mesh %dx%d tp=%d  min/med/max ms over %d runs: %.1f / %.1f / %.1f  (%.0f actions/s at the median)"
            % (label, *mesh_shape, model.fused_cfg.tp, args.runs, mn, md, mx, 50 * 1000.0 / md)
        )
        out2 = model.sample_actions_fused(images, tokens)
        print("repeat determinism max |diff|: %.3e" % (out2 - out).abs().max().item())
        model.release_trace()
    finally:
        ttnn_ccl.close_mesh(device)


if __name__ == "__main__":
    main()
