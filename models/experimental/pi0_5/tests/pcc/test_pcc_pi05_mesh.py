# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DEVICE test: the fused / traced pi0.5 graph on a Blackhole MeshDevice (PI05_MESH, default 1x4 = 2x p300)
vs the torch reference, on the served shape (PI05_NUM_IMAGES cameras, PI05_TOKEN_LEN tokens, 10 steps).

    PI05_MESH=1x4 PI05_WEIGHTS_DIR=... pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_mesh.py -v -s -o timeout=3600

Gates: e2e PCC(mesh, torch) >= 0.93 on two random observations (same gate as test_pcc_pi05_fused),
traced repeat bit-exact, every chip holds the same output (the expert is replicated), and a second
observation moves the output about as much as it moves the torch reference.
Optionally PI05_PCC_REF=<.pt> (the single-chip fused output for observation 1, e.g. saved by
tests/perf) is compared too and reported.
"""

import os

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0_5.tt import ttnn_ccl
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

CHECKPOINT_PATH = os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base")
TOKEN_LEN = int(os.environ.get("PI05_TOKEN_LEN", "224"))
NUM_IMAGES = int(os.environ.get("PI05_NUM_IMAGES", "2"))
MESH = tuple(int(v) for v in os.environ.get("PI05_MESH", "1x4").lower().replace("x", ",").split(","))
SEED = 42
PCC_E2E = 0.93
RESPONSE_RATIO_MIN = 0.6


def create_pi05_config() -> PI0ModelConfig:
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


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1, t2 = a.flatten().float(), b.flatten().float()
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 == 0 or s2 == 0:
        return 1.0 if torch.allclose(t1, t2) else 0.0
    return (torch.mean((t1 - t1.mean()) * (t2 - t2.mean())) / (s1 * s2)).item()


def make_inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(NUM_IMAGES)]
    tokens = torch.randint(0, 256000, (1, TOKEN_LEN), generator=g)
    noise = torch.randn(1, 50, 32, generator=g)
    return images, tokens, noise


def run_torch(model_torch: PI0ModelTorch, images, tokens, noise):
    masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    lm = torch.ones(1, TOKEN_LEN, dtype=torch.bool)
    torch.manual_seed(SEED)
    model_torch.denoising.sample_noise = lambda *a, **k: noise.clone()
    return model_torch.sample_actions(images, masks, tokens, lm, torch.zeros(1, 32)).float()


@pytest.fixture(scope="module")
def mesh():
    fused = FusedConfig.from_env()
    if not fused.enabled:
        pytest.skip("the mesh path needs TT_FUSED=1")
    dev = ttnn_ccl.open_mesh(fused, MESH, l1_small_size=24576)
    yield dev
    ttnn_ccl.close_mesh(dev)


def test_mesh_vs_torch(mesh):
    fused_cfg = FusedConfig.from_env()
    loader = PI0WeightLoader(CHECKPOINT_PATH)
    config = create_pi05_config()

    images, tokens, noise = make_inputs(1)
    images_b, tokens_b, _ = make_inputs(2)

    torch.manual_seed(SEED)
    model_torch = PI0ModelTorch(config, loader)
    ref_a = run_torch(model_torch, images, tokens, noise)
    ref_b = run_torch(model_torch, images_b, tokens_b, noise)
    del model_torch

    torch.manual_seed(SEED)
    model = PI0ModelTTNN(config, loader, mesh, fused=fused_cfg)
    print(f"\nmesh {MESH} tp={model.fused_cfg.tp} trace={model.fused_cfg.trace}")

    out_a = model.sample_actions_fused(images, tokens, noise)
    out_a2 = model.sample_actions_fused(images, tokens, noise)
    per_chip = [ttnn.to_torch(t).float()[:, :50] for t in ttnn.get_device_tensors(model._fused_out)]
    out_b = model.sample_actions_fused(images_b, tokens_b, noise)

    pcc_a = compute_pcc(out_a, ref_a)
    pcc_b = compute_pcc(out_b, ref_b)
    chip_diff = max([(c - per_chip[0]).abs().max().item() for c in per_chip[1:]] or [0.0])
    resp_tt = (out_b - out_a).abs().mean().item()
    resp_ref = (ref_b - ref_a).abs().mean().item()
    print(
        f"PCC(mesh, torch) obs1 {pcc_a:.4f} obs2 {pcc_b:.4f}; repeat max|diff| {(out_a2 - out_a).abs().max().item():.2e}; "
        f"chips max|diff| {chip_diff:.2e}; response tt/ref {resp_tt:.4f}/{resp_ref:.4f}"
    )
    ref_path = os.environ.get("PI05_PCC_REF")
    if ref_path:
        single = torch.load(ref_path).float()
        print(
            f"PCC(mesh, single-chip fused) obs1 {compute_pcc(out_a, single):.6f} max|diff| {(out_a - single).abs().max().item():.4f}"
        )
    model.release_trace()

    assert pcc_a >= PCC_E2E and pcc_b >= PCC_E2E, (pcc_a, pcc_b)
    assert torch.equal(out_a2, out_a), "traced repeat must be bit-exact"
    assert chip_diff == 0.0, "the replicated expert must give identical outputs on every chip"
    assert resp_tt >= RESPONSE_RATIO_MIN * resp_ref, (resp_tt, resp_ref)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-o", "timeout=3600"])
