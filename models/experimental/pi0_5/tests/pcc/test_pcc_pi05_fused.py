# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DEVICE test (not run in the host-only pass): fused / traced graph vs torch reference and vs the
legacy ttnn path, on the served shape (2 cameras, PI05_TOKEN_LEN tokens, 10 steps).

    # from the model repo root ($ROOT/models/pi05-base-p150), tree python, PYTHONPATH=code
    TT_FUSED=1 [PI05_TRACE=0] [PI05_FUSED_RESIDUAL=legacy] \
        pytest code/models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s

Gates (same as test_pcc_pi05_model / test_pcc_pi05_multireplan): e2e PCC(fused, torch) >= 0.93,
PCC(fused, legacy ttnn) >= 0.99 for the exact levers with PI05_FUSED_RESIDUAL=legacy (rounding-level
differences only), traced == eager fused bit-for-bit, repeat == repeat bit-for-bit, and a second
observation must move the output about as much as it moves the torch reference (multi-replan).

The fused-vs-legacy-ttnn comparison runs ONLY with PI05_FUSED_RESIDUAL=legacy: under ``bf16`` /
``mixed`` the model stores the expert o_proj / down_proj in bf16 and the legacy expert block (bf8
residual into dit_minimal_matmul_addcmul_fused) cannot run on it -- ``sample_actions`` raises a
RuntimeError (see FusedConfig.legacy_sample_actions_available); the true legacy baseline for those
modes is the TT_FUSED-unset run (DEVICE_VALIDATION.md step 0).
"""

import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

CHECKPOINT_PATH = os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base")
TOKEN_LEN = int(os.environ.get("PI05_TOKEN_LEN", "224"))
NUM_IMAGES = int(os.environ.get("PI05_NUM_IMAGES", "2"))
SEED = 42
PCC_E2E = 0.93
PCC_VS_LEGACY_EXACT = 0.99
RESPONSE_RATIO_MIN = 0.6


def create_pi05_config() -> PI0ModelConfig:
    config = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
    config.siglip_config = SigLIPConfig(
        hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16, image_size=224, patch_size=14
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


def run_legacy(model: PI0ModelTTNN, device, images, tokens, noise):
    images_ttnn = [
        ttnn.from_torch(im, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for im in images
    ]
    masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    tok = ttnn.from_torch(tokens, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    lm = ttnn.from_torch(torch.ones(1, TOKEN_LEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    st = ttnn.from_torch(torch.zeros(1, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = model.sample_actions(images=images_ttnn, img_masks=masks, lang_tokens=tok, lang_masks=lm, state=st, noise=noise)
    return ttnn.to_torch(out).float()


def run_torch(model_torch: PI0ModelTorch, images, tokens, noise):
    masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    lm = torch.ones(1, TOKEN_LEN, dtype=torch.bool)
    torch.manual_seed(SEED)
    # the torch reference draws its own noise: monkeypatch the sampler to use ours
    model_torch.denoising.sample_noise = lambda *a, **k: noise.clone()
    return model_torch.sample_actions(images, masks, tokens, lm, torch.zeros(1, 32)).float()


@pytest.fixture(scope="module")
def device():
    fused = FusedConfig.from_env()
    kwargs = dict(device_id=int(os.environ.get("PI0_DEVICE_ID", "0")), l1_small_size=24576)
    if fused.enabled and fused.trace:
        kwargs["trace_region_size"] = fused.trace_region_size
    dev = ttnn.open_device(**kwargs)
    dev.enable_program_cache()
    yield dev
    ttnn.close_device(dev)


def test_fused_vs_torch_and_legacy(device):
    fused_cfg = FusedConfig.from_env()
    if not fused_cfg.enabled:
        pytest.skip("set TT_FUSED=1")
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
    model = PI0ModelTTNN(config, loader, device, fused=fused_cfg)

    fused_a = model.sample_actions_fused(images, tokens, noise)  # compile + capture (+ execute)
    fused_a2 = model.sample_actions_fused(images, tokens, noise)  # replay
    fused_b = model.sample_actions_fused(images_b, tokens_b, noise)
    assert fused_a.shape == (1, 50, 32) and torch.isfinite(fused_a).all()
    assert torch.equal(fused_a, fused_a2), "trace replay must be deterministic"

    pcc_a = compute_pcc(fused_a, ref_a)
    pcc_b = compute_pcc(fused_b, ref_b)
    print(f"PCC fused vs torch: A={pcc_a:.4f} B={pcc_b:.4f} (trace={model._fused_trace_id is not None})")
    assert pcc_a >= PCC_E2E and pcc_b >= PCC_E2E

    # multi-replan responsiveness (stale prefix KV would collapse the fused response)
    resp_torch = (ref_b - ref_a).norm().item()
    resp_fused = (fused_b - fused_a).norm().item()
    print(f"response to a new observation: torch {resp_torch:.4f} fused {resp_fused:.4f}")
    assert resp_fused >= RESPONSE_RATIO_MIN * resp_torch

    # legacy ttnn path on the same model object for the A/B number -- only when the model keeps the
    # bf8 expert o_proj / down_proj (PI05_FUSED_RESIDUAL=legacy); with bf16 weights the legacy block
    # would hit TT_FATAL(ternary_a_data_format == in1_data_format) and sample_actions refuses to run.
    if fused_cfg.legacy_sample_actions_available:
        legacy_a = run_legacy(model, device, images, tokens, noise)
        pcc_fl = compute_pcc(fused_a, legacy_a)
        print(f"PCC fused vs legacy-ttnn (same weights, residual={fused_cfg.residual}): {pcc_fl:.4f}")
        assert pcc_fl >= PCC_VS_LEGACY_EXACT
    else:
        print(
            f"PI05_FUSED_RESIDUAL={fused_cfg.residual}: legacy sample_actions is not runnable on this model "
            "(bf16 expert o_proj / down_proj); compare against the TT_FUSED-unset run (step 0) instead"
        )
        with pytest.raises(RuntimeError, match="not available"):
            run_legacy(model, device, images, tokens, noise)
    model.release_trace()
