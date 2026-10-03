# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
pi0.5 megakernel latency (device), host clock: the median of ``sample_actions`` (host inputs + upload + trace
replay + readback) and of the trace replay alone, at both compiled shapes.

    pytest models/experimental/pi0/tests/perf/test_perf_pi05_megakernel.py

``base``: lerobot/pi05_base, 2 x 224^2, 224-token prompt (40 real ids), H = 50.
``libero``: lerobot/pi05_libero (``PI05_LIBERO_WEIGHTS``), 2 x 224^2, 32-token prompt (12 real ids), H = 10.
"""

import os
import statistics
import time

import pytest
import torch
from loguru import logger

from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.tt.ttnn_pi05_model import PI05_DEVICE_PARAMS
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN

PI05_BASE_WEIGHTS = os.environ.get("PI05_BASE_WEIGHTS", "lerobot/pi05_base")
PI05_LIBERO_WEIGHTS = os.environ.get("PI05_LIBERO_WEIGHTS", "lerobot/pi05_libero")
NUM_WARMUP_ITERATIONS = 3
NUM_INFERENCE_ITERATIONS = 60
# (horizon, prompt length, real prompt ids, weights)
SHAPES = {"base": (50, 224, 40, PI05_BASE_WEIGHTS), "libero": (10, 32, 12, PI05_LIBERO_WEIGHTS)}


def weights_available(path: str) -> bool:
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "model.safetensors"))
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(path, "model.safetensors"), str)
    except Exception:
        return False


@pytest.mark.timeout(900)
@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("device_params", [PI05_DEVICE_PARAMS], indirect=True)
def test_perf_pi05_megakernel(device, shape):
    horizon, prompt_len, n_real, weights = SHAPES[shape]
    if not weights_available(weights):
        pytest.skip(f"{weights} not available")
    config = PI0ModelConfig(action_dim=32, action_horizon=horizon, state_dim=32, pi05=True)
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    g = torch.Generator().manual_seed(1)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    img_masks = [torch.ones(1, dtype=torch.bool)] * 2
    tokens = torch.zeros(1, prompt_len, dtype=torch.long)
    tokens[0, :n_real] = torch.randint(1, 256000, (n_real,), generator=g)
    lang_masks = tokens != 0
    noise = torch.randn(1, horizon, 32, generator=g)

    t0 = time.perf_counter()
    model = PI0ModelTTNN(config, PI0WeightLoader(weights), device)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.sample_actions(images, img_masks, tokens, lang_masks, None, noise=noise)  # compile + trace capture
    first_s = time.perf_counter() - t0
    for _ in range(NUM_WARMUP_ITERATIONS):
        model.sample_actions(images, img_masks, tokens, lang_masks, None, noise=noise)

    call_ms = []
    for _ in range(NUM_INFERENCE_ITERATIONS):
        t0 = time.perf_counter()
        model.sample_actions(images, img_masks, tokens, lang_masks, None, noise=noise)
        call_ms.append((time.perf_counter() - t0) * 1e3)
    pi05 = model.pi05_model
    key = pi05.host_inputs(images, img_masks, tokens, lang_masks, noise)["preset"].key  # the request's preset
    replay_ms = []
    for _ in range(NUM_INFERENCE_ITERATIONS):
        t0 = time.perf_counter()
        pi05.replay(key)
        replay_ms.append((time.perf_counter() - t0) * 1e3)
    pi05.release_trace()

    logger.info(
        f"pi0.5 megakernel [{shape}: 2 x 224^2, {n_real} prompt tokens (preset {key}), H {horizon}]: build {build_s:.1f} s, "
        f"first call {first_s:.1f} s, sample_actions median {statistics.median(call_ms):.2f} ms "
        f"(min {min(call_ms):.2f}), replay median {statistics.median(replay_ms):.2f} ms (min {min(replay_ms):.2f}), "
        f"{NUM_INFERENCE_ITERATIONS} iterations"
    )
    assert statistics.median(replay_ms) <= statistics.median(call_ms)
