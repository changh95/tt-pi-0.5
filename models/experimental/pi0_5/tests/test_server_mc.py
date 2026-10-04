# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the server's multi-config backend (no device, no weights): backend selection, start-up refusals,
the openpi-exact image normalisation."""
import base64
import io

import numpy as np
import pytest
import torch
from PIL import Image

from models.experimental.pi0_5.server import app as A
from models.experimental.pi0_5.server import mc_backend as M


def _cfg(monkeypatch, **env):
    for k in ("PI05_MEGAKERNEL", "TT_MESH_SHAPE", "PI05_NUM_IMAGES", "PI05_ACTION_HORIZON", "PI05_NUM_STEPS"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return A.load_config()


def test_default_backend_is_mc_on_one_chip(monkeypatch):
    cfg = _cfg(monkeypatch)
    assert cfg.backend == "mc" and cfg.num_images == 2 and cfg.action_horizon == 50 and cfg.num_steps == 10
    assert _cfg(monkeypatch, PI05_MEGAKERNEL="mc").backend == "mc"
    assert _cfg(monkeypatch, PI05_MEGAKERNEL="whole").backend == "whole"
    assert _cfg(monkeypatch, TT_MESH_SHAPE="1x4").backend == "legacy"  # unset on a mesh: the stock-op path


@pytest.mark.parametrize(
    "env, why",
    [
        ({"PI05_NUM_IMAGES": "5"}, "cameras = 5"),
        ({"PI05_NUM_IMAGES": "0"}, "cameras = 0"),
        ({"PI05_ACTION_HORIZON": "65"}, "action_horizon=65"),
        ({"PI05_ACTION_HORIZON": "0"}, "action_horizon=0"),
        ({"PI05_NUM_STEPS": "17"}, "num_denoising_steps=17"),
        ({"PI05_MEGAKERNEL": "mc", "TT_MESH_SHAPE": "1x4"}, "single-chip"),
        ({"PI05_BATCH_SIZES": "1,2"}, "batch 1 only"),
        ({"PI05_TOKEN_LEN": "256"}, "PI05_TOKEN_LEN=256"),
    ],
)
def test_mc_refuses_by_name(monkeypatch, env, why):
    monkeypatch.delenv("PI05_BATCH_SIZES", raising=False)
    monkeypatch.delenv("PI05_TOKEN_LEN", raising=False)
    with pytest.raises(RuntimeError, match=why):
        _cfg(monkeypatch, **env)


@pytest.mark.parametrize("cams, h, n", [(1, 1, 1), (2, 50, 10), (3, 32, 5), (4, 64, 16), (2, 33, 1)])
def test_mc_accepts_the_compiled_range(monkeypatch, cams, h, n):
    cfg = _cfg(monkeypatch, PI05_NUM_IMAGES=str(cams), PI05_ACTION_HORIZON=str(h), PI05_NUM_STEPS=str(n))
    assert (cfg.backend, cfg.num_images, cfg.action_horizon, cfg.num_steps) == ("mc", cams, h, n)


def test_single_config_paths_keep_h50(monkeypatch):
    with pytest.raises(RuntimeError, match="serve 50 only"):
        _cfg(monkeypatch, PI05_MEGAKERNEL="whole", PI05_ACTION_HORIZON="10")


def test_decode_image_is_openpi_exact():
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    got, size = A.decode_image(base64.b64encode(buf.getvalue()).decode(), 0)
    x = torch.from_numpy(u8).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    want = x * torch.tensor(1.0 / 255.0, dtype=torch.float32) * 2.0 - 1.0  # openpi on CUDA
    assert size == (224, 224) and got.dtype == torch.float32 and torch.equal(got, want)
    assert float(got.min()) >= -1.0 and float(got.max()) <= 1.0


def test_refusal_helper_matches_the_compiled_presets():
    from models.experimental.pi0.tt.megakernel import presets as PS

    assert M.CAMERAS == PS.CAMERAS and M.PROMPT_BUCKETS == PS.PROMPT_BUCKETS
    assert M.refusal(2, 50, 10, (1, 1), "mesh", (1,), 224) is None
