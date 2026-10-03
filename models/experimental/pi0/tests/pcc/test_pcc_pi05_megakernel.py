# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
pi0.5 megakernel PCC tests (device): ``PI0ModelTTNN(PI0ModelConfig(pi05=True), ...)``.

``test_pi05_base_vs_reference``: lerobot/pi05_base at the base shape (2 cameras, 224-token prompt, H = 50) with
padded prompts (a real prefix of ``N_REAL`` ids, then ``<pad>`` = 0) vs the fp32 torch reference (openpi's padding
mask and positions). Also: the mask is live (the pads unmasked change the output) and exact (other pad ids under the
mask give a bit-identical output), 10 replays are bit-identical, and a call is exactly three device ops (vision,
prefix, expert).

``test_pi05_libero_vs_openpi``: lerobot/pi05_libero at the LIBERO shape (2 cameras, 32-token prompt, H = 10) vs
openpi's GPU policy on recorded LIBERO observations with fixed noise (``PI05_OPENPI_GOLDEN``, skipped when absent):
PCC over the 7 LIBERO action dims of every record.

    pytest models/experimental/pi0/tests/pcc/test_pcc_pi05_megakernel.py

Weights: ``PI05_BASE_WEIGHTS`` / ``PI05_LIBERO_WEIGHTS`` (local directory or HuggingFace id).
"""

import os

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0.tt.ttnn_pi05_model import PI05_DEVICE_PARAMS
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN

PI05_BASE_WEIGHTS = os.environ.get("PI05_BASE_WEIGHTS", "lerobot/pi05_base")
PI05_LIBERO_WEIGHTS = os.environ.get("PI05_LIBERO_WEIGHTS", "lerobot/pi05_libero")
PI05_OPENPI_GOLDEN = os.environ.get("PI05_OPENPI_GOLDEN", "")

BASE_PROMPT_LEN = 224
N_REAL = (40, 97, 12, 150, 201, 224)  # real prompt ids of the base observations (the rest is <pad>; 224 = unpadded)
PCC_BASE_MIN = 0.95  # bf16 / bfp8 device vs fp32 torch on random inputs, per observation
PCC_BASE_MEAN = 0.98
LIBERO_H = 10
LIBERO_PROMPT_LEN = 32
PCC7_LIBERO_MEAN = 0.9995
PCC7_LIBERO_MIN = 0.999


def pi05_config(action_horizon: int) -> PI0ModelConfig:
    config = PI0ModelConfig(action_dim=32, action_horizon=action_horizon, state_dim=32, pi05=True)
    config.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    return config


def weights_available(path: str) -> bool:
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "model.safetensors"))
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(path, "model.safetensors"), str)
    except Exception:
        return False


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def padded_prompt(seed: int, n_real: int, prompt_len: int = BASE_PROMPT_LEN, horizon: int = 50, cameras: int = 2):
    """(images, tokens, lang_mask, noise): ``cameras`` random images in [-1, 1], ``n_real`` random ids then
    ``<pad>`` = 0."""
    g = torch.Generator().manual_seed(seed)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(cameras)]
    tokens = torch.zeros(1, prompt_len, dtype=torch.long)
    tokens[0, :n_real] = torch.randint(1, 256000, (n_real,), generator=g)
    mask = torch.zeros(1, prompt_len, dtype=torch.bool)
    mask[0, :n_real] = True
    noise = torch.randn(1, horizon, 32, generator=g)
    return images, tokens, mask, noise


def call(model, obs, lang_mask=None, tokens=None):
    images, toks, mask, noise = obs
    cams = [torch.ones(1, dtype=torch.bool)] * len(images)
    toks = toks if tokens is None else tokens
    mask = mask if lang_mask is None else lang_mask
    return model.sample_actions(images, cams, toks, mask, state=None, noise=noise)


def device_op_count(model) -> int:
    """Device operations launched by one execution of the traced device graph (graph capture of an eager run)."""
    pi05 = model.pi05_model
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        pi05.forward_device(pi05.presets[0].key)
    finally:
        captured = ttnn.graph.end_graph_capture()
    ttnn.synchronize_device(pi05.device)
    ops = [
        n
        for n in captured
        if n.get("node_type") == "function_start" and "DeviceOperation" in str(n["params"].get("name", ""))
    ]
    logger.info(f"device ops in one call: {[n['params']['name'] for n in ops]}")
    return len(ops)


@pytest.mark.skipif(not weights_available(PI05_BASE_WEIGHTS), reason=f"{PI05_BASE_WEIGHTS} not available")
@pytest.mark.timeout(1800)  # the fp32 torch reference of 6 observations runs on the host
@pytest.mark.parametrize("device_params", [PI05_DEVICE_PARAMS], indirect=True)
def test_pi05_base_vs_reference(device):
    loader = PI0WeightLoader(PI05_BASE_WEIGHTS)
    obs = [padded_prompt(i + 1, n) for i, n in enumerate(N_REAL)]
    ref = PI0ModelTorch(pi05_config(50), loader)
    refs = []
    for images, tokens, mask, noise in obs:
        ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone()
        with torch.no_grad():
            cams = [torch.ones(1, dtype=torch.bool)] * 2
            refs.append(ref.sample_actions(images, cams, tokens, mask, torch.zeros(1, 32)).float())
    del ref

    model = PI0ModelTTNN(pi05_config(50), loader, device)
    outs = [call(model, o) for o in obs]
    pccs = [pcc(o, r) for o, r in zip(outs, refs)]
    logger.info(f"base PCC vs fp32 reference (n_real {N_REAL}): {[round(p, 6) for p in pccs]}")
    assert min(pccs) >= PCC_BASE_MIN and sum(pccs) / len(pccs) >= PCC_BASE_MEAN

    # the padding mask is live: the same prompt with every pad treated as a real token changes the actions
    unmasked = call(model, obs[0], lang_mask=torch.ones_like(obs[0][2]))
    assert float((outs[0] - unmasked).abs().max()) > 0.0, "the padding mask is not applied"
    # ... and exact: other ids in the pad slots under the same mask leave the actions bit-identical
    other = obs[0][1].clone()
    g = torch.Generator().manual_seed(9)
    other[0, N_REAL[0] :] = torch.randint(1, 256000, (BASE_PROMPT_LEN - N_REAL[0],), generator=g)
    assert torch.equal(call(model, obs[0], tokens=other), outs[0])
    # lang_masks=None -> tokens != 0, the same as the explicit mask for a <pad> = 0 prompt; img_masks=None -> valid
    images, tokens, _, noise = obs[0]
    assert torch.equal(model.sample_actions(images, None, tokens, None, None, noise=noise), outs[0])

    # 10 replays bit-identical
    assert all(torch.equal(call(model, obs[0]), outs[0]) for _ in range(10))

    # three device ops per call: vision, prefix, expert
    model.pi05_model.release_trace()
    assert device_op_count(model) == 3
    assert torch.equal(call(model, obs[0]), outs[0])  # re-captured trace, same actions
    model.pi05_model.release_trace()


@pytest.mark.skipif(not weights_available(PI05_BASE_WEIGHTS), reason=f"{PI05_BASE_WEIGHTS} not available")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("horizon", [10, 50])  # both suffix buckets (32 / 64 rows)
@pytest.mark.parametrize("steps", [1, 5])
@pytest.mark.parametrize("device_params", [PI05_DEVICE_PARAMS], indirect=True)
def test_pi05_base_steps_vs_reference(device, steps, horizon):
    """N = 1 / 5 denoising steps: the padded prompts (each in its prompt bucket) vs the fp32 reference with N steps."""
    loader = PI0WeightLoader(PI05_BASE_WEIGHTS)
    obs = [padded_prompt(i + 1, n, horizon=horizon) for i, n in enumerate(N_REAL)]
    config = pi05_config(horizon)
    config.num_denoising_steps = steps
    ref = PI0ModelTorch(config, loader)
    refs = []
    for images, tokens, mask, noise in obs:
        ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone()
        with torch.no_grad():
            cams = [torch.ones(1, dtype=torch.bool)] * 2
            refs.append(ref.sample_actions(images, cams, tokens, mask, torch.zeros(1, 32)).float())
    del ref
    model = PI0ModelTTNN(config, loader, device)
    pccs = [pcc(call(model, o), r) for o, r in zip(obs, refs)]
    logger.info(f"base N {steps} H {horizon} PCC vs fp32 reference (n_real {N_REAL}): {[round(p, 6) for p in pccs]}")
    assert min(pccs) >= PCC_BASE_MIN and sum(pccs) / len(pccs) >= PCC_BASE_MEAN
    model.pi05_model.release_trace()


@pytest.mark.skipif(not weights_available(PI05_BASE_WEIGHTS), reason=f"{PI05_BASE_WEIGHTS} not available")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("horizon", [10, 50])  # both suffix buckets
@pytest.mark.parametrize("cameras", [1, 3, 4])
@pytest.mark.parametrize("device_params", [PI05_DEVICE_PARAMS], indirect=True)
def test_pi05_cameras_vs_reference(device, cameras, horizon):
    """num_cameras = 1 / 3: padded prompts (each in its prompt bucket) vs the fp32 reference with that many cameras;
    replays identical; a masked camera is refused with how to fix the call."""
    loader = PI0WeightLoader(PI05_BASE_WEIGHTS)
    obs = [padded_prompt(i + 1, n, horizon=horizon, cameras=cameras) for i, n in enumerate(N_REAL)]
    config = pi05_config(horizon)
    config.num_cameras = cameras
    ref = PI0ModelTorch(config, loader)
    refs = []
    for images, tokens, mask, noise in obs:
        ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone()
        with torch.no_grad():
            valid = [torch.ones(1, dtype=torch.bool)] * cameras
            refs.append(ref.sample_actions(images, valid, tokens, mask, torch.zeros(1, 32)).float())
    del ref
    model = PI0ModelTTNN(config, loader, device)
    outs = [call(model, o) for o in obs]
    pccs = [pcc(o, r) for o, r in zip(outs, refs)]
    logger.info(f"{cameras} cameras H {horizon} PCC vs fp32 reference (n_real {N_REAL}): {[round(p, 6) for p in pccs]}")
    assert min(pccs) >= PCC_BASE_MIN and sum(pccs) / len(pccs) >= PCC_BASE_MEAN
    assert all(torch.equal(call(model, obs[0]), outs[0]) for _ in range(10))
    images, tokens, mask, noise = obs[0]
    with pytest.raises(RuntimeError, match="drop the masked image slots"):
        masks = [torch.ones(1, dtype=torch.bool)] * (cameras - 1) + [torch.zeros(1, dtype=torch.bool)]
        model.sample_actions(images, masks, tokens, mask, None, noise=noise)
    model.pi05_model.release_trace()


@pytest.mark.skipif(not weights_available(PI05_BASE_WEIGHTS), reason=f"{PI05_BASE_WEIGHTS} not available")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [PI05_DEVICE_PARAMS], indirect=True)
def test_pi05_close_and_rebuild(device):
    """One process: a second model while the first is live is refused by name; close() frees the first model's trace
    region and L1; a model built afterwards (with `with`) reproduces the first model's output; then a 3-camera model,
    closed, and a 4-camera model (DRAM caches) built and run after it (the verifier's CB / L1 clash)."""
    loader = PI0WeightLoader(PI05_BASE_WEIGHTS)
    config = pi05_config(10)
    config.num_denoising_steps = 1
    obs = padded_prompt(1, 20, prompt_len=32, horizon=10)
    view = lambda bt: int(ttnn.get_memory_view(device, bt).total_bytes_allocated_per_bank)
    t0, l0 = view(ttnn.BufferType.TRACE), view(ttnn.BufferType.L1)
    a = PI0ModelTTNN(config, loader, device)
    out_a = call(a, obs).clone()
    assert view(ttnn.BufferType.TRACE) > t0 and view(ttnn.BufferType.L1) > l0
    with pytest.raises(RuntimeError, match="other model"):
        PI0ModelTTNN(config, loader, device)
    a.close()
    assert view(ttnn.BufferType.TRACE) == t0 and view(ttnn.BufferType.L1) == l0  # a is still referenced
    with pytest.raises(RuntimeError, match="closed"):
        call(a, obs)
    with PI0ModelTTNN(config, loader, device) as b:
        assert torch.equal(call(b, obs), out_a)
    assert view(ttnn.BufferType.TRACE) == t0 and view(ttnn.BufferType.L1) == l0
    for cams in (3, 4):
        config = pi05_config(50)
        config.num_denoising_steps = 1
        config.num_cameras = cams
        with PI0ModelTTNN(config, loader, device) as m:
            out = call(m, padded_prompt(2, 30, prompt_len=224, horizon=50, cameras=cams))
            assert bool(torch.isfinite(out).all())
    assert view(ttnn.BufferType.TRACE) == t0 and view(ttnn.BufferType.L1) == l0


@pytest.mark.skipif(not weights_available(PI05_BASE_WEIGHTS), reason=f"{PI05_BASE_WEIGHTS} not available")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [PI05_DEVICE_PARAMS], indirect=True)
def test_pi05_spill_merge_identity(device):
    """The VLM attention's spill merge (the 7-part path, MERGE_SPILL) is bit-identical to the in-DST merge where both
    apply: the 2-camera / 224-token / 64-row preset (4 parts) with the spill forced (PShape.force_spill) vs the
    production PREFIX program, end to end (eager)."""
    import dataclasses

    from models.experimental.pi0.common.pi05_host import unpad_rows
    from models.experimental.pi0.tt.megakernel.pe_program import PREFIX_OPS, PrefixEngineProgram

    with PI0ModelTTNN(pi05_config(50), PI0WeightLoader(PI05_BASE_WEIGHTS), device) as w:
        m = w.pi05_model
        cams = [torch.ones(1, dtype=torch.bool)] * 2
        args = (m.kv_caches, m.exp_mask, m.tables, m.noise, m.out)
        for seed, n_real in ((1, 150), (2, 224)):
            images, tokens, mask, noise = padded_prompt(seed, n_real)
            host = m.host_inputs(images, cams, tokens, mask, noise, 224)
            m.write_inputs(host)
            pr = m.programs[host["preset"].key]
            ps = pr.prefix.ps
            assert ps.v_np <= 6 and not ps.merge_spill
            spill = PrefixEngineProgram(
                device, pr.expert, m.prefix_tensors, dataclasses.replace(ps, force_spill=True), PREFIX_OPS
            )
            outs = []
            for prefix in (pr.prefix, spill):
                for v in pr.visions:
                    v.run(*args)
                prefix.run(*args)
                pr.expert.run(*args)
                ttnn.synchronize_device(device)
                outs.append(unpad_rows(ttnn.to_torch(m.out).float(), 50).clone())
            assert torch.equal(outs[0], outs[1]), seed


def load_golden():
    rec = torch.load(PI05_OPENPI_GOLDEN, weights_only=False)["records"]
    obs = []
    for r in rec:
        mi = r["model_inputs"]
        images = [mi["images"]["base_0_rgb"].float(), mi["images"]["left_wrist_0_rgb"].float()]
        tok, mask = mi["tokenized_prompt"][0].long(), mi["tokenized_prompt_mask"][0].bool()
        n = int(mask.sum())
        assert bool(mask[:n].all()) and n <= LIBERO_PROMPT_LEN
        tokens = torch.zeros(1, LIBERO_PROMPT_LEN, dtype=torch.long)
        tokens[0, :n] = tok[:n]
        lang_mask = torch.zeros(1, LIBERO_PROMPT_LEN, dtype=torch.bool)
        lang_mask[0, :n] = True
        obs.append(((images, tokens, lang_mask, r["noise"].float().reshape(1, LIBERO_H, 32)), r["actions_model_norm"]))
    return obs


@pytest.mark.skipif(not os.path.exists(PI05_OPENPI_GOLDEN), reason="PI05_OPENPI_GOLDEN (openpi records) not set")
@pytest.mark.skipif(not weights_available(PI05_LIBERO_WEIGHTS), reason=f"{PI05_LIBERO_WEIGHTS} not available")
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device_params", [PI05_DEVICE_PARAMS], indirect=True)
def test_pi05_libero_vs_openpi(device):
    """openpi feeds a third all -1 camera with mask False: a masked camera only adds masked keys and does not
    advance the positions, so the two real cameras are the same computation."""
    golden = load_golden()
    model = PI0ModelTTNN(pi05_config(LIBERO_H), PI0WeightLoader(PI05_LIBERO_WEIGHTS), device)
    pcc7 = []
    for obs, actions in golden:
        out = call(model, obs)
        pcc7.append(pcc(out[..., :7], actions.reshape(1, LIBERO_H, 32)[..., :7]))
    logger.info(f"LIBERO PCC7 vs openpi: mean {sum(pcc7) / len(pcc7):.6f} min {min(pcc7):.6f}")
    first = call(model, golden[0][0])
    assert all(torch.equal(call(model, golden[0][0]), first) for _ in range(10))
    assert sum(pcc7) / len(pcc7) >= PCC7_LIBERO_MEAN and min(pcc7) >= PCC7_LIBERO_MIN
    model.pi05_model.release_trace()
