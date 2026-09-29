# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DEVICE tests of the fused / traced graph (``sample_actions_fused``), with openpi as ground truth.

    TT_FUSED=1 pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s -k libero   # one model each:
    TT_FUSED=1 pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s -k base     # run separately
    python models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py {libero|base} --out results.json

``libero``: lerobot/pi05_libero at openpi's pi05_libero shape (2 cameras, prompt right-padded to 32, H = 10) on the
openpi GPU golden (``golden_openpi.py``): PCC over the 7 action dims of every record; ten trace replays bit-identical.

``base``: lerobot/pi05_base at the served shape (2 cameras, 224 tokens, H = 50) with PADDED prompts (a real prefix
of ``N_REAL`` ids, then ``<pad>`` = 0): PCC vs the torch reference (which applies openpi's padding mask and suffix
positions, see ``test_reference_vs_openpi.py``); the mask is live (treating the pads as real tokens changes the
output) and exact (changing the pad ids under the mask leaves the output bit-identical); ten replays bit-identical.

Both report the traced latency: ``sample_actions_fused`` wall (upload + replay + readback) and ``execute_trace``
alone, median of ``PI05_BENCH_RUNS`` (default 60).
"""

import json
import os
import statistics
import sys
import time

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model as PI0ModelTorch
from models.experimental.pi0_5.tests.pcc.golden_openpi import (
    LIBERO_WEIGHTS,
    libero_config,
    load_records,
    pcc,
    pcc7,
    record_inputs,
)
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

BASE_WEIGHTS = os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base")
BENCH_RUNS = int(os.environ.get("PI05_BENCH_RUNS", "60"))
LIBERO_LANG_LEN = 32
BASE_TOKEN_LEN = 224
N_REAL = (40, 97, 12, 150, 201, 224)  # real prompt ids of the base observations (the rest is <pad>; 224 = unpadded)
PCC7_LIBERO_MIN = 0.95  # per record; the mean gate is below
PCC7_LIBERO_MEAN = 0.9828  # the eager workaround's mean (tt_pi05_policy.py mode=openpi, 2026-09-2x)
PCC_BASE_MIN = 0.95  # per observation: bf16 device vs fp32 torch on random inputs (unpadded: 0.90-0.999, README)
PCC_BASE_MEAN = 0.98


def base_config() -> PI0ModelConfig:
    config = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
    config.siglip_config = SigLIPConfig(
        hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16, image_size=224, patch_size=14
    )
    return config


def open_device():
    from models.experimental.pi0_5.common.device_open import open_pi05_device

    return open_pi05_device(FusedConfig.from_env(), device_id=int(os.environ.get("PI0_DEVICE_ID", "0")))


def stamp_of(model) -> dict:
    """The constructed model's backend stamp (DESIGN.md §4.12): read from the object, never from the env."""
    return {"megakernel_backend": getattr(model, "megakernel_backend", "off"),
            "megakernel_program": getattr(model, "megakernel_program", None)}


@pytest.fixture(scope="module")
def device():
    dev = open_device()
    yield dev
    ttnn.close_device(dev)


def bench(model, device, args, runs=BENCH_RUNS):
    """``args`` = (images, tokens, noise, lang_masks)."""
    call, replay = [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.sample_actions_fused(*args[:3], lang_masks=args[3])
        call.append((time.perf_counter() - t0) * 1e3)
    for _ in range(runs):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, model._fused_trace_id, cq_id=0, blocking=True)
        replay.append((time.perf_counter() - t0) * 1e3)
    return {
        "runs": runs,
        "call_ms_median": statistics.median(call),
        "call_ms_min": min(call),
        "replay_ms_median": statistics.median(replay),
        "replay_ms_min": min(replay),
    }


def replays_identical(model, args, n=10):
    first = model.sample_actions_fused(*args[:3], lang_masks=args[3])
    return all(torch.equal(first, model.sample_actions_fused(*args[:3], lang_masks=args[3])) for _ in range(n - 1))


def run_libero(device):
    records = load_records()
    torch.manual_seed(42)
    model = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), device, fused=FusedConfig.from_env())
    rows = []
    for rec in records:
        images, tokens, lang_mask, noise = record_inputs(rec, LIBERO_LANG_LEN)
        out = model.sample_actions_fused(images, tokens, noise, lang_masks=lang_mask)
        rows.append({"tag": rec["tag"], "n_lang": int(lang_mask.sum()), "pcc7": pcc7(out, rec["actions_model_norm"]),
                     "pcc32": pcc(out, rec["actions_model_norm"]),
                     "maxabs7": float((out[..., :7] - rec["actions_model_norm"][..., :7]).abs().max())})
        print(json.dumps(rows[-1]), flush=True)
    images, tokens, lang_mask, noise = record_inputs(records[0], LIBERO_LANG_LEN)
    args = (images, tokens, noise, lang_mask)
    res = {
        "shape": "libero 2x224^2, lang_len 32, H 10",
        "traced": model._fused_trace_id is not None,
        "fused_cfg": {k: str(v) for k, v in model.fused_cfg.describe().items()},
        "stamp": stamp_of(model),
        "rows": rows,
        "pcc7_mean": sum(r["pcc7"] for r in rows) / len(rows),
        "pcc7_min": min(r["pcc7"] for r in rows),
        "ten_replays_bit_identical": replays_identical(model, args),
        "latency": bench(model, device, args),
    }
    model.release_trace()
    return res


def padded_prompt(seed: int, n_real: int, pad_fill: int = 0):
    g = torch.Generator().manual_seed(seed)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    tokens = torch.full((1, BASE_TOKEN_LEN), pad_fill, dtype=torch.long)
    tokens[0, :n_real] = torch.randint(1, 256000, (n_real,), generator=g)
    mask = torch.zeros(1, BASE_TOKEN_LEN, dtype=torch.bool)
    mask[0, :n_real] = True
    noise = torch.randn(1, 50, 32, generator=g)
    return images, tokens, noise, mask


def run_base(device):
    loader = PI0WeightLoader(BASE_WEIGHTS)
    obs = [padded_prompt(i + 1, n) for i, n in enumerate(N_REAL)]
    ref = PI0ModelTorch(base_config(), loader)
    refs = []
    for images, tokens, noise, mask in obs:
        ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone()
        with torch.no_grad():
            refs.append(ref.sample_actions(images, [torch.ones(1, dtype=torch.bool)] * 2, tokens, mask, torch.zeros(1, 32)).float())
    del ref
    torch.manual_seed(42)
    model = PI0ModelTTNN(base_config(), loader, device, fused=FusedConfig.from_env())
    outs = [model.sample_actions_fused(i, t, n, lang_masks=m) for (i, t, n, m) in obs]
    images, tokens, noise, mask = obs[0]
    # mask live: the same padded prompt with every pad treated as a real token
    unmasked = model.sample_actions_fused(images, tokens, noise, lang_masks=torch.ones_like(mask))
    # mask exact: different ids in the pad slots, same mask -> the pads are invisible
    other_pads = tokens.clone()
    other_pads[0, N_REAL[0]:] = torch.randint(1, 256000, (BASE_TOKEN_LEN - N_REAL[0],), generator=torch.Generator().manual_seed(9))
    repadded = model.sample_actions_fused(images, other_pads, noise, lang_masks=mask)
    # default mask (tokens != 0) == explicit mask for a <pad>=0 prompt
    default_mask = model.sample_actions_fused(images, tokens, noise)
    res = {
        "shape": "base 2x224^2, 224 tokens, H 50",
        "n_real": list(N_REAL),
        "traced": model._fused_trace_id is not None,
        "stamp": stamp_of(model),
        "fused_cfg": {k: str(v) for k, v in model.fused_cfg.describe().items()},
        "pcc_vs_reference": [pcc(o, r) for o, r in zip(outs, refs)],
        "masked_vs_unmasked_pcc": pcc(outs[0], unmasked),
        "masked_vs_unmasked_maxabs": float((outs[0] - unmasked).abs().max()),
        "unmasked_pcc_vs_reference": pcc(unmasked, refs[0]),
        "pad_ids_invisible_bit_identical": bool(torch.equal(outs[0], repadded)),
        "default_mask_bit_identical": bool(torch.equal(outs[0], default_mask)),
        "ten_replays_bit_identical": replays_identical(model, obs[0]),
        "latency": bench(model, device, obs[0]),
    }
    model.release_trace()
    return res


def test_libero_vs_openpi(device):
    res = run_libero(device)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
    assert res["stamp"]["megakernel_backend"] == FusedConfig.from_env().megakernel
    assert res["traced"] and res["ten_replays_bit_identical"]
    assert res["pcc7_mean"] >= PCC7_LIBERO_MEAN and res["pcc7_min"] >= PCC7_LIBERO_MIN


def test_base_padded_vs_reference(device):
    res = run_base(device)
    print(json.dumps(res, indent=1))
    assert res["stamp"]["megakernel_backend"] == FusedConfig.from_env().megakernel
    assert res["traced"] and res["ten_replays_bit_identical"]
    assert min(res["pcc_vs_reference"]) >= PCC_BASE_MIN
    assert sum(res["pcc_vs_reference"]) / len(res["pcc_vs_reference"]) >= PCC_BASE_MEAN
    assert res["masked_vs_unmasked_maxabs"] > 0.0, "the padding mask is not applied"
    assert res["pad_ids_invisible_bit_identical"] and res["default_mask_bit_identical"]


if __name__ == "__main__":
    which = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    dev = open_device()
    try:
        res = run_libero(dev) if which == "libero" else run_base(dev)
    finally:
        ttnn.close_device(dev)
    res["time"] = time.strftime("%F %T")
    print("RESULT", json.dumps({k: v for k, v in res.items() if k not in ("rows", "fused_cfg")}), flush=True)
    if out:
        json.dump(res, open(out, "w"), indent=1)
