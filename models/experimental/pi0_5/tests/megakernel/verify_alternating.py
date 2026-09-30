# Copied from the fused-fix session scratchpad (verify_alternating.py); device open via common/device_open.py.
#!/usr/bin/env python3
"""
Device verification - finding 3 (trace refresh on alternating prompts) + shape switch + golden PCC.

Checks:
  A. alternating_base   : n_real=12 / n_real=150 at base shape, 6 alternating calls.
                          Reference for each prompt = FRESH model that has only seen that prompt.
                          Positive control: the two fresh references differ.
  B. alternating_libero : two LIBERO records with different n_lang, 6 alternating calls.
                          Same fresh-reference design.
  C. shape_switch       : base model released, then LIBERO model, in same process.
  D. golden_libero      : all LIBERO records vs openpi golden, pcc7_mean >= 0.9828.

Pass criterion for A/B: each of the 6 alternating outputs is bit-identical or PCC >= 1-1e-6
vs its FRESH-MODEL reference (not vs a within-process reference).

Usage:
  TT_FUSED=1 PYTHONPATH=<repo> python3 verify_alternating.py [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import ttnn

REPO = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tests.pcc.golden_openpi import (
    LIBERO_WEIGHTS,
    libero_config,
    load_records,
    pcc,
    pcc7,
    record_inputs,
)
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

BASE_WEIGHTS = os.environ.get(
    "PI05_BASE_WEIGHTS",
    "/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
)
BASE_TOKEN_LEN = 224
LIBERO_LANG_LEN = 32
PCC_THRESHOLD = 1.0 - 1e-6


def base_config() -> PI0ModelConfig:
    cfg = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
    cfg.siglip_config = SigLIPConfig(
        hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
        num_attention_heads=16, image_size=224, patch_size=14,
    )
    return cfg


def padded_obs(seed: int, n_real: int, token_len: int = BASE_TOKEN_LEN):
    """Returns (images, tokens, noise, lang_mask); positional order matches sample_actions_fused."""
    g = torch.Generator().manual_seed(seed)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    tokens = torch.zeros(1, token_len, dtype=torch.long)
    tokens[0, :n_real] = torch.randint(1, 256000, (n_real,), generator=g)
    mask = torch.zeros(1, token_len, dtype=torch.bool)
    mask[0, :n_real] = True
    noise = torch.randn(1, 50, 32, generator=g)
    return images, tokens, noise, mask


def open_device():
    fused = FusedConfig.from_env()
    from models.experimental.pi0_5.common.device_open import device_kwargs  # the megakernel's 64 KiB worker-L1 cut

    dev = ttnn.open_device(**device_kwargs(fused, device_id=int(os.environ.get("PI0_DEVICE_ID", "0"))))
    dev.enable_program_cache()
    return dev, fused


def assert_equiv(a: torch.Tensor, b: torch.Tensor, label: str,
                 threshold: float = PCC_THRESHOLD) -> dict:
    bit = bool(torch.equal(a, b))
    p = float(pcc(a, b))
    ok = bit or p >= threshold
    tag = "BIT_IDENTICAL" if bit else f"PCC={p:.8f}"
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}: {tag}", flush=True)
    return {"label": label, "bit_identical": bit, "pcc": p, "ok": ok}


def fresh_ref(device, fused_cfg, cfg, weights_path: str, obs):
    """
    Build a FRESH model that has seen only one prompt (obs), capture and return its output,
    then release the trace. The returned tensor is the reference for that prompt.
    obs = (images, tokens, noise, lang_mask).
    """
    loader = PI0WeightLoader(weights_path)
    torch.manual_seed(42)
    model = PI0ModelTTNN(cfg, loader, device, fused=fused_cfg)
    out = model.sample_actions_fused(*obs[:3], lang_masks=obs[3]).clone()
    model.release_trace()
    return out


def _pick_two_different_lang(records):
    """Return two records with maximally distinct n_lang values."""
    by_nlang = {}
    for rec in records:
        _, _, mask, _ = record_inputs(rec, LIBERO_LANG_LEN)
        n = int(mask.sum())
        by_nlang.setdefault(n, rec)
    if len(by_nlang) < 2:
        raise RuntimeError(f"All LIBERO records share n_lang; values: {list(by_nlang.keys())}")
    nlang_sorted = sorted(by_nlang.keys())
    return by_nlang[nlang_sorted[0]], by_nlang[nlang_sorted[-1]]


# ---------------------------------------------------------------------------
# A. Alternating base prompts (finding 3, base shape)
# ---------------------------------------------------------------------------

def test_alternating_base(device, fused_cfg) -> dict:
    """
    Three model instances, all sequential (each released before the next is built):
      1. model_ref_a : sees only prompt A (n_real=12, seed 3)  -> ref_a
      2. model_ref_b : sees only prompt B (n_real=150, seed 5) -> ref_b
      3. model_alt   : alternates A B A B A B; each output compared vs ref_a / ref_b

    Positive control: pcc(ref_a, ref_b) << 1 (the two prompts produce different outputs).
    Secondary: each alternating call also compared vs first alternating call for same prompt.
    """
    print("\n[A] alternating_base: capturing fresh-model references ...", flush=True)
    obs_a = padded_obs(3, 12)
    obs_b = padded_obs(5, 150)

    ref_a = fresh_ref(device, fused_cfg, base_config(), BASE_WEIGHTS, obs_a)
    ref_b = fresh_ref(device, fused_cfg, base_config(), BASE_WEIGHTS, obs_b)

    pcc_ab = float(pcc(ref_a, ref_b))
    print(f"  positive control pcc(ref_a, ref_b) = {pcc_ab:.6f}  (must be << 1)", flush=True)
    if pcc_ab > 0.999:
        print("  WARNING: positive control WEAK - the two prompts produce nearly identical outputs", flush=True)
    positive_ok = pcc_ab < 0.999

    print("[A] building alternating model ...", flush=True)
    loader = PI0WeightLoader(BASE_WEIGHTS)
    torch.manual_seed(42)
    model_alt = PI0ModelTTNN(base_config(), loader, device, fused=fused_cfg)

    # First call of each prompt in the alternating model (secondary / consistency column)
    first_a_in_alt = None
    first_b_in_alt = None

    pattern = ["A", "B", "A", "B", "A", "B"]
    checks_vs_fresh = []
    checks_vs_within = []
    for i, which in enumerate(pattern):
        obs, ref_fresh, lbl = (obs_a, ref_a, "n_real=12") if which == "A" else (obs_b, ref_b, "n_real=150")
        out = model_alt.sample_actions_fused(*obs[:3], lang_masks=obs[3])

        # Primary: vs fresh-model reference (this is what detects a stale mask)
        c_fresh = assert_equiv(out, ref_fresh, f"call{i+1} {lbl} vs fresh-ref")
        checks_vs_fresh.append(c_fresh)

        # Secondary: vs first call of same prompt within alternating model
        if which == "A":
            if first_a_in_alt is None:
                first_a_in_alt = out.clone()
                checks_vs_within.append({"label": f"call{i+1} {lbl} first-in-alt", "bit_identical": True, "pcc": 1.0, "ok": True})
            else:
                c_w = assert_equiv(out, first_a_in_alt, f"call{i+1} {lbl} vs first-in-alt")
                checks_vs_within.append(c_w)
        else:
            if first_b_in_alt is None:
                first_b_in_alt = out.clone()
                checks_vs_within.append({"label": f"call{i+1} {lbl} first-in-alt", "bit_identical": True, "pcc": 1.0, "ok": True})
            else:
                c_w = assert_equiv(out, first_b_in_alt, f"call{i+1} {lbl} vs first-in-alt")
                checks_vs_within.append(c_w)

    model_alt.release_trace()
    all_ok = positive_ok and all(c["ok"] for c in checks_vs_fresh)
    print(f"[A] alternating_base all_ok={all_ok}  positive_control_ok={positive_ok}", flush=True)
    return {
        "test": "alternating_base",
        "pcc_ref_a_vs_ref_b": pcc_ab,
        "positive_control_ok": positive_ok,
        "checks_vs_fresh": checks_vs_fresh,
        "checks_vs_within_alt": checks_vs_within,
        "all_ok": all_ok,
    }


# ---------------------------------------------------------------------------
# B. Alternating LIBERO prompts (finding 3, LIBERO shape)
# ---------------------------------------------------------------------------

def test_alternating_libero(device, fused_cfg) -> dict:
    """
    Same design as A but for LIBERO (H=10, lang_len=32).
    Two records with different n_lang (max spread from golden file).
    """
    print("\n[B] alternating_libero: selecting records ...", flush=True)
    records = load_records()
    rec_a, rec_b = _pick_two_different_lang(records)
    imgs_a, tok_a, mask_a, noise_a = record_inputs(rec_a, LIBERO_LANG_LEN)
    imgs_b, tok_b, mask_b, noise_b = record_inputs(rec_b, LIBERO_LANG_LEN)
    na, nb = int(mask_a.sum()), int(mask_b.sum())
    print(f"  record A tag={rec_a['tag']} n_lang={na}", flush=True)
    print(f"  record B tag={rec_b['tag']} n_lang={nb}", flush=True)

    obs_a = (imgs_a, tok_a, noise_a, mask_a)
    obs_b = (imgs_b, tok_b, noise_b, mask_b)

    print("[B] capturing fresh-model references ...", flush=True)
    ref_a = fresh_ref(device, fused_cfg, libero_config(), LIBERO_WEIGHTS, obs_a)
    ref_b = fresh_ref(device, fused_cfg, libero_config(), LIBERO_WEIGHTS, obs_b)

    p7a = float(pcc7(ref_a, rec_a["actions_model_norm"]))
    p7b = float(pcc7(ref_b, rec_b["actions_model_norm"]))
    pcc_ab = float(pcc(ref_a, ref_b))
    print(f"  fresh A PCC7 vs openpi {p7a:.6f}  B {p7b:.6f}", flush=True)
    print(f"  positive control pcc(ref_a, ref_b) = {pcc_ab:.6f}  (must be << 1)", flush=True)
    if pcc_ab > 0.999:
        print("  WARNING: positive control WEAK", flush=True)
    positive_ok = pcc_ab < 0.999

    print("[B] building alternating model ...", flush=True)
    loader = PI0WeightLoader(LIBERO_WEIGHTS)
    torch.manual_seed(42)
    model_alt = PI0ModelTTNN(libero_config(), loader, device, fused=fused_cfg)

    first_a_in_alt = None
    first_b_in_alt = None
    pattern = ["A", "B", "A", "B", "A", "B"]
    checks_vs_fresh = []
    checks_vs_within = []
    for i, which in enumerate(pattern):
        if which == "A":
            out = model_alt.sample_actions_fused(imgs_a, tok_a, noise_a, lang_masks=mask_a)
            c_fresh = assert_equiv(out, ref_a, f"call{i+1} rec_A n_lang={na} vs fresh-ref")
            checks_vs_fresh.append(c_fresh)
            if first_a_in_alt is None:
                first_a_in_alt = out.clone()
                checks_vs_within.append({"label": f"call{i+1} rec_A first-in-alt", "bit_identical": True, "pcc": 1.0, "ok": True})
            else:
                checks_vs_within.append(assert_equiv(out, first_a_in_alt, f"call{i+1} rec_A vs first-in-alt"))
        else:
            out = model_alt.sample_actions_fused(imgs_b, tok_b, noise_b, lang_masks=mask_b)
            c_fresh = assert_equiv(out, ref_b, f"call{i+1} rec_B n_lang={nb} vs fresh-ref")
            checks_vs_fresh.append(c_fresh)
            if first_b_in_alt is None:
                first_b_in_alt = out.clone()
                checks_vs_within.append({"label": f"call{i+1} rec_B first-in-alt", "bit_identical": True, "pcc": 1.0, "ok": True})
            else:
                checks_vs_within.append(assert_equiv(out, first_b_in_alt, f"call{i+1} rec_B vs first-in-alt"))

    model_alt.release_trace()
    all_ok = positive_ok and all(c["ok"] for c in checks_vs_fresh)
    print(f"[B] alternating_libero all_ok={all_ok}", flush=True)
    return {
        "test": "alternating_libero",
        "n_lang": [na, nb],
        "pcc7_vs_openpi": [p7a, p7b],
        "pcc_ref_a_vs_ref_b": pcc_ab,
        "positive_control_ok": positive_ok,
        "checks_vs_fresh": checks_vs_fresh,
        "checks_vs_within_alt": checks_vs_within,
        "all_ok": all_ok,
    }


# ---------------------------------------------------------------------------
# C. Shape switch: base model released, then LIBERO model, same process
# ---------------------------------------------------------------------------

def test_shape_switch(device, fused_cfg) -> dict:
    """
    Same process: build base model, 2 replays, release_trace(), build LIBERO model, 2 replays.
    Fresh references come from the first call to each model (the LIBERO model has never seen
    base prompts, so its first call is a genuine fresh reference for that shape).
    """
    print("\n[C] shape_switch: building base model ...", flush=True)
    obs_a = padded_obs(3, 12)
    obs_b = padded_obs(5, 150)

    loader = PI0WeightLoader(BASE_WEIGHTS)
    torch.manual_seed(42)
    base_model = PI0ModelTTNN(base_config(), loader, device, fused=fused_cfg)

    ref_a = base_model.sample_actions_fused(*obs_a[:3], lang_masks=obs_a[3]).clone()
    ref_b = base_model.sample_actions_fused(*obs_b[:3], lang_masks=obs_b[3]).clone()
    checks = [
        assert_equiv(base_model.sample_actions_fused(*obs_a[:3], lang_masks=obs_a[3]), ref_a, "base n_real=12 replay1"),
        assert_equiv(base_model.sample_actions_fused(*obs_b[:3], lang_masks=obs_b[3]), ref_b, "base n_real=150 replay1"),
    ]
    base_model.release_trace()
    print("[C] base released; building LIBERO model ...", flush=True)

    records = load_records()
    rec = records[0]
    imgs_lib, tok_lib, mask_lib, noise_lib = record_inputs(rec, LIBERO_LANG_LEN)
    n_lang = int(mask_lib.sum())

    libero_loader = PI0WeightLoader(LIBERO_WEIGHTS)
    torch.manual_seed(42)
    libero_model = PI0ModelTTNN(libero_config(), libero_loader, device, fused=fused_cfg)

    lib_ref = libero_model.sample_actions_fused(imgs_lib, tok_lib, noise_lib, lang_masks=mask_lib).clone()
    p7_lib = float(pcc7(lib_ref, rec["actions_model_norm"]))
    checks += [
        assert_equiv(libero_model.sample_actions_fused(imgs_lib, tok_lib, noise_lib, lang_masks=mask_lib), lib_ref, f"libero n_lang={n_lang} replay1"),
        assert_equiv(libero_model.sample_actions_fused(imgs_lib, tok_lib, noise_lib, lang_masks=mask_lib), lib_ref, f"libero n_lang={n_lang} replay2"),
    ]
    libero_model.release_trace()

    all_ok = all(c["ok"] for c in checks)
    print(f"[C] shape_switch all_ok={all_ok}  libero_pcc7_vs_openpi={p7_lib:.6f}", flush=True)
    return {
        "test": "shape_switch",
        "libero_pcc7_vs_openpi": p7_lib,
        "checks": checks,
        "all_ok": all_ok,
    }


# ---------------------------------------------------------------------------
# D. Golden PCC re-run
# ---------------------------------------------------------------------------

def test_golden_libero(device, fused_cfg) -> dict:
    """All LIBERO records vs openpi golden; requires pcc7_mean >= 0.9828, pcc7_min >= 0.95."""
    print("\n[D] golden_libero: building model ...", flush=True)
    loader = PI0WeightLoader(LIBERO_WEIGHTS)
    torch.manual_seed(42)
    model = PI0ModelTTNN(libero_config(), loader, device, fused=fused_cfg)
    rows = []
    for rec in load_records():
        imgs, tok, mask, noise = record_inputs(rec, LIBERO_LANG_LEN)
        out = model.sample_actions_fused(imgs, tok, noise, lang_masks=mask)
        p7 = float(pcc7(out, rec["actions_model_norm"]))
        n = int(mask.sum())
        rows.append({"tag": rec["tag"], "n_lang": n, "pcc7": p7})
        print(f"  {rec['tag']} n_lang={n} pcc7={p7:.6f}", flush=True)
    model.release_trace()
    mean7 = sum(r["pcc7"] for r in rows) / len(rows)
    min7 = min(r["pcc7"] for r in rows)
    ok = mean7 >= 0.9828 and min7 >= 0.95
    print(f"[D] golden_libero pcc7_mean={mean7:.6f} pcc7_min={min7:.6f} ok={ok}", flush=True)
    return {"test": "golden_libero", "rows": rows, "pcc7_mean": mean7, "pcc7_min": min7, "ok": ok}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip", nargs="*", default=[], help="skip tests by letter: A B C D")
    args = parser.parse_args()

    device, fused_cfg = open_device()
    results = {}
    try:
        t0 = time.time()
        if "A" not in args.skip:
            results["A_alternating_base"] = test_alternating_base(device, fused_cfg)
        if "B" not in args.skip:
            results["B_alternating_libero"] = test_alternating_libero(device, fused_cfg)
        if "C" not in args.skip:
            results["C_shape_switch"] = test_shape_switch(device, fused_cfg)
        if "D" not in args.skip:
            results["D_golden_libero"] = test_golden_libero(device, fused_cfg)

        all_ok = all(r.get("all_ok", r.get("ok", False)) for r in results.values())
        results["all_ok"] = all_ok
        results["megakernel"] = fused_cfg.megakernel
        results["elapsed_s"] = round(time.time() - t0, 1)
        results["time"] = time.strftime("%F %T")

        print("\n=== SUMMARY ===", flush=True)
        for name, r in results.items():
            if isinstance(r, dict):
                ok_val = r.get("all_ok", r.get("ok", "n/a"))
                print(f"  {name}: ok={ok_val}", flush=True)
        print(f"OVERALL all_ok={all_ok}", flush=True)

        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
            print(f"Results written to {args.out}", flush=True)

    finally:
        ttnn.close_device(device)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
