#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the pi-0.5 tt-dit-server: posts two observations and checks the actions.

    python smoke_test.py --url http://127.0.0.1:20000 [--images base.png wrist.png] \
        [--prompt "pick up the cube"] [--timeout 1800] [--max-abs 3.0]

Without ``--images`` two deterministic synthetic 224x224 images are generated (the same
kind as ``media/sample_base.png`` / ``media/sample_wrist.png``). The test:

1. waits for ``GET /health`` to report ``ok`` (the server is warm),
2. reads ``GET /info`` for the contract (number of camera slots, token budget),
3. ``POST /predict`` with the prompt and a zero state -> asserts HTTP 200, ``actions`` of
   shape (50, 32), all finite, ``max|a| <= --max-abs`` (normalised action space is ~[-1, 1]),
   not constant, ``normalized: true``,
4. ``POST /predict`` again with the same inputs -> the policy is deterministic (fixed
   initial noise), so the two chunks must match closely,
5. ``POST /predict`` with different images and a different prompt -> the chunk must CHANGE
   (guards the stale-prefix-KV bug the fork fixed: every request must refresh the prefix).

Prints exactly one ``PASS ...`` / ``FAIL ...`` line last and exits 0 / 1. Only stdlib + PIL.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from typing import List, Optional

from PIL import Image, ImageDraw

DEFAULT_PROMPT = "pick up the cube"
ACTION_HORIZON = 50
ACTION_DIM = 32


def _get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post(url: str, payload: dict, timeout: float = 600.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            body = json.loads(body)
        except Exception:  # noqa: BLE001
            pass
        return e.code, body


def synthetic_image(kind: str, size: int = 224) -> Image.Image:
    """A flat 'tabletop' with a coloured cube -- deterministic, clearly synthetic."""
    if kind == "base":
        im = Image.new("RGB", (size, size), (196, 178, 150))
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, size, size // 3], fill=(120, 130, 150))  # wall
        d.rectangle([size // 2 - 18, size // 2 - 12, size // 2 + 18, size // 2 + 24], fill=(200, 40, 40))  # cube
        d.rectangle([size // 2 - 18, size // 2 - 24, size // 2 + 18, size // 2 - 12], fill=(230, 90, 90))
    else:
        im = Image.new("RGB", (size, size), (170, 150, 120))
        d = ImageDraw.Draw(im)
        d.rectangle([40, 60, size - 40, size - 30], fill=(200, 40, 40))
        d.rectangle([40, 30, size - 40, 60], fill=(230, 90, 90))
        d.rectangle([0, size - 40, 30, size], fill=(60, 60, 60))  # gripper finger
        d.rectangle([size - 30, size - 40, size, size], fill=(60, 60, 60))
    return im


def b64_png(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def load_images(paths: Optional[List[str]], n: int) -> List[str]:
    if paths:
        return [base64.b64encode(open(p, "rb").read()).decode() for p in paths[:n]]
    kinds = ["base", "wrist", "wrist"]
    return [b64_png(synthetic_image(kinds[i])) for i in range(n)]


def wait_ready(url: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            h = _get(f"{url}/health", timeout=10)
            last = h
            if h.get("status") == "ok":
                return h
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(5)
    raise SystemExit(f"FAIL server at {url} not ready after {timeout:.0f}s (last: {last})")


def check_actions(resp: dict, max_abs: float) -> dict:
    acts = resp.get("actions")
    if not isinstance(acts, list) or len(acts) != ACTION_HORIZON or any(len(r) != ACTION_DIM for r in acts):
        raise AssertionError(f"actions shape != ({ACTION_HORIZON}, {ACTION_DIM})")
    flat = [float(v) for r in acts for v in r]
    if not all(math.isfinite(v) for v in flat):
        raise AssertionError("non-finite action values")
    amax = max(abs(v) for v in flat)
    mean = sum(flat) / len(flat)
    std = math.sqrt(sum((v - mean) ** 2 for v in flat) / len(flat))
    if amax > max_abs:
        raise AssertionError(f"max|a|={amax:.3f} > {max_abs} (normalised actions should be ~[-1,1])")
    if std < 1e-4:
        raise AssertionError("actions are constant")
    if resp.get("normalized") is not True:
        raise AssertionError("response must say normalized: true")
    if resp.get("action_horizon") != ACTION_HORIZON or resp.get("action_dim") != ACTION_DIM:
        raise AssertionError("action_horizon/action_dim mismatch")
    return {"max_abs": amax, "mean": mean, "std": std, "flat": flat}


def max_diff(a: List[float], b: List[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:20000")
    ap.add_argument("--images", nargs="*", default=None, help="PNG/JPEG files, ordered [base, wrist]")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for /health == ok")
    ap.add_argument("--max-abs", type=float, default=3.0)
    ap.add_argument(
        "--skip-change-check", action="store_true", help="do not require a different observation to change the actions"
    )
    args = ap.parse_args()
    url = args.url.rstrip("/")

    try:
        wait_ready(url, args.timeout)
        info = _get(f"{url}/info")
        n_images = int(info.get("inputs", {}).get("num_images", 2))
        token_len = int(info.get("inputs", {}).get("token_len", 224))
        tokenizer_ok = bool(info.get("tokenizer", {}).get("available", False))
        print(
            f"info: num_images={n_images} token_len={token_len} tokenizer={tokenizer_ok} "
            f"weights={info.get('weights')} warmup_ms={info.get('warmup_latency_ms')}"
        )

        images = load_images(args.images, n_images)
        payload = {"images": images, "state": [0.0] * 32}
        if tokenizer_ok:
            payload["prompt"] = args.prompt
        else:
            payload["tokens"] = [2, 7071, 235292, 235248]  # PaliGemma ids of "Task: " (tokens-only server)
        status, r1 = _post(f"{url}/predict", payload)
        if status != 200:
            raise AssertionError(f"/predict -> HTTP {status}: {r1}")
        s1 = check_actions(r1, args.max_abs)
        lat1 = r1.get("timing_ms", {}).get("inference")

        status, r2 = _post(f"{url}/predict", payload)
        if status != 200:
            raise AssertionError(f"/predict (repeat) -> HTTP {status}: {r2}")
        s2 = check_actions(r2, args.max_abs)
        repeat_diff = max_diff(s1["flat"], s2["flat"])
        if repeat_diff > 0.05:
            raise AssertionError(f"repeat call differs by {repeat_diff:.4f} (> 0.05): policy should be deterministic")

        change_diff = None
        if not args.skip_change_check:
            other = {
                "images": [b64_png(synthetic_image("wrist")), b64_png(synthetic_image("base"))][:n_images],
                "state": [0.5] * 8,
            }
            if tokenizer_ok:
                other["prompt"] = "open the drawer"
            else:
                other["tokens"] = [
                    2,
                    7071,
                    235292,
                    2174,
                    573,
                    39635,
                    235289,
                    108,
                    4022,
                    235292,
                    235248,
                ]  # "Task: open the drawer;\nAction: "
            status, r3 = _post(f"{url}/predict", other)
            if status != 200:
                raise AssertionError(f"/predict (other observation) -> HTTP {status}: {r3}")
            s3 = check_actions(r3, args.max_abs)
            change_diff = max_diff(s1["flat"], s3["flat"])
            if change_diff < 1e-3:
                raise AssertionError("a different observation produced the same actions (stale prefix KV?)")

        lat2 = r2.get("timing_ms", {}).get("inference")
        print(
            f"PASS pi05 actions=(50,32) max|a|={s1['max_abs']:.3f} std={s1['std']:.3f} "
            f"repeat_maxdiff={repeat_diff:.4f} change_maxdiff={change_diff if change_diff is None else round(change_diff, 4)} "
            f"inference_ms={lat1}/{lat2} tokens={r1.get('num_tokens')}/{token_len} truncated={r1.get('prompt_truncated')}"
        )
        return 0
    except SystemExit as e:
        print(str(e))
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL pi05: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
