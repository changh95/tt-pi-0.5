# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Served latency of a running server: N POST /predict with the smoke test's synthetic images and prompt; prints the
median of the server-side `timing_ms.total` / `inference` and the client wall, plus GET /info's megakernel stamp."""
import argparse
import json
import statistics
import time
import urllib.request

from models.experimental.pi0_5.server.smoke_test import DEFAULT_PROMPT, b64_png, synthetic_image


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:20000")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    body = {"images": [b64_png(synthetic_image("base")), b64_png(synthetic_image("wrist"))], "prompt": DEFAULT_PROMPT,
            "state": [0.0] * 32}
    tot, inf, wall = [], [], []
    for _ in range(a.n):
        t0 = time.perf_counter()
        r = post(a.url + "/predict", body)
        wall.append((time.perf_counter() - t0) * 1e3)
        tot.append(r["timing_ms"]["total"])
        inf.append(r["timing_ms"]["inference"])
    info = json.loads(urllib.request.urlopen(a.url + "/info", timeout=30).read())
    res = {"n": a.n, "total_ms_median": statistics.median(tot), "inference_ms_median": statistics.median(inf),
           "client_wall_ms_median": statistics.median(wall), "megakernel": info.get("megakernel"),
           "hardware": info.get("hardware"), "time": time.strftime("%F %T")}
    print("RESULT " + json.dumps(res), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
