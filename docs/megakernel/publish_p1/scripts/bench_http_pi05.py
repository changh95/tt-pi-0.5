#!/usr/bin/env python3
"""Served-latency bench for pi05-base-p150: N warm POST /predict with the card's curl request (media/sample_*.png,
"pick up the cube", the card's state). Records timing_ms.{preprocess,inference,total} and the client wall per request,
plus the first response (for the card's Response example). Stdlib only."""
import argparse, base64, json, statistics as st, time, urllib.request

def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read().decode())
    return body, (time.perf_counter() - t0) * 1e3

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:20000"); ap.add_argument("--media", required=True)
ap.add_argument("--n", type=int, default=100); ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--out", required=True)
a = ap.parse_args()
imgs = [base64.b64encode(open(f"{a.media}/sample_{k}.png", "rb").read()).decode() for k in ("base", "wrist")]
payload = {"images": imgs, "prompt": "pick up the cube", "state": [0.1, -0.2, 0.3, 0, 0, 0, 0.5, -0.5]}
first, _ = post(f"{a.url}/predict", payload)
for _ in range(a.warmup - 1):
    post(f"{a.url}/predict", payload)
rows = []
for _ in range(a.n):
    b, w = post(f"{a.url}/predict", payload)
    rows.append({**{k: b["timing_ms"][k] for k in b["timing_ms"]}, "client_wall": w})
    assert b["actions"] == first["actions"], "served actions changed between identical requests"
def stats(k):
    v = sorted(r[k] for r in rows); n = len(v)
    return {"median": round(st.median(v), 2), "p10": round(v[int(0.1 * (n - 1))], 2), "p90": round(v[int(0.9 * (n - 1))], 2), "min": round(v[0], 2), "max": round(v[-1], 2)}
out = {"n": a.n, "warmup": a.warmup, "request": {k: v for k, v in payload.items() if k != "images"} | {"images": "media/sample_base.png, media/sample_wrist.png"},
       "stats": {k: stats(k) for k in rows[0]}, "identical_actions_all": True,
       "first_response": {k: v for k, v in first.items() if k != "actions"} | {"actions_head": [[round(x, 4) for x in first["actions"][0][:8]]]},
       "rows": rows, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
json.dump(out, open(a.out, "w"), indent=1)
print("BENCH", json.dumps(out["stats"]))
