"""integrate-p1: the SERVED path (batcher) passes the request's own mask. Pre-tokenised X = [2, 0] (2 real tokens, the
second is id 0) and Y = [2] (1 real token). With the mask passed they are different prompts (n_valid 2 vs 1) -> different
actions; with the old tokens != 0 fallback X would be read as Y -> identical actions. Determinism control: X twice."""
import json
import sys
import time
import urllib.request

import numpy as np

from models.experimental.pi0_5.server.smoke_test import b64_png, synthetic_image


def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


url = sys.argv[1]
imgs = [b64_png(synthetic_image("base")), b64_png(synthetic_image("wrist"))]
X1 = post(url + "/predict", {"images": imgs, "tokens": [2, 0]})
X2 = post(url + "/predict", {"images": imgs, "tokens": [2, 0]})
Y = post(url + "/predict", {"images": imgs, "tokens": [2]})
ax1, ax2, ay = (np.asarray(r["actions"], np.float64) for r in (X1, X2, Y))
res = {"time": time.strftime("%F %T"), "num_tokens": [X1["num_tokens"], Y["num_tokens"]],
       "X_repeat_identical": bool(np.array_equal(ax1, ax2)), "X_vs_Y_identical": bool(np.array_equal(ax1, ay)),
       "X_vs_Y_maxabs": float(np.abs(ax1 - ay).max()), "batched_as": [X1.get("batched_as"), Y.get("batched_as")]}
res["pass"] = res["X_repeat_identical"] and not res["X_vs_Y_identical"] and res["num_tokens"] == [2, 1]
print("RESULT " + json.dumps(res), flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
