# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HOST test: the fp32 torch reference (the oracle of every device PCC test) against openpi's GPU pi05_libero outputs.

The reference must apply openpi's padding mask and suffix positions (``prefix_attention_inputs``); the same records
padded to two different prompt lengths must then give the same actions (padding is invisible), and both must match
openpi.

    python -m pytest models/experimental/pi0_5/tests/pcc/test_reference_vs_openpi.py -s   # CPU, ~10 min
    python models/experimental/pi0_5/tests/pcc/test_reference_vs_openpi.py [--out results.json]
"""

import json
import sys
import time

import torch

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

PCC7_MIN = 0.99


def run_reference(model, images, tokens, lang_mask, noise):
    model.denoising.sample_noise = lambda *a, **k: noise.clone()
    masks = [torch.ones(1, dtype=torch.bool) for _ in images]
    with torch.no_grad():
        return model.sample_actions(images, masks, tokens, lang_mask, torch.zeros(1, 32)).float()


def evaluate(lang_lens=(32, 224)):
    model = PI0ModelTorch(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS))
    rows = []
    for rec in load_records():
        row = {"tag": rec["tag"]}
        outs = {}
        for L in lang_lens:
            t0 = time.time()
            outs[L] = run_reference(model, *record_inputs(rec, L))
            row[f"pcc7_L{L}"] = pcc7(outs[L], rec["actions_model_norm"])
            row[f"pcc32_L{L}"] = pcc(outs[L], rec["actions_model_norm"])
            row[f"s_L{L}"] = round(time.time() - t0, 1)
        row["maxabs_L_vs_L"] = float((outs[lang_lens[0]] - outs[lang_lens[-1]]).abs().max())
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def test_reference_matches_openpi():
    rows = evaluate()
    for r in rows:
        assert r["pcc7_L32"] >= PCC7_MIN and r["pcc7_L224"] >= PCC7_MIN, r
        assert r["maxabs_L_vs_L"] < 1e-3, r  # padding must be invisible


if __name__ == "__main__":
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    rows = evaluate()
    summ = {
        k: {"mean": sum(r[k] for r in rows) / len(rows), "min": min(r[k] for r in rows)}
        for k in ("pcc7_L32", "pcc7_L224", "pcc32_L32", "pcc32_L224")
    }
    summ["max_maxabs_L_vs_L"] = max(r["maxabs_L_vs_L"] for r in rows)
    print("SUMMARY", json.dumps(summ), flush=True)
    if out:
        json.dump({"rows": rows, "summary": summ}, open(out, "w"), indent=1)
