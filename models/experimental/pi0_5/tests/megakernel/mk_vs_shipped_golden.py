# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DESIGN.md P1-4: x_0 of the megakernel vs the current (shipped) path per golden LIBERO observation (8), one process
under the 64 KiB cut; also both paths' PCC7 vs the openpi golden."""
import dataclasses
import json
import sys
import time

import torch
import ttnn

from models.experimental.pi0_5.common.device_open import open_pi05_device
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, pcc, pcc7, record_inputs
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

env = FusedConfig.from_env()
dev = open_pi05_device(dataclasses.replace(env, megakernel="expert"))
recs = load_records()
outs = {}
try:
    for name in ("off", "expert"):
        torch.manual_seed(42)
        m = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), dev, fused=dataclasses.replace(env, megakernel=name))
        outs[name] = []
        for rec in recs:
            im, tok, mask, noise = record_inputs(rec, 32)
            outs[name].append(m.sample_actions_fused(im, tok, noise, lang_masks=mask))
        m.release_trace()
        del m
finally:
    ttnn.close_device(dev)
res = {"time": time.strftime("%F %T"), "tags": [r["tag"] for r in recs],
       "pcc_mk_vs_off": [pcc(a, b) for a, b in zip(outs["expert"], outs["off"])],
       "pcc7_off_vs_golden": [pcc7(o, r["actions_model_norm"]) for o, r in zip(outs["off"], recs)],
       "pcc7_mk_vs_golden": [pcc7(o, r["actions_model_norm"]) for o, r in zip(outs["expert"], recs)]}
res["min_mk_vs_off"] = min(res["pcc_mk_vs_off"])
print("RESULT " + json.dumps(res), flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
