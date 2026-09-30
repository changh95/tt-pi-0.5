# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One soak run (DESIGN.md §7 exit gate "no hang in 20 consecutive runs"): one process = model build + warm-up +
trace capture + 30 calls; prints one RESULT line with the output digest (identical across runs = deterministic)."""
import hashlib
import json
import sys
import time

import torch
import ttnn

from models.experimental.pi0_5.common.device_open import open_pi05_device
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, record_inputs
from models.experimental.pi0_5.tests.pcc.test_pcc_pi05_fused import BASE_WEIGHTS, base_config, padded_prompt
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

shape = sys.argv[1]
t0 = time.time()
cfg = FusedConfig.from_env()
dev = open_pi05_device(cfg)
try:
    torch.manual_seed(42)
    if shape == "base":
        m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=cfg)
        images, tokens, noise, mask = padded_prompt(1, 40)
    else:
        m = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), dev, fused=cfg)
        images, tokens, mask, noise = record_inputs(load_records()[0], 32)
    outs = [m.sample_actions_fused(images, tokens, noise, lang_masks=mask) for _ in range(31)]
    same = all(torch.equal(outs[0], o) for o in outs)
    dig = hashlib.sha256(outs[0].numpy().tobytes()).hexdigest()[:16]
    print("RESULT " + json.dumps({"shape": shape, "backend": m.megakernel_backend, "calls": 31, "all_equal": same,
                                  "digest": dig, "wall_s": round(time.time() - t0, 1), "time": time.strftime("%F %T")}), flush=True)
    m.release_trace()
finally:
    ttnn.close_device(dev)
