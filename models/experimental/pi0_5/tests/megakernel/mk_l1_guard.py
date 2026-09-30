# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DESIGN.md §4.12 replay L1 guard + the free-L1 measurement (P1-0): after capture the model records the L1 allocator
signature; a later L1 allocation makes sample_actions_fused raise before execute_trace; freeing it restores replays
(bit-identical). Also: the guard's cost and the largest free L1 block per bank at capture vs the megakernel CB union."""
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

res = {"time": time.strftime("%F %T")}
cfg = FusedConfig.from_env()
dev = open_pi05_device(cfg)
try:
    for shape in ("base", "libero"):
        torch.manual_seed(42)
        if shape == "base":
            m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=cfg)
            args = padded_prompt(1, 40)
            images, tokens, noise, mask = args
        else:
            m = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), dev, fused=cfg)
            images, tokens, mask, noise = record_inputs(load_records()[0], 32)
        a = m.sample_actions_fused(images, tokens, noise, lang_masks=mask)
        t0 = time.perf_counter()
        for _ in range(100):
            m.l1_signature()
        r = {"megakernel_l1": m.megakernel_l1, "guard_cost_us": (time.perf_counter() - t0) / 100 * 1e6}
        extra = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                memory_config=ttnn.L1_MEMORY_CONFIG)
        try:
            m.sample_actions_fused(images, tokens, noise, lang_masks=mask)
            r["raised_on_l1_alloc"] = False
        except RuntimeError as e:
            r["raised_on_l1_alloc"] = "L1 allocation state changed" in str(e)
        ttnn.deallocate(extra)
        b = m.sample_actions_fused(images, tokens, noise, lang_masks=mask)
        r["replay_after_free_bit_identical"] = bool(torch.equal(a, b))
        res[shape] = r
        print(shape, json.dumps(r), flush=True)
        m.release_trace()
        del m
finally:
    ttnn.close_device(dev)
print("RESULT " + json.dumps(res), flush=True)
if len(sys.argv) > 1:
    json.dump(res, open(sys.argv[1], "w"), indent=1)
