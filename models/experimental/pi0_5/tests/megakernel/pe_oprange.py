# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run one op range of the prefix-only program on real weights and synthetic-but-plausible inputs (hang triage)."""
import argparse
import os
import time

import torch
import ttnn

from models.experimental.pi0_5.common.fused_host import im2col_patches, prefix_valid_mask
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tests.megakernel.pe_bringup import Rig, stamp
from models.experimental.pi0_5.tt.megakernel import geometry as G
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P
from models.experimental.pi0_5.tt.megakernel import pe_host as H

ap = argparse.ArgumentParser()
ap.add_argument("--first", type=int, required=True)
ap.add_argument("--stop", type=int, required=True)
ap.add_argument("--shape", default="base")
a = ap.parse_args()
sh = G.SHAPES[a.shape]
ps = P.pshape_for(sh)
wl = PI0WeightLoader(os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
cat = wl.categorized_weights
pp = H.prefix_params(cat)
ew = cat["vlm_language"]["lm_head.weight"]
torch.set_num_threads(16)
dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1395712)
try:
    emb = ttnn.from_torch(ew, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rig = Rig(dev, a.shape, pp, emb)
    g = torch.Generator().manual_seed(0)
    tokens = torch.zeros(1, ps.ntok, dtype=torch.int32)
    tokens[0, :100] = torch.randint(1, 257152, (100,), generator=g).to(torch.int32)
    rig.put(rig.t.tokens, tokens, ttnn.uint32)
    stamp(f"run [{a.first}, {a.stop})")
    dt = rig.run(a.first, a.stop)
    stamp(f"ok in {dt:.3f} s")
finally:
    ttnn.close_device(dev)
