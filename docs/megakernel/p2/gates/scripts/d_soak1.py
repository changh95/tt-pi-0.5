"""verify-p1-r1 soak run: one process = device open (repo helper) + model build + warm-up/capture + 30 more calls
cycling three prompts of different lengths (1 / 128 / 224); every call must equal the first call of its prompt."""
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc import BASE_WEIGHTS, base_config, now, obs_for  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.device_open import open_pi05_device  # noqa: E402
from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

t0 = time.time()
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
dev = open_pi05_device(env)
try:
    torch.manual_seed(42)
    m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=env)
    ob = [obs_for(701, 1), obs_for(712, 128), obs_for(6, 224)]
    outs = [m.sample_actions_fused(*ob[i % 3][:3], lang_masks=ob[i % 3][3]) for i in range(31)]
    same = all(torch.equal(outs[i % 3], outs[i]) for i in range(31))
    d = hashlib.sha256(b"".join(o.numpy().tobytes() for o in outs[:3])).hexdigest()[:16]
    print("RESULT " + json.dumps({"backend": m.megakernel_backend,
                                  "kernel_digest": (m.megakernel_program or {}).get("kernel_digest"),
                                  "calls": 31, "all_equal": same, "digest": d, "finite": all(bool(torch.isfinite(o).all()) for o in outs),
                                  "wall_s": round(time.time() - t0, 1), "time": now()}), flush=True)
    m.release_trace()
finally:
    ttnn.close_device(dev)
