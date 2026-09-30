"""One soak run (verify-p1-r0): process = device open + model build + warm-up + capture + 30 calls; digest of outputs."""
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import BASE_WEIGHTS, base_config, now, obs_for  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

t0 = time.time()
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=env.trace_region_size, worker_l1_size=1_395_712)
dev.enable_program_cache()
try:
    torch.manual_seed(42)
    m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=env)
    ob = [obs_for(2, 97), obs_for(304, 150)]
    outs = [m.sample_actions_fused(*ob[i % 2][:3], lang_masks=ob[i % 2][3]) for i in range(31)]
    same = all(torch.equal(outs[i % 2], outs[i]) for i in range(31))
    d = hashlib.sha256(outs[0].numpy().tobytes() + outs[1].numpy().tobytes()).hexdigest()[:16]
    print("RESULT " + json.dumps({"backend": m.megakernel_backend, "digest_kernel": m.megakernel_program["kernel_digest"],
                                  "calls": 31, "all_equal": same, "digest": d, "wall_s": round(time.time() - t0, 1),
                                  "time": now()}), flush=True)
    m.release_trace()
finally:
    ttnn.close_device(dev)
