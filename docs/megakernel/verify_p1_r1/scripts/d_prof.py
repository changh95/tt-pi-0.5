"""verify-p1-r1 profiled structural run: build, first call (compile + capture), drain, then N traced replays, each
drained with ReadDeviceProfiler. Run under python -m tracy -r -p -v ... -m d_prof SHAPE N OUT."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc import BASE_WEIGHTS, base_config, now, obs_for  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.device_open import open_pi05_device  # noqa: E402
from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, record_inputs  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

shape, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
dev = open_pi05_device(env)
res = {"t0": now(), "arm": env.megakernel, "shape": shape}
try:
    torch.manual_seed(42)
    if shape == "base":
        m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=env)
        im, tk, nz, mk = obs_for(712, 128)
    else:
        m = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), dev, fused=env)
        im, tk, mk, nz = record_inputs(load_records()[3], 32)
    first = m.sample_actions_fused(im, tk, nz, lang_masks=mk).clone()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    eq = []
    for r in range(n):
        ttnn.execute_trace(dev, m._fused_trace_id, cq_id=0, blocking=True)
        eq.append(bool(torch.equal(ttnn.to_torch(m._fused_out).float()[:, : first.shape[1]], first)))
        ttnn.ReadDeviceProfiler(dev)
    res["replays_equal_first"] = eq
    res["stamp"] = {"backend": m.megakernel_backend, "program": m.megakernel_program}
    m.release_trace()
finally:
    ttnn.close_device(dev)
res["t1"] = now()
json.dump(res, open(out, "w"), indent=1, default=str)
print("RESULT", json.dumps(res, default=str), flush=True)
