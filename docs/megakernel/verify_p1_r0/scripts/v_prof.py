"""Profiled structural run (verify-p1-r0): build, first call (compile + capture), drain, then N traced replays each
drained. Run under: python -m tracy -r -p -v --no-op-info-cache --op-support-count 16000 -o DIR -m v_prof SHAPE N OUT"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vcommon import BASE_WEIGHTS, base_config, now, obs_for  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config, load_records, record_inputs  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

shape, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
kw = dict(device_id=0, l1_small_size=24576, trace_region_size=env.trace_region_size)
if env.megakernel != "off":
    kw["worker_l1_size"] = 1_395_712
dev = ttnn.open_device(**kw)
dev.enable_program_cache()
res = {"t0": now(), "arm": env.megakernel, "shape": shape, "kw": kw}
try:
    torch.manual_seed(42)
    if shape == "base":
        m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=env)
        im, tk, nz, mk = obs_for(2, 97)
    else:
        m = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), dev, fused=env)
        im, tk, mk, nz = record_inputs(load_records()[0], 32)
    first = m.sample_actions_fused(im, tk, nz, lang_masks=mk).clone()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    ttnn.tracy_message("`TT_SIGNPOST: replays_start`")
    eq = []
    for r in range(n):
        ttnn.execute_trace(dev, m._fused_trace_id, cq_id=0, blocking=True)
        y = ttnn.to_torch(m._fused_out).float()[:, : first.shape[1]]
        eq.append(bool(torch.equal(y, first)))
        ttnn.ReadDeviceProfiler(dev)
    res["replays_equal_first"] = eq
    res["stamp"] = {"backend": m.megakernel_backend, "program": m.megakernel_program}
    m.release_trace()
finally:
    ttnn.close_device(dev)
res["t1"] = now()
json.dump(res, open(out, "w"), indent=1, default=str)
print("RESULT", json.dumps(res, default=str), flush=True)
