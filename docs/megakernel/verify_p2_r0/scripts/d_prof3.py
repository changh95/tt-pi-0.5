"""verify-p2-r0: profiled run through the REAL request path: first call (compile + capture), drain, then N
sample_actions_fused calls alternating 3 prompts (n 1 / 128 / 224: the key-mask rewrite path), each drained. Any device
op a request issues outside execute_trace appears as a non-session row after the first call's rows."""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc2 import BASE_WEIGHTS, base_config, now, obs_for  # noqa: E402
import ttnn  # noqa: E402
from models.experimental.pi0_5.common.device_open import open_pi05_device  # noqa: E402
from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402
n, out = int(sys.argv[1]), sys.argv[2]
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
dev = open_pi05_device(env)
res = {"t0": now(), "arm": env.megakernel}
try:
    torch.manual_seed(42)
    m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=env)
    ob = [obs_for(701, 1), obs_for(712, 128), obs_for(6, 224)]
    first = [None] * 3
    first[0] = m.sample_actions_fused(*ob[0][:3], lang_masks=ob[0][3]).clone()
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    res["marker_after_first"] = now()
    eq = []
    for i in range(n):
        k = (i + 1) % 3
        y = m.sample_actions_fused(*ob[k][:3], lang_masks=ob[k][3]).clone()
        if first[k] is None:
            first[k] = y
        eq.append(bool(torch.equal(y, first[k])))
        ttnn.ReadDeviceProfiler(dev)
    res["calls_equal_first_of_prompt"] = eq
    m.release_trace()
finally:
    ttnn.close_device(dev)
json.dump(res, open(out, "w"), indent=1)
print("RESULT", json.dumps(res), flush=True)
