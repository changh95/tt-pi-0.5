"""verify-p2-r0 LIBERO edge arm (PI05_MEGAKERNEL picks it): outputs for edge.EDGE.  python d_edge.py OUT.pt"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge import edge_obs  # noqa: E402
import ttnn  # noqa: E402
from models.experimental.pi0_5.common.device_open import open_pi05_device  # noqa: E402
from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tests.pcc.golden_openpi import LIBERO_WEIGHTS, libero_config  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402
torch.set_grad_enabled(False)
env = FusedConfig.from_env()
dev = open_pi05_device(env)
try:
    torch.manual_seed(42)
    m = PI0ModelTTNN(libero_config(), PI0WeightLoader(LIBERO_WEIGHTS), dev, fused=env)
    res = {"backend": m.megakernel_backend, "outs": {}}
    for tag, (im, tk, nz, mk) in edge_obs():
        res["outs"][tag] = m.sample_actions_fused(im, tk, nz, lang_masks=mk).clone()
        print("obs", tag, flush=True)
    m.release_trace()
finally:
    ttnn.close_device(dev)
torch.save(res, sys.argv[1])
print("RESULT ok", res["backend"])
