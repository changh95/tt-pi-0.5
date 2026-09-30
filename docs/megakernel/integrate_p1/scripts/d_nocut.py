"""integrate-p1: DESIGN §4.12 refusal (d) on the device. (1) memory views with the repo's cut (device_kwargs of the
default config) and without it; (2) the DEFAULT config (PI05_MEGAKERNEL unset) on a device opened WITHOUT the cut must
raise the named refusal before any program runs. Positive control for the rule's arithmetic = (1)."""
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc import BASE_WEIGHTS, base_config, now  # noqa: E402

import ttnn  # noqa: E402

from models.experimental.pi0_5.common.device_open import MEGAKERNEL_WORKER_L1_SIZE, device_kwargs  # noqa: E402
from models.experimental.pi0_5.common.fused_config import FusedConfig  # noqa: E402
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader  # noqa: E402
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN  # noqa: E402

torch.set_grad_enabled(False)
env = FusedConfig.from_env()
assert "PI05_MEGAKERNEL" not in os.environ and env.megakernel == "expert", env.megakernel
res = {"t0": now(), "env_megakernel": env.megakernel, "MEGAKERNEL_WORKER_L1_SIZE": MEGAKERNEL_WORKER_L1_SIZE}


def views(dev):
    return {str(bt): int(ttnn.get_memory_view(dev, bt).total_bytes_per_bank)
            for bt in (ttnn.BufferType.L1, ttnn.BufferType.L1_SMALL)}


for label, kw in (("cut", device_kwargs(env)), ("nocut", {k: v for k, v in device_kwargs(env).items() if k != "worker_l1_size"})):
    dev = ttnn.open_device(**kw)
    try:
        res[f"views_{label}"] = views(dev)
        res[f"refusal_{label}"] = PI0ModelTTNN.megakernel_device_refusal(dev)
        res[f"kwargs_{label}"] = kw
    finally:
        ttnn.close_device(dev)
kw = {k: v for k, v in device_kwargs(env).items() if k != "worker_l1_size"}
dev = ttnn.open_device(**kw)
dev.enable_program_cache()
t0 = time.time()
try:
    torch.manual_seed(42)
    try:
        m = PI0ModelTTNN(base_config(), PI0WeightLoader(BASE_WEIGHTS), dev, fused=env)
        res["model_built_without_cut"] = True
        res["stamp"] = m.megakernel_backend
    except RuntimeError as e:
        res["model_built_without_cut"] = False
        res["error"] = str(e)
        res["error_names_cut"] = "worker-L1 cut" in str(e)
        res["tb_tail"] = traceback.format_exc().splitlines()[-4:]
    res["seconds"] = round(time.time() - t0, 1)
finally:
    ttnn.close_device(dev)
res["t1"] = now()
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
print("RESULT", json.dumps(res, default=str), flush=True)
