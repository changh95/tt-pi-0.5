"""Device-profiler run inside the image: one model at a time (c2 H50 N10, then c3 H32 N5), warm-up (compiles + captures
every prompt bucket), then 6 calls over 3 prompt lengths; the profiler dump is read by prof_analyze.py."""
import sys, numpy as np, torch, ttnn
from models.experimental.pi0.common.configs import PI0ModelConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.tt.ttnn_pi05_model import PI05_DEVICE_PARAMS, PI05MegakernelTTNN
L = PI0WeightLoader("/hf/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"); _ = L.categorized_weights
dev = ttnn.open_device(device_id=0, **PI05_DEVICE_PARAMS)
try:
    for cams, H, N in [(2, 50, 10), (3, 32, 5)]:
        torch.manual_seed(42)
        with PI05MegakernelTTNN(PI0ModelConfig(action_horizon=H, num_denoising_steps=N, num_cameras=cams, pi05=True), L, dev) as m:
            m.warmup()
            ims = [torch.rand(1, 3, 224, 224) * 2 - 1 for _ in range(cams)]
            for i, n in enumerate([5, 150, 40, 5, 150, 40]):
                ids = torch.zeros(1, 224, dtype=torch.long); ids[0, 0] = 2; ids[0, 1:n] = 1000 + i
                m.sample_actions(ims, None, ids, lang_masks=torch.arange(224)[None] < n)
                print("call", cams, n, m.preset_for(n).key, flush=True)
            try:
                ttnn.ReadDeviceProfiler(dev)
            except Exception as e:  # noqa: BLE001
                print('ReadDeviceProfiler:', e)
finally:
    ttnn.close_device(dev)
print("PROF DONE")
