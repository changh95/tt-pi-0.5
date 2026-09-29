"""Pre-fix p300x2 code at the LIBERO shape (2 cameras, 32 tokens, H = 10): does it run?"""
import time, traceback, torch, ttnn
from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
from models.experimental.pi0_5.common.fused_config import FusedConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN
cfg = PI0ModelConfig(action_dim=32, action_horizon=10, state_dim=32, pi05=True)
cfg.siglip_config = SigLIPConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16, image_size=224, patch_size=14)
fc = FusedConfig.from_env()
dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=fc.trace_region_size)
try:
    m = PI0ModelTTNN(cfg, PI0WeightLoader("/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_libero/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd"), dev, fused=fc)
    g = torch.Generator().manual_seed(0)
    imgs = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    try:
        out = m.sample_actions_fused(imgs, torch.randint(1, 256000, (1, 32), generator=g), torch.randn(1, 10, 32, generator=g))
        print("RESULT ran", tuple(out.shape))
    except Exception as e:
        print("RESULT refused:", type(e).__name__, str(e)[:300]); traceback.print_exc()
finally:
    ttnn.close_device(dev)
