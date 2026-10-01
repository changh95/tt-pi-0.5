"""verify-p1-r1 shared helpers (own code; independent of the implementer's tests/megakernel/* and of verify_p1_r0)."""
import os
import time

import torch

BASE_WEIGHTS = "/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
# the 6 seeds of tests/pcc/test_pcc_pi05_fused.py (seed 2 = the contested one) + 16 NEW seeds whose prompt lengths sit on
# and around 32-token tile boundaries (n_valid = 512 + n: key-tile and chunk edges) -- not verify_p1_r0's set.
SPEC = [(1, 40), (2, 97), (3, 12), (4, 150), (5, 201), (6, 224)] + list(
    zip(range(701, 717), [1, 2, 31, 32, 33, 63, 64, 65, 96, 128, 129, 159, 191, 192, 223, 224]))


def now():
    return time.strftime("%F %T")


def obs_for(seed, n_real, token_len=224, horizon=50):
    """Same construction as test_pcc_pi05_fused.padded_prompt (so seeds 1..6 are the test's observations)."""
    g = torch.Generator().manual_seed(seed)
    images = [torch.rand(1, 3, 224, 224, generator=g) * 2 - 1 for _ in range(2)]
    tokens = torch.zeros(1, token_len, dtype=torch.long)
    tokens[0, :n_real] = torch.randint(1, 256000, (n_real,), generator=g)
    mask = torch.zeros(1, token_len, dtype=torch.bool)
    mask[0, :n_real] = True
    noise = torch.randn(1, horizon, 32, generator=g)
    return images, tokens, noise, mask


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def base_config():
    from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig

    c = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
    c.siglip_config = SigLIPConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                                   num_attention_heads=16, image_size=224, patch_size=14)
    return c
