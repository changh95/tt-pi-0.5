"""Verifier (verify-p1-r0) shared helpers. Independent of the implementer's tests/megakernel/* scripts:
observations are generated here, the expert oracle is the torch REFERENCE model's own denoise loop
(reference/torch_pi0_model.PI0Model.denoising + backbone.forward_expert), not host_model.loop_reference."""
import os
import time

import torch

BASE_WEIGHTS = "/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

# 6 seeds of test_pcc_pi05_fused (incl. the contested seed 2) + 14 verifier seeds with verifier prompt lengths
SPEC = [(1, 40), (2, 97), (3, 12), (4, 150), (5, 201), (6, 224)] + list(
    zip(range(301, 315), [1, 224, 128, 150, 64, 33, 180, 97, 5, 210, 128, 77, 160, 20]))


def now():
    return time.strftime("%F %T")


def obs_for(seed, n_real, token_len=224, horizon=50):
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
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def base_config():
    from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig

    c = PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)
    c.siglip_config = SigLIPConfig(hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
                                   num_attention_heads=16, image_size=224, patch_size=14)
    return c


def oracle(ref, kv, prefix_valid, noise, horizon):
    """fp32 expert loop of the torch reference fed with the given prefix K/V (list of 18 (K,V) [1,1,P,256])."""
    from models.experimental.pi0_5.reference.torch_pi0_model import prefix_attention_inputs

    _, _, emask, epos = prefix_attention_inputs(prefix_valid.reshape(1, -1), horizon)
    ref.denoising.sample_noise = lambda *a, _n=noise, **k: _n.clone().float()
    with torch.no_grad():
        return ref.denoising.sample_actions(1, prefix_kv_cache=kv, device=None, state=torch.zeros(1, 32),
                                            attention_mask=emask, position_ids=epos).float()
