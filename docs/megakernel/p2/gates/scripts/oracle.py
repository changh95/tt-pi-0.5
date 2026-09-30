"""verify-p1-r1 oracle, re-derived: my own Euler loop (t_i = 1 - i/10, dt = -1/10, 10 steps) around the fp32 torch
reference's velocity function (PI0Model._denoise_forward = suffix embed + 18 expert layers + out proj), with the expert
attention mask and positions built here from the prefix validity (not prefix_attention_inputs; cross-checked)."""
import torch

NEG = -2.3819763e38


def expert_inputs(valid, H):
    """valid [1, P] bool -> additive mask [1,1,H,P+H] fp32 (pad prefix keys hidden, action keys all visible),
    positions [1,H] = n_valid + [0,H)."""
    P = valid.shape[1]
    bias = torch.zeros(1, P + H)
    bias[0, :P][~valid[0]] = NEG
    mask = bias.view(1, 1, 1, P + H).expand(1, 1, H, P + H).contiguous()
    pos = int(valid.sum()) + torch.arange(H).view(1, H)
    return mask, pos


def expert_loop(ref, kv, valid, noise, H, steps=10):
    mask, pos = expert_inputs(valid, H)
    x = noise.clone().float()
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor([1.0 - i / steps], dtype=torch.float32)
            v = ref._denoise_forward(x, t, kv, state=torch.zeros(1, 32), attention_mask=mask, position_ids=pos)
            x = x + (-1.0 / steps) * v
    return x.float()
