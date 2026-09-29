# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""openpi ground truth for pi0.5 LIBERO (``pi05_libero`` conventions), shared by the reference and device checks.

The golden file (``PI05_OPENPI_GOLDEN``) holds real LIBERO observations run through openpi's GPU pi05_libero policy
with fixed noise: per record ``model_inputs`` (images in [-1, 1], ``tokenized_prompt`` right-padded to 200 with
``tokenized_prompt_mask``, 32-dim normalised state), ``noise`` [1, 10, 32] and ``actions_model_norm`` [1, 10, 32].
openpi feeds a third all -1 camera with mask False; a masked camera adds only masked keys and does not advance the
positions, so the two real cameras (base, left wrist) are the same computation.

Weights: lerobot/pi05_libero (``PI05_LIBERO_WEIGHTS``). The prompt carries no state tokens
(``discrete_state_input=False``), H = 10, 10 Euler steps.
"""

import os
from typing import Dict, List

import torch

GOLDEN = os.environ.get(
    "PI05_OPENPI_GOLDEN", "/home/deepgadget/experiments/gr00t/libero_eval/pi05/openloop_golden.pt"
)
LIBERO_WEIGHTS = os.environ.get(
    "PI05_LIBERO_WEIGHTS",
    "/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_libero/snapshots/a217bfd3b14673cf2ce597e69997ab21866438dd",
)
LIBERO_H = 10


def libero_config():
    from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig

    cfg = PI0ModelConfig(action_dim=32, action_horizon=LIBERO_H, state_dim=32, pi05=True)
    cfg.siglip_config = SigLIPConfig(
        hidden_size=1152, intermediate_size=4304, num_hidden_layers=27, num_attention_heads=16, image_size=224, patch_size=14
    )
    return cfg


def load_records() -> List[Dict]:
    return torch.load(GOLDEN, weights_only=False)["records"]


def record_inputs(rec: Dict, lang_len: int):
    """-> (images [2 x [1,3,224,224]], tokens [1, lang_len] int64, lang_mask [1, lang_len] bool, noise [1,10,32]).
    The golden prompt (200 ids, right-padded) is re-padded to ``lang_len``; a prompt longer than that raises."""
    mi = rec["model_inputs"]
    images = [mi["images"]["base_0_rgb"].float(), mi["images"]["left_wrist_0_rgb"].float()]
    tok, mask = mi["tokenized_prompt"][0].long(), mi["tokenized_prompt_mask"][0].bool()
    n = int(mask.sum())
    if not bool(mask[:n].all()):
        raise ValueError("golden prompt is not right-padded")
    if n > lang_len:
        raise ValueError(f"prompt has {n} tokens > lang_len {lang_len}")
    tokens = torch.zeros(1, lang_len, dtype=torch.long)
    tokens[0, :n] = tok[:n]
    lang_mask = torch.zeros(1, lang_len, dtype=torch.bool)
    lang_mask[0, :n] = True
    return images, tokens, lang_mask, rec["noise"].float().reshape(1, LIBERO_H, 32)


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def pcc7(actions: torch.Tensor, golden: torch.Tensor) -> float:
    """PCC over the 7 LIBERO action dims of the normalised chunk (the metric of the earlier workaround)."""
    return pcc(actions.reshape(-1, LIBERO_H, 32)[..., :7], golden.reshape(-1, LIBERO_H, 32)[..., :7])
