# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The default serving backend (``PI05_MEGAKERNEL`` unset or ``mc``): the multi-config pi0.5 megakernel.

``models/experimental/pi0`` (``tt/ttnn_pi05_model.py``, ``PI05MegakernelTTNN``) runs the whole ``sample_actions`` as
three persistent ``ttnn.generic_op`` programs per call on one Blackhole chip -- VISION (SigLIP + the
projector; one program per group of <= 2 cameras, so 3 or 4 cameras take two), PREFIX (language embedding + the VLM
prefill writing the 18 K / V caches) and EXPERT (the N-step x 18-layer action expert, action in / out, Euler) --
replayed from one Metal trace per prompt bucket.

A model fixes at construction (here: at server start, from the environment):

* ``PI05_NUM_IMAGES``      cameras, 1..4 (default 2); a request must send exactly that many images;
* ``PI05_ACTION_HORIZON``  H, 1..64 (default 50; suffix buckets of 32 / 64 action rows);
* ``PI05_NUM_STEPS``       flow-matching steps N, 1..16 (default 10; the adaRMS folds depend on the schedule).
* ``PI05_DISPATCH``        the device profile: ``eth`` (non-scalable, the default: ethernet dispatch, a 12 x 10 worker
                           grid) or ``tensix`` (scalable: Tensix dispatch, 11 x 10 workers, the ethernet cores stay free);
                           read by ``models/experimental/pi0/tt/megakernel/profile.py`` at import.

A request runs in the smallest prompt bucket (32 / 64 / 128 / 224 tokens) that holds its real tokens, or in the
bucket it names (``prompt_bucket``). Batch 1, 224 x 224 images, a single chip opened with ``open_pi05_device``
(the profile's dispatch cores and the 64 KiB worker-L1 cut), one live model per device. Anything else is refused with a message that names it.

Importing this module has no side effects (no ttnn import).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

CAMERAS = (1, 2, 3, 4)
PROMPT_BUCKETS = (32, 64, 128, 224)
ACTION_DIM = 32


def refusal(num_images: int, action_horizon: int, num_steps: int, mesh: Tuple[int, int], layout: str,
            batch_sizes: Tuple[int, ...], token_len: int) -> Optional[str]:
    """Why the server configuration cannot run on the multi-config megakernel (None = it can)."""
    if mesh != (1, 1):
        return f"TT_MESH_SHAPE={mesh[0]}x{mesh[1]} (the megakernel is single-chip: use 1x1)"
    if layout != "mesh":
        return f"PI05_LAYOUT={layout} (single-chip megakernel)"
    if batch_sizes != (1,):
        return f"PI05_BATCH_SIZES={','.join(map(str, batch_sizes))} (the megakernel serves batch 1 only)"
    # the model's own checks and messages (PI05MegakernelTTNN.__init__ runs the same two; pure Python, no ttnn)
    from models.experimental.pi0.tt.megakernel import geometry as G
    from models.experimental.pi0.tt.megakernel import presets as PS

    why = G.megakernel_refusal(num_steps, action_horizon)
    if why is None and num_images not in PS.CAMERAS:
        why = f"cameras = {num_images} (compiled: {', '.join(map(str, PS.CAMERAS))})"
    if why is not None:
        return f"pi0.5 megakernel refused: {why} [PI05_NUM_IMAGES={num_images}, PI05_ACTION_HORIZON={action_horizon}, PI05_NUM_STEPS={num_steps}]"
    if token_len > PROMPT_BUCKETS[-1]:
        return f"PI05_TOKEN_LEN={token_len} (the largest prompt bucket holds {PROMPT_BUCKETS[-1]} tokens)"
    return None


def open_device(device_id: int):
    """The device for the profile (``open_pi05_device``: the dispatch cores of ``PI05_DISPATCH``, the 64 KiB worker-L1
    cut); returns it and a printable description of the parameters."""
    from models.experimental.pi0.tt.megakernel import profile as PR
    from models.experimental.pi0.tt.ttnn_pi05_model import PI05_DEVICE_PARAMS, open_pi05_device

    params = {k: str(v) for k, v in PI05_DEVICE_PARAMS.items()}
    params.update(profile=PR.NAME, dispatch=PR.DISPATCH)
    return open_pi05_device(device_id), params


def weight_loader(weights_dir):
    from models.experimental.pi0.common.weight_loader import PI0WeightLoader

    return PI0WeightLoader(str(weights_dir))


def build_model(loader, device, num_images: int, action_horizon: int, num_steps: int):
    """``PI05MegakernelTTNN`` for the configuration (raises by name on anything it does not compile). The caller seeds
    torch first: the model draws its default noise at construction."""
    from models.experimental.pi0.common.configs import PI0ModelConfig
    from models.experimental.pi0.tt.ttnn_pi05_model import PI05MegakernelTTNN

    config = PI0ModelConfig(
        action_dim=ACTION_DIM,
        action_horizon=action_horizon,
        state_dim=32,
        num_denoising_steps=num_steps,
        pi05=True,
        num_cameras=num_images,
    )
    return PI05MegakernelTTNN(config, loader, device)


def device_ops_per_call(model) -> int:
    """Programs one call enqueues (= ops in its trace): the vision programs, the prefix and the expert."""
    pr = next(iter(model.programs.values()))
    return len(pr.visions) + 2


def describe(model) -> Dict[str, Any]:
    """``/info`` ``megakernel`` block."""
    from models.experimental.pi0.tt.megakernel import profile as PR

    return {
        "backend": "mc",
        "profile": {"name": PR.NAME, "dispatch": PR.DISPATCH, "grid": list(PR.PE_GRID)},
        "program": {
            "kernel_digest": str(model.kernel_digest),
            "cameras": model.cameras,
            "action_horizon": model.horizon,
            "suffix_rows": model.suffix_rows,
            "num_steps": model.config.num_denoising_steps,
            "prompt_buckets": [p.prompt_len for p in model.presets],
            "kv_dram": bool(model.kv_dram),
            "device_ops_per_call": device_ops_per_call(model),
            "class": f"{type(model).__module__}.{type(model).__name__}",
        },
    }


def infer(model, images: List[torch.Tensor], ids: torch.Tensor, mask: torch.Tensor, noise: Optional[torch.Tensor],
          prompt_bucket: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    """One call: ``(1, H, 32)`` float32 normalised actions and the prompt bucket it ran in."""
    preset = model.preset_for(int(mask.sum()), prompt_bucket)
    actions = model.sample_actions(
        images, [torch.ones(1, dtype=torch.bool)] * len(images), ids, lang_masks=mask, noise=noise,
        prompt_bucket=prompt_bucket,
    )
    return actions, preset.prompt_len
