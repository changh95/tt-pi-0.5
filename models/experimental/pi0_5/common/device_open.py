# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The one device-open helper of the pi0.5 port (DESIGN.md §6.1): the megakernel needs the 64 KiB worker-L1 cut
(a 136,192 B kernel-config ring for its program), which is a process-wide device-open option."""
from typing import Optional

from models.experimental.pi0_5.common.fused_config import FusedConfig

MEGAKERNEL_WORKER_L1_SIZE = 1_395_712  # default 1,461,248 minus 64 KiB


def device_kwargs(fused: Optional[FusedConfig] = None, device_id: int = 0, l1_small_size: int = 24576) -> dict:
    fused = FusedConfig.from_env() if fused is None else fused
    kw = dict(device_id=device_id, l1_small_size=l1_small_size)
    if fused.trace:
        kw["trace_region_size"] = fused.trace_region_size
    if fused.megakernel != "off":
        kw["worker_l1_size"] = MEGAKERNEL_WORKER_L1_SIZE
    return kw


def open_pi05_device(fused: Optional[FusedConfig] = None, device_id: int = 0, l1_small_size: int = 24576):
    import ttnn

    dev = ttnn.open_device(**device_kwargs(fused, device_id, l1_small_size))
    dev.enable_program_cache()
    return dev
