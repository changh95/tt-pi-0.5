# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The device profile of the pi0.5 megakernel, chosen per process with ``PI05_DISPATCH``:

* ``eth`` ("non-scalable", the default): ethernet dispatch cores, 12 x 10 worker grid (the 10 former dispatch cores
  compute); needs a tt-metal runtime whose Blackhole ethernet dispatch opens (harvested ETH descriptor + idle-ERISC
  kernel budget, tt-metal PR #57142). ``open_pi05_device`` names the fix when the runtime cannot open it.
* ``tensix`` ("scalable"): Tensix dispatch cores, 11 x 10 worker grid; the ethernet cores stay free for a fabric. Runs
  on any runtime.

The prefix engine's grid (``PE_NCOL`` columns x 10 rows) follows the profile; the device is opened with the matching
``DispatchCoreConfig`` (``ttnn_pi05_model.PI05_DEVICE_PARAMS``) and the model refuses a device whose grid differs.
"""

import os

PROFILES = {"tensix": ("scalable", 11), "eth": ("non-scalable", 12)}
DISPATCH = os.environ.get("PI05_DISPATCH", "eth")
if DISPATCH not in PROFILES:
    raise ValueError(f"PI05_DISPATCH={DISPATCH!r}: expected one of {', '.join(PROFILES)}")
NAME, PE_NCOL = PROFILES[DISPATCH]
PE_GRID = (PE_NCOL, 10)
