# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HTTP serving adapter for the pi-0.5 tt-nn port (``kind: tt-dit-server``).

``app`` is the ASGI application uvicorn serves; ``smoke_test`` is the client-side
sanity check the hardware phase runs against a live server.
"""
