# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline (mock-cluster) compile of the expert megakernel + its kernel-config ring footprint (DESIGN.md §4.9).

    python -m models.experimental.pi0_5.tt.megakernel.size_check [--shape base|libero|both] [--json out.json]

Sets ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` (UMD's single-P150 descriptor) before importing ttnn, so no card is opened;
the JIT builds exactly the binaries a device run would (memory note mock-cluster-offline-kernel-compile). Point
``TT_METAL_CACHE`` at a private directory. The footprint is the sum over the five RISC binaries (text + data, 16 B
aligned) plus the per-core and common runtime args of the three kernels, 16 B per CB id and the semaphores: the
quantities tt-metal places in the 136,192 B kernel-config ring of the 64 KiB worker-L1 cut. Exit 1 when over the
128 KB gate (8 KB margin).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

RING_BYTES = 136_192
GATE_BYTES = 128 * 1024
WORKER_L1_SIZE = 1_395_712  # the 64 KiB cut
RISCS = ("brisc", "ncrisc", "trisc0", "trisc1", "trisc2")
KERNEL_NAMES = {"brisc": "mk_brisc", "ncrisc": "mk_ncrisc", "trisc0": "mk_trisc", "trisc1": "mk_trisc",
                "trisc2": "mk_trisc"}


def elf_text_data(path: str) -> Tuple[int, int]:
    with open(path, "rb") as fh:
        head = fh.read(52)
        (e_shoff,) = struct.unpack_from("<I", head, 32)
        e_shentsize, e_shnum = struct.unpack_from("<HH", head, 46)
        fh.seek(e_shoff)
        raw = fh.read(e_shentsize * e_shnum)
    text = data = 0
    for i in range(e_shnum):
        _n, sh_type, sh_flags, _a, _o, sh_size = struct.unpack_from("<IIIIII", raw, i * e_shentsize)
        if not sh_flags & 0x2 or sh_type == 8:
            continue
        if sh_flags & 0x1:
            data += sh_size
        else:
            text += sh_size
    return text, data


def find_elfs(root: Path, since: float) -> Dict[str, Tuple[int, str]]:
    out = {}
    for risc in RISCS:
        paths = glob.glob(str(root / f"**/kernels/{KERNEL_NAMES[risc]}/*/{risc}/{risc}.elf"), recursive=True)
        best = None
        for p in paths:
            t, d = elf_text_data(p)
            key = (os.path.getmtime(p) >= since - 1, t + d)
            if best is None or key > best[0]:
                best = (key, t + d, p)
        if best is not None:
            out[risc] = (best[1], best[2])
    return out


def mock_compile(shape_name: str) -> Dict[str, object]:
    import torch
    import ttnn

    from . import geometry as G
    from .program import ExpertMegakernel
    from .host_model import ExpertParams

    shape = G.SHAPES[shape_name]
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=WORKER_L1_SIZE)
    try:
        def dram(shape_, dtype=ttnn.bfloat16):
            return ttnn.from_torch(torch.zeros(shape_), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def l1(shape_, dtype=ttnn.bfloat16):
            return ttnn.from_torch(torch.zeros(shape_), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev,
                                   memory_config=ttnn.L1_MEMORY_CONFIG)

        z = torch.zeros(1)
        params = ExpertParams(wqkv=[], wo=[], wug=[], wd=[], mods=[], final=[], w_in=z, b_in=z, w_out=z, b_out=z,
                              eps=1e-6, dts=tuple([-0.1] * G.N_STEPS))
        w8 = [dram((32, 256), ttnn.bfloat8_b) for _ in range(G.N_STEPS)]
        w16 = [dram((32, 256)) for _ in range(G.N_STEPS)]
        mk = ExpertMegakernel(dev, params, shape, arenas=(w8, w16))
        cache_len = 800 if shape_name == "base" else 640
        kv = [(l1((1, 1, cache_len, 256), ttnn.bfloat8_b), l1((1, 1, cache_len, 256), ttnn.bfloat8_b))
              for _ in range(G.N_LAYERS)]
        mask = dram((1, 1, 32, shape.prefix_len + shape.suffix_rows))
        tables = [l1((1, 1, shape.suffix_rows, 256)) for _ in range(4)]
        noise = l1((1, shape.suffix_rows, 32))
        t0 = time.time()
        mk.run(kv, mask, tables, noise)
        ttnn.synchronize_device(dev)
        return {"shape": shape_name, "compile_s": round(time.time() - t0, 1), "since": t0}
    finally:
        ttnn.close_device(dev)


def footprint(shape_name: str, elfs: Dict[str, Tuple[int, str]]) -> Dict[str, object]:
    from . import geometry as G

    bins = {r: (s + 15) // 16 * 16 for r, (s, _) in elfs.items()}
    args = 3 * 4 * (G.N_RT_ARGS + G.N_COMMON_ARGS)
    cbs = 16 * G.N_CBS
    sems = 16 * 2
    total = sum(bins.values()) + args + cbs + sems
    return {"shape": shape_name, "binaries": bins, "args_bytes": args, "cb_config_bytes": cbs,
            "total_bytes": total, "ring_bytes": RING_BYTES, "gate_bytes": GATE_BYTES, "fits_gate": total <= GATE_BYTES,
            "paths": {r: p for r, (_, p) in elfs.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="both", choices=["base", "libero", "both"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-mock", action="store_true", help="compile on the real device (hold the lock!)")
    a = ap.parse_args()
    if not a.no_mock:
        os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = os.path.join(
            os.environ["TT_METAL_HOME"], "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml")
    shapes = ["base", "libero"] if a.shape == "both" else [a.shape]
    root = Path(os.environ.get("TT_METAL_CACHE", str(Path.home() / ".cache" / "tt-metal-cache")))
    rows = []
    ok = True
    for sh in shapes:
        info = mock_compile(sh)
        f = footprint(sh, find_elfs(root, info["since"]))
        f["compile_s"] = info["compile_s"]
        rows.append(f)
        ok &= bool(f["fits_gate"])
        print(json.dumps({k: v for k, v in f.items() if k != "paths"}), flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"time": time.strftime("%F %T"), "rows": rows}, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
