# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline (mock-cluster) compile of the WHOLE-model megakernel (phase 2) + its kernel-config ring footprint.

    python -m models.experimental.pi0_5.tt.megakernel.pe_size_check [--shape base|libero|both] [--json out.json]
        [--prefix-only]

Same method as size_check.py (UMD single-P150 mock descriptor, private TT_METAL_CACHE, ELF text + data of the five
binaries + runtime args + 16 B per CB id + semaphores); the gate is 128 KB (DESIGN.md §7 P2-0).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

from .size_check import GATE_BYTES, RING_BYTES, RISCS, WORKER_L1_SIZE, elf_text_data

KERNEL_NAMES2 = {"brisc": "whole_brisc", "ncrisc": "whole_ncrisc", "trisc0": "whole_trisc", "trisc1": "whole_trisc",
                 "trisc2": "whole_trisc"}


def find_elfs2(root: Path, since: float) -> Dict[str, Tuple[int, str]]:
    out = {}
    for risc in RISCS:
        paths = glob.glob(str(root / f"**/kernels/{KERNEL_NAMES2[risc]}/*/{risc}/{risc}.elf"), recursive=True)
        best = None
        for p in paths:
            t, d = elf_text_data(p)
            key = (os.path.getmtime(p) >= since - 1, os.path.getmtime(p))
            if best is None or key > best[0]:
                best = (key, t + d, p)
        if best is not None:
            out[risc] = (best[1], best[2])
    return out


def dummy_prefix_params(n_sig: int = 1, n_vlm: int = 1):
    import torch

    from . import pe_host as H

    z = torch.zeros
    sig = [H.SigLayer(wqkv=z(1152, 4608), bqkv=z(4608), wo=z(1536, 1152), bo=z(1152), ln1w=z(1152), ln1b=z(1152),
                      ln2w=z(1152), ln2b=z(1152), wfc1=z(1152, 4352), bfc1=z(4352), wfc2=z(4352, 1152), bfc2=z(1152))
           for _ in range(n_sig)]
    vlm = [H.VlmLayer(wqkv=z(2048, 2560), wo=z(2048, 2048), wug=z(2048, 32768), wd=z(16384, 2048), g1=z(2048),
                      g2=z(2048)) for _ in range(n_vlm)]
    return H.PrefixParams(patch_w=z(608, 1152), pos_b=z(256, 1152), sig=sig, post_w=z(1152), post_b=z(1152),
                          proj_w=z(1152, 2048), proj_b=z(2048), vlm=vlm)


def build_dummy(dev, shape_name: str):
    """(WholeMegakernel, kv, mask, tables, noise) with zero weights and one layer arena per model."""
    import torch
    import ttnn

    from . import geometry as G
    from . import pe_geometry as P
    from .host_model import ExpertParams
    from .pe_program import PrefixTensors, WholeMegakernel
    from .program import ExpertMegakernel

    shape = G.SHAPES[shape_name]
    ps = P.pshape_for(shape)

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
    emb = ttnn.from_torch(torch.zeros(256, 2048), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pt = PrefixTensors(dev, dummy_prefix_params(), ps, n_sig=1, n_vlm=1, embed=emb)
    return WholeMegakernel(dev, mk, pt), kv, mask, tables, noise


def mock_compile(shape_name: str, whole: bool) -> Dict[str, object]:
    import ttnn

    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=WORKER_L1_SIZE)
    try:
        wm, kv, mask, tables, noise = build_dummy(dev, shape_name)
        t0 = time.time()
        wm.run(kv, mask, tables, noise, whole=whole)
        ttnn.synchronize_device(dev)
        return {"shape": shape_name, "compile_s": round(time.time() - t0, 1), "since": t0,
                "arena_bytes": wm.arena_bytes, "need": wm.need, "tail": wm.tail}
    finally:
        ttnn.close_device(dev)


def footprint2(elfs) -> Dict[str, object]:
    from . import geometry as G
    from . import pe_geometry as P

    bins = {r: (s + 15) // 16 * 16 for r, (s, _) in elfs.items()}
    args = 4 * (2 * (P.PR_N + P.PA_N) + G.N_RT_ARGS + P.PA_TRISC_N)  # BRISC + NCRISC full lists, TRISC cut
    cbs = 16 * (P.P_LAST + 1)
    sems = 16 * 4
    total = sum(bins.values()) + args + cbs + sems
    return {"binaries": bins, "args_bytes": args, "cb_config_bytes": cbs, "total_bytes": total,
            "ring_bytes": RING_BYTES, "gate_bytes": GATE_BYTES, "fits_gate": total <= GATE_BYTES,
            "paths": {r: p for r, (_, p) in elfs.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="both", choices=["base", "libero", "both"])
    ap.add_argument("--json", default=None)
    ap.add_argument("--prefix-only", action="store_true")
    a = ap.parse_args()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = os.path.join(
        os.environ["TT_METAL_HOME"], "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml")
    shapes = ["base", "libero"] if a.shape == "both" else [a.shape]
    root = Path(os.environ.get("TT_METAL_CACHE", str(Path.home() / ".cache" / "tt-metal-cache")))
    rows, ok = [], True
    for sh in shapes:
        info = mock_compile(sh, whole=not a.prefix_only)
        f = footprint2(find_elfs2(root, info["since"]))
        f.update({k: v for k, v in info.items() if k != "since"})
        rows.append(f)
        ok &= bool(f["fits_gate"])
        print(json.dumps({k: v for k, v in f.items() if k != "paths"}), flush=True)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"time": time.strftime("%F %T"), "rows": rows}, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
