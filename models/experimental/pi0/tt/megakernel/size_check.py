# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline (mock-cluster) compile of the pi0.5 megakernel programs and their kernel-config ring footprints.

    TT_METAL_CACHE=$(mktemp -d) python -m models.experimental.pi0.tt.megakernel.size_check [--preset c2_l224_s64|...|all]
        [--json out.json]

Sets ``TT_METAL_MOCK_CLUSTER_DESC_PATH`` (UMD's single-P150 descriptor) before ttnn opens a device, so no card is
needed: the JIT builds exactly the binaries a device run would, with zero weights. Use an empty ``TT_METAL_CACHE``:
only ELFs built by this compile are counted. The footprint is what tt-metal places in the kernel-config ring of the
64 KiB worker-L1 cut (136,192 B) for ONE program: its five RISC binaries (text + data, 16 B aligned), the per-core and
common runtime args of its three kernels, 16 B per CB id and the semaphores. Each of the three programs of a call
(vision, prefix, expert) must fit on its own. Exit 1 when one is over the 128 KiB gate.
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
SOURCE = {"brisc": "brisc", "ncrisc": "ncrisc", "trisc0": "trisc", "trisc1": "trisc", "trisc2": "trisc"}
PROGRAMS = ("vision", "prefix", "expert")


def elf_text_data(path: str) -> Tuple[int, int]:
    """(text, data) bytes of the allocated sections of a 32-bit ELF (NOBITS excluded)."""
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


def find_elfs(root: Path, prefix: str, since: float) -> Dict[str, Tuple[int, str]]:
    """{risc: (text + data, path)} of the newest ``<prefix><source>`` ELF of each RISC built at or after ``since``
    (empty when the program's binaries came from the cache, i.e. an earlier program of this process built them)."""
    out = {}
    for risc in RISCS:
        best = None
        for p in glob.glob(str(root / f"**/kernels/{prefix}{SOURCE[risc]}/*/{risc}/{risc}.elf"), recursive=True):
            if os.path.getmtime(p) < since - 1:
                continue
            t, d = elf_text_data(p)
            if best is None or os.path.getmtime(p) > best[0]:
                best = (os.path.getmtime(p), t + d, p)
        if best is not None:
            out[risc] = (best[1], best[2])
    return out


def dummy_prefix_params(n_sig: int = 1, n_vlm: int = 1):
    """Zero prefix-engine parameters with ``n_sig`` / ``n_vlm`` layers (the binaries do not depend on the values)."""
    import torch

    from . import pe_host as H

    z = torch.zeros
    sig = [
        H.SigLayer(
            wqkv=z(1152, 4608),
            bqkv=z(4608),
            wo=z(1536, 1152),
            bo=z(1152),
            ln1w=z(1152),
            ln1b=z(1152),
            ln2w=z(1152),
            ln2b=z(1152),
            wfc1=z(1152, 4352),
            bfc1=z(4352),
            wfc2=z(4352, 1152),
            bfc2=z(1152),
        )
        for _ in range(n_sig)
    ]
    vlm = [
        H.VlmLayer(wqkv=z(2048, 2560), wo=z(2048, 2048), wug=z(2048, 32768), wd=z(16384, 2048), g1=z(2048), g2=z(2048))
        for _ in range(n_vlm)
    ]
    return H.PrefixParams(
        patch_w=z(608, 1152),
        pos_b=z(256, 1152),
        sig=sig,
        post_w=z(1152),
        post_b=z(1152),
        proj_w=z(1152, 2048),
        proj_b=z(2048),
        vlm=vlm,
    )


def build_dummy(dev, key):
    """{program: launch()} of preset ``key`` with zero weights and one layer arena per model."""
    import torch
    import ttnn

    from . import geometry as G
    from . import pe_geometry as P
    from . import presets as PS
    from .host_model import ExpertParams
    from .pe_program import PREFIX_OPS, VISION_OPS, PrefixEngineProgram, PrefixTensors
    from .program import ExpertMegakernel

    preset = PS.PRESETS[key]
    shape = preset.shape

    def dram(shape_, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            torch.zeros(shape_), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def l1(shape_, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            torch.zeros(shape_), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
        )

    z = torch.zeros(1)
    params = ExpertParams(
        wqkv=[],
        wo=[],
        wug=[],
        wd=[],
        mods=[],
        final=[],
        w_in=z,
        b_in=z,
        w_out=z,
        b_out=z,
        eps=1e-6,
        dts=tuple([-0.1] * G.N_STEPS),
    )
    w8 = [dram((32, 256), ttnn.bfloat8_b) for _ in range(G.N_STEPS)]
    w16 = [dram((32, 256)) for _ in range(G.N_STEPS)]
    kv_dram = PS.kv_in_dram(PS.presets_for(preset.cameras, preset.suffix_rows))  # the model's placement
    mk = ExpertMegakernel(dev, params, shape, arenas=(w8, w16), kv_dram=kv_dram)
    cache_len = preset.cache_rows
    kvbuf = dram if kv_dram else l1
    kv = [
        (kvbuf((1, 1, cache_len, 256), ttnn.bfloat8_b), kvbuf((1, 1, cache_len, 256), ttnn.bfloat8_b))
        for _ in range(G.N_LAYERS)
    ]
    mask = dram((1, 1, 32, preset.prefix_keys + shape.suffix_rows))
    tables = [l1((1, 1, shape.suffix_rows, 256)) for _ in range(4)]
    noise = l1((1, shape.suffix_rows, 32))
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, shape.suffix_rows, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.L1_MEMORY_CONFIG
    )
    emb = ttnn.from_torch(
        torch.zeros(256, 2048),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pt = PrefixTensors(dev, dummy_prefix_params(), preset.pshape, n_sig=1, n_vlm=1, embed=emb)
    args = (kv, mask, tables, noise, out)
    visions = {
        ("vision" if i == 0 else f"vision{i}"): PrefixEngineProgram(dev, mk, pt, preset.pshape, VISION_OPS, group=g)
        for i, g in enumerate(P.vision_groups(preset.pshape.img))
    }
    prefix = PrefixEngineProgram(dev, mk, pt, preset.pshape, PREFIX_OPS)
    progs = {**visions, "prefix": prefix, "expert": mk}
    info = {"arena_bytes": {n: p.arena_bytes for n, p in progs.items() if n != "expert"}}
    ps, sh = preset.pshape, shape
    pe_sig = (sh.rt, sh.pt, sh.chunk_tiles, sh.nch, ps.mt, ps.ptv, ps.rv, ps.lt, ps.ntok, ps.ncg)
    sigs = {n: pe_sig + (p.ps.img, p.arena_bytes) for n, p in progs.items() if n != "expert"}
    sigs["expert"] = (sh.rt, sh.pt, sh.chunk_tiles, sh.nch)
    return {name: (lambda p=p: p.run(*args)) for name, p in progs.items()}, info, sigs


def program_args_bytes(program: str) -> Tuple[int, int]:
    """(runtime-arg bytes, CB-config bytes) of one program in the ring."""
    from . import geometry as G
    from . import pe_geometry as P

    if program == "expert":
        return 4 * 3 * (G.N_RT_ARGS + G.N_COMMON_ARGS), 16 * G.N_CBS
    # NCRISC full lists, BRISC / TRISC cut
    return 4 * (2 * P.PR_N + P.PA_N + P.PA_BRISC_N + G.N_RT_ARGS + P.PA_TRISC_N), 16 * (P.P_LAST + 1)


_BUILT: Dict[Tuple, Dict[str, Tuple[int, str]]] = {}  # binary signature -> its ELFs (this process's compiles)


def mock_compile(key) -> Dict[str, object]:
    """Compile the three programs of preset ``key``; {program: footprint}."""
    import ttnn

    root = Path(os.environ.get("TT_METAL_CACHE", str(Path.home() / ".cache" / "tt-metal-cache")))
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=WORKER_L1_SIZE)
    rows = {}
    try:
        launch, info, sigs = build_dummy(dev, key)
        for name in launch:
            prefix = "mk_" if name == "expert" else "whole_"
            t0 = time.time()
            launch[name]()
            ttnn.synchronize_device(dev)
            elfs = find_elfs(root, prefix, t0)
            sig = (prefix,) + sigs[name]
            if sorted(elfs) != sorted(RISCS):  # (some of) the same binaries as an earlier program (kernel cache hit)
                elfs = {**_BUILT.get(sig, {}), **elfs}
            if sorted(elfs) != sorted(RISCS):
                raise RuntimeError(
                    f"{name}: no ELF of {sorted(set(RISCS) - set(elfs))} under {root}: use an empty cache"
                )
            _BUILT[sig] = elfs
            rows[name] = footprint(name, elfs)
            rows[name]["compile_s"] = round(time.time() - t0, 1)
            rows[name].update({k: v[name] for k, v in info.items() if name in v})
    finally:
        ttnn.close_device(dev)
    return rows


def footprint(program: str, elfs: Dict[str, Tuple[int, str]]) -> Dict[str, object]:
    bins = {r: (s + 15) // 16 * 16 for r, (s, _) in elfs.items()}
    args, cbs = program_args_bytes(program)
    sems = 16 * 4
    total = sum(bins.values()) + args + cbs + sems
    return {
        "binaries": bins,
        "args_bytes": args,
        "cb_config_bytes": cbs,
        "total_bytes": total,
        "ring_bytes": RING_BYTES,
        "gate_bytes": GATE_BYTES,
        "fits_gate": total <= GATE_BYTES,
        "paths": {r: p for r, (_, p) in elfs.items()},
    }


def main() -> int:
    from . import presets as PS

    names = {p.shape.name: k for k, p in PS.PRESETS.items()}
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="all", choices=[*names, "all"])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = os.path.join(
        os.environ["TT_METAL_HOME"], "tt_metal/third_party/umd/tests/cluster_descriptor_examples/blackhole_P150.yaml"
    )
    out, ok = {}, True
    for n in names if a.preset == "all" else [a.preset]:
        rows = mock_compile(names[n])
        for prog, f in rows.items():
            ok &= bool(f["fits_gate"])
            print(
                json.dumps({"preset": n, "program": prog, **{k: v for k, v in f.items() if k != "paths"}}), flush=True
            )
        out[n] = rows
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"time": time.strftime("%F %T"), "presets": out}, fh, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
