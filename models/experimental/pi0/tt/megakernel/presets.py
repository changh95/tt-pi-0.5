# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Presets of the pi0.5 megakernel: (cameras, prompt bucket, suffix bucket) -> the geometry of the three programs.

A call runs three programs: VISION (SigLIP on the cameras + projector), PREFIX (language embedding + the VLM prefill
that writes the K / V caches) and EXPERT (the denoising loop). A preset fixes their shapes: the prefix shape
(``pe_geometry.PShape``: VLM row tiles, bands, prompt tokens) and the expert shape (``geometry.Shape``: prefix key
tiles, suffix rows, attention chunking). Pure Python (no ttnn).

A model fixes the cameras, the suffix bucket (from the action horizon) and the denoising steps at construction; a
request runs in the smallest prompt bucket that holds its prompt. ``PRESETS`` lists what is compiled; anything else is
refused by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from . import geometry as G
from . import pe_geometry as P

TOKENS_PER_IMAGE = 256


@dataclass(frozen=True)
class Preset:
    cameras: int
    prompt_len: int  # prompt bucket L (tokens, a multiple of 32)
    suffix_rows: int  # suffix bucket S (action rows, 32 or 64)
    shape: G.Shape  # the EXPERT program (its prefix padded to the chunking: prefix_keys >= prefix_len)
    pshape: P.PShape  # the VISION and PREFIX programs

    @property
    def key(self) -> Tuple[int, int, int]:
        return (self.cameras, self.prompt_len, self.suffix_rows)

    @property
    def prefix_len(self) -> int:
        """Prefix tokens: the camera patches, then the prompt bucket (pads included)."""
        return TOKENS_PER_IMAGE * self.cameras + self.prompt_len

    @property
    def prefix_keys(self) -> int:
        """Prefix keys the expert reads: ``prefix_len`` plus masked pad keys up to its chunking."""
        return self.shape.prefix_len

    @property
    def cache_rows(self) -> int:
        """K / V cache rows the programs touch: the VLM row tiles (rounded up to 4-tile chunks) and the expert's keys."""
        return G.TILE * max(-(-self.pshape.mt // 4) * 4, self.shape.nkt)

    def check(self) -> None:
        self.shape.check()
        P.check_ops(self.pshape)
        if self.pshape.ntok != self.prompt_len or self.pshape.ptv * G.TILE != self.prefix_len:
            raise AssertionError(f"preset {self.key}: prefix shape {self.pshape} does not match")
        if self.shape.suffix_rows != self.suffix_rows or self.prefix_keys < self.prefix_len:
            raise AssertionError(f"preset {self.key}: expert shape {self.shape} does not match")


def make_preset(cameras: int, prompt_len: int, suffix_rows: int) -> Preset:
    name = f"c{cameras}_l{prompt_len}_s{suffix_rows}"
    ps = P.prefix_shape(cameras, prompt_len, name)
    return Preset(cameras, prompt_len, suffix_rows, G.expert_shape(name, ps.ptv, suffix_rows), ps)


CAMERAS = (1, 2, 3, 4)
PROMPT_BUCKETS = (32, 64, 128, 224)
SUFFIX_BUCKETS = (32, 64)
PRESETS: Dict[Tuple[int, int, int], Preset] = {
    p.key: p for p in (make_preset(c, l, s) for c in CAMERAS for l in PROMPT_BUCKETS for s in SUFFIX_BUCKETS)
}
for _p in PRESETS.values():
    _p.check()


def presets_for(cameras: int, suffix_rows: int) -> List[Preset]:
    """The presets a model with ``cameras`` and suffix bucket ``suffix_rows`` serves, by prompt bucket."""
    return sorted(
        (p for p in PRESETS.values() if p.cameras == cameras and p.suffix_rows == suffix_rows),
        key=lambda p: p.prompt_len,
    )


def alloc_pshape(presets: List[Preset]) -> P.PShape:
    """The prefix shape that sizes the shared buffers: the largest of every dimension over ``presets``."""
    return P.PShape(
        "alloc",
        mt=max(p.pshape.mt for p in presets),
        ptv=max(p.pshape.ptv for p in presets),
        rv=max(p.pshape.rv for p in presets),
        lt=max(p.pshape.lt for p in presets),
        ntok=max(p.pshape.ntok for p in presets),
        img=max(p.pshape.img for p in presets),
    )


# ======================================================================================================== L1 / K-V
L1_PER_CORE = 1_371_136  # allocatable L1 per core with the 64 KiB worker-L1 cut (memory view `per_bank`)
L1_OTHER = 12_288  # the persistent L1 besides the caches: noise, the RoPE tables, out (memory view, WP4)
L1_MIN_FREE = 16_384  # every program keeps >= 16 KB of L1 free (MULTICONFIG gate)
L1_BANKS = 110  # L1-interleaved buffers spread their pages over the worker cores


def kv_l1_bytes(cache_rows: int) -> int:
    """Per-core L1 of the 18 x (K, V) bfp8 caches of ``cache_rows`` rows (256 wide: 8 tiles per row tile)."""
    pages = cache_rows // G.TILE * 8
    return 2 * G.N_LAYERS * -(-pages // L1_BANKS) * G.TILE_BYTES["bfp8"]


def program_cb_bytes(p: Preset) -> Dict[str, int]:
    """L1 CB bytes of each program of preset ``p`` (the descriptor lists of pe_program / program)."""
    import dataclasses

    p1 = [(c.cb_id, c.page_bytes) for c in G.cb_table(p.shape)]
    p1_arena = sum(pb for i, pb in p1 if i != G.CB_SYNC)
    fixed = P.PSYNC_BYTES + 2 * P.OPD_BYTES + (P.T16 + P.T8 + P.T32 + 64) + sum(pb for i, pb in p1 if i == G.CB_SYNC)
    out = {}
    progs = [
        (f"vision{i}", dataclasses.replace(p.pshape, img=n), (0, P.OP_EMBED))
        for i, (_, n) in enumerate(P.vision_groups(p.cameras))
    ] + [("prefix", p.pshape, (P.OP_EMBED, P.N_OPS))]
    for name, ps, ops in progs:
        out[name] = fixed + p1_arena + max(64, P.arena_need(ps, ops) - p1_arena)
    out["expert"] = G.cb_union_bytes(p.shape)
    return out


def l1_free(presets: List[Preset], kv_in_l1: bool) -> Dict[Tuple[int, int, int], int]:
    """Per preset: L1 left free by its largest program when a model holds ``presets`` (its caches are sized for the
    longest of them)."""
    rows = max(p.cache_rows for p in presets)
    persistent = L1_OTHER + (kv_l1_bytes(rows) if kv_in_l1 else 0)
    return {p.key: L1_PER_CORE - persistent - max(program_cb_bytes(p).values()) for p in presets}


def kv_in_dram(presets: List[Preset]) -> bool:
    """User decision Q3 (K / V in L1 as far as possible) with the lead's fallback (WP4): DRAM caches (+ the row
    multicast) only when some program of the model's presets would keep < 16 KB of L1 free with the caches in L1."""
    return min(l1_free(presets, kv_in_l1=True).values()) < L1_MIN_FREE
