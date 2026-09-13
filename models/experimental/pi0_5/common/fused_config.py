# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Knobs for the fused / traced pi0.5 device graph (``TT_FUSED=1``).

Everything new in the ``opt/pi05-base-p150-megakernel`` branch is opt-in behind ONE env knob
that is read ONCE when the model is built (``PI0ModelTTNN.__init__`` -> ``FusedConfig.from_env``).
``TT_FUSED`` unset / ``1`` -> ``FusedConfig.enabled == True`` (the DEFAULT since the device pass of
2026-09-13); ``TT_FUSED=0`` -> every module runs the legacy code path untouched (bit-for-bit the
behaviour shipped before that pass).

Sub-knobs (only read when the fused path is enabled; every default is the recipe of
``reports/megakernel/pi05-base-p150.md`` for the device pass):

``PI05_TRACE``              ``1`` (default): capture the whole device graph in one Metal trace during
                            the first ``sample_actions_fused`` call (the server's warm-up) and replay
                            it per request; ``0``: run the same fused graph eagerly (debug / A/B).
``PI05_TRACE_REGION_SIZE``  bytes reserved for the trace at ``ttnn.open_device`` (default 160 MB;
                            estimate, see DEVICE_VALIDATION.md).
``PI05_FUSED_RESIDUAL``     expert gated residuals:
                            ``bf16`` (default): o_proj / down_proj stored in bf16, hidden stream and
                            gates stay bf16, ``dit_minimal_matmul_addcmul_fused`` with bf16 act x bf16
                            weight x bf16 residual x bf16 gate (the rf-detr-validated combination);
                            the attention input is typecast bf8 -> bf16 first (1 launch).
                            ``mixed``: same but the bf8 activations feed the fused op directly
                            (bf8 act x bf16 weight, path unexercised on Blackhole by this port).
                            ``legacy``: bf8 weights, linear + mac and the 3 typecasts (numerics of the
                            shipped path) while keeping the other fused levers.
                            NOTE: with ``bf16`` / ``mixed`` the expert o_proj / down_proj are bf16 on
                            the device, and the LEGACY ``sample_actions`` (which typecasts its
                            residual / gate to bf8 before ``dit_minimal_matmul_addcmul_fused``) can no
                            longer run on that model: the op requires the residual format == the
                            weight format (``minimal_matmul_program_factory.cpp``
                            ``TT_FATAL(ternary_a_data_format == in1_data_format)``). Only
                            ``PI05_FUSED_RESIDUAL=legacy`` (or ``TT_FUSED`` unset) keeps the legacy
                            entry points usable -- see ``legacy_sample_actions_available``.
``PI05_MLP_CHUNK``          VLM MLP sequence chunk (default 256 = the legacy chunking; 384 / 512 / 0 were
                            measured far slower in the graph: the auto matmul program collapses above 8 M
                            tiles, and the unchunked path (``0``) also lowered the e2e PCC on 3 of 8
                            observations below the legacy path -- device pass 2026-09-13; kept for A/B).
``PI05_VLM_DOWN_PC``        ``mcast2d`` (default) | ``auto``: program of the UNCHUNKED down projection
                            (``PI05_MLP_CHUNK=0`` only): explicit 2D multicast config (0.20 ms in isolation
                            for [736,16384]x[16384,2048] vs 2.90 ms for the auto program).
``PI05_VLM_GATEUP_PC``      ``auto`` (default) | ``mcast2d``: program of the unchunked gate (+GELU) / up
                            projections [S,2048]x[2048,16384] (2D multicast, in0_block_w 4: 0.56 ms each).
``PI05_SIGLIP_PC``          ``auto`` | ``mcast2d`` | ``mcast2d_fp32``: program of the SigLIP qkv / wo / fc1(+GELU) /
                            fc2 linears (512 rows for 2 cameras); isolated probe 2026-09-13: fc2 auto 0.247 ->
                            0.102 ms, fc1 -> 0.09 ms. ``_fp32`` = the same program with fp32 destination
                            accumulation (the explicit programs otherwise round the K-block partial sums to
                            bf16, which moved single observations' e2e PCC by -0.03).
``PI05_VLM_ATTN_PC``        ``auto`` | ``mcast2d`` | ``mcast2d_fp32``: program of the VLM qkv / o_proj linears
                            ([736,2048]x[2048,2560] and x[2048,2048]); isolated probe 2026-09-13: auto 0.387 /
                            0.405 ms, 2D multicast (in0_block_w 8) 0.035 / 0.029 ms.
``PI05_SIGLIP_BATCHED``     ``1`` (default): both camera images through SigLIP as one batch;
                            ``0``: one image at a time (legacy order, fused ops).
``PI05_SKIP_VLM_TAIL``      ``1`` (default): the last VLM layer stops after writing its K/V and the
                            unused final VLM norm is skipped (exact: nothing downstream reads them).
``PI05_SDPA_VLM_CHUNKS``    ``q,k`` chunk sizes (multiples of 32) for the VLM / expert / SigLIP
``PI05_SDPA_EXPERT_CHUNKS`` scaled_dot_product_attention on the FULL compute grid; ``legacy`` = the
``PI05_SDPA_SIGLIP_CHUNKS`` legacy program config (None -> 32/32 for Gemma, 8x8 grid 256/256 SigLIP).
                            All three default to the legacy config. Device pass 2026-09-13 (traced, on top
                            of the tuned dit blocks): expert ``64,256`` -0.3 ms (PCC / per-step unchanged),
                            VLM ``128,256`` -0.9 ms (one observation's e2e PCC 0.9984 -> 0.9931), SigLIP
                            ``64,256`` -0.2 ms (one observation 0.9984 -> 0.9403): within the +-0.3 ms noise
                            or precision-affecting -> knobs only.
``PI05_DIT_BLOCKS``         ``M,K,N,subblock_h,subblock_w[,grid_x,grid_y]`` (tiles; grid 0 = the device
                            extent) -> ``ttnn.MinimalMatmulConfig`` of the expert o_proj / down_proj fused
                            matmul+residual calls. Default ``1,8,4,1,4,0,2`` (measured on the p150a,
                            2026-09-13: o_proj 40.2 -> 21.6 us, down_proj 58.5 -> 39.4 us in-trace, output
                            bit-identical); ``op`` = the op's own default blocks (8x8x8 on the full grid:
                            ~1.2 MB of static circular buffers per core, which clashed with the model's L1
                            buffers on the device -- program.cpp validate_circular_buffer_region).
``PI05_EULER_DIT_BLOCKS``   same for the Euler-step fused op ([64,1024]x[1024,32]); default ``1,8,1,1,1,0,2``
                            (27.3 -> 10.0 us).
``PI05_EXPERT_MM``          ``mcast1d_fp32`` (default since the device pass 2026-09-13: 130.1 -> 125.9 ms, PCC
                            0.9977 / 0.9987, 16-observation robustness mean 0.9840 vs legacy 0.9786) | ``linear``
                            (the port's ``ttnn.linear`` auto programs) | ``mcast1d`` (bf16 partial sums: one
                            observation 0.0106 below legacy, rejected) | ``minimal``. ``mcast1d[_fp32]``: qkv and up through
                            ``ttnn.linear`` with an explicit ``MatmulMultiCoreReuseMultiCast1DProgramConfig``
                            (full grid, in0 multicast, per_core_N 2; isolated probe 2026-09-13: qkv 23.5 ->
                            10.0 us, up 22.1 -> 14.2 us, PCC unchanged; the gate keeps the auto program
                            because that config's fused GELU was numerically wrong, PCC 0.984);
                            ``minimal``: ``ttnn.experimental.minimal_matmul`` with ``PI05_EXPERT_MM_BLOCKS``
                            (same format as the dit blocks; default ``1,8,4,1,4,0,2``) -- measured slower
                            than the auto program on these shapes (19.7-60 us), kept for A/B only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Mapping, Optional, Tuple

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off", "")

RESIDUAL_MODES = ("bf16", "mixed", "legacy")
EXPERT_MM_MODES = ("linear", "mcast1d", "mcast1d_fp32", "minimal")


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r}: expected one of {_TRUE + _FALSE}")


def _int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None
    if v < minimum:
        raise ValueError(f"{name}={raw!r} must be >= {minimum}")
    return v


def _chunks(env: Mapping[str, str], name: str, default: Optional[Tuple[int, int]] = None) -> Optional[Tuple[int, int]]:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    if raw.strip().lower() in ("legacy", "op", "default", "none"):
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name}={raw!r}: expected 'q_chunk,k_chunk'")
    try:
        q, k = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"{name}={raw!r}: chunk sizes must be integers") from None
    for v in (q, k):
        if v <= 0 or v % 32 != 0:
            raise ValueError(f"{name}={raw!r}: SDPA chunk sizes must be positive multiples of 32")
    return (q, k)


Blocks = Tuple[int, int, int, int, int, int, int]  # M, K, N, subblock_h, subblock_w, grid_x, grid_y (0 = device extent)

# Measured on the p150a (logs/megakernel-validate/pi05-base/dit_blocks.json): the expert dits are
# fastest with one M tile per core, K blocks of 8, one N block of 4 tiles on an 11x2 core grid (M = 2
# tiles -> 2 rows of cores, N = 32 tiles over the 11 columns); the Euler dit (N = 1 tile) with N block 1.
DEFAULT_DIT_BLOCKS: Blocks = (1, 8, 4, 1, 4, 0, 2)
DEFAULT_EULER_DIT_BLOCKS: Blocks = (1, 8, 1, 1, 1, 0, 2)


def _blocks(env: Mapping[str, str], name: str, default: Optional[Blocks]) -> Optional[Blocks]:
    """``M,K,N,subblock_h,subblock_w[,grid_x,grid_y]`` -> 7-tuple; ``op`` -> None (the op's defaults);
    unset -> ``default``."""
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    if raw.strip().lower() in ("op", "default", "none"):
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) not in (5, 7):
        raise ValueError(f"{name}={raw!r}: expected 'M,K,N,subblock_h,subblock_w[,grid_x,grid_y]' (tiles; grid 0 = device)")
    try:
        vals = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"{name}={raw!r}: block sizes must be integers") from None
    if any(v <= 0 for v in vals[:5]):
        raise ValueError(f"{name}={raw!r}: block sizes must be positive")
    if vals[0] % vals[3] != 0 or vals[2] % vals[4] != 0:
        raise ValueError(f"{name}={raw!r}: M_block % subblock_h and N_block % subblock_w must be 0")
    if len(vals) == 5:
        vals += [0, 0]
    if any(v < 0 for v in vals[5:]):
        raise ValueError(f"{name}={raw!r}: grid sizes must be >= 0 (0 = device extent)")
    return tuple(vals)  # type: ignore[return-value]


@dataclass(frozen=True)
class FusedConfig:
    """Resolved knobs. ``enabled=False`` means: legacy path everywhere, nothing else is consulted."""

    enabled: bool = False
    trace: bool = True
    trace_region_size: int = 160_000_000
    residual: str = "bf16"
    mlp_chunk: int = 256
    vlm_down_pc: str = "mcast2d"
    vlm_gateup_pc: str = "auto"
    vlm_attn_pc: str = "auto"
    siglip_pc: str = "auto"
    siglip_batched: bool = True
    skip_vlm_tail: bool = True
    sdpa_vlm: Optional[Tuple[int, int]] = None
    sdpa_expert: Optional[Tuple[int, int]] = None
    sdpa_siglip: Optional[Tuple[int, int]] = None
    dit_blocks: Optional[Blocks] = DEFAULT_DIT_BLOCKS
    euler_dit_blocks: Optional[Blocks] = DEFAULT_EULER_DIT_BLOCKS
    expert_mm: str = "mcast1d_fp32"
    expert_mm_blocks: Optional[Blocks] = DEFAULT_DIT_BLOCKS

    def __post_init__(self):
        if self.residual not in RESIDUAL_MODES:
            raise ValueError(f"PI05_FUSED_RESIDUAL={self.residual!r} must be one of {RESIDUAL_MODES}")
        if self.mlp_chunk < 0 or self.mlp_chunk % 32 != 0:
            raise ValueError(f"PI05_MLP_CHUNK={self.mlp_chunk} must be 0 (unchunked) or a positive multiple of 32")
        if self.trace_region_size <= 0:
            raise ValueError("PI05_TRACE_REGION_SIZE must be positive")
        if self.expert_mm not in EXPERT_MM_MODES:
            raise ValueError(f"PI05_EXPERT_MM={self.expert_mm!r} must be one of {EXPERT_MM_MODES}")
        if self.vlm_down_pc not in ("mcast2d", "auto"):
            raise ValueError(f"PI05_VLM_DOWN_PC={self.vlm_down_pc!r} must be 'mcast2d' or 'auto'")
        if self.vlm_gateup_pc not in ("mcast2d", "auto"):
            raise ValueError(f"PI05_VLM_GATEUP_PC={self.vlm_gateup_pc!r} must be 'mcast2d' or 'auto'")
        if self.vlm_attn_pc not in ("mcast2d", "mcast2d_fp32", "auto"):
            raise ValueError(f"PI05_VLM_ATTN_PC={self.vlm_attn_pc!r} must be 'mcast2d', 'mcast2d_fp32' or 'auto'")
        if self.siglip_pc not in ("mcast2d", "mcast2d_fp32", "auto"):
            raise ValueError(f"PI05_SIGLIP_PC={self.siglip_pc!r} must be 'mcast2d', 'mcast2d_fp32' or 'auto'")

    @classmethod
    def legacy(cls) -> "FusedConfig":
        return cls(enabled=False)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "FusedConfig":
        """Read the knobs once. ``env`` defaults to ``os.environ`` (tests pass a dict)."""
        env = os.environ if env is None else env
        enabled = _bool(env, "TT_FUSED", True)  # default ON (device pass 2026-09-13); TT_FUSED=0 = legacy
        if not enabled:
            return cls.legacy()
        return cls(
            enabled=True,
            trace=_bool(env, "PI05_TRACE", True),
            trace_region_size=_int(env, "PI05_TRACE_REGION_SIZE", 160_000_000, minimum=1),
            residual=env.get("PI05_FUSED_RESIDUAL", "bf16").strip().lower() or "bf16",
            mlp_chunk=_int(env, "PI05_MLP_CHUNK", 256, minimum=0),
            vlm_down_pc=env.get("PI05_VLM_DOWN_PC", "mcast2d").strip().lower() or "mcast2d",
            vlm_gateup_pc=env.get("PI05_VLM_GATEUP_PC", "auto").strip().lower() or "auto",
            vlm_attn_pc=env.get("PI05_VLM_ATTN_PC", "auto").strip().lower() or "auto",
            siglip_pc=env.get("PI05_SIGLIP_PC", "auto").strip().lower() or "auto",
            siglip_batched=_bool(env, "PI05_SIGLIP_BATCHED", True),
            skip_vlm_tail=_bool(env, "PI05_SKIP_VLM_TAIL", True),
            sdpa_vlm=_chunks(env, "PI05_SDPA_VLM_CHUNKS"),
            sdpa_expert=_chunks(env, "PI05_SDPA_EXPERT_CHUNKS"),
            sdpa_siglip=_chunks(env, "PI05_SDPA_SIGLIP_CHUNKS"),
            dit_blocks=_blocks(env, "PI05_DIT_BLOCKS", DEFAULT_DIT_BLOCKS),
            euler_dit_blocks=_blocks(env, "PI05_EULER_DIT_BLOCKS", DEFAULT_EULER_DIT_BLOCKS),
            expert_mm=env.get("PI05_EXPERT_MM", "mcast1d_fp32").strip().lower() or "mcast1d_fp32",
            expert_mm_blocks=_blocks(env, "PI05_EXPERT_MM_BLOCKS", DEFAULT_DIT_BLOCKS),
        )

    # ---- derived choices (pure python so the host tests can check them without ttnn) ----

    @property
    def fused_residual(self) -> bool:
        """True when the expert gated residuals use the fused matmul+addcmul with a bf16 stream."""
        return self.enabled and self.residual != "legacy"

    def expert_residual_weight_dtype(self) -> str:
        """dtype name of the expert o_proj / down_proj weights (the fused op needs residual == weight format)."""
        return "bfloat16" if self.fused_residual else "bfloat8_b"

    @property
    def legacy_sample_actions_available(self) -> bool:
        """True when the legacy ``sample_actions`` / ``sample_actions_traced`` can run on a model built
        with this config. The legacy expert block typecasts hidden + gate to bf8 and feeds them with
        ``mlp.down_proj`` into ``dit_minimal_matmul_addcmul_fused``, whose program factory requires the
        residual format == the weight format; with bf16 o_proj / down_proj (``fused_residual``) that is
        a hard device failure, so the legacy path is only available with bf8 expert weights."""
        return self.expert_residual_weight_dtype() == "bfloat8_b"

    def expert_mlp_act_dtype(self) -> str:
        """dtype name of the expert gate/up/gelu*up activations feeding the down-proj."""
        if not self.enabled or self.residual == "legacy":
            return "bfloat8_b"
        return "bfloat16" if self.residual == "bf16" else "bfloat8_b"

    def action_out_weight_dtype(self) -> str:
        """The Euler+out-proj fused op needs x_t (bf16) == weight format -> bf16 when fused."""
        return "bfloat16" if self.enabled else "bfloat8_b"

    def keep_fused_gate_up_copy(self) -> bool:
        """Legacy keeps a fused [hidden, 2*mlp] gate_up copy next to the separate weights; fused mode does not."""
        return not self.enabled

    def describe(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


__all__ = ["FusedConfig", "RESIDUAL_MODES", "EXPERT_MM_MODES", "DEFAULT_DIT_BLOCKS", "DEFAULT_EULER_DIT_BLOCKS"]
