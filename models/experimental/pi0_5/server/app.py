# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ASGI serving app for pi-0.5 (``lerobot/pi05_base``) on one Tenstorrent Blackhole p150a.

Served by tt-model-manager as ``kind: tt-dit-server``::

    runtime:
      app: models.experimental.pi0_5.server.app:app

Default backend (``PI05_MEGAKERNEL`` unset or ``mc``, single chip): the multi-config pi0.5 megakernel of
``models/experimental/pi0`` (``PI05MegakernelTTNN``; ``server/mc_backend.py``): three persistent ``ttnn.generic_op``
programs per call (VISION | PREFIX | EXPERT; four with 3-4 cameras, one vision program per <= 2 cameras), replayed from
one Metal trace per prompt bucket. ``PI05_NUM_IMAGES`` 1..4, ``PI05_ACTION_HORIZON`` 1..64 and ``PI05_NUM_STEPS``
1..10 are fixed at server start; each request runs in the smallest 32 / 64 / 128 / 224-token prompt bucket that holds
its prompt (``prompt_bucket`` overrides it). The single-config paths below (``whole`` / ``expert`` / ``off``, the
mesh and dp layouts) stay selectable as comparators; they were validated on tt-metal 668c2907575, not re-validated on
the tree this package pins.

uvicorn runs this module. Importing it has no side effects (no device, no weights, no
network): the image's ``verify.sh`` imports it as an unprivileged user with no card.
Everything heavy happens inside the ASGI **lifespan**, so uvicorn's
``Application startup complete`` -- the line ``tt-model serve`` waits for -- means the
chip is claimed, the 14.5 GB checkpoint is converted to device tensors and the fused graph
has run (kernels compiled, trace captured).

Recipe (the one the port validated -- ``tests/pcc/test_pcc_pi05_fused.py`` and the LIBERO
closed-loop harness ``tests/pcc/test_rollout_libero.py``):

* ``device = ttnn.open_device(**common.device_open.device_kwargs(...))`` (single chip: l1_small_size=24576,
  trace_region_size=..., and the 64 KiB worker-L1 cut the default ``PI05_MEGAKERNEL=whole`` needs) or a
  MeshDevice (``TT_MESH_SHAPE``); ``PI0ModelTTNN.sample_actions_fused``.
* ``PI0ModelConfig(action_dim=32, action_horizon=50, state_dim=32, pi05=True)`` with the
  default ``SigLIPConfig`` (224 px, patch 14 -> 256 tokens per image).
* weights: ``PI0WeightLoader(<HF snapshot dir>)`` -- resolved with ``hf_hub_download`` of
  ``model.safetensors`` + ``config.json`` at ``TT_WEIGHTS_REVISION``.
* inputs: N images (default 2: base/exterior camera, wrist camera) -> RGB, bilinear resize
  to 224x224, ``/255``, ``(x-0.5)/0.5`` -> torch ``(1, 3, 224, 224)`` float in [-1, 1];
  language tokens ``(1, L)`` + mask ``(1, L)`` bool (right-padded; the padded tokens are
  hidden from every query); state ``(1, 32)`` (accepted by the API, not a graph input: the
  pi0.5 expert does not read it -- the state reaches the model through the prompt, discretised
  into 256 bins like lerobot's ``Pi05PrepareStateTokenizerProcessorStep``).
* output: ``model.sample_actions_fused(...)`` -> torch ``(1, 50, 32)`` actions in lerobot's
  normalised (QUANTILES, ~[-1, 1]) action space, zero-padded to 32 dims.

Configuration is read from the environment inside the lifespan (never at import):

``HF_MODEL``               weights repo id (default ``lerobot/pi05_base``)
``TT_WEIGHTS_REVISION``    commit sha of the weights to load (default: repo default branch)
``PI05_WEIGHTS_DIR``       local directory holding model.safetensors + config.json (offline override)
``TT_MESH_SHAPE``          ``1x1`` (one chip, ``ttnn.open_device``) or ``1x4`` (2x p300: MeshDevice + Ethernet
                           fabric, SigLIP / VLM prefill tensor-parallel, expert replicated -- tt/ttnn_ccl.py)
``TT_DEVICE_ID``           chip to open (default 0)
``PI05_L1_SMALL_SIZE``     l1_small_size for ttnn.open_device (default 24576, validated)
``PI05_TOKEN_LEN``         language token budget L (default 224 = the LIBERO-validated length;
                           32 reproduces the README PCC/perf configuration but cannot hold the state)
``PI05_NUM_IMAGES``        camera slots (default 2 = validated; fewer images are padded with black)
``PI05_NUM_STEPS``         flow-matching denoising steps (default 10, fixed at build time)
``PI05_SEED``              seed for the model's fixed initial noise (default 42, as in the tests)
``PI05_TOKENIZER``         tokenizer repo (default ``google/paligemma-3b-pt-224``, gated -> needs a token)
``PI05_TOKENIZER_DIR``     local tokenizer directory (offline override)
``PI05_TOKENIZER_REQUIRED`` ``1`` (default) fails startup when the tokenizer cannot be loaded;
                           ``0`` starts anyway and serves pre-tokenised ``tokens`` only
``PI05_WARMUP_RUNS``       warm-up forwards before READY (default 2)
``PI05_FREE_HOST_WEIGHTS`` ``1`` (default) drops the fp32 host copy of the checkpoint after conversion
``PI05_MODEL_NAME``        name reported by ``/info`` and ``/v1/models`` (default ``pi05-base-p150``)
``PI05_SOURCE_COMMIT``     commit of github.com/changh95/tt-pi-0.5 reported by ``/info`` (set by the package)

Fused / traced device graph (the only inference path; knobs read once by ``FusedConfig.from_env()``):
the whole device graph (host im2col -> SigLIP batched over the cameras -> VLM prefill writing the
KV caches -> 10 expert steps) reads persistent device inputs and is captured into ONE Metal trace on the
first warm-up call, i.e. before READY; every request then does small host->device input copies,
``execute_trace`` and one readback. With the default ``PI05_MEGAKERNEL=whole`` that trace holds exactly one
device op, the whole-model megakernel. The device is opened with ``trace_region_size``.

``PI05_MEGAKERNEL``        ``whole`` (default since 2026-10-01): the ENTIRE sample_actions (SigLIP x2, projector,
                           language embedding, VLM prefill -> K / V caches, the 10-step expert loop, action in / out,
                           Euler) is ONE persistent generic_op (tt/megakernel/pe_program.py, docs/megakernel/DESIGN.md
                           §11); the trace replays that one device op. The device is opened with the 64 KiB worker-L1
                           cut (common/device_open.py). Single chip, batch 1, 2 cameras (PI05_NUM_IMAGES=2), bf8 K/V,
                           10 steps: any other configuration refuses at startup with a log line.
                           Comparator knobs only: ``expert`` (phase 1: the expert loop is one generic_op, the SigLIP /
                           VLM prefix is traced stock ttnn ops) and ``off`` (the previous path: stock ttnn ops + 3
                           custom generic_ops). On a multi-chip mesh the UNSET default is ``off`` (the megakernels are
                           single-p150a programs).
``TT_FUSED``               unset or ``1``; ``0`` / ``false`` / ``off`` fails startup (the unfused path was removed)
``PI05_TRACE``             ``0`` runs the fused graph eagerly (no trace; debug / A/B). Default ``1``.
``PI05_TRACE_REGION_SIZE`` bytes for the trace region (default 160000000, an estimate).
``PI05_FUSED_RESIDUAL``    ``bf16`` (default) | ``mixed`` | ``legacy`` -- see ``common/fused_config.py``
``PI05_MLP_CHUNK``         VLM MLP chunk rows (default 256; ``0`` = unchunked, device-gated)
``PI05_SIGLIP_BATCHED``    ``1`` (default) both cameras in one SigLIP batch
``PI05_SKIP_VLM_TAIL``     ``1`` (default) skip the dead tail of the last VLM layer (exact)
``PI05_SDPA_{VLM,EXPERT,SIGLIP}_CHUNKS``  ``q,k`` SDPA chunks on the full grid (default: the op's program)
``PI05_DIT_BLOCKS`` / ``PI05_EULER_DIT_BLOCKS``  ``M,K,N,sh,sw[,gx,gy]`` MinimalMatmulConfig blocks (+ core
                           grid) of the fused matmul+residual ops; measured defaults, ``op`` = op defaults
``PI05_EXPERT_MM``         ``linear`` | ``mcast1d`` | ``minimal``: program of the expert qkv / up projections
``PI05_VLM_ATTN_PC``, ``PI05_SIGLIP_PC``  ``auto`` | ``mcast2d``: explicit 2D-multicast matmul program configs
``PI05_VLM_DOWN_PC``, ``PI05_VLM_GATEUP_PC``  programs of the unchunked VLM MLP (``PI05_MLP_CHUNK=0`` only)
The request / response contract, warm-up-before-READY and seed semantics are unchanged.
"""
from __future__ import annotations

import base64
import gc
import io
import logging
import os
import sys
import queue
import threading
import time
import traceback
from concurrent.futures import Future
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------------------
# constants (the validated contract)
# ----------------------------------------------------------------------------------------

MODEL_NAME = os.environ.get("PI05_MODEL_NAME", "pi05-base-p150")  # /info and the /v1/models stub
DEFAULT_WEIGHTS_REPO = "lerobot/pi05_base"
DEFAULT_TOKENIZER_REPO = "google/paligemma-3b-pt-224"
IMAGE_SIZE = 224
ACTION_DIM = 32
ACTION_HORIZON = 50
STATE_DIM = 32
PALIGEMMA_VOCAB_SIZE = 257152
PAD_TOKEN_ID = 0  # PaliGemma <pad>
NUM_STATE_BINS = 256
SOURCE_REPO = "https://github.com/changh95/tt-pi-0.5"
# The commit of SOURCE_REPO the served code was taken from; the package sets it in serve.env
# (a commit cannot name itself). The tt-metal tree is in tt_kernel_manifest.json (container.built.tt_metal).
SOURCE_COMMIT = os.environ.get("PI05_SOURCE_COMMIT") or None
TT_METAL_NOTE = "see the package's tt_kernel_manifest.json (container.built.tt_metal)"
LICENSE_NOTE = (
    "Weights: lerobot/pi05_base under the Gemma Terms of Use "
    "(https://ai.google.dev/gemma/terms); the port code (Tenstorrent / Hyunggi Chang) is "
    "Apache-2.0 but is distributed under the same Gemma terms. The tokenizer repo "
    "google/paligemma-3b-pt-224 is gated: accept the Gemma terms on huggingface.co and "
    "provide a token (HF_TOKEN or the token file under HF_HOME)."
)

_STATE_BIN_EDGES = np.linspace(-1.0, 1.0, NUM_STATE_BINS + 1)[:-1]

# ----------------------------------------------------------------------------------------
# logging: uvicorn owns its own loggers; give ours a stdout handler so the boot phrases
# ("Loading weights", "Warming up", "Warmup complete") reach `tt-model serve`'s checklist.
# ----------------------------------------------------------------------------------------

LOG = logging.getLogger("pi05.server")
if not LOG.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s pi05.server: %(message)s"))
    LOG.addHandler(_h)
    LOG.setLevel(logging.INFO)
    LOG.propagate = False

# ----------------------------------------------------------------------------------------
# process-wide state. The TT model is NOT reentrant (persistent trace inputs, backbone-owned KV
# caches, one captured trace) and ttnn is not thread-safe -> one lock around every device call.
# ----------------------------------------------------------------------------------------

STATE: Dict[str, Any] = {}
LOCK = threading.Lock()
TOK_LOCK = threading.Lock()  # the HF fast tokenizer is not thread-safe ("Already borrowed" under concurrent requests)


# ----------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{name}={raw!r} is not an integer") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


@dataclass(frozen=True)
class ServerConfig:
    weights_repo: str
    weights_revision: Optional[str]
    weights_dir: Optional[str]
    tokenizer_repo: str
    tokenizer_dir: Optional[str]
    tokenizer_required: bool
    device_id: int
    l1_small_size: int
    token_len: int
    num_images: int
    num_steps: int
    seed: int
    warmup_runs: int
    free_host_weights: bool
    batch_sizes: Tuple[int, ...] = (1,)
    batch_window_ms: float = 4.0
    layout: str = "mesh"  # PI05_LAYOUT: "mesh" (all chips on one request group) | "dp" (independent chip groups, each the
    # whole model) | "pipeline" (prefix chips -> expert chips)
    dp_group: int = 2  # dp: chips per group (the prefix is tensor-parallel inside a group, the expert replicated)
    profile: str = "single-robot"  # PI05_PROFILE: the serving profile's name, reported by /info
    prefix_chips: int = 2  # pipeline: chips of the prefix sub-mesh (the remaining chips run the expert)
    pipeline_margin_ms: float = 4.0  # pipeline: safety margin on the measured expert time when waiting for the next group
    backend: str = "mc"  # PI05_MEGAKERNEL: "mc" (default on one chip: the multi-config megakernel) | whole | expert | off
    action_horizon: int = ACTION_HORIZON  # PI05_ACTION_HORIZON (mc: 1..64; the single-config paths: 50 only)


def load_config() -> ServerConfig:
    """Read the serving configuration from the environment (lifespan only, never at import)."""
    from models.experimental.pi0_5.common.fused_config import FusedConfig

    layout = os.environ.get("PI05_LAYOUT", "mesh").strip().lower() or "mesh"
    backend = os.environ.get("PI05_MEGAKERNEL", "").strip().lower()
    if not backend:
        # the multi-config megakernel is single-chip; the UNSET default on a multi-chip mesh stays the stock-op path
        backend = "mc" if parse_mesh_shape(os.environ.get("TT_MESH_SHAPE")) == (1, 1) else ""
    if backend != "mc":
        FusedConfig.from_env()  # fail fast on a bad knob (TT_FUSED=0: the unfused path was removed)
    batch_sizes = tuple(sorted({int(b) for b in os.environ.get("PI05_BATCH_SIZES", "1").split(",") if b.strip()}))
    profile = os.environ.get("PI05_PROFILE", "").strip().lower() or (
        "multi-robot" if layout in ("dp", "pipeline") else ("single-robot" if batch_sizes == (1,) else "custom")
    )
    cfg = ServerConfig(
        weights_repo=os.environ.get("HF_MODEL", "").strip() or DEFAULT_WEIGHTS_REPO,
        weights_revision=os.environ.get("TT_WEIGHTS_REVISION", "").strip() or None,
        weights_dir=os.environ.get("PI05_WEIGHTS_DIR", "").strip() or None,
        tokenizer_repo=os.environ.get("PI05_TOKENIZER", "").strip() or DEFAULT_TOKENIZER_REPO,
        tokenizer_dir=os.environ.get("PI05_TOKENIZER_DIR", "").strip() or None,
        tokenizer_required=_env_bool("PI05_TOKENIZER_REQUIRED", True),
        device_id=_env_int("TT_DEVICE_ID", 0),
        l1_small_size=_env_int("PI05_L1_SMALL_SIZE", 24576),
        token_len=_env_int("PI05_TOKEN_LEN", 224),
        num_images=_env_int("PI05_NUM_IMAGES", 2),
        num_steps=_env_int("PI05_NUM_STEPS", 10),
        seed=_env_int("PI05_SEED", 42),
        warmup_runs=_env_int("PI05_WARMUP_RUNS", 2),
        free_host_weights=_env_bool("PI05_FREE_HOST_WEIGHTS", True),
        batch_sizes=batch_sizes,
        batch_window_ms=float(os.environ.get("PI05_BATCH_WINDOW_MS", "4")),
        layout=layout,
        profile=profile,
        prefix_chips=_env_int("PI05_PREFIX_CHIPS", 2),
        dp_group=_env_int("PI05_DP_GROUP", 2),
        pipeline_margin_ms=float(os.environ.get("PI05_PIPELINE_MARGIN_MS", "4")),
        backend=backend or "legacy",
        action_horizon=_env_int("PI05_ACTION_HORIZON", ACTION_HORIZON),
    )
    if cfg.token_len < 1 or cfg.token_len % 32 != 0:
        raise RuntimeError(
            f"PI05_TOKEN_LEN={cfg.token_len} must be a positive multiple of 32 (tile alignment; "
            "validated values: 224 (LIBERO rollout) and 32 (README PCC/perf))"
        )
    if cfg.backend == "mc":
        from models.experimental.pi0_5.server import mc_backend

        why = mc_backend.refusal(cfg.num_images, cfg.action_horizon, cfg.num_steps,
                                 parse_mesh_shape(os.environ.get("TT_MESH_SHAPE")), cfg.layout, cfg.batch_sizes,
                                 cfg.token_len)
        if why is not None:
            LOG.error("PI05_MEGAKERNEL=mc refused at startup: %s", why)
            raise RuntimeError(f"PI05_MEGAKERNEL=mc refused: {why}")
    elif cfg.num_images < 1 or cfg.num_images > 3:
        raise RuntimeError(f"PI05_NUM_IMAGES={cfg.num_images} must be 1..3 (validated: 2)")
    elif cfg.action_horizon != ACTION_HORIZON:
        raise RuntimeError(f"PI05_ACTION_HORIZON={cfg.action_horizon}: the single-config paths serve {ACTION_HORIZON} only")
    if cfg.num_steps < 1:
        raise RuntimeError(f"PI05_NUM_STEPS={cfg.num_steps} must be >= 1 (validated: 10)")
    if cfg.warmup_runs < 1:
        raise RuntimeError("PI05_WARMUP_RUNS must be >= 1: READY must mean warm")
    if not cfg.batch_sizes or any(b < 1 for b in cfg.batch_sizes):
        raise RuntimeError("PI05_BATCH_SIZES must be a comma-separated list of positive batch sizes (e.g. 1,2,4)")
    if cfg.batch_window_ms < 0:
        raise RuntimeError("PI05_BATCH_WINDOW_MS must be >= 0")
    if cfg.layout == "pipeline":
        # the disaggregated prefix / expert pipeline (tt/ttnn_disagg.py, p300x2) was removed on 2026-09-29: it did not
        # carry the padding mask / action-token RoPE fix of the fused graph
        raise RuntimeError("PI05_LAYOUT=pipeline was removed; use 'mesh' or 'dp'")
    if cfg.layout not in ("mesh", "dp"):
        raise RuntimeError(f"PI05_LAYOUT={cfg.layout!r} must be 'mesh' or 'dp'")
    if cfg.dp_group < 1:
        raise RuntimeError("PI05_DP_GROUP must be >= 1")
    if cfg.prefix_chips < 1:
        raise RuntimeError("PI05_PREFIX_CHIPS must be >= 1")
    if cfg.pipeline_margin_ms < 0:
        raise RuntimeError("PI05_PIPELINE_MARGIN_MS must be >= 0")
    return cfg


def parse_mesh_shape(value: Optional[str]) -> Tuple[int, int]:
    """Parse ``TT_MESH_SHAPE`` -- ``1x1`` (what the tt-dit-server launcher exports),
    ``(1, 1)`` or ``1,1``. This port drives a single chip through ``ttnn.open_device``, so
    anything but a 1x1 mesh is refused with a clear message."""
    if value is None or not value.strip():
        return (1, 1)
    cleaned = value.strip().strip("()[]").replace("X", "x").replace("x", ",")
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    try:
        if len(parts) != 2:
            raise ValueError
        rows, cols = int(parts[0]), int(parts[1])
    except ValueError:
        raise RuntimeError(f"TT_MESH_SHAPE={value!r} is not a mesh shape; expected '1x1', '(1, 1)' or '1,1'") from None
    if rows < 1 or cols < 1:
        raise RuntimeError(f"TT_MESH_SHAPE={value!r} must be a positive mesh shape (1x1 or 1x4)")
    return (rows, cols)


# ----------------------------------------------------------------------------------------
# weights / tokenizer
# ----------------------------------------------------------------------------------------


def resolve_weights_dir(cfg: ServerConfig) -> Tuple[Path, Dict[str, Any]]:
    """Return the directory ``PI0WeightLoader`` reads (``model.safetensors`` + ``config.json``).

    ``PI05_WEIGHTS_DIR`` wins (offline/host use); otherwise each file is resolved with
    ``hf_hub_download(repo_id=HF_MODEL, revision=TT_WEIGHTS_REVISION)`` -- a cache hit when
    ``tt-model serve`` pre-downloaded the pinned snapshot into the mounted HF cache, a
    download otherwise. A sha-pinned snapshot has no ``refs/main``, hence the revision.
    """
    needed = ("model.safetensors", "config.json")
    if cfg.weights_dir:
        d = Path(cfg.weights_dir).expanduser()
        missing = [f for f in needed if not (d / f).is_file()]
        if missing:
            raise FileNotFoundError(f"PI05_WEIGHTS_DIR={d} lacks {missing}")
        LOG.info("Loading weights from local directory %s", d)
        return d, {"repo": None, "revision": None, "local_dir": str(d)}

    from huggingface_hub import hf_hub_download

    LOG.info(
        "Loading weights: %s @ %s (%s)",
        cfg.weights_repo,
        cfg.weights_revision or "default branch",
        ", ".join(needed),
    )
    paths = [hf_hub_download(repo_id=cfg.weights_repo, filename=f, revision=cfg.weights_revision) for f in needed]
    snapshot_dir = Path(paths[0]).parent  # snapshots/<sha>/ -- keep the symlinks, do not resolve
    return snapshot_dir, {
        "repo": cfg.weights_repo,
        "revision": cfg.weights_revision,
        "snapshot_dir": str(snapshot_dir),
    }


def load_tokenizer(cfg: ServerConfig):
    """PaliGemma tokenizer (gated repo). Returns None when unavailable and not required."""
    src = cfg.tokenizer_dir or cfg.tokenizer_repo
    try:
        from transformers import AutoTokenizer

        kwargs: Dict[str, Any] = {}
        if cfg.tokenizer_dir:
            kwargs["local_files_only"] = True
        LOG.info("Loading tokenizer %s", src)
        tok = AutoTokenizer.from_pretrained(src, **kwargs)
        tok.padding_side = "right"
        return tok
    except Exception as e:  # noqa: BLE001 - the message matters more than the type
        msg = (
            f"could not load the PaliGemma tokenizer from {src!r}: {type(e).__name__}: {e}. "
            "google/paligemma-3b-pt-224 is gated: accept the Gemma terms on huggingface.co, "
            "then `hf auth login` (the token file under HF_HOME is mounted into the container) "
            "or export HF_TOKEN before `tt-model serve`. Set PI05_TOKENIZER_REQUIRED=0 to start "
            "without it and send pre-tokenised `tokens` instead."
        )
        if cfg.tokenizer_required:
            raise RuntimeError(msg) from e
        LOG.warning("%s -- continuing in tokens-only mode", msg)
        return None


# ----------------------------------------------------------------------------------------
# preprocessing (host side)
# ----------------------------------------------------------------------------------------


def discretize_state(state32: np.ndarray) -> List[int]:
    """lerobot ``Pi05PrepareStateTokenizerProcessorStep``: 256 bins over [-1, 1]."""
    s = np.clip(np.asarray(state32, dtype=np.float32), -1.0, 1.0)
    return [int(b) for b in (np.digitize(s, _STATE_BIN_EDGES) - 1)]


def build_prompt(task: str, state32: np.ndarray) -> str:
    """pi0.5 prompt: ``Task: <task>, State: b0 b1 ... b31;\\nAction: `` (lerobot format)."""
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    bins = discretize_state(state32)
    return f"Task: {cleaned}, State: {' '.join(str(b) for b in bins)};\nAction: "


def tokenize_prompt(tokenizer, prompt: str, token_len: int) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
    """Right-padded ``(1, L)`` int64 ids + ``(1, L)`` bool mask, real-token count, truncated flag."""
    enc = tokenizer(
        prompt,
        padding="max_length",
        max_length=token_len,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc["input_ids"].to(torch.long)
    mask = enc["attention_mask"].to(torch.bool)
    n_real = int(mask.sum().item())
    n_full = len(tokenizer(prompt)["input_ids"])
    return ids, mask, n_real, n_full > token_len


def tokens_from_request(tokens: List[int], token_len: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pre-tokenised path (bypasses the gated tokenizer). Pads with PaliGemma <pad>=0."""
    if not tokens:
        raise HTTPException(status_code=400, detail="tokens must contain at least one id")
    if len(tokens) > token_len:
        raise HTTPException(
            status_code=400,
            detail=f"tokens has {len(tokens)} ids; this server accepts at most {token_len} (PI05_TOKEN_LEN)",
        )
    bad = [t for t in tokens if t < 0 or t >= PALIGEMMA_VOCAB_SIZE]
    if bad:
        raise HTTPException(status_code=400, detail=f"token ids out of range [0, {PALIGEMMA_VOCAB_SIZE}): {bad[:5]}")
    ids = torch.full((1, token_len), PAD_TOKEN_ID, dtype=torch.long)
    ids[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    mask = torch.zeros((1, token_len), dtype=torch.bool)
    mask[0, : len(tokens)] = True
    return ids, mask, len(tokens)


def decode_image(b64: str, idx: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """base64 PNG/JPEG -> ``(1, 3, 224, 224)`` float32 in [-1, 1]: RGB, bilinear squash-resize to 224x224 (a
    224x224 image is not resampled), then ``x * float32(1/255) * 2 - 1`` -- bit-exact to openpi's PyTorch policy input
    (its ``x / 255.0 * 2.0 - 1.0`` runs on CUDA as a multiply by the float32 reciprocal; a true division differs by
    1 ulp on some pixels). Returns the original size."""
    try:
        payload = b64.strip()
        if payload.startswith("data:") and "," in payload:
            payload = payload.split(",", 1)[1]
        raw = base64.b64decode(payload, validate=False)
        im = Image.open(io.BytesIO(raw))
        im.load()
        im = im.convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"images[{idx}]: cannot decode base64 PNG/JPEG: {e}") from None
    orig_w, orig_h = im.size
    im = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    return (x * torch.tensor(1.0 / 255.0, dtype=torch.float32) * 2.0 - 1.0).contiguous(), (orig_w, orig_h)


def black_image() -> torch.Tensor:
    """What lerobot feeds a missing camera slot: an all -1 (black) image."""
    return torch.full((1, 3, IMAGE_SIZE, IMAGE_SIZE), -1.0, dtype=torch.float32)


def state_from_request(state: Optional[List[float]]) -> np.ndarray:
    if state is None:
        return np.zeros(STATE_DIM, dtype=np.float32)
    if len(state) > STATE_DIM:
        raise HTTPException(status_code=400, detail=f"state has {len(state)} values; at most {STATE_DIM} are accepted")
    arr = np.asarray(state, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise HTTPException(status_code=400, detail="state contains non-finite values")
    out = np.zeros(STATE_DIM, dtype=np.float32)
    out[: arr.size] = arr
    return out


# ----------------------------------------------------------------------------------------
# device inference (call under LOCK)
# ----------------------------------------------------------------------------------------


def run_inference(
    images: List[torch.Tensor],
    lang_tokens: torch.Tensor,
    lang_masks: torch.Tensor,
    state: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    prompt_bucket: Optional[int] = None,
) -> torch.Tensor:
    """One ``sample_actions_fused`` call (``PI05_MEGAKERNEL=mc``: one ``PI05MegakernelTTNN.sample_actions`` call). Returns ``(1, 50, 32)`` float32: the host builds the im2col /
    token / noise inputs, the model copies them into its persistent device buffers and replays the trace
    (captured on the first call = warm-up 1, before READY). ``state`` is not a graph input (pi0.5)."""
    model = STATE["model"]
    if STATE["config"].backend == "mc":
        from models.experimental.pi0_5.server import mc_backend

        return mc_backend.infer(model, images, lang_tokens, lang_masks, noise, prompt_bucket)[0]
    if prompt_bucket is not None:
        raise ValueError("prompt_bucket is a multi-config (PI05_MEGAKERNEL=mc) option")
    return model.sample_actions_fused(images=images, lang_tokens=lang_tokens, noise=noise, lang_masks=lang_masks)


class _Batcher:
    """Dynamic request batching for the fused graph: requests queue up, a worker thread gathers up to the largest
    configured batch (waiting at most ``window_s`` after the first arrival), pads the group to the smallest
    configured batch size that fits (repeating the last request) and runs ONE traced forward for the group; each
    request gets its own action chunk back. One trace per batch size is captured at warm-up (the model keeps one
    prepared shape per batch size and switches between them)."""

    def __init__(self, model, sizes: Tuple[int, ...], window_s: float):
        self.model = model
        self.sizes = tuple(sorted(sizes))
        self.window_s = window_s
        self.q: "queue.Queue" = queue.Queue()
        self.stats = {"forwards": 0, "requests": 0, "by_batch": {str(b): 0 for b in self.sizes}}
        self._thread = threading.Thread(target=self._loop, name="pi05-batcher", daemon=True)
        self._thread.start()

    def submit(self, images: List[torch.Tensor], ids: torch.Tensor, mask: torch.Tensor,
               noise: Optional[torch.Tensor]) -> Future:
        """``mask``: the request's own ``(1, L)`` bool language mask (tokenizer attention mask, or the pre-tokenised
        length). It is passed to the model as ``lang_masks``; the batcher never re-derives it from ``ids != 0``."""
        if mask is None or tuple(mask.shape) != tuple(ids.shape):
            raise ValueError(f"lang mask {None if mask is None else tuple(mask.shape)} != tokens {tuple(ids.shape)}")
        fut: Future = Future()
        self.q.put((images, ids, mask, noise, fut))
        return fut

    def _pick_size(self, n: int) -> int:
        for b in self.sizes:
            if b >= n:
                return b
        return self.sizes[-1]

    def _loop(self) -> None:
        import ttnn  # noqa: F401  (device work happens on this thread only, after warm-up)

        while True:
            first = self.q.get()
            group = [first]
            deadline = time.perf_counter() + self.window_s
            while len(group) < self.sizes[-1]:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    group.append(self.q.get(timeout=remaining))
                except queue.Empty:
                    break
            n = len(group)
            b = self._pick_size(n)
            try:
                padded = group + [group[-1]] * (b - n)
                images = [img for (imgs, _ids, _mask, _noise, _fut) in padded for img in imgs]
                ids = torch.cat([r[1] for r in padded], dim=0)
                masks = torch.cat([r[2] for r in padded], dim=0).bool()
                noise = None
                if any(r[3] is not None for r in padded):
                    default = self.model._default_noise_torch.reshape(1, ACTION_HORIZON, ACTION_DIM)
                    noise = torch.cat([r[3] if r[3] is not None else default for r in padded], dim=0)
                with torch.inference_mode():
                    actions = self.model.sample_actions_fused(images=images, lang_tokens=ids, noise=noise,
                                                              lang_masks=masks)
                if tuple(actions.shape) != (b, ACTION_HORIZON, ACTION_DIM):
                    raise RuntimeError(f"unexpected action shape {tuple(actions.shape)} for batch {b}")
                self.stats["forwards"] += 1
                self.stats["requests"] += n
                self.stats["by_batch"][str(b)] += 1
                for i, (_imgs, _ids, _mask, _noise, fut) in enumerate(group):
                    fut.set_result((actions[i : i + 1].clone(), b))
            except BaseException as e:  # noqa: BLE001
                for _imgs, _ids, _mask, _noise, fut in group:
                    if not fut.done():
                        fut.set_exception(e)


class _DPRouter:
    """``PI05_LAYOUT=dp`` (the multi-robot profile): independent chip groups, each running the whole model with its
    own ``_Batcher`` thread. A request goes to the group with the fewest requests in flight, so two robots land on
    two groups (one request each, the single-request latency) and four robots become two groups of two. Requests
    beyond the groups' capacity queue at the least-loaded group and are batched there."""

    def __init__(self, batchers: List[_Batcher], chips: List[List[int]]):
        self.batchers = batchers
        self.chips = chips
        self._inflight = [0] * len(batchers)
        self._lock = threading.Lock()
        self.stats = {"routed": [0] * len(batchers), "groups": [b.stats for b in batchers]}

    def submit(self, images: List[torch.Tensor], ids: torch.Tensor, mask: torch.Tensor,
               noise: Optional[torch.Tensor]) -> Future:
        with self._lock:
            g = min(range(len(self.batchers)), key=lambda i: (self._inflight[i], i))
            self._inflight[g] += 1
            self.stats["routed"][g] += 1
        fut = self.batchers[g].submit(images, ids, mask, noise)

        def _done(_f, g=g):
            with self._lock:
                self._inflight[g] -= 1

        fut.add_done_callback(_done)
        return fut


def _warmup_inputs(cfg: ServerConfig, tokenizer) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    images = [black_image() for _ in range(cfg.num_images)]
    state32 = np.zeros(STATE_DIM, dtype=np.float32)
    if tokenizer is not None:
        ids, mask, _, _ = tokenize_prompt(tokenizer, build_prompt("warm up", state32), cfg.token_len)
    else:
        ids, mask, _ = tokens_from_request([2], cfg.token_len)  # <bos>
    return images, ids, mask, torch.from_numpy(state32).unsqueeze(0)


def _free_host_weights(model, loader) -> None:
    """Every tensor was converted to device memory in ``PI0ModelTTNN.__init__``; the fp32
    host copy (~14.5 GB) is only referenced by the loader caches and the backbone's unused
    ``torch_weights`` attribute."""
    try:
        if hasattr(model, "backbone"):
            model.backbone.torch_weights = None
        loader._state_dict = None
        loader._categorized = None
        gc.collect()
        LOG.info("Released the fp32 host copy of the checkpoint")
    except Exception as e:  # noqa: BLE001
        LOG.warning("could not release host weights: %s", e)


def _start_dp(cfg: ServerConfig, config, loader, fused_cfg, device, tokenizer, n_tensors: int) -> None:
    """``PI05_LAYOUT=dp`` (the multi-robot profile): the mesh is carved into ``cols / PI05_DP_GROUP`` sub-meshes; every
    sub-mesh runs the whole model (prefix tensor-parallel inside the group, expert replicated) with its own traces and
    batching worker; ``_DPRouter`` spreads requests over the groups."""
    import ttnn
    from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

    rows, cols = STATE["mesh_shape"]
    n_chips = rows * cols
    if n_chips % cfg.dp_group != 0 or n_chips < cfg.dp_group:
        raise RuntimeError(f"PI05_DP_GROUP={cfg.dp_group} does not divide the {rows}x{cols} mesh")
    n_groups = n_chips // cfg.dp_group
    if n_groups == 1:
        raise RuntimeError("PI05_LAYOUT=dp with one group is the mesh layout; use PI05_LAYOUT=mesh")
    subs = [device.create_submesh(ttnn.MeshShape(1, cfg.dp_group), ttnn.MeshCoordinate(0, g * cfg.dp_group)) for g in range(n_groups)]
    for sub in subs:
        sub.enable_program_cache()
    cfg_g = fused_cfg.resolved(cfg.dp_group)
    LOG.info(
        "Loading %d models: converting %d tensors to device per %d-chip group (TP=%d inside a group) ...",
        n_groups,
        n_tensors,
        cfg.dp_group,
        cfg_g.tp,
    )
    t0 = time.perf_counter()
    models = []
    for sub in subs:
        torch.manual_seed(cfg.seed)  # every group draws the same fixed initial noise (as the tests do)
        models.append(PI0ModelTTNN(config, loader, sub, fused=cfg_g))
    chips = [list(sub.get_device_ids()) for sub in subs]
    LOG.info("Models built in %.1f s on chip groups %s", time.perf_counter() - t0, chips)
    STATE["dp_models"] = models
    STATE["dp_submeshes"] = subs
    STATE["dp_chips"] = chips
    STATE["model"] = None
    STATE["fused"] = models[0].fused_cfg.describe()
    STATE["tokenizer"] = tokenizer
    if cfg.free_host_weights:
        for m in models[1:]:
            m.backbone.torch_weights = None
        _free_host_weights(models[0], loader)

    images, ids, mask, _state = _warmup_inputs(cfg, tokenizer)
    latencies: List[float] = []
    for gi, m in enumerate(models):
        for b in cfg.batch_sizes:
            for i in range(cfg.warmup_runs):
                t0 = time.perf_counter()
                with torch.inference_mode():
                    acts = m.sample_actions_fused(images=images * b, lang_tokens=ids.repeat(b, 1),
                                                  lang_masks=mask.repeat(b, 1))
                dt = (time.perf_counter() - t0) * 1000.0
                if tuple(acts.shape) != (b, ACTION_HORIZON, ACTION_DIM) or not torch.isfinite(acts).all():
                    raise RuntimeError(f"warm-up of group {gi} at batch {b} produced {tuple(acts.shape)}")
                if b == cfg.batch_sizes[0] and gi == 0:
                    latencies.append(dt)
                LOG.info("Warmup group %d batch %d %d/%d: %.1f ms", gi, b, i + 1, cfg.warmup_runs, dt)
        if m.fused_cfg.trace and getattr(m, "_fused_trace_id", None) is None:
            raise RuntimeError(f"group {gi}: no trace was captured during warm-up")
    STATE["warmup_ms"] = {"first": round(latencies[0], 1), "last": round(latencies[-1], 1)}
    STATE["traced"] = bool(models[0].fused_cfg.trace)
    batchers = [_Batcher(m, cfg.batch_sizes, cfg.batch_window_ms / 1000.0) for m in models]
    STATE["batcher"] = _DPRouter(batchers, chips)
    STATE["ready"] = True
    LOG.info("Warmup complete (first %.1f ms, steady %.1f ms) -- %d groups traced", latencies[0], latencies[-1], n_groups)





# ----------------------------------------------------------------------------------------
# lifespan
# ----------------------------------------------------------------------------------------


def _start_mc(cfg: ServerConfig) -> None:
    """The default backend: weights -> tokenizer -> device -> ``PI05MegakernelTTNN`` -> compile + capture every prompt
    bucket -> warm-up requests. Leaves STATE ready; on any failure releases the model and the device and re-raises."""
    from models.experimental.pi0_5.server import mc_backend

    weights_dir, weights_info = resolve_weights_dir(cfg)
    t0 = time.perf_counter()
    loader = mc_backend.weight_loader(weights_dir)
    n_tensors = len(loader.state_dict)  # materialises the fp32 safetensors in host RAM
    LOG.info("Loading weights done: %d tensors in %.1f s", n_tensors, time.perf_counter() - t0)
    STATE["weights"] = weights_info
    tokenizer = load_tokenizer(cfg)

    device, open_kwargs = mc_backend.open_device(cfg.device_id)
    LOG.info("Opening device %s (multi-config megakernel)", {"device_id": cfg.device_id, **open_kwargs})
    STATE["device"] = device
    STATE["mesh_shape"] = [1, 1]
    try:
        LOG.info(
            "Loading pipeline: converting %d tensors to device (PI05MegakernelTTNN: %d camera(s), H=%d, %d steps) ...",
            n_tensors, cfg.num_images, cfg.action_horizon, cfg.num_steps,
        )
        t0 = time.perf_counter()
        torch.manual_seed(cfg.seed)  # the model draws its default initial noise here
        model = mc_backend.build_model(loader, device, cfg.num_images, cfg.action_horizon, cfg.num_steps)
        STATE["model"] = model
        STATE["tokenizer"] = tokenizer
        LOG.info("Model built in %.1f s", time.perf_counter() - t0)
        if cfg.free_host_weights:
            _free_host_weights(model, loader)
        del loader
        LOG.info("Warming up: compiling and capturing the %s-token prompt buckets ...",
                 " / ".join(str(p.prompt_len) for p in model.presets))
        t0 = time.perf_counter()
        with LOCK:
            model.warmup()
        LOG.info("Warmup traces captured in %.1f s", time.perf_counter() - t0)
        images, ids, mask, state = _warmup_inputs(cfg, tokenizer)
        latencies = []
        for i in range(cfg.warmup_runs):
            t0 = time.perf_counter()
            with LOCK, torch.inference_mode():
                actions = run_inference(images, ids, mask, state)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if tuple(actions.shape) != (1, cfg.action_horizon, ACTION_DIM) or not torch.isfinite(actions).all():
                raise RuntimeError(f"warm-up produced actions of shape {tuple(actions.shape)} (or non-finite)")
            LOG.info("Warmup %d/%d: %.1f ms", i + 1, cfg.warmup_runs, latencies[-1])
        STATE["warmup_ms"] = {"first": round(latencies[0], 1), "last": round(latencies[-1], 1)}
        STATE["traced"] = True
        STATE["megakernel"] = mc_backend.describe(model)
        STATE["fused"] = None
        STATE["batcher"] = None
        STATE["ready"] = True
        LOG.info("Warmup complete (first %.1f ms, steady %.1f ms) -- %d device ops per call, one trace per prompt bucket",
                 latencies[0], latencies[-1], STATE["megakernel"]["program"]["device_ops_per_call"])
    except BaseException:
        _stop_mc()
        raise


def _stop_mc() -> None:
    STATE["ready"] = False
    STATE.pop("tokenizer", None)
    model = STATE.pop("model", None)
    try:
        if model is not None:
            model.close()  # traces + device tensors
    finally:
        del model
        gc.collect()
        dev = STATE.pop("device", None)
        if dev is not None:
            import ttnn

            LOG.info("Closing device")
            ttnn.close_device(dev)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    torch.set_grad_enabled(False)
    cfg = load_config()
    STATE["config"] = cfg
    STATE["ready"] = False
    mesh = parse_mesh_shape(os.environ.get("TT_MESH_SHAPE"))
    LOG.info("config: %s mesh=%sx%s", asdict(cfg), *mesh)
    if cfg.backend == "mc":
        _start_mc(cfg)
        try:
            yield
        finally:
            _stop_mc()
        return

    # 1. weights on the host first (no device yet): a download/auth failure is cheap here
    weights_dir, weights_info = resolve_weights_dir(cfg)
    from models.experimental.pi0_5.common.configs import PI0ModelConfig, SigLIPConfig
    from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader

    t0 = time.perf_counter()
    loader = PI0WeightLoader(weights_dir)
    n_tensors = len(loader.state_dict)  # materialises the fp32 safetensors in host RAM
    LOG.info("Loading weights done: %d tensors in %.1f s", n_tensors, time.perf_counter() - t0)
    STATE["weights"] = weights_info

    # 2. tokenizer (gated) -- before the device so a missing token fails fast and loudly
    tokenizer = load_tokenizer(cfg)

    # 3. device + model, the validated recipe
    import ttnn
    from models.experimental.pi0_5.common.fused_config import FusedConfig
    from models.experimental.pi0_5.tt.ttnn_pi0_model import PI0ModelTTNN

    fused_cfg = FusedConfig.from_env()  # read once here; passed to the model explicitly
    STATE["fused"] = fused_cfg.describe()
    if cfg.layout == "dp" and mesh == (1, 1):
        raise RuntimeError("PI05_LAYOUT=dp needs a multi-chip mesh (TT_MESH_SHAPE)")
    if fused_cfg.megakernel != "off" and not fused_cfg.megakernel_explicit and mesh != (1, 1):
        # the UNSET default on a multi-chip mesh is the stock-op path (FusedConfig.resolved does the same in the model)
        from dataclasses import replace as _replace

        LOG.info("PI05_MEGAKERNEL unset on a %sx%s mesh: the single-chip megakernel does not apply -> off", *mesh)
        fused_cfg = _replace(fused_cfg, megakernel="off")
        STATE["fused"] = fused_cfg.describe()
    if fused_cfg.megakernel != "off":
        # docs/megakernel/DESIGN.md §4.12 refusals: the megakernel is single-chip, batch 1, one prepared shape
        from models.experimental.pi0_5.tt.megakernel.geometry import megakernel_refusal

        why = megakernel_refusal(fused_cfg.kv_dtype, cfg.num_steps)  # kv dtype and step count the kernels compile
        if why is None and mesh != (1, 1):
            why = f"TT_MESH_SHAPE={mesh[0]}x{mesh[1]} (the megakernel is single-chip: use 1x1)"
        elif why is None and cfg.layout != "mesh":
            why = f"PI05_LAYOUT={cfg.layout} (single-chip megakernel: use mesh on a 1x1 device)"
        elif why is None and cfg.batch_sizes != (1,):
            why = f"PI05_BATCH_SIZES={','.join(map(str, cfg.batch_sizes))} (the megakernel serves batch 1 only)"
        elif why is None and fused_cfg.megakernel == "whole" and cfg.num_images != 2:
            why = f"PI05_NUM_IMAGES={cfg.num_images} (the whole-model megakernel is built for 2 camera slots)"
        if why is not None:
            LOG.error("PI05_MEGAKERNEL=%s refused at startup: %s", fused_cfg.megakernel, why)
            raise RuntimeError(f"PI05_MEGAKERNEL={fused_cfg.megakernel} refused: {why}")
    if mesh != (1, 1):
        # Multi-chip (2x p300 = 1x4 Ethernet ring): fabric + MeshDevice; the SigLIP tower and the VLM
        # prefill are tensor-parallel over the chips, the expert replicated (tt/ttnn_ccl.py).
        from models.experimental.pi0_5.tt import ttnn_ccl

        LOG.info("Opening mesh %sx%s (fabric %s) fused=%s", *mesh, fused_cfg.ccl_topology, fused_cfg.describe())
        device = ttnn_ccl.open_mesh(fused_cfg, mesh, l1_small_size=cfg.l1_small_size)
    else:
        from models.experimental.pi0_5.common.device_open import device_kwargs

        # the one device-open helper: adds the megakernel's 64 KiB worker-L1 cut when PI05_MEGAKERNEL != off
        open_kwargs: Dict[str, Any] = device_kwargs(fused_cfg, device_id=cfg.device_id, l1_small_size=cfg.l1_small_size)
        LOG.info("Opening device %s%s", open_kwargs, f" fused={fused_cfg.describe()}")
        device = ttnn.open_device(**open_kwargs)
    STATE["device"] = device
    STATE["mesh_shape"] = list(mesh)
    try:
        device.enable_program_cache()
        config = PI0ModelConfig(
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
            state_dim=STATE_DIM,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            num_denoising_steps=cfg.num_steps,
            pi05=True,
        )
        config.siglip_config = SigLIPConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=IMAGE_SIZE,
            patch_size=14,
        )
        if cfg.layout == "dp":
            _start_dp(cfg, config, loader, fused_cfg, device, tokenizer, n_tensors)
        else:
            LOG.info("Loading pipeline: converting %d tensors to device (PI0ModelTTNN, pi05=True) ...", n_tensors)
            t0 = time.perf_counter()
            torch.manual_seed(cfg.seed)  # the model draws its fixed initial noise here (as the tests do)
            model = PI0ModelTTNN(config, loader, device, fused=fused_cfg)
            LOG.info(
                "Model built in %.1f s%s",
                time.perf_counter() - t0,
                " (fused graph; the first warm-up call compiles and captures the trace)",
            )
            STATE["model"] = model
            STATE["fused"] = model.fused_cfg.describe()  # resolved knobs (PI05_TP=0 -> mesh size, TP-mode defaults)
            STATE["tokenizer"] = tokenizer
            if cfg.free_host_weights:
                _free_host_weights(model, loader)

            # 4. warm-up: first call compiles every kernel; the last one gives the steady-state latency
            LOG.info(
                "Warming up: %d x sample_actions on %d black image(s), L=%d tokens, %d steps ...",
                cfg.warmup_runs,
                cfg.num_images,
                cfg.token_len,
                cfg.num_steps,
            )
            images, ids, mask, state = _warmup_inputs(cfg, tokenizer)
            latencies = []
            if cfg.batch_sizes != (1,):
                # one prepared shape (inputs, KV caches, trace) per batch size; batch 1 first (its caches stay in L1)
                for b in cfg.batch_sizes:
                    if b == 1:
                        continue
                    for i in range(cfg.warmup_runs):
                        t0 = time.perf_counter()
                        with LOCK, torch.inference_mode():
                            acts = model.sample_actions_fused(images=images * b, lang_tokens=ids.repeat(b, 1),
                                                              lang_masks=mask.repeat(b, 1))
                        if tuple(acts.shape) != (b, ACTION_HORIZON, ACTION_DIM) or not torch.isfinite(acts).all():
                            raise RuntimeError(f"warm-up at batch {b} produced {tuple(acts.shape)}")
                        LOG.info("Warmup batch %d %d/%d: %.1f ms", b, i + 1, cfg.warmup_runs, (time.perf_counter() - t0) * 1000)
            for i in range(cfg.warmup_runs):
                t0 = time.perf_counter()
                with LOCK, torch.inference_mode():
                    actions = run_inference(images, ids, mask, state)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                if tuple(actions.shape) != (1, ACTION_HORIZON, ACTION_DIM):
                    raise RuntimeError(f"warm-up produced actions of shape {tuple(actions.shape)}")
                if not torch.isfinite(actions).all():
                    raise RuntimeError("warm-up produced non-finite actions")
                LOG.info("Warmup %d/%d: %.1f ms", i + 1, cfg.warmup_runs, latencies[-1])
            STATE["warmup_ms"] = {"first": round(latencies[0], 1), "last": round(latencies[-1], 1)}
            if fused_cfg.trace and getattr(model, "_fused_trace_id", None) is None:
                raise RuntimeError("PI05_TRACE=1 but no trace was captured during warm-up")
            STATE["traced"] = getattr(model, "_fused_trace_id", None) is not None
            STATE["megakernel"] = {"backend": getattr(model, "megakernel_backend", "off"),
                                   "program": getattr(model, "megakernel_program", None)}
            STATE["batcher"] = _Batcher(model, cfg.batch_sizes, cfg.batch_window_ms / 1000.0)
            STATE["ready"] = True
            LOG.info(
                "Warmup complete (first %.1f ms, steady %.1f ms)%s",
                latencies[0],
                latencies[-1],
                " -- fused graph traced" if STATE["traced"] else "",
            )
        yield
    finally:
        STATE["ready"] = False
        model = STATE.pop("model", None)
        STATE.pop("tokenizer", None)
        batcher = STATE.pop("batcher", None)  # holds the model / pipeline reference
        if batcher is not None:
            batcher.pipe = None if hasattr(batcher, "pipe") else None
            batcher.model = None if hasattr(batcher, "model") else None
            del batcher
        dp_models = STATE.pop("dp_models", None) or []
        dp_subs = STATE.pop("dp_submeshes", None) or []
        for m in dp_models:
            try:
                m.release_trace()
            except Exception:  # noqa: BLE001
                pass
        del dp_models
        gc.collect()
        for sub in dp_subs:
            try:
                import ttnn as _ttnn

                _ttnn.close_mesh_device(sub)  # sub-meshes before the parent (see PI05DisaggPipeline.close)
            except Exception as e:  # noqa: BLE001
                LOG.warning("closing a sub-mesh: %s", e)
        del dp_subs
        pipe = STATE.pop("pipeline", None)
        if pipe is not None:
            try:
                pipe.close()  # traces, sockets, buffers, models, then the two sub-meshes (before the parent)
            except Exception as e:  # noqa: BLE001
                LOG.warning("pipeline teardown: %s", e)
            del pipe
        if model is not None:
            try:
                model.release_trace()  # no-op unless the traced path was used
            except Exception:  # noqa: BLE001
                pass
            del model
        gc.collect()
        dev = STATE.pop("device", None)
        if dev is not None:
            LOG.info("Closing device")
            from models.experimental.pi0_5.tt import ttnn_ccl

            ttnn_ccl.close_mesh(dev)  # single chip or mesh (disables the fabric after a real mesh)
        if cfg.layout in ("dp", "pipeline") and _env_bool("PI05_RESET_ON_EXIT", False):
            # Safety net for the pipeline layout: reset the chips after the mesh is closed so the next process
            # never finds an Ethernet core still busy (tt-smi ships in the image; ~10 s).
            import shutil
            import subprocess

            tt_smi = shutil.which("tt-smi")
            if tt_smi:
                LOG.info("PI05_RESET_ON_EXIT: resetting the chips with tt-smi")
                subprocess.run([tt_smi, "-r", "all"], check=False, timeout=120)
            else:
                LOG.warning("PI05_RESET_ON_EXIT set but tt-smi is not on PATH")


app = FastAPI(title="pi-0.5 on Tenstorrent Blackhole", lifespan=lifespan)


# ----------------------------------------------------------------------------------------
# schemas
# ----------------------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """One observation -> one action chunk."""

    images: List[str] = Field(
        ...,
        min_length=1,
        description="base64 PNG/JPEG, ordered [base/exterior camera, wrist camera, ...]. The multi-config "
        "megakernel (default) takes exactly PI05_NUM_IMAGES; the single-config paths pad missing slots with a black "
        "image (degraded fidelity, documented).",
    )
    prompt: Optional[str] = Field(
        None, description="Task instruction, e.g. 'pick up the cube'. Required unless tokens is given."
    )
    state: Optional[List[float]] = Field(
        None,
        description="Robot proprioceptive state, <=32 floats ALREADY normalised to [-1, 1] with your dataset's "
        "QUANTILES stats (2*(x-q01)/(q99-q01)-1). Zero-padded to 32; discretised into the prompt. Default: zeros.",
    )
    tokens: Optional[List[int]] = Field(
        None, description="Pre-tokenised PaliGemma prompt (bypasses the gated tokenizer); <= PI05_TOKEN_LEN ids."
    )
    num_steps: Optional[int] = Field(
        None, description="Must equal the server's PI05_NUM_STEPS (default 10); informational."
    )
    seed: Optional[int] = Field(
        None, description="Seed for the initial flow-matching noise. Default: the model's fixed seeded noise."
    )
    prompt_bucket: Optional[int] = Field(
        None, description="Multi-config megakernel only: run in this prompt bucket (32 / 64 / 128 / 224 tokens) "
        "instead of the smallest one that holds the prompt."
    )


# ----------------------------------------------------------------------------------------
# routes
# ----------------------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    cfg = STATE.get("config")
    return {
        "status": "ok" if STATE.get("ready") else "starting",
        "model": MODEL_NAME,
        "device": f"blackhole:{cfg.device_id}" if cfg else None,
    }


def _hardware_string(mesh_shape, layout: str = "mesh", chips=None) -> str:
    rows, cols = (mesh_shape or (1, 1))
    if layout == "dp":
        groups = chips or []
        return (
            f"Tenstorrent Blackhole, {rows}x{cols} mesh ({rows * cols} chips, 2x p300, Ethernet ring) via tt-nn, "
            f"{len(groups)} independent chip groups {groups}: each runs the whole model (SigLIP + VLM prefill "
            "tensor-parallel inside the group, action expert replicated) with its own traces and batching worker; "
            "requests are routed to the least-loaded group; "
        )
    if layout == "pipeline":
        chips = chips or {}
        return (
            f"Tenstorrent Blackhole, {rows}x{cols} mesh ({rows * cols} chips, 2x p300, Ethernet ring) via tt-nn, "
            f"prefix / expert pipeline: SigLIP + VLM prefill tensor-parallel on chips {chips.get('prefix')}, action "
            f"expert replicated on chips {chips.get('expert')}, prefix K/V over mesh sockets; the prefix of request "
            "group n+1 overlaps the expert of group n; "
        )
    if rows * cols > 1:
        return (
            f"Tenstorrent Blackhole, {rows}x{cols} mesh ({rows * cols} chips, 2x p300, Ethernet ring) via tt-nn: "
            "SigLIP + VLM prefill tensor-parallel over the chips, action expert replicated; "
        )
    return "Tenstorrent Blackhole, single chip (mesh 1x1) via tt-nn; "


def _graph_string(cfg: Optional[ServerConfig], traced: bool) -> str:
    layout = cfg.layout if cfg is not None else "mesh"
    if layout == "pipeline":
        return "fused prefix and expert graphs" + (", one Metal trace per stage per batch size" if traced else ", eager")
    if layout == "dp":
        return "fused whole-graph sample_actions_fused per group" + (", one Metal trace per group per batch size" if traced else ", eager")
    mk = (STATE.get("megakernel") or {}).get("backend", "off")
    if mk == "mc":
        prog = STATE["megakernel"]["program"]
        return (f"multi-config megakernel: the entire sample_actions as {prog['device_ops_per_call']} persistent "
                "generic_op programs (vision | VLM prefix | action expert) on 110 cores"
                + (", one Metal trace per prompt bucket" if traced else ", eager"))
    if mk == "whole":
        return ("whole-model megakernel: the entire sample_actions (SigLIP x2, projector, language embedding, VLM "
                "prefill -> K / V caches, the 10-step expert loop) as ONE persistent generic_op"
                + (", one Metal trace" if traced else ", eager"))
    if mk == "expert":
        return ("fused whole-graph sample_actions_fused: traced stock-ttnn SigLIP / VLM prefix + the phase-1 expert "
                "megakernel (ONE persistent generic_op for the whole 10-step action expert loop)"
                + (", one Metal trace" if traced else ", eager"))
    return "fused whole-graph sample_actions_fused" + (", one Metal trace per batch size" if traced else ", eager")


@app.get("/info")
def info() -> dict:
    cfg: Optional[ServerConfig] = STATE.get("config")
    return {
        "model": "pi-0.5 (Physical Intelligence pi0.5 vision-language-action policy, lerobot/pi05_base)",
        "name": MODEL_NAME,
        "status": "ok" if STATE.get("ready") else "starting",
        "task": f"images + language instruction (+ normalised state) -> {cfg.action_horizon if cfg else ACTION_HORIZON}"
        "-step chunk of 32-dim normalised actions (flow matching, Euler, on device)",
        "profile": cfg.profile if cfg else None,
        "layout": cfg.layout if cfg else None,
        "hardware": _hardware_string(
            STATE.get("mesh_shape"), cfg.layout if cfg else "mesh", STATE.get("dp_chips") if (cfg and cfg.layout == "dp") else STATE.get("pipeline_chips")
        )
        + _graph_string(cfg, bool(STATE.get("traced"))),
        "fused": STATE.get("fused"),
        "megakernel": STATE.get("megakernel"),
        "mesh_shape": STATE.get("mesh_shape"),
        "weights": STATE.get("weights")
        or {
            "repo": cfg.weights_repo if cfg else DEFAULT_WEIGHTS_REPO,
            "revision": cfg.weights_revision if cfg else None,
        },
        "tokenizer": {
            "repo": cfg.tokenizer_repo if cfg else DEFAULT_TOKENIZER_REPO,
            "available": STATE.get("tokenizer") is not None,
            "note": "gated google/paligemma-3b-pt-224; send `tokens` to bypass",
        },
        "source": {
            "repo": SOURCE_REPO,
            "commit": SOURCE_COMMIT,
            "path": "models/experimental/pi0 (PI05MegakernelTTNN) + models/experimental/pi0_5 (server)"
            if cfg and cfg.backend == "mc" else "models/experimental/pi0_5",
            "tt_metal": TT_METAL_NOTE,
        },
        "inputs": {
            "num_images": cfg.num_images if cfg else 2,
            "image_order": ["base/exterior camera", "wrist camera", "second wrist camera"][
                : cfg.num_images if cfg else 2
            ],
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "image_preprocess": "RGB, bilinear squash-resize to 224x224 (no aspect padding), x * float32(1/255) * 2 - 1",
            "token_len": cfg.token_len if cfg else 224,
            "prompt_buckets": (STATE.get("megakernel") or {}).get("program", {}).get("prompt_buckets"),
            "missing_images": "refused (exactly num_images)" if cfg and cfg.backend == "mc" else "padded with black",
            "prompt_format": "Task: <prompt>, State: b0 ... b31;\\nAction:  (state discretised into 256 bins over [-1,1])",
            "state_dim": STATE_DIM,
            "state_semantics": "normalised [-1,1] QUANTILES space; enters the model only through the prompt (pi0.5)",
            "batch": {
                "sizes": list(cfg.batch_sizes) if cfg else [1],
                "window_ms": cfg.batch_window_ms if cfg else 0,
                "note": "concurrent requests are batched into one traced forward (padded to the next configured size)"
                + ("; the pipeline runs the prefix of one group while the expert denoises the previous one" if cfg and cfg.layout == "pipeline" else "")
                + ("; requests are spread over independent chip groups first and batched only within a busy group" if cfg and cfg.layout == "dp" else ""),
                "stats": STATE["batcher"].stats if STATE.get("batcher") else None,
                "pipeline_stage_ms": STATE.get("pipeline_stage_ms"),
            },
        },
        "outputs": {
            "actions": [cfg.action_horizon if cfg else ACTION_HORIZON, ACTION_DIM],
            "normalized": True,
            "note": "lerobot QUANTILES-normalised actions padded to 32 dims; pi05_base ships no per-feature stats, "
            "so denormalise with your dataset's action q01/q99 and slice to your action dim",
            "denoising_steps": cfg.num_steps if cfg else 10,
        },
        "limits": {
            "max_images": cfg.num_images if cfg else 2,
            "min_images": (cfg.num_images if cfg.backend == "mc" else 1) if cfg else 1,
            "max_state": STATE_DIM,
            "max_tokens": cfg.token_len if cfg else 224,
            "batch": max(cfg.batch_sizes) if cfg else 1,
            "per_request_num_steps": False,
        },
        "warmup_latency_ms": STATE.get("warmup_ms"),
        "license": LICENSE_NOTE,
    }


@app.get("/v1/models")
def v1_models() -> dict:
    cfg: Optional[ServerConfig] = STATE.get("config")
    return {
        "object": "list",
        "data": [
            {
                "id": cfg.weights_repo if cfg else DEFAULT_WEIGHTS_REPO,
                "object": "model",
                "owned_by": "changh95",
            }
        ],
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    if not STATE.get("ready"):
        raise HTTPException(status_code=503, detail="model is still starting")
    cfg: ServerConfig = STATE["config"]
    t_start = time.perf_counter()

    # --- images -------------------------------------------------------------------------
    if cfg.backend == "mc" and len(req.images) != cfg.num_images:
        raise HTTPException(
            status_code=400,
            detail=f"{len(req.images)} images given; this server's model is built for exactly {cfg.num_images} "
            "camera(s) (PI05_NUM_IMAGES). Masked or padded camera slots are not served: send only the real cameras "
            "to a server started with PI05_NUM_IMAGES = that count",
        )
    if len(req.images) > cfg.num_images:
        raise HTTPException(
            status_code=400,
            detail=f"{len(req.images)} images given; this server has {cfg.num_images} camera slot(s) (PI05_NUM_IMAGES)",
        )
    images: List[torch.Tensor] = []
    original_sizes = []
    for i, b64 in enumerate(req.images):
        t, size = decode_image(b64, i)
        images.append(t)
        original_sizes.append(list(size))
    images_padded = cfg.num_images - len(images)
    images.extend(black_image() for _ in range(images_padded))

    # --- state + language -----------------------------------------------------------------
    state32 = state_from_request(req.state)
    if req.num_steps is not None and req.num_steps != cfg.num_steps:
        raise HTTPException(
            status_code=400,
            detail=f"num_steps={req.num_steps} cannot be changed per request: the port precomputes the per-step "
            f"conditioning at build time (server runs {cfg.num_steps}; set PI05_NUM_STEPS to change it)",
        )
    prompt: Optional[str] = None
    truncated = False
    if req.tokens is not None:
        ids, mask, n_real = tokens_from_request(req.tokens, cfg.token_len)
    elif req.prompt is not None:
        tokenizer = STATE.get("tokenizer")
        if tokenizer is None:
            raise HTTPException(
                status_code=400,
                detail="this server has no tokenizer (gated google/paligemma-3b-pt-224 not available); send `tokens`",
            )
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt is empty")
        prompt = build_prompt(req.prompt, state32)
        with TOK_LOCK:
            ids, mask, n_real, truncated = tokenize_prompt(tokenizer, prompt, cfg.token_len)
    else:
        raise HTTPException(status_code=400, detail="either prompt or tokens is required")

    noise = None
    if req.seed is not None:
        gen = torch.Generator().manual_seed(int(req.seed))
        noise = torch.randn(1, cfg.action_horizon, ACTION_DIM, generator=gen)
    state_t = torch.from_numpy(state32).unsqueeze(0)
    bucket = None
    if cfg.backend == "mc":
        try:
            bucket = STATE["model"].preset_for(int(mask.sum()), req.prompt_bucket).prompt_len
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
    elif req.prompt_bucket is not None:
        raise HTTPException(status_code=400, detail="prompt_bucket is a multi-config (PI05_MEGAKERNEL=mc) option")
    t_pre = time.perf_counter()

    # --- device -----------------------------------------------------------------------------
    batched_as = 1
    try:
        batcher = STATE.get("batcher")
        if batcher is not None:
            actions, batched_as = batcher.submit(images, ids, mask, noise).result(timeout=120.0)
        else:
            with LOCK, torch.inference_mode():
                actions = run_inference(images, ids, mask, state_t, noise, req.prompt_bucket)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        LOG.error("inference failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from None
    t_end = time.perf_counter()

    if tuple(actions.shape) != (1, cfg.action_horizon, ACTION_DIM):
        raise HTTPException(status_code=500, detail=f"unexpected action shape {tuple(actions.shape)}")
    return {
        "actions": actions[0].tolist(),
        "action_horizon": cfg.action_horizon,
        "action_dim": ACTION_DIM,
        "normalized": True,
        "denoising_steps": cfg.num_steps,
        "prompt": prompt,
        "num_tokens": n_real,
        "token_len": cfg.token_len,
        "prompt_bucket": bucket,
        "prompt_truncated": truncated,
        "images_used": len(req.images),
        "images_padded": images_padded,
        "original_sizes": original_sizes,
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "seed": req.seed,
        "batched_as": batched_as,
        "timing_ms": {
            "preprocess": round((t_pre - t_start) * 1000.0, 2),
            "inference": round((t_end - t_pre) * 1000.0, 2),
            "total": round((t_end - t_start) * 1000.0, 2),
        },
    }
