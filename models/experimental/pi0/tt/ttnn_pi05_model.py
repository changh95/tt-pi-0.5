# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
pi0.5 inference as THREE device operations per call (Blackhole, single chip).

``PI05MegakernelTTNN`` runs the whole ``sample_actions`` as three persistent ``ttnn.generic_op`` programs on 110 cores
(``tt/megakernel``), captured in one Metal trace per preset:

1. VISION: SigLIP on the cameras and the projector;
2. PREFIX: the language embedding and the VLM prefill that writes the 18 K / V caches;
3. EXPERT: the N-step x 18-layer action expert with the time MLP / adaRMS conditioning, the action in / out
   projections and the Euler steps.

``PI0ModelTTNN`` delegates to it when ``config.pi05`` is set. Semantics are openpi's pi0.5: pad keys of the prompt are
masked, the prefix is bidirectional, the action tokens sit at positions ``n_valid + [0, H)``, the time embedding
conditions every expert norm through adaRMS.

A model fixes the cameras, the action horizon and the denoising steps at construction; a request picks one of the
model's presets (``megakernel/presets.py``). Every device buffer is allocated at construction at the largest size the
model's presets need. Per request the host writes the inputs into those fixed buffers (camera patches, token ids, the
VLM key mask, the expert key row and RoPE rows, the noise) and replays the preset's trace. The first call of a preset
compiles its programs (or loads them from the kernel cache), runs them once eagerly and captures the trace.

Supported: batch 1, 1 or 2 cameras of 224 x 224 (``config.num_cameras``), 1..10 denoising steps
(``num_denoising_steps``), an action horizon of 1..64 (suffix buckets of 32 / 64 rows) and right-padded prompts in a
token buffer of any length with up to 224 real tokens: a request runs in the smallest prompt bucket (32 / 64 / 128 /
224 tokens) that holds its real tokens (``prompt_bucket=`` overrides it).

The device must be opened with ``PI05_DEVICE_PARAMS`` (the 64 KiB worker-L1 cut leaves each program its 136,192 B
kernel-config ring).
"""

import weakref
from typing import Dict, List, Optional, Tuple

import torch
import ttnn

from models.experimental.pi0.common.configs import PI0ModelConfig
from models.experimental.pi0.common.pi05_host import (
    MASK_NEG,
    attention_inputs,
    im2col_patches,
    kv_cache_plan,
    pad_rows,
    prefix_valid_mask,
    round_up,
    unpad_rows,
)
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from .megakernel import geometry as G
from .megakernel import pe_geometry as P
from .megakernel import presets as PS
from .megakernel.host_model import expert_params
from .megakernel.pe_host import prefix_params, vlm_key_mask
from .megakernel.pe_program import PREFIX_OPS, VISION_OPS, PrefixEngineProgram, PrefixTensors, kernel_digest
from .megakernel.program import ExpertMegakernel
from .ttnn_gemma import precompute_freqs_cis_meta_format

# The megakernel needs the 64 KiB worker-L1 cut: without it a program misses the kernel-config ring (136,192 B).
MEGAKERNEL_WORKER_L1_SIZE = 1_395_712
PI05_DEVICE_PARAMS = {
    "l1_small_size": 24576,
    "worker_l1_size": MEGAKERNEL_WORKER_L1_SIZE,
    "trace_region_size": 10_000_000,
}

PATCH_FEATURES = 608  # 14 * 14 * 3 = 588 im2col features, tile-padded


class _Programs:
    """The programs of one preset: VISION (one per camera group of <= 2), PREFIX, EXPERT."""

    def __init__(self, preset: PS.Preset, visions: List[PrefixEngineProgram], prefix: PrefixEngineProgram, expert):
        self.preset, self.visions, self.prefix, self.expert = preset, list(visions), prefix, expert
        self.vision = self.visions[0]


# models alive (built, not closed, still referenced): a second one on a device is refused by name. Its persistent L1
# (K / V caches, tables, noise, out) sits at the top of L1, where the first model's buffers would collide with the
# second model's program CB regions (an anonymous "circular buffers clash with L1 buffers" TT_THROW at its first call)
# and its traces would hold the trace region.
_LIVE: "weakref.WeakSet[PI05MegakernelTTNN]" = weakref.WeakSet()


def live_models(device) -> List["PI05MegakernelTTNN"]:
    """The live (not closed) models on ``device``."""
    return [m for m in list(_LIVE) if m.device is device and not m._closed]


def _device_tensors(root) -> List[ttnn.Tensor]:
    """Every ttnn.Tensor reachable from ``root`` through this package's objects, lists, tuples and dicts (each once)."""
    seen, out, stack = set(), [], [root]
    while stack:
        x = stack.pop()
        if id(x) in seen:
            continue
        seen.add(id(x))
        if isinstance(x, ttnn.Tensor):
            out.append(x)
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif type(x).__module__.startswith("models.experimental.pi0") and hasattr(x, "__dict__"):
            stack.extend(vars(x).values())
    return out


class PI05MegakernelTTNN:
    """pi0.5 ``sample_actions`` as three ``ttnn.generic_op`` programs (vision | prefix | expert), replayed from a Metal
    trace per preset."""

    def __init__(
        self,
        config: PI0ModelConfig,
        weight_loader: PI0WeightLoader,
        device: ttnn.Device,
        kv_dram: Optional[bool] = None,
    ):
        """``kv_dram``: the K / V caches in DRAM (None: ``presets.kv_in_dram`` -- L1 unless a program would keep
        < 16 KB of L1 free, i.e. 4 cameras; True / False force it: the kernels read the caches through
        TensorAccessors, so only the memory config and the expert's row multicast change)."""
        if not config.pi05:
            raise ValueError("PI05MegakernelTTNN needs PI0ModelConfig(pi05=True)")
        others = live_models(device)
        if others:
            desc = ", ".join(
                f"{m.cameras} cameras / {m.suffix_rows} action rows ({m.l1_bytes:,} B of L1 per core)" for m in others
            )
            raise RuntimeError(
                f"pi0.5 megakernel: {len(others)} other model(s) on this device still hold device memory ({desc}); "
                "close() them or build them in a `with` block before building another"
            )
        why = self.device_refusal(device)
        if why is None:
            why = G.megakernel_refusal(config.num_denoising_steps, config.action_horizon)
        if why is None and config.num_cameras not in PS.CAMERAS:
            why = f"cameras = {config.num_cameras} (compiled: {', '.join(map(str, PS.CAMERAS))})"
        if why is None and config.action_dim != 32:
            why = f"action_dim={config.action_dim} (the megakernel is compiled for 32)"
        if why is not None:
            raise RuntimeError(f"pi0.5 megakernel refused: {why}")
        self.config = config
        self.device = device
        self._traces: Dict[Tuple[int, int, int], int] = {}
        self._closed = False
        self.cameras = config.num_cameras
        self.horizon = config.action_horizon
        self.suffix_rows = round_up(self.horizon)
        self.presets = PS.presets_for(self.cameras, self.suffix_rows)
        self.kv_dram = PS.kv_in_dram(self.presets) if kv_dram is None else bool(kv_dram)
        rows = max(p.cache_rows for p in self.presets)
        self.l1_bytes = PS.L1_OTHER + (0 if self.kv_dram else PS.kv_l1_bytes(rows))  # persistent L1 per core
        _LIVE.add(self)

        # Default flow-matching noise, drawn once (as PI0ModelTTNN draws its x_t) for a reproducible policy.
        self.default_noise = torch.randn(1, self.horizon, config.action_dim)

        cw = weight_loader.categorized_weights
        exp_params = expert_params(
            cw["action_expert"],
            cw["pi0_projections"],
            eps=config.expert_config.rms_norm_eps,
            num_steps=config.num_denoising_steps,
        )
        pre_params = prefix_params(cw)
        embed = cw["vlm_language"].get("model.embed_tokens.weight")
        if embed is None:
            embed = cw["vlm_language"]["lm_head.weight"]  # tied embeddings
        # The prefix engine gathers the prompt rows from the bf16 ROW_MAJOR table (DRAM).
        self.embed_tokens = ttnn.from_torch(embed, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

        # Expert RoPE table (split-half layout), read back once: the host writes the action tokens' rows per prompt.
        cos, sin = precompute_freqs_cis_meta_format(config.expert_config.head_dim, config.max_seq_len, device)
        self._rope = (ttnn.to_torch(cos).float(), ttnn.to_torch(sin).float())
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        # Device state shared by the presets, at the largest size any of them needs: the inputs below and the K / V
        # caches stay at fixed addresses, so a trace replay only needs their contents rewritten. Every L1 buffer is
        # allocated here, before any program is compiled or captured.
        l1, dram = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        self._set_horizon(self.horizon)
        cache_len = max(max(self.plans[p.key]["cache_len"], p.cache_rows) for p in self.presets)
        cache_shape = [1, config.expert_config.num_kv_heads, cache_len, config.expert_config.head_dim]
        s = self.suffix_rows
        self.noise = ttnn.from_torch(
            torch.zeros(1, s, config.action_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=l1,
        )
        self.kv_caches: List[Tuple[ttnn.Tensor, ttnn.Tensor]] = [
            tuple(
                ttnn.zeros(
                    cache_shape,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=dram if self.kv_dram else l1,
                )
                for _ in range(2)
            )
            for _ in range(config.expert_config.depth)
        ]
        self.tables = tuple(
            ttnn.from_torch(
                torch.zeros(1, 1, s, 256), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=l1
            )
            for _ in range(4)
        )  # cosq, sinq, cosk, sink
        self.out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, s, config.action_dim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, l1
        )
        self.mask_width = max(p.prefix_keys for p in self.presets) + s
        self.exp_mask = ttnn.from_torch(
            torch.zeros(1, 1, 32, self.mask_width),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=dram,
        )
        self.alloc_ps = PS.alloc_pshape(self.presets)
        self.prefix_tensors = PrefixTensors(device, pre_params, self.alloc_ps, embed=self.embed_tokens)
        experts = {}
        for p in self.presets:  # the weight arenas do not depend on the attention chunking: one upload
            first = next(iter(experts.values()), None)
            if first is not None and G.plan_banks(p.shape) != first.plan:
                raise AssertionError(f"preset {p.key}: the expert's DRAM bank plan differs from {first.shape.name}'s")
            experts[p.key] = ExpertMegakernel(
                device,
                exp_params,
                p.shape,
                arenas=None if first is None else (first.w8, first.w16),
                kv_dram=self.kv_dram,
            )
        self.programs: Dict[Tuple[int, int, int], _Programs] = {
            p.key: _Programs(
                p,
                [
                    PrefixEngineProgram(device, experts[p.key], self.prefix_tensors, p.pshape, VISION_OPS, group=g)
                    for g in P.vision_groups(p.pshape.img)
                ],
                PrefixEngineProgram(device, experts[p.key], self.prefix_tensors, p.pshape, PREFIX_OPS),
                experts[p.key],
            )
            for p in self.presets
        }
        self.kernel_digest = kernel_digest()

        self._inputs_key = None  # (preset, prefix validity) the attention inputs were last written for
        self._l1_at_capture = None

    def _set_horizon(self, horizon: int) -> None:
        """The host-side state of the action horizon: the K / V cache plan per preset (the key row masks the action
        rows >= H; the device programs depend on the suffix bucket only)."""
        if round_up(horizon) != self.suffix_rows:
            raise ValueError(f"action_horizon={horizon} is outside the model's {self.suffix_rows}-row suffix bucket")
        self.horizon = horizon
        self.plans = {p.key: kv_cache_plan(p.prefix_len, horizon) for p in self.presets}
        self._inputs_key = None  # the expert key row masks the action rows >= H: rewrite it on the next request
        self._inputs_key = None

    # ------------------------------------------------------------------ refusals
    @staticmethod
    def device_refusal(device) -> Optional[str]:
        """Why this device cannot run the megakernel (None = it can)."""
        try:
            n = int(device.get_num_devices())
        except AttributeError:
            n = 1
        if n != 1:
            return f"single chip only (the device has {n} chips)"
        if not ttnn.device.is_blackhole(device):
            return "Blackhole only"
        grid = device.compute_with_storage_grid_size()
        if grid.x < G.GRID[0] or grid.y < G.GRID[1]:
            return f"the programs run on an {G.GRID[0]} x {G.GRID[1]} core grid; the device has {grid.x} x {grid.y}"
        worker = sum(
            int(ttnn.get_memory_view(device, bt).total_bytes_per_bank)
            for bt in (ttnn.BufferType.L1, ttnn.BufferType.L1_SMALL)
        )
        if worker > MEGAKERNEL_WORKER_L1_SIZE:
            return (
                f"the device was opened without the 64 KiB worker-L1 cut (worker L1 {worker} B per bank > "
                f"{MEGAKERNEL_WORKER_L1_SIZE}); open it with worker_l1_size={MEGAKERNEL_WORKER_L1_SIZE} "
                "(PI05_DEVICE_PARAMS)"
            )
        return None

    def l1_signature(self) -> Tuple[int, int, int]:
        """L1 allocator state. A trace replay does not re-validate the programs' circular buffers against L1 buffers
        allocated after the capture, so ``sample_actions`` refuses to replay when this changed."""
        mv = ttnn.get_memory_view(self.device, ttnn.BufferType.L1)
        return (
            int(mv.total_bytes_allocated_per_bank),
            int(mv.largest_contiguous_bytes_free_per_bank),
            len(mv.block_table),
        )

    # ------------------------------------------------------------------ inputs
    def preset_for(self, n_tokens: int, prompt_bucket: Optional[int] = None) -> PS.Preset:
        """The preset of a request with ``n_tokens`` real prompt tokens: the smallest prompt bucket that holds them,
        or ``prompt_bucket`` when given (the model's cameras and suffix bucket)."""
        lens = " / ".join(str(p.prompt_len) for p in self.presets)
        for p in self.presets:
            if prompt_bucket is None and p.prompt_len >= n_tokens:
                return p
            if p.prompt_len == prompt_bucket:
                if n_tokens > prompt_bucket:
                    raise RuntimeError(
                        f"pi0.5 megakernel: prompt_bucket={prompt_bucket} cannot hold {n_tokens} real tokens"
                    )
                return p
        if prompt_bucket is not None:
            raise RuntimeError(f"pi0.5 megakernel: prompt_bucket={prompt_bucket} (compiled: {lens})")
        raise RuntimeError(
            f"pi0.5 megakernel: the prompt has {n_tokens} real tokens; the prompt buckets hold at most "
            f"{self.presets[-1].prompt_len} ({lens})"
        )

    def host_inputs(
        self,
        images: List[torch.Tensor],
        img_masks: Optional[List[torch.Tensor]],
        lang_tokens: torch.Tensor,
        lang_masks: Optional[torch.Tensor],
        noise: Optional[torch.Tensor],
        prompt_bucket: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Request -> its preset and the host tensors of the device inputs (checked against the compiled presets)."""
        to_torch = lambda t: ttnn.to_torch(t) if isinstance(t, ttnn.Tensor) else torch.as_tensor(t)
        images = [to_torch(img) for img in images]
        if img_masks is not None and not all(bool(to_torch(m).bool().all()) for m in img_masks):
            raise RuntimeError(
                "pi0.5 megakernel: every camera must be valid (img_masks all True); drop the masked image slots and "
                "pass only the real cameras (a model built with num_cameras = that count): openpi's masked slots are "
                "padding keys that do not advance the positions, so this is the same computation"
            )
        if len(images) != self.cameras:
            raise RuntimeError(f"pi0.5 megakernel serves {self.cameras} cameras (got {len(images)})")
        tokens = to_torch(lang_tokens)
        if tokens.dim() > 1 and tokens.shape[0] != 1:
            raise RuntimeError(f"pi0.5 megakernel serves batch 1 (got batch {tokens.shape[0]})")
        tokens = tokens.reshape(1, -1)
        lmask = (tokens != 0) if lang_masks is None else to_torch(lang_masks).reshape(1, -1).bool()
        if lmask.shape != tokens.shape:
            raise ValueError(f"lang_masks {tuple(lmask.shape)} != lang_tokens {tuple(tokens.shape)}")
        n_lang = int(lmask.sum())
        if not bool(lmask[0, :n_lang].all()):
            # the prefix positions 0..P-1 equal openpi's cumsum(valid) - 1 only for right padding
            raise ValueError("lang_masks must mark a right-padded prompt (real tokens first, then padding)")
        preset = self.preset_for(n_lang, prompt_bucket)
        # into the bucket: the tail beyond it is padding (right-padded), a shorter prompt gets masked <pad> = 0 ids
        pad = preset.prompt_len - tokens.shape[1]
        tokens = torch.nn.functional.pad(tokens, (0, pad)) if pad > 0 else tokens[:, : preset.prompt_len]
        lmask = torch.nn.functional.pad(lmask, (0, pad)) if pad > 0 else lmask[:, : preset.prompt_len]
        pixels = torch.cat([img.reshape(1, *img.shape[-3:]).float() for img in images], dim=0)
        if tuple(pixels.shape[-2:]) != (224, 224):
            raise RuntimeError(f"pi0.5 megakernel serves 224 x 224 images (got {tuple(pixels.shape[-2:])})")
        noise = self.default_noise if noise is None else to_torch(noise)
        noise = noise.reshape(1, self.horizon, self.config.action_dim).float()
        valid = prefix_valid_mask(self.cameras, lmask)
        return {
            "preset": preset,
            "im2col": im2col_patches(pixels, self.config.siglip_config.patch_size, pad_to=PATCH_FEATURES),
            "tokens": tokens,
            "valid": valid,
            "noise": pad_rows(noise, self.suffix_rows),
        }

    def write_inputs(self, host: Dict[str, torch.Tensor]) -> None:
        """Host -> the fixed device buffers. The masks and RoPE rows only when the preset or the prompt validity
        changed. A buffer wider than the preset needs gets its unused pages padded (never read)."""
        self._check_open()
        t = self.prefix_tensors
        preset = host["preset"]

        def write(x, dtype, layout, dev):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(x, dtype=dtype, layout=layout), dev)

        def pad_last(x, n, value):
            return torch.nn.functional.pad(x, (0, n - x.shape[-1]), value=value)

        im2col = host["im2col"].reshape(self.cameras * 256, -1)
        im2col = torch.nn.functional.pad(im2col, (0, 0, 0, t.im2col.shape[0] - im2col.shape[0]))  # unread rows
        write(im2col, ttnn.bfloat16, ttnn.TILE_LAYOUT, t.im2col)
        write(pad_last(host["tokens"].to(torch.int32), t.ps.ntok, 0), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, t.tokens)
        write(host["noise"], ttnn.bfloat16, ttnn.TILE_LAYOUT, self.noise)
        valid = host["valid"]
        key = (preset.key, valid.numpy().tobytes())
        if key != self._inputs_key:
            vmask = vlm_key_mask(valid.reshape(-1), preset.pshape)
            write(pad_last(vmask, t.ps.ptv * 32, MASK_NEG), ttnn.bfloat16, ttnn.TILE_LAYOUT, t.vmask)
            cos, sin = self._rope
            plan = self.plans[preset.key]
            att = attention_inputs(
                valid,
                plan,
                cos,
                sin,
                1.0 / float(self.config.expert_config.head_dim) ** 0.5,
                prefix_keys=preset.prefix_keys,
            )
            exp_mask = pad_last(att["exp_mask"], self.mask_width, MASK_NEG)
            write(exp_mask, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.exp_mask)
            for name, dev in zip(("cosq", "sinq", "cosk", "sink"), self.tables):
                write(att[name], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
            self._inputs_key = key

    # ------------------------------------------------------------------ device
    def forward_device(self, key: Tuple[int, int, int]) -> ttnn.Tensor:
        """The three device ops of preset ``key`` on the current inputs; returns x_0 ``[1, round_up(H), 32]`` bf16
        (L1). This is what a trace captures, so it does not touch the host."""
        pr = self.programs[key]
        args = (self.kv_caches, self.exp_mask, self.tables, self.noise, self.out)
        for v in pr.visions:
            v.run(*args)
        pr.prefix.run(*args)
        pr.expert.run(*args)
        return self.out

    def capture(self, key: Tuple[int, int, int]) -> None:
        """Compile (or load from the kernel cache) and capture the trace of preset ``key``: one eager pass, then the
        three ops in one trace."""
        self._check_open()
        if key in self._traces:
            return
        self.forward_device(key)  # compile + program cache, outside the trace
        ttnn.synchronize_device(self.device)
        trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        try:
            self.forward_device(key)
        except BaseException:
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.release_trace(self.device, trace_id)
            raise
        ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        sig = self.l1_signature()
        if self._l1_at_capture is not None and sig != self._l1_at_capture:
            ttnn.release_trace(self.device, trace_id)
            raise RuntimeError(f"pi0.5 megakernel: L1 allocation state changed between captures ({sig})")
        self._traces[key] = trace_id
        self._l1_at_capture = sig

    def warmup(self, prompt_lens: Optional[List[int]] = None) -> None:
        """Compile and capture the presets of ``prompt_lens`` (default: every preset of the model) ahead of the first
        requests: every preset's programs first (so no program is built while a trace exists), then the traces. The
        programs run on the current buffer contents; the outputs are discarded."""
        keys = [p.key for p in self.presets if prompt_lens is None or p.prompt_len in prompt_lens]
        for key in keys:
            if key not in self._traces:
                self.forward_device(key)
        ttnn.synchronize_device(self.device)
        for key in keys:
            self.capture(key)

    def replay(self, key: Tuple[int, int, int]) -> ttnn.Tensor:
        """Replay the captured trace of preset ``key`` (blocking); returns the output buffer."""
        self._check_open()
        now = self.l1_signature()
        if now != self._l1_at_capture:
            raise RuntimeError(
                f"pi0.5 megakernel: L1 allocation state changed since the trace capture ({self._l1_at_capture} -> "
                f"{now}: allocated bytes / largest free block / blocks per bank); an L1 buffer allocated after the "
                "capture can overlap the programs' circular buffers during a replay. Free it, or call release_trace()."
            )
        ttnn.execute_trace(self.device, self._traces[key], cq_id=0, blocking=True)
        return self.out

    def release_trace(self) -> None:
        for trace_id in self._traces.values():
            ttnn.release_trace(self.device, trace_id)
        self._traces = {}
        self._l1_at_capture = None

    def close(self) -> None:
        """Release this model's traces and deallocate its device tensors (L1 first, then DRAM), so a closed model
        frees its memory even while referenced; later calls raise. Idempotent. ``with`` runs it."""
        if self._closed:
            return
        self.release_trace()
        tensors = _device_tensors(self)
        in_l1 = lambda t: t.memory_config().buffer_type == ttnn.BufferType.L1
        for t in sorted(tensors, key=lambda t: not in_l1(t)):
            ttnn.deallocate(t)
        self._closed = True
        _LIVE.discard(self)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("pi0.5 megakernel: the model was closed (close() released its traces and memory)")

    def __enter__(self) -> "PI05MegakernelTTNN":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def sample_actions(
        self,
        images: List[torch.Tensor],
        img_masks: Optional[List[torch.Tensor]],
        lang_tokens: torch.Tensor,
        lang_masks: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        prompt_bucket: Optional[int] = None,
    ) -> torch.Tensor:
        """Actions ``[1, action_horizon, 32]`` float32 (normalised, as the policy outputs them).

        ``images``: ``num_cameras`` x ``[1, 3, 224, 224]`` in [-1, 1]. ``lang_tokens``: ``[1, L]`` ids, right-padded, with
        at most 224 real tokens; ``lang_masks`` marks the real tokens (None -> ``tokens != 0``). ``state`` is
        unused: pi0.5 carries the state in the prompt. ``noise``: ``[1, H, 32]`` (None -> the model's default).
        ``prompt_bucket``: run in this prompt bucket instead of the smallest one that holds the prompt (the buffer is
        truncated, its tail must be padding, or padded with masked ``<pad>`` ids).
        """
        host = self.host_inputs(images, img_masks, lang_tokens, lang_masks, noise, prompt_bucket)
        self.write_inputs(host)
        key = host["preset"].key
        self.capture(key)
        return unpad_rows(ttnn.to_torch(self.replay(key)).float(), self.horizon)
