# π0.5 Model for Tenstorrent

π0.5 (Physical Intelligence 0.5) is a vision-language-action (VLA) model for
robotics that combines a vision encoder, language model, and action expert for
end-to-end robot control. This repository is a port of π0.5 to Tenstorrent
hardware via TTNN, derived from `lerobot/pi05_base`, running on a single
Blackhole p150a.

The port runs one **traced device graph** per shape (`PI0ModelTTNN.sample_actions_fused`). Since
2026-09-30 its default (`PI05_MEGAKERNEL=expert`) is **phase 1 of the megakernel**
([`docs/megakernel/`](docs/megakernel/DESIGN.md)):

| Part of one call | How it runs on the device | Ops per trace replay |
|---|---|---:|
| host im2col, SigLIP (both cameras in one batch), projector, language embedding, Gemma-2B VLM prefill writing 18 bf8 K/V caches | stock TT-NN ops, captured in the Metal trace (**not** a megakernel yet) | 891 |
| the whole flow-matching loop: action in-projection, 10 Euler steps × 18 Gemma-300M expert layers (adaRMS, GQA attention over the prefix K/V with the padding mask and the offset RoPE, GeGLU MLP, gated residuals), action out-projection | **ONE persistent `ttnn.generic_op`**: custom kernels (`tt/megakernel/kernels/mk_{brisc,ncrisc,trisc}.cpp`) on all 110 worker cores, in-kernel step and layer loops, weights streamed from per-core DRAM arenas (bfp8 matmuls at HiFi2, fp32 residual), activations in L1 | 1 |

A replay is 892 device ops; the previous path (`PI05_MEGAKERNEL=off`) is 2,551, of which 1,660 run the
expert loop. Phase 2 (SigLIP + VLM inside the same persistent program, one op per replay) is the next
stage and is not built yet. The megakernel needs the device opened with the 64 KiB worker-L1 cut
(`common/device_open.py`: `open_pi05_device` / `device_kwargs`; `worker_l1_size=1395712`); the model refuses a
device without it by name. It serves batch 1, one chip, bf8 K/V caches and 10 steps; any other
configuration requested together with it refuses at build / server start.

`PI05_MEGAKERNEL=off` selects the previous shipped path (stock TT-NN expert ops plus three custom
`generic_op` programs: fused expert attention `tt/ttnn_fused_attn.py`, adaRMS row rsqrt and GeGLU
`tt/ttnn_fused_norm.py`). It is kept **only as the comparator / oracle** for the megakernel's gates. On a
multi-chip mesh the unset default resolves to `off` (the megakernel is a single-p150a program).

Since 2026-09-29 the graph applies openpi's attention semantics: right-padded prompt tokens are
masked out of every query, and the action tokens are rotated at positions `n_valid_prefix + [0, H)`.
Before that fix, both the port and its torch reference attended the pad tokens and rotated the
action tokens at `[0, H)`, so the PCC tests could not see either bug. Details and raw results:
[`docs/FUSED_FIX_2026-09-29.md`](docs/FUSED_FIX_2026-09-29.md).

## Results

Measured 2026-09-30 on one Blackhole p150a (AICLK 1350 MHz) with tt-metal `main` @
[`668c2907575`](https://github.com/tenstorrent/tt-metal/commit/668c290757550588d0ce46b180c344a462a2aaf5)
(v0.79.0-dev20260914), batch 1, 10 flow-matching steps, branch `megakernel-2026-09-29` (kernel digest
`328761c8a1ce3fd9`). "default" = `PI05_MEGAKERNEL` unset (the phase-1 expert megakernel); "off" = the previous
shipped path, measured in the same session as the comparator. Raw files: `docs/megakernel/integrate_p1/results/`
and the phase-1 record in `docs/megakernel/JOURNAL.md`.

### Accuracy

| Check | default (phase-1 megakernel) | off (previous path) |
|---|---:|---:|
| vs the openpi GPU golden: `lerobot/pi05_libero`, 8 real LIBERO observations, prompt right-padded to 32 tokens, H = 10, PCC over the 7 action dims | mean **0.999884**, min **0.999778** | mean 0.999839, min 0.999712 |
| Expert vs an fp32 expert-loop oracle fed the device's own prefix K/V and noise, served shape, 22 seeds (prompts with 1-224 real tokens) | mean **0.99974**, min 0.99916; closer than off on **22 / 22** seeds | mean 0.99596, min 0.98575 |
| Whole call vs the fixed fp32 torch reference, served shape, same 22 seeds | mean 0.97535 (closer than off on 19 / 22) | mean 0.96921 |
| Fixed torch reference (fp32, CPU) vs the openpi golden, prompt padded to 32 / 224 tokens | 0.999995 / 0.999995 | |
| Ten trace replays; alternating prompts and a shape switch (20 calls vs fresh-model outputs); output buffer poisoned between calls | bit-identical | bit-identical |
| 20 consecutive processes × 31 calls (prompt lengths 1 / 128 / 224) | 20 / 20, no hang, one output digest | |

The whole-call fp32 comparison on random inputs is dominated by the prefix (SigLIP / VLM in bf16, bf8 K/V
caches), which the megakernel does not change: the fp32 oracle fed the device's own K/V isolates the expert,
and that is the phase-1 gate (DESIGN.md §7, amended 2026-09-30). `tests/pcc/test_pcc_pi05_fused.py -k base`
still fails its own 0.95 whole-call floor on seed 3 in **both** paths (default 0.94833, off 0.93811); this
was already true on `main`. The LIBERO golden, on real observations, is the meaningful whole-model check.

### Latency

Host wall time per call (upload + trace replay + readback), median of 60 calls; device time from the
profiler (median of 21 profiled replays):

| Shape | default: per call | default: replay | off: per call | off: replay |
|---|---:|---:|---:|---:|
| Served shape: 2 × 224² images, 224 tokens, H = 50 | **70.7 ms** | 69.5 ms | 84.3 ms | 82.8 ms |
| LIBERO shape: 2 × 224² images, 32 tokens, H = 10 | **65.8 ms** | 64.4 ms | 76.7 ms | 75.6 ms |

| Device time of the expert loop | default: ONE generic_op | off: 1,660 stock / custom ops |
|---|---:|---:|
| Served shape | **17.49 ms** | 30.31 ms |
| LIBERO shape | **16.07 ms** | 26.83 ms |

The prefix (891 traced stock TT-NN ops, identical in both paths) is the remaining ~52 ms and is phase 2's target.
Served over HTTP by `server/app.py` (served shape, 30 warm requests per server, arms alternated default / off /
default / off, `integrate_p1/results/S_*`): `timing_ms.inference` median **70.93 / 70.88 ms** default vs 84.04 / 84.10 ms
off; `timing_ms.total` 72.3 vs 85.5 ms. 50 actions / 70.7 ms = 707 actions/s.

### LIBERO closed loop

`lerobot/pi05_libero` @ `a217bfd3` with openpi's `pi05_libero` norm stats, libero_spatial
(10 tasks × official init states 0-9), openpi's evaluation loop (5 of each 10-action chunk
executed), served through openpi's websocket protocol by the default path (2026-09-30):

| Policy | Device | Success |
|---|---|---:|
| this port, default (phase-1 expert megakernel) | p150a | **99 / 100** (0 errors, 0 timeouts; t9/i4 hit the step cap) |
| same, paired subset (init states 0-4) | p150a | 49 / 50 |
| previous path (`off`, 2026-09-29) | p150a | 98 / 100 (paired 48 / 50) |
| openpi `PI0Pytorch`, same weights and client (init states 0-4) | RTX 5090 | 50 / 50 |

Server-side policy latency: median **66.4 ms** per call (p10 66.1, p90 66.7; 2,130 calls), vs 77.6 ms for the
previous path. The LIBERO server wrapper and the client are not part of this repository.

## What the fused graph does

With the default `PI05_MEGAKERNEL=expert`, everything from the VLM's last K/V-cache write to the output
actions is the one megakernel op (`tt/megakernel/`: `geometry.py` core map and shapes parsed from
`kernels/mk_defs.hpp`, `arena.py` per-core consumption-ordered weight streams, `host_model.py` the host
parameters and a CPU model of the kernel's decomposition, `program.py` the `ProgramDescriptor`). The list
below describes the traced stock-op part (the prefix) and, for `off`, the previous expert.

The knobs are read once when `PI0ModelTTNN` is built (`FusedConfig.from_env()` in
`common/fused_config.py`; `FusedConfig.resolved()` applies the measured defaults):

- persistent device inputs, a compile pass, then one `begin/end_trace_capture` of the whole graph;
  per call: `copy_host_to_device_tensor` + `execute_trace`;
- the attention inputs (VLM key mask, expert key row, expert RoPE rows) are persistent trace inputs,
  rewritten outside the capture only when the prompt's valid length changes; `lang_masks=None`
  means `tokens != 0` (PaliGemma `<pad>` = 0) and prompts must be right-padded;
- the VLM writes the expert's K/V caches directly (no per-layer-per-step prefix refill);
- the expert attention is ONE `generic_op` program per layer (`tt/ttnn_fused_attn.py`: RoPE, masked
  attention over the prefix and the suffix, per-request RoPE tables and mask rows); the two adaRMS
  norms per expert layer are folded into per-step weights (`tt/ttnn_fused_norm.py`);
- expert gated residuals as `dit_minimal_matmul_addcmul_fused` in bf16 and the Euler step folded into
  the output projection, with `MinimalMatmulConfig` blocks measured on the p150a;
- both cameras through SigLIP as one batch from a host im2col with a precomputed positional table and
  fused biases; expert qkv / up through a 1D-multicast matmul program with fp32 accumulation.

Knobs (the defaults are the validated recipe; each is described in the module docstring of
`common/fused_config.py`):

| Env | Default | Meaning |
|-----|---------|---------|
| `PI05_MEGAKERNEL` | `expert` | `expert` = the expert loop as ONE persistent generic_op (phase 1); `off` = the previous stock-op expert (comparator only); `whole` = phase 2, not built (refuses) |
| `PI05_TRACE` | `1` | capture the graph in one Metal trace and replay it; `0` runs the same graph eagerly (debug / A/B) |
| `PI05_TRACE_REGION_SIZE` | 160 MB | `trace_region_size` passed to `ttnn.open_device` |
| `PI05_EXPERT_ATTN` | `fused` | `fused` = the generic_op attention; `ttnn` = SDPA ops (one prompt length per batch) |
| `PI05_FUSED_RESIDUAL` | `bf16` | expert gated residuals: `bf16`, `mixed` (bf8 activation × bf16 weight), `legacy` (bf8 weights) |
| `PI05_DIT_BLOCKS` / `PI05_EULER_DIT_BLOCKS` | `1,8,4,1,4,0,2` / `1,8,1,1,1,0,2` | `M,K,N,subblock_h,subblock_w[,grid_x,grid_y]` tiles of the fused matmul+residual ops |
| `PI05_EXPERT_MM` | `mcast1d_fp32` | program of the expert qkv / up matmuls: `linear`, `mcast1d`, `mcast1d_fp32`, `minimal` |
| `PI05_MLP_CHUNK` | `256` | VLM MLP sequence chunk; `0` = unchunked |
| `PI05_SIGLIP_BATCHED` | `1` | both cameras through SigLIP as one batch |
| `PI05_SKIP_VLM_TAIL` | `1` | skip the unused tail of the last VLM layer and the final VLM norm (exact) |

The graph is shape-bound: the number of images, the token length (a multiple of 32) and the action
horizon are fixed when a trace is captured; each shape gets its own trace.

## Directory Structure

The package lives at `models/experimental/pi0_5/`, the same path as in a tt-metal tree and in the
Hugging Face package `changh95/pi05-base-p150` (`code/models/experimental/pi0_5`), so the repo root
can be put on `PYTHONPATH` directly (imports are `models.experimental.pi0_5.*`).

```
tt-pi-0.5/
├── models/experimental/pi0_5/
│   ├── common/
│   │   ├── configs.py              # Model configurations
│   │   ├── fused_config.py         # PI05_* knobs (FusedConfig, read once at build)
│   │   ├── device_open.py          # the one device-open helper (adds the megakernel's worker-L1 cut)
│   │   ├── fused_host.py           # Torch-side inputs of the graph (im2col, Euler fold, attention masks / RoPE rows)
│   │   ├── weight_loader.py        # Checkpoint loading (pi05_base / pi05_libero)
│   │   └── utils.py
│   ├── reference/                  # PyTorch reference (the PCC oracle; openpi mask + positions)
│   ├── tt/
│   │   ├── ttnn_pi0_model.py       # PI0ModelTTNN.sample_actions_fused
│   │   ├── ttnn_paligemma.py       # SigLIP + VLM + expert backbone
│   │   ├── ttnn_siglip.py, ttnn_gemma.py, ttnn_prefix.py, ttnn_suffix.py, ttnn_common.py
│   │   ├── ttnn_fused_attn.py      # generic_op expert attention
│   │   ├── ttnn_fused_norm.py      # adaRMS fold, row rsqrt, fused GeGLU
│   │   ├── ttnn_ccl.py             # mesh / tensor-parallel helpers (not validated after the fix)
│   │   ├── kernels/                # fused_attn, geglu_rc, row_rsqrt (reader / compute / writer; the off path)
│   │   └── megakernel/             # phase-1 expert megakernel: geometry, arena, host_model, program, size_check,
│   │                               #   kernels/mk_{brisc,ncrisc,trisc}.cpp + mk_defs.hpp / mk_dm.hpp
│   ├── server/                     # app.py (FastAPI, served by tt-model-manager), smoke_test.py
│   └── tests/
│       ├── test_fused_host.py      # Torch-only proofs of the reformulations and the attention inputs (no device)
│       ├── test_server_masks.py    # CPU: every server path passes the request's own language mask
│       ├── megakernel/             # CPU tests of the megakernel host side + device tools (bring-up, oracle, soak)
│       ├── pcc/                    # test_pcc_pi05_fused.py, test_reference_vs_openpi.py, golden_openpi.py, LIBERO rollout
│       ├── perf/                   # test_perf_pi05_fused.py
│       ├── unit/                   # adaRMS, suffix, time embedding
│       └── demo/                   # sample-image extraction from the ALOHA / LIBERO datasets
├── docs/megakernel/             # DESIGN.md (gates), JOURNAL.md (every measurement, with files), PROFILE.md, results
├── docs/FUSED_FIX_2026-09-29.md    # the mask / RoPE fix, proof and raw results (docs/fused_fix_2026-09-29/)
└── docs/history/                   # earlier validation records (superseded)
```

## Quick Start

### 1. Environment

The port runs against a built tt-metal checkout at `668c2907575` (tenstorrent/tt-metal `main`,
v0.79.0-dev20260914). Its `generic_op` kernels use that tree's compute API and do not build on
the older `changh95/pi05` fork. Put this repo's root on `PYTHONPATH` in front of the tree:

```bash
export TT_METAL_HOME=/path/to/tt-metal           # @ 668c2907575, built
export PYTHONPATH=$(pwd):$TT_METAL_HOME:$TT_METAL_HOME/ttnn
export ARCH_NAME=blackhole
source $TT_METAL_HOME/python_env/bin/activate
export PI0_DEVICE_ID=0                            # optional
```

### 2. Weights

```bash
huggingface-cli download lerobot/pi05_base        # served checkpoint; resolved from the HF cache
huggingface-cli download lerobot/pi05_libero      # LIBERO fine-tune (the golden test)
```

The device tests also accept `PI05_WEIGHTS_DIR=<dir with model.safetensors + config.json>`.
`weights/` is git-ignored.

## Running Tests

```bash
# Host only (torch, no device, ~2 s): the reformulations, the knob plumbing, and the attention inputs
# vs openpi's mask / positions (fp64 emulation of the kernel math, batch 2 with two prompt lengths)
python models/experimental/pi0_5/tests/test_fused_host.py

# Host only: the megakernel's host side (knob / default / refusals, core map and sync-word invariants,
# the kernel decomposition vs the reference loop) and the server's mask plumbing (needs fastapi)
pytest models/experimental/pi0_5/tests/megakernel/test_cpu_mk.py models/experimental/pi0_5/tests/test_server_masks.py

# CPU: fixed torch reference vs the openpi golden (~80 s)
python models/experimental/pi0_5/tests/pcc/test_reference_vs_openpi.py

# Device: fused traced graph vs the openpi golden (LIBERO shape) and vs the torch reference (served shape).
# One model per process; each also reports the traced latency.
pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s -k libero
pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s -k base

# Device: the same tests on the comparator path
PI05_MEGAKERNEL=off pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s -k libero

# Device: traced timing (min / median / max over --runs)
python models/experimental/pi0_5/tests/perf/test_perf_pi05_fused.py --runs 10
```

The golden file (`PI05_OPENPI_GOLDEN`, 8 real LIBERO observations with fixed noise, produced by openpi on
a GPU) is not in this repository; `tests/pcc/golden_openpi.py` documents its format.

## Serving

`models/experimental/pi0_5/server/app.py` is the HTTP server packaged as the tt-model-manager container
[`changh95/pi05-base-p150`](https://huggingface.co/changh95/pi05-base-p150):

```bash
tt-model pull  changh95/pi05-base-p150 --with-weights
tt-model serve changh95/pi05-base-p150
python models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000
```

## Sample images

```bash
python -m pip install "imageio[pyav]"
python models/experimental/pi0_5/tests/demo/extract_aloha_samples.py    # ALOHA (MuJoCo) samples
python models/experimental/pi0_5/tests/demo/extract_libero_samples.py   # LIBERO samples
```

## Troubleshooting

- **`Checkpoint not found` / `No model.safetensors found`**: download `lerobot/pi05_base` into the HF
  cache or point `PI05_WEIGHTS_DIR` at a directory with `model.safetensors` + `config.json`.
- **`lang_masks must mark a right-padded prompt`**: the prefix RoPE uses positions `0..P-1`, which equal openpi's
  `cumsum(valid) - 1` only for right-padded prompts; right-pad the tokens.
- **`'mul_init' ... was not declared` while compiling `fused_attn/compute.cpp`**: the tt-metal tree is too
  old; use `668c2907575` or a tree re-validated with the tests above.
- **`PI05_MEGAKERNEL=expert refused: the device was opened without the 64 KiB worker-L1 cut`**: open the device
  with `common/device_open.py` (`open_pi05_device(fused)` or `ttnn.open_device(**device_kwargs(fused))`), or set
  `PI05_MEGAKERNEL=off` for the comparator path. Other `refused:` messages name the knob the kernels do not compile
  (`PI05_KV_DTYPE=bf16`, `PI05_NUM_STEPS` != 10, batch > 1, a mesh).
- **`Statically allocated circular buffers ... clash with L1 buffers`**: seen with `PI05_DIT_BLOCKS=op`;
  keep the default blocks.

## Model Specifications

| Component | Details |
|-----------|---------|
| Vision Encoder | SigLIP (27 transformer blocks, 1152 hidden dim) |
| VLM Backbone | Gemma 2B (18 transformer blocks) |
| Action Expert | Gemma 300M (18 transformer blocks, adaRMS) |
| Image Size | 224×224 |
| Action Dimension | 32 |
| Action Horizon | 50 (served), 10 (LIBERO) |
| Denoising Steps | 10 (flow matching, Euler) |
| HF Checkpoint | `lerobot/pi05_base` |

## Comparison with an RTX 5090 (same host)

Action chunk 50×32, 2×224² + 224 tokens, 10 denoising steps; ratio = p150a ms / GPU ms. The GPU rows
were measured on 2026-09-14 with this repo's torch reference as it was then (before the mask / RoPE
fix, which changes the attention mask and positions; the tensor shapes are unchanged); they were not
re-measured. The p150a row is the current served number.

| setting | ms | vs p150a |
|---|---:|---|
| p150a, default path with the phase-1 expert megakernel (served `timing_ms.inference`, 2026-09-30) | 70.9 | — |
| RTX 5090 fp32 strict | 144.1 | p150a 2.03× faster |
| RTX 5090 bf16 autocast | 121.5 | p150a 1.71× faster |
| RTX 5090 fp16 autocast | 123.7 | p150a 1.74× faster |
| RTX 5090 bf16 weights resident (eager) | 99.9 | p150a 1.41× faster |
| RTX 5090 bf16 weights + whole-request `torch.compile` | 46.6 | GPU 1.52× |

Methodology: same host, the torch reference (same weights and preprocessing as the served path) run
eagerly in PyTorch 2.11 cu128 (fp32 weights + `torch.autocast` unless stated; no TensorRT), batch 1,
medians of 50 iterations after warm-up, H2D/D2H included. p150a power was not measured, so no
efficiency comparison is made. The p150a is faster than every eager GPU row; the GPU needs resident bf16 weights and a compiled whole-request graph to be faster (1.52×). Full table: [`GPU_COMPARISON.md`](GPU_COMPARISON.md).
