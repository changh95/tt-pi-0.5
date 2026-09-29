# π0.5 Model for Tenstorrent

π0.5 (Physical Intelligence 0.5) is a vision-language-action (VLA) model for
robotics that combines a vision encoder, language model, and action expert for
end-to-end robot control. This repository is a port of π0.5 to Tenstorrent
hardware via TTNN, derived from `lerobot/pi05_base`, running on a single
Blackhole p150a.

The port runs one **fused / traced device graph**: host im2col → SigLIP (both cameras in one
batch) → Gemma-2B VLM prefill writing the expert's KV caches → 10 flow-matching steps of the
Gemma-300M action expert, captured in one Metal trace and replayed per call
(`PI0ModelTTNN.sample_actions_fused`). The expert attention, the adaRMS folds and the GeGLU are
custom `ttnn.generic_op` programs (`tt/kernels/`). This is the only inference path.

Since 2026-09-29 the graph applies openpi's attention semantics: right-padded prompt tokens are
masked out of every query, and the action tokens are rotated at positions `n_valid_prefix + [0, H)`.
Before that fix, both the port and its torch reference attended the pad tokens and rotated the
action tokens at `[0, H)`, so the PCC tests could not see either bug. Details and raw results:
[`docs/FUSED_FIX_2026-09-29.md`](docs/FUSED_FIX_2026-09-29.md).

## Results

Measured 2026-09-29 on one Blackhole p150a (AICLK 1350 MHz) with tt-metal `main` @
[`668c2907575`](https://github.com/tenstorrent/tt-metal/commit/668c290757550588d0ce46b180c344a462a2aaf5)
(v0.79.0-dev20260914), batch 1, 10 flow-matching steps.

### Accuracy

| Check | Result |
|---|---:|
| Fused traced graph vs the openpi GPU golden: `lerobot/pi05_libero`, 8 real LIBERO observations, prompt right-padded to 32 tokens, H = 10, PCC over the 7 action dims | mean **0.99984**, min **0.99971** |
| Same golden, the earlier eager path with a Python workaround for the mask (for comparison) | mean 0.9828, min 0.9037 |
| Fixed torch reference (fp32, CPU) vs the openpi golden, prompt padded to 32 / 224 tokens | 0.999995 / 0.999995 |
| Fused graph vs the fixed torch reference, served shape (`pi05_base`, 2 × 224², 224 tokens, H = 50), random inputs, prompts with 12-224 real tokens | 0.938-0.998 (mean 0.980; the spread follows the random seed, not the padding) |
| Mask is exact: random ids in the padded slots under the same mask | bit-identical output |
| Ten trace replays; alternating prompts; switching shapes | bit-identical |

The base-shape PCC compares bf16 device arithmetic with fp32 torch on random inputs, which has
always spread this widely (the pre-fix card reported a minimum of 0.8865 over 16 random
observations). The LIBERO golden, on real observations, is the meaningful accuracy check.

### Latency

Host wall time per 50-action chunk (upload + trace replay + readback), median of 60 calls:

| Shape | Per call | Trace replay only |
|---|---:|---:|
| Served shape: 2 × 224² images, 224 tokens, H = 50 | **84.2 ms** | 82.8 ms |
| LIBERO shape: 2 × 224² images, 32 tokens, H = 10 | **76.9 ms** | 75.6 ms |
| Previous release (fused graph before the fix, on the `changh95/pi05` fork @ `4c9fbfcceb9`), served shape | 125.9 ms | 124.8 ms |

Served over HTTP by the container `changh95/pi05-base-p150` (100 warm requests, served shape):
`timing_ms.inference` median **84.0 ms** (p90 84.2), `timing_ms.total` 85.3 ms, client wall 86.9 ms.

The fix costs 1.2 ms at the served shape (the VLM mask is read by 18 SDPAs and the expert
attention adds a mask to every key tile). 50 actions / 84.2 ms = 594 actions/s.

### LIBERO closed loop

`lerobot/pi05_libero` @ `a217bfd3` with openpi's `pi05_libero` norm stats, libero_spatial
(10 tasks × official init states 0-9), openpi's evaluation loop (5 of each 10-action chunk
executed), served by this fused graph through openpi's websocket protocol:

| Policy | Device | Success |
|---|---|---:|
| this port, fused traced graph | p150a | **98 / 100** (0 errors, 0 timeouts; t3/i2 and t4/i0 hit the step cap) |
| same, paired subset (init states 0-4) | p150a | 48 / 50 |
| openpi `PI0Pytorch`, same weights and client (init states 0-4) | RTX 5090 | 50 / 50 |

Server-side policy latency: median **77.6 ms** per call (p10 77.3, p90 78.0; 2,180 calls). The
LIBERO server wrapper and the client are not part of this repository.

## What the fused graph does

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
│   │   └── kernels/                # fused_attn, geglu_rc, row_rsqrt (reader / compute / writer)
│   ├── server/                     # app.py (FastAPI, served by tt-model-manager), smoke_test.py
│   └── tests/
│       ├── test_fused_host.py      # Torch-only proofs of the reformulations and the attention inputs (no device)
│       ├── pcc/                    # test_pcc_pi05_fused.py, test_reference_vs_openpi.py, golden_openpi.py, LIBERO rollout
│       ├── perf/                   # test_perf_pi05_fused.py
│       ├── unit/                   # adaRMS, suffix, time embedding
│       └── demo/                   # sample-image extraction from the ALOHA / LIBERO datasets
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

# CPU: fixed torch reference vs the openpi golden (~80 s)
python models/experimental/pi0_5/tests/pcc/test_reference_vs_openpi.py

# Device: fused traced graph vs the openpi golden (LIBERO shape) and vs the torch reference (served shape).
# One model per process; each also reports the traced latency.
pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s -k libero
pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s -k base

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
| p150a, bf16 fused trace (served `timing_ms.inference`, 2026-09-29) | 84.0 | — |
| RTX 5090 fp32 strict | 144.1 | p150a 1.72× faster |
| RTX 5090 bf16 autocast | 121.5 | p150a 1.45× faster |
| RTX 5090 fp16 autocast | 123.7 | p150a 1.47× faster |
| RTX 5090 bf16 weights resident (eager) | 99.9 | p150a 1.19× faster |
| RTX 5090 bf16 weights + whole-request `torch.compile` | 46.6 | GPU 1.80× |

Methodology: same host, the torch reference (same weights and preprocessing as the served path) run
eagerly in PyTorch 2.11 cu128 (fp32 weights + `torch.autocast` unless stated; no TensorRT), batch 1,
medians of 50 iterations after warm-up, H2D/D2H included. p150a power was not measured, so no
efficiency comparison is made. The p150a is faster than every eager GPU row; the GPU needs resident bf16 weights and a compiled whole-request graph to be faster (1.80×). Full table: [`GPU_COMPARISON.md`](GPU_COMPARISON.md).
