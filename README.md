# π0.5 Model for Tenstorrent

π0.5 (Physical Intelligence 0.5) is a vision-language-action (VLA) model for
robotics that combines a vision encoder, language model, and action expert for
end-to-end robot control. This repository is a port of π0.5 to Tenstorrent
hardware via TTNN, derived from `lerobot/pi05_base`.

Since 2026-09-13 the port runs a **fused / traced device graph by default**
(`TT_FUSED` unset or `1`): the whole SigLIP + VLM + 10-step action-expert graph is
captured in one Metal trace and replayed per request. `TT_FUSED=0` restores the
previous (legacy, untraced) path bit-for-bit. The hardware pass that measured it is
documented in [`DEVICE_VALIDATION.md`](DEVICE_VALIDATION.md).

## PCC Results

PCC (Pearson Correlation Coefficient) of the TTNN implementation against the
PyTorch reference, measured on a single Tenstorrent Blackhole p150a.

### Fused / traced path (default), served shape

Measured 2026-09-13 on a p150a with tt-metal `changh95/pi05` @ `4c9fbfcceb9`
(`DEVICE_VALIDATION.md` "Results (device, 2026-09-13)"). Shape: 2 x 224x224 images,
224 language tokens, 10 flow-matching steps, batch 1, action horizon 50.

| Metric | Legacy (`TT_FUSED=0`) | Fused / traced (default) |
|--------|-----------------------|--------------------------|
| PCC vs. torch reference, `test_pcc_pi05_fused.py` (observations A / B) | -- (see 32-token table below) | **0.9977 / 0.9987** |
| Per-step velocity PCC (worst of 10 steps) | -- | 0.9990 |
| PCC of the fused served actions vs. the legacy served actions (card example) | -- | 0.9993 (max abs 0.066) |
| e2e PCC on a 16-observation robustness set (mean; >= legacy on 15/16) | 0.9786 | **0.9853** |
| `test_perf_pi05_fused.py`, device time per 50-action chunk (min / median / max, 10 runs) | median 143.2-144.0 ms across 4 processes (min 142.1, max 144.6; in-process `sample_actions`) | **125.8 / 126.1 / 126.6 ms** |
| Served, 100 warm requests, `timing_ms.inference` median / min / max | 171.7 / 142.1 / 188.1 ms (drifts 142 -> 188 over the run; first-20 median 143.4) | **125.9 / 125.6 / 126.5 ms** (flat) |
| Determinism (repeat == repeat) | bit-identical | bit-identical (trace replay == replay, traced == eager) |
| Trace region used | -- | 26.6 MB of the 160 MB reserved |

Derived: 50 actions / 125.9 ms = 397 actions/sec at the served shape (fused), vs. 50 / 143.5 ms
= 348 actions/sec for the legacy path at the same shape (median 143.5).

Not re-measured for the fused path: the LIBERO closed-loop rollout (the simulator stack is not
installed on the validation host). The e2e PCC of a 10-step chunk varies with the observation on
both paths (the lowest of 16 synthetic observations scores 0.89-0.90 on either path; the fused
path is at or above legacy on 15 of the 16). The first `sample_actions_fused` call on a fresh kernel
cache JIT-compiles the fused kernels and captures the trace (8.4 s warm-up measured with the legacy
kernels already cached).

### Legacy path (`TT_FUSED=0`), 32-token port configuration

Numbers as originally published (source-built tt-metal v0.65.1rc17); re-checked on 2026-09-13 with
`TT_FUSED` unset on the tree above: `test_pcc_pi05_model.py` PCC 0.9928, `test_perf_pi05.py`
132.9 / 133.1 / 135.1 ms.

| Metric              | Value                 |
|---------------------|-----------------------|
| PCC (vs. reference) | **0.9921**            |
| Latency             | 132.7 ms / action batch |
| Throughput          | 376.7 actions/sec     |
| Denoising steps     | 10 (flow matching)    |
| Action horizon      | 50                    |
| Hardware            | Blackhole p150a (single chip) |
| Checkpoint          | `lerobot/pi05_base`   |

Optimization trajectory (PyTorch-reference PCC throughout):

| PCC    | Latency  | Throughput | Notes |
|--------|----------|------------|-------|
| 0.9977 | 183.4 ms | 272.7 a/s  | baseline, TTNN 0.67.4 wheel |
| 0.9967 | 169.2 ms | 295.5 a/s  | bfloat8_b SigLIP + pre-baked adaRMS |
| 0.9933 | 166.3 ms | 300.6 a/s  | source-built tt-metal v0.65.1rc17 |
| 0.9933 | 151.4 ms | 330.3 a/s  | pre-allocated KV cache (no first-call concat) |
| 0.9921 | 145.6 ms | 343.4 a/s  | KV cache in L1 (SDPA 75µs → 53µs) |
| 0.9921 | 144.6 ms | 345.5 a/s  | fused `rotary_embedding_to_cache` op |
| 0.9921 | 144.1 ms | 347.0 a/s  | precomputed per-step adarms_cond + cached suffix mask |
| **0.9921** | **132.7 ms** | **376.7 a/s** | precomputed per-(step,layer) adaRMS modulations in DRAM (32 tokens) |
| 0.9977 / 0.9987 | **125.9 ms** (served, 224 tokens) | 397 a/s (derived) | `TT_FUSED=1` (default): whole-graph Metal trace, backbone-owned expert KV cache, 64-row suffix, bf16 fused gated residuals + fused Euler step with measured `MinimalMatmulConfig` blocks, batched SigLIP from a host im2col, 1D-multicast expert qkv / up with fp32 accumulation |

Reproduce (from the repo root, see *Quick Start* for the environment):

```bash
# Fused / traced path (default): PCC vs torch on the served shape, then device timing
pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s
python models/experimental/pi0_5/tests/perf/test_perf_pi05_fused.py --runs 10 --skip-legacy

# Legacy path (32-token config): PCC (accuracy) and performance
TT_FUSED=0 python models/experimental/pi0_5/tests/pcc/test_pcc_pi05_model.py
TT_FUSED=0 python models/experimental/pi0_5/tests/perf/test_perf_pi05.py
```

## Fused / traced path (`TT_FUSED`)

All fused levers live behind one env knob, read once when `PI0ModelTTNN` is built
(`FusedConfig.from_env()` in `common/fused_config.py`). What the default path does:

- persistent device inputs, a compile pass, then one `begin/end_trace_capture` of the whole graph;
  per call: `copy_host_to_device_tensor` + `execute_trace` (`sample_actions_fused`);
- the VLM writes the expert's K/V caches directly (no per-layer-per-step prefix refill);
- 64-row suffix (no `slice(q)`), TILE-layout language embedding, mask-free prefix, cos/sin slice cache,
  last-VLM-layer tail skip -- all exact reformulations;
- expert gated residuals as `dit_minimal_matmul_addcmul_fused` in bf16 (o_proj / down_proj + residual
  in one launch) and the Euler step folded into the output projection, with `MinimalMatmulConfig`
  blocks measured on the p150a;
- GeGLU gate matmul with the fused GELU; both cameras through SigLIP as one batch from a host im2col
  with a precomputed positional table and fused biases;
- expert qkv / up through a 1D-multicast matmul program with fp32 accumulation.

`TT_FUSED=0` disables every one of them and runs the previously shipped code path bit-for-bit.

Sub-knobs (read only when the fused path is enabled; defaults are the validated recipe -- the full
description of each is the module docstring of `common/fused_config.py`, the measurements behind
each default are in `DEVICE_VALIDATION.md`):

| Env | Default | Meaning |
|-----|---------|---------|
| `PI05_TRACE` | `1` | capture the graph in one Metal trace and replay it; `0` runs the same fused graph eagerly (debug / A/B) |
| `PI05_TRACE_REGION_SIZE` | 160 MB | `trace_region_size` passed to `ttnn.open_device` (26.6 MB used) |
| `PI05_FUSED_RESIDUAL` | `bf16` | expert gated residuals: `bf16` (bf16 o_proj / down_proj / stream), `mixed` (bf8 activation x bf16 weight), `legacy` (bf8 weights, shipped numerics) |
| `PI05_DIT_BLOCKS` / `PI05_EULER_DIT_BLOCKS` | `1,8,4,1,4,0,2` / `1,8,1,1,1,0,2` | `M,K,N,subblock_h,subblock_w[,grid_x,grid_y]` tiles of the fused matmul+residual ops; `op` = the op's own defaults (clash with L1 on this model) |
| `PI05_EXPERT_MM` | `mcast1d_fp32` | program of the expert qkv / up matmuls: `linear` (auto), `mcast1d`, `mcast1d_fp32`, `minimal` (+ `PI05_EXPERT_MM_BLOCKS`) |
| `PI05_MLP_CHUNK` | `256` | VLM MLP sequence chunk; `0` = unchunked (slower and less accurate in the graph -- A/B only) |
| `PI05_VLM_DOWN_PC` / `PI05_VLM_GATEUP_PC` | `mcast2d` / `auto` | matmul programs of the unchunked VLM MLP (`PI05_MLP_CHUNK=0` only) |
| `PI05_VLM_ATTN_PC` / `PI05_SIGLIP_PC` | `auto` | explicit 2D-multicast programs (`mcast2d`, `mcast2d_fp32`) for the VLM attention / SigLIP linears; faster but rejected on accuracy, kept as knobs |
| `PI05_SIGLIP_BATCHED` | `1` | both cameras through SigLIP as one batch; `0` = one image at a time |
| `PI05_SKIP_VLM_TAIL` | `1` | skip the unused tail of the last VLM layer and the final VLM norm (exact) |
| `PI05_SDPA_VLM_CHUNKS` / `PI05_SDPA_EXPERT_CHUNKS` / `PI05_SDPA_SIGLIP_CHUNKS` | legacy config | `q,k` chunk sizes for the three SDPA sites on the full grid |

The fused graph is shape-bound: `PI05_NUM_IMAGES` (default 2) and `PI05_TOKEN_LEN` (default 224, a
multiple of 32) are fixed when the trace is captured. With `PI05_FUSED_RESIDUAL=bf16` / `mixed` the
expert weights are stored in bf16, so the legacy `sample_actions` / `sample_actions_traced` entry
points raise a `RuntimeError` on that model; use `PI05_FUSED_RESIDUAL=legacy` for an in-process
fused-vs-legacy comparison, or `TT_FUSED=0` for the true legacy baseline.

### Fused-path tests

```bash
# Host-only proofs of every exact reformulation + the knob plumbing (torch only, no device, ~2 s)
pytest models/experimental/pi0_5/tests/test_fused_host.py -q
# (also runs as a plain script: python models/experimental/pi0_5/tests/test_fused_host.py)

# Device: fused vs torch (and vs the legacy ttnn path when PI05_FUSED_RESIDUAL=legacy)
pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s
PI05_TRACE=0 PI05_FUSED_RESIDUAL=legacy PI05_SIGLIP_BATCHED=0 \
    pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s   # exact levers, eager

# Device: fused traced timing (min / median / max over --runs); --skip-legacy omits the
# in-process legacy timing (which needs PI05_FUSED_RESIDUAL=legacy)
python models/experimental/pi0_5/tests/perf/test_perf_pi05_fused.py --runs 10 --skip-legacy

# Legacy regression (the model's own tests, TT_FUSED=0)
TT_FUSED=0 pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_model.py \
    models/experimental/pi0_5/tests/pcc/test_pcc_pi05_multireplan.py \
    models/experimental/pi0_5/tests/pcc/test_determinism_pi05.py -v -s
```

The device tests take `PI05_WEIGHTS_DIR=<dir with model.safetensors + config.json>` (default: the
`lerobot/pi05_base` HF cache), `PI0_DEVICE_ID`, `PI05_NUM_IMAGES`, `PI05_TOKEN_LEN`.

## Directory Structure

The package lives at `models/experimental/pi0_5/`, the same path as in the tt-metal fork
(`changh95/pi05`) and in the Hugging Face package `changh95/pi05-base-p150` (`code/models/experimental/pi0_5`),
so the repo root can be put on `PYTHONPATH` directly (imports are `models.experimental.pi0_5.*`).

```
tt-pi-0.5/
├── models/experimental/pi0_5/
│   ├── common/                     # Shared configs and utilities
│   │   ├── configs.py              # Model configurations (GemmaConfig.use_adarms, etc.)
│   │   ├── fused_config.py         # TT_FUSED / PI05_* knobs (FusedConfig, read once at build)
│   │   ├── fused_host.py           # Torch reformulations used by the fused graph (im2col, Euler fold, ...)
│   │   ├── weight_loader.py        # Checkpoint loading (pi05_base)
│   │   └── utils.py                # Common utilities
│   ├── reference/                  # PyTorch reference implementation
│   │   ├── torch_pi0_model.py      # Main π0.5 model
│   │   ├── torch_paligemma.py      # PaliGemma backbone
│   │   ├── torch_siglip.py         # SigLIP vision tower
│   │   ├── torch_gemma.py          # Gemma attention/MLP (with adaRMS)
│   │   ├── torch_prefix.py         # Prefix embedding
│   │   ├── torch_suffix.py         # Suffix embedding
│   │   └── torch_denoise.py        # Flow-matching denoising logic
│   ├── tt/                         # TTNN implementation (legacy + fused paths)
│   │   ├── ttnn_pi0_model.py       # Main π0.5 model (TTNN): sample_actions / sample_actions_fused
│   │   ├── ttnn_paligemma.py       # PaliGemma backbone (TTNN)
│   │   ├── ttnn_siglip.py          # SigLIP vision tower (TTNN)
│   │   ├── ttnn_gemma.py           # Gemma attention/MLP + adaRMS (TTNN)
│   │   ├── ttnn_prefix.py          # Prefix embedding (TTNN)
│   │   ├── ttnn_suffix.py          # Suffix embedding (TTNN)
│   │   └── ttnn_common.py          # Common TTNN utilities
│   └── tests/
│       ├── test_fused_host.py      # Torch-only proofs for the fused graph (no device)
│       ├── pcc/                    # PCC (accuracy) tests, incl. test_pcc_pi05_model.py, test_pcc_pi05_fused.py
│       ├── perf/                   # Performance benchmarks, incl. test_perf_pi05.py, test_perf_pi05_fused.py
│       ├── unit/                   # Component unit tests (adaRMS, suffix, time embedding)
│       ├── demo/                   # Demo scripts with ALOHA (MuJoCo) / LIBERO datasets
│       └── download_pretrained_weights.py
├── DEVICE_VALIDATION.md            # Hardware pass of the fused path (plan, results, knobs, gates)
└── Dockerfile
```

## Quick Start

### 1. Environment Setup

This model targets Tenstorrent hardware and runs against a built `tt-metal`
checkout. Either put this repo's root on `PYTHONPATH` in front of the tree, or
copy `models/experimental/pi0_5/` into the tree at the same path.

```bash
# Set required environment variables for tt-metal
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$(pwd):$TT_METAL_HOME        # this repo's root first
export ARCH_NAME=blackhole          # or wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml  # wormhole only

# Activate virtual environment
source $TT_METAL_HOME/python_env/bin/activate

# (Optional) Select device
export PI0_DEVICE_ID=2
```

The numbers above were measured on the author's tt-metal fork `changh95/pi05` @ `4c9fbfcceb9`
(v0.72.0-dev: main @ 2026-06-03 + the fused `rotary_embedding_to_cache` op + the Blackhole SDPA
JIT header fix), which both paths use.

### 2. Download Pretrained Weights

π0.5 weights live on HuggingFace as `lerobot/pi05_base`. The tests pass
`lerobot/pi05_base` to the weight loader, which resolves it through the
HuggingFace cache (`snapshot_download`) when no local directory of that name exists,
so a plain `huggingface-cli download lerobot/pi05_base` is enough:

```bash
# Install huggingface CLI
pip install -U huggingface_hub

# Download into the HF cache (used automatically) ...
huggingface-cli download lerobot/pi05_base

# ... or into a local weights directory
huggingface-cli download lerobot/pi05_base \
    --local-dir models/experimental/pi0_5/weights/pi05_base
```

The fused device tests also accept `PI05_WEIGHTS_DIR=<dir>` pointing at a snapshot directory
with `model.safetensors` + `config.json`. `weights/` is git-ignored.

> Note: `tests/download_pretrained_weights.py` is a legacy helper for the
> original π0 Google-Drive checkpoint and is **not** used for π0.5.

## Running Tests

All commands run from the repo root with the environment above.

### PCC Tests (Accuracy Validation)

PCC (Pearson Correlation Coefficient) tests compare TTNN outputs against the
PyTorch reference.

**Fused / traced path (default):**

```bash
pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s
```

**Full π0.5 Model PCC Test (legacy path):**

```bash
TT_FUSED=0 pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_model.py -v -s
# or direct execution
TT_FUSED=0 python models/experimental/pi0_5/tests/pcc/test_pcc_pi05_model.py
```

**Component PCC Tests:**

```bash
# Run all component tests
python models/experimental/pi0_5/tests/pcc/run_all_pcc_tests.py

# Individual component tests
pytest models/experimental/pi0_5/tests/pcc/test_pcc_suffix.py -v
pytest models/experimental/pi0_5/tests/pcc/test_pcc_prefix.py -v
pytest models/experimental/pi0_5/tests/pcc/test_pcc_gemma.py -v
pytest models/experimental/pi0_5/tests/pcc/test_pcc_siglip.py -v
pytest models/experimental/pi0_5/tests/pcc/test_pcc_paligemma.py -v

# Determinism, multi-replan responsiveness, per-step velocity PCC (legacy API)
TT_FUSED=0 pytest models/experimental/pi0_5/tests/pcc/test_determinism_pi05.py \
    models/experimental/pi0_5/tests/pcc/test_pcc_pi05_multireplan.py \
    models/experimental/pi0_5/tests/pcc/test_pcc_pi05_per_step.py -v -s
```

### Performance Tests (Benchmarking)

```bash
# Fused / traced path: device time per 50-action chunk at the served shape
python models/experimental/pi0_5/tests/perf/test_perf_pi05_fused.py --runs 10 --skip-legacy

# Legacy π0.5 performance test (action throughput / latency, 32-token config)
TT_FUSED=0 python models/experimental/pi0_5/tests/perf/test_perf_pi05.py

# Legacy Metal Trace variant
TT_FUSED=0 python models/experimental/pi0_5/tests/perf/test_perf_pi05_trace.py

# Profiling helper
python models/experimental/pi0_5/tests/perf/profile_pi05.py
```

### Performance Test (end-to-end 2CQ + Trace, legacy)

```bash
TT_FUSED=0 pytest models/experimental/pi0_5/tests/perf/test_perf_e2e.py
```

Recommended invocation (Blackhole p300c, device 2, source-built tt-metal):

```bash
TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME/build_Release/libexec/tt-metalium \
PI0_DEVICE_ID=2 \
PYTHONPATH=$(pwd):$TT_METAL_HOME \
TT_METAL_HOME=$TT_METAL_HOME \
python models/experimental/pi0_5/tests/perf/test_perf_pi05_fused.py --runs 10 --skip-legacy
```

## Demo Scripts

Demo scripts visualize π0.5 inference on robotics datasets.

- **ALOHA sim** uses MuJoCo-based bimanual setups.
- **LIBERO** uses the standard LIBERO benchmark suite.

**Extract Sample Images (required first):**

```bash
# imageio[pyav] is needed to extract frames from the dataset videos
python -m pip install "imageio[pyav]"

# Extract ALOHA (MuJoCo) samples (downloads from HuggingFace)
python models/experimental/pi0_5/tests/demo/extract_aloha_samples.py

# Extract LIBERO samples (downloads from HuggingFace)
python models/experimental/pi0_5/tests/demo/extract_libero_samples.py
```

Output layout:

```
sample_images/
├── aloha_sim/
│   ├── sample_0_top.png
│   ├── sample_1_top.png
│   └── metadata.txt
└── libero/
    ├── sample_0_main.png
    ├── sample_0_wrist.png
    └── metadata.txt
```

**Run Demos:**

```bash
# ALOHA (MuJoCo) simulation demo
python models/experimental/pi0_5/tests/demo/run_aloha_sim_demo.py

# LIBERO demo
python models/experimental/pi0_5/tests/demo/run_libero_demo.py

# Visualize results
python models/experimental/pi0_5/tests/demo/visualize_demo.py
```

## Troubleshooting

### `Checkpoint not found` / `No model.safetensors found`

Make sure `lerobot/pi05_base` is in the HuggingFace cache, or that a local
`models/experimental/pi0_5/weights/pi05_base/` holds `model.safetensors` + `config.json`:

```bash
huggingface-cli download lerobot/pi05_base \
    --local-dir models/experimental/pi0_5/weights/pi05_base
```

### `PI0ModelTTNN.sample_actions is not available on a model built with TT_FUSED=1 ...`

The default `PI05_FUSED_RESIDUAL=bf16` stores the expert o_proj / down_proj in bf16, which the
legacy expert block cannot consume. Call `sample_actions_fused`, or build the model with
`TT_FUSED=0` (true legacy) / `PI05_FUSED_RESIDUAL=legacy` (fused levers with bf8 expert weights).

### `Statically allocated circular buffers ... clash with L1 buffers`

Seen with `PI05_DIT_BLOCKS=op` (the fused op's own 8x8x8 blocks). Keep the default measured blocks
(`1,8,4,1,4,0,2` / `1,8,1,1,1,0,2`).

## Model Specifications

| Component | Details |
|-----------|---------|
| Vision Encoder | SigLIP (27 transformer blocks, 1152 hidden dim) |
| VLM Backbone | Gemma 2B (18 transformer blocks, static RMSNorm) |
| Action Expert | Gemma 300M (18 transformer blocks, **adaRMS**) |
| Image Size | 224×224 |
| Action Dimension | 32 |
| Action Horizon | 50 |
| Denoising Steps | 10 (flow matching) |
| HF Checkpoint | `lerobot/pi05_base` |
