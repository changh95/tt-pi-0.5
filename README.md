# π0.5 Model for Tenstorrent

π0.5 (Physical Intelligence 0.5) is a vision-language-action (VLA) model for
robotics that combines a vision encoder, language model, and action expert for
end-to-end robot control. This repository is a port of π0.5 to Tenstorrent
hardware via TTNN, derived from `lerobot/pi05_base`.

## PCC Results

PCC (Pearson Correlation Coefficient) of the TTNN implementation against the
PyTorch reference, measured on a single Tenstorrent Blackhole p150a
(source-built tt-metal v0.65.1rc17).

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
| **0.9921** | **132.7 ms** | **376.7 a/s** | precomputed per-(step,layer) adaRMS modulations in DRAM |

Reproduce:

```bash
# PCC (accuracy)
python tests/pcc/test_pcc_pi05_model.py

# Performance (latency / throughput)
python tests/perf/test_perf_pi05.py
```

## Directory Structure

```
tt-pi-05/
├── common/                     # Shared configs and utilities
│   ├── configs.py              # Model configurations (GemmaConfig.use_adarms, etc.)
│   ├── weight_loader.py        # Checkpoint loading (pi05_base)
│   └── utils.py                # Common utilities
├── reference/                  # PyTorch reference implementation
│   ├── torch_pi0_model.py      # Main π0.5 model
│   ├── torch_paligemma.py      # PaliGemma backbone
│   ├── torch_siglip.py         # SigLIP vision tower
│   ├── torch_gemma.py          # Gemma attention/MLP (with adaRMS)
│   ├── torch_prefix.py         # Prefix embedding
│   ├── torch_suffix.py         # Suffix embedding
│   └── torch_denoise.py        # Flow-matching denoising logic
├── tt/                         # TTNN implementation
│   ├── ttnn_pi0_model.py       # Main π0.5 model (TTNN)
│   ├── ttnn_paligemma.py       # PaliGemma backbone (TTNN)
│   ├── ttnn_siglip.py          # SigLIP vision tower (TTNN)
│   ├── ttnn_gemma.py           # Gemma attention/MLP + adaRMS (TTNN)
│   ├── ttnn_prefix.py          # Prefix embedding (TTNN)
│   ├── ttnn_suffix.py          # Suffix embedding (TTNN)
│   └── ttnn_common.py          # Common TTNN utilities
├── tests/
│   ├── pcc/                    # PCC (accuracy) tests, incl. test_pcc_pi05_model.py
│   ├── perf/                   # Performance benchmarks, incl. test_perf_pi05.py
│   ├── demo/                   # Demo scripts with ALOHA (MuJoCo) / LIBERO datasets
│   └── download_pretrained_weights.py
└── weights/                    # Pretrained checkpoints (git-ignored)
    └── pi05_base/              # π0.5 base checkpoint (symlink or download)
```

## Quick Start

### 1. Environment Setup

This model targets Tenstorrent hardware and is designed to be dropped into a
`tt-metal` checkout at `models/experimental/pi0/`.

```bash
# Set required environment variables for tt-metal
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=blackhole          # or wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml  # wormhole only

# Activate virtual environment
source $TT_METAL_HOME/python_env/bin/activate

# (Optional) Select device
export PI0_DEVICE_ID=2
```

### 2. Download Pretrained Weights

π0.5 weights live on HuggingFace as `lerobot/pi05_base`.

```bash
# Install huggingface CLI
pip install -U huggingface_hub

# Download into the standard weights path
huggingface-cli download lerobot/pi05_base \
    --local-dir $TT_METAL_HOME/models/experimental/pi0/weights/pi05_base
```

Or, symlink an existing HF cache:

```bash
ln -s ~/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/<REVISION> \
      $TT_METAL_HOME/models/experimental/pi0/weights/pi05_base
```

Verify the structure:

```
$TT_METAL_HOME/models/experimental/pi0/weights/
└── pi05_base/
    ├── model.safetensors
    └── config.json
```

> Note: `tests/download_pretrained_weights.py` is a legacy helper for the
> original π0 Google-Drive checkpoint and is **not** used for π0.5.

## Running Tests

### PCC Tests (Accuracy Validation)

PCC (Pearson Correlation Coefficient) tests compare TTNN outputs against the
PyTorch reference.

**Full π0.5 Model PCC Test:**

```bash
pytest models/experimental/pi0/tests/pcc/test_pcc_pi05_model.py -v -s
# or direct execution
python models/experimental/pi0/tests/pcc/test_pcc_pi05_model.py
```

**Component PCC Tests:**

```bash
# Run all component tests
python models/experimental/pi0/tests/pcc/run_all_pcc_tests.py

# Individual component tests
pytest models/experimental/pi0/tests/pcc/test_pcc_suffix.py -v
pytest models/experimental/pi0/tests/pcc/test_pcc_prefix.py -v
pytest models/experimental/pi0/tests/pcc/test_pcc_gemma.py -v
pytest models/experimental/pi0/tests/pcc/test_pcc_siglip.py -v
pytest models/experimental/pi0/tests/pcc/test_pcc_paligemma.py -v
```

### Performance Tests (Benchmarking)

```bash
# π0.5 performance test (action throughput / latency)
python models/experimental/pi0/tests/perf/test_perf_pi05.py

# Metal Trace variant
python models/experimental/pi0/tests/perf/test_perf_pi05_trace.py

# Profiling helper
python models/experimental/pi0/tests/perf/profile_pi05.py
```

### Performance Test (end-to-end 2CQ + Trace)

```bash
pytest models/experimental/pi0/tests/perf/test_perf_e2e.py
```

Recommended invocation (Blackhole p300c, device 2, source-built tt-metal):

```bash
TT_METAL_RUNTIME_ROOT=$TT_METAL_HOME/build_Release/libexec/tt-metalium \
PI0_DEVICE_ID=2 \
PYTHONPATH=$TT_METAL_HOME \
TT_METAL_HOME=$TT_METAL_HOME \
python models/experimental/pi0/tests/perf/test_perf_pi05.py
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
python models/experimental/pi0/tests/demo/extract_aloha_samples.py

# Extract LIBERO samples (downloads from HuggingFace)
python models/experimental/pi0/tests/demo/extract_libero_samples.py
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
python models/experimental/pi0/tests/demo/run_aloha_sim_demo.py

# LIBERO demo
python models/experimental/pi0/tests/demo/run_libero_demo.py

# Visualize results
python models/experimental/pi0/tests/demo/visualize_demo.py
```

## Troubleshooting

### `Checkpoint not found`

Ensure `pi05_base` is at `$TT_METAL_HOME/models/experimental/pi0/weights/pi05_base/`:

```bash
huggingface-cli download lerobot/pi05_base \
    --local-dir $TT_METAL_HOME/models/experimental/pi0/weights/pi05_base
```

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
