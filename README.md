# Pi0.5 Model for Tenstorrent

Pi0.5 (Physical Intelligence 0.5) is a vision-language-action (VLA) model for
robotics that combines a vision encoder, language model, and action expert for
end-to-end robot control. This repository is a port of Pi0.5 to Tenstorrent
hardware via TTNN, derived from `lerobot/pi05_base`.

**What makes Pi0.5 different from Pi0:**
- **Adaptive RMSNorm (adaRMS)** in the action expert: per-layer scale/shift/gate
  modulations are conditioned on the flow-matching timestep, replacing the
  static RMSNorm used in Pi0.
- Uses the `lerobot/pi05_base` HuggingFace checkpoint (not the original
  Google-Drive `pi0_base`).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             Pi0.5 Model                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────┐   ┌──────────────────────────┐│
│  │         PREFIX EMBEDDING            │   │    SUFFIX EMBEDDING      ││
│  │                                     │   │                          ││
│  │  ┌───────────┐   ┌───────────────┐  │   │  ┌────────┐  ┌────────┐ ││
│  │  │  Images   │   │ Language      │  │   │  │ State  │  │ Noisy  │ ││
│  │  │  (224x224)│   │ Tokens        │  │   │  │ (32)   │  │Actions │ ││
│  │  └─────┬─────┘   └───────┬───────┘  │   │  └───┬────┘  └───┬────┘ ││
│  │        │                 │          │   │      │           │      ││
│  │        ▼                 │          │   │      └─────┬─────┘      ││
│  │  ┌───────────┐           │          │   │            │            ││
│  │  │  SigLIP   │           │          │   │   ┌────────▼─────────┐  ││
│  │  │  Vision   │           │          │   │   │ Action+Time MLP  │  ││
│  │  │  Tower    │           │          │   │   │ (fuse_action_    │  ││
│  │  │(27 blocks)│           │          │   │   │  time)           │  ││
│  │  └─────┬─────┘           │          │   │   └────────┬─────────┘  ││
│  │        │                 │          │   │            │            ││
│  │        ▼                 │          │   └────────────┼────────────┘│
│  │  ┌───────────┐           │          │                │             │
│  │  │Projector  │           │          │                │             │
│  │  │(1152→2048)│           │          │                │             │
│  │  └─────┬─────┘           │          │                │             │
│  │        │                 │          │                │             │
│  │        ▼                 ▼          │                │             │
│  │  ┌───────────────────────────────┐  │                │             │
│  │  │  Image Embeds + Lang Embeds   │  │                │             │
│  │  │  (Gemma 2B embedding)         │  │                │             │
│  │  └───────────────┬───────────────┘  │                │             │
│  │                  │                  │                │             │
│  └──────────────────┼──────────────────┘                │             │
│                     │                                   │             │
│                     ▼                                   ▼             │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │               DUAL-EXPERT TRANSFORMER (18 layers)                │ │
│  │  ┌────────────────────────┐    ┌────────────────────────┐        │ │
│  │  │     Gemma 2B VLM       │    │   Gemma 300M Expert    │        │ │
│  │  │   (processes prefix)   │◄──►│  (processes suffix)    │        │ │
│  │  │                        │    │  + adaRMS(timestep)    │        │ │
│  │  │  Q_vlm ──┐             │    │  Q_exp ──┐             │        │ │
│  │  │  K_vlm ──┼─► SHARED ◄──┼────┼─ K_exp   │             │        │ │
│  │  │  V_vlm ──┘   ATTN      │    │  V_exp ──┘             │        │ │
│  │  │                        │    │                        │        │ │
│  │  │  MLP_vlm               │    │  adaRMS · MLP_exp      │        │ │
│  │  └────────────────────────┘    └────────────────────────┘        │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                   │                                   │
│                                   ▼                                   │
│                    ┌──────────────────────────────┐                   │
│                    │     FLOW MATCHING DENOISER   │                   │
│                    │     (10 denoising steps)     │                   │
│                    │                              │                   │
│                    │  for t in [1.0 → 0.0]:       │                   │
│                    │    noise_pred = expert_out   │                   │
│                    │    actions = euler_step()    │                   │
│                    └──────────────┬───────────────┘                   │
│                                   │                                   │
│                                   ▼                                   │
│                         ┌───────────────────┐                         │
│                         │   Action Output   │                         │
│                         │ [batch=1, 50, 32] │                         │
│                         └───────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key architectural details:**
- **Shared Attention**: VLM and Expert share K,V tensors (concatenated), but have separate Q and MLPs
- **Adaptive RMSNorm (Pi0.5)**: Expert RMSNorm layers take `(scale, shift, gate)` modulations derived from the flow-matching timestep embedding — enabling timestep-aware denoising.
- **Flow Matching**: Iterative denoising from pure noise to actions over 10 steps
- **Dual Experts**: VLM (2B) processes images+language, Expert (300M) processes actions

## Directory Structure

```
tt-pi-05/
├── common/                     # Shared configs and utilities
│   ├── configs.py              # Model configurations (GemmaConfig.use_adarms, etc.)
│   ├── weight_loader.py        # Checkpoint loading (pi05_base)
│   └── utils.py                # Common utilities
├── reference/                  # PyTorch reference implementation
│   ├── torch_pi0_model.py      # Main Pi0.5 model
│   ├── torch_paligemma.py      # PaliGemma backbone
│   ├── torch_siglip.py         # SigLIP vision tower
│   ├── torch_gemma.py          # Gemma attention/MLP (with adaRMS)
│   ├── torch_prefix.py         # Prefix embedding
│   ├── torch_suffix.py         # Suffix embedding
│   └── torch_denoise.py        # Flow-matching denoising logic
├── tt/                         # TTNN implementation
│   ├── ttnn_pi0_model.py       # Main Pi0.5 model (TTNN)
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
    └── pi05_base/              # Pi0.5 base checkpoint (symlink or download)
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

Pi0.5 weights live on HuggingFace as `lerobot/pi05_base`.

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
> original Pi0 Google-Drive checkpoint and is **not** used for Pi0.5.

## Running Tests

### PCC Tests (Accuracy Validation)

PCC (Pearson Correlation Coefficient) tests compare TTNN outputs against the
PyTorch reference.

**Full Pi0.5 Model PCC Test:**

```bash
pytest models/experimental/pi0/tests/pcc/test_pcc_pi05_model.py -v -s
# or direct execution
python models/experimental/pi0/tests/pcc/test_pcc_pi05_model.py
```

**Code Flow (what gets tested):**

```
Pi0_5ModelTTNN.sample_actions()
│
├─► self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
│   └─► PrefixEmbeddingTTNN.embed_prefix()
│       │
│       ├─► self.embed_image_fn(img)  [backbone.embed_image]
│       │   └─► PaliGemmaBackboneTTNN.embed_image()
│       │       ├─► SigLIPVisionTowerTTNN.forward()
│       │       │   └─► SigLIPBlockTTNN.forward() × 27 layers
│       │       │       ├─► SigLIPAttentionTTNN.forward()
│       │       │       └─► SigLIPMLPTTNN.forward()
│       │       │
│       │       └─► MultiModalProjectorTTNN.forward() [1152 → 2048]
│       │
│       └─► self.embed_language_fn(tokens)  [backbone.embed_language_tokens]
│           └─► ttnn.embedding(tokens, vlm_embed_tokens)
│
├─► self.backbone.forward_vlm(prefix_embs, use_cache=True)
│   └─► PaliGemmaBackboneTTNN.forward_vlm()
│       └─► GemmaBlockTTNN.forward() × 18 layers (VLM blocks, static RMSNorm)
│
├─► [DENOISING LOOP × 10 steps]
│   │
│   ├─► precompute per-step adaRMS modulations (scale/shift/gate)
│   │   from timestep embedding → stored in DRAM
│   │
│   ├─► self.embed_suffix(state, x_t, timestep)
│   │   └─► SuffixEmbeddingTTNN.embed_suffix()
│   │       └─► fuse_action_time MLP + action_embed + state_embed
│   │
│   └─► self.backbone.forward_expert(suffix_embs, past_key_values=prefix_kv_cache)
│       └─► PaliGemmaBackboneTTNN.forward_expert()
│           └─► GemmaBlockTTNN.forward() × 18 layers (Expert blocks)
│               ├─► adaRMSNorm(hidden, scale, shift)
│               ├─► GemmaAttentionTTNN.forward()
│               │   ├─► ttnn.linear() for fused QKV
│               │   ├─► ttnn.experimental.rotary_embedding() for RoPE
│               │   └─► ttnn.transformer.scaled_dot_product_attention()
│               └─► GemmaMLPTTNN.forward() + gate * residual
│
└─► return denoised_actions [batch=1, 50, 32]
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
# Pi0.5 performance test (action throughput / latency)
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

Demo scripts visualize Pi0.5 inference on robotics datasets.

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

## License

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
SPDX-License-Identifier: Apache-2.0
