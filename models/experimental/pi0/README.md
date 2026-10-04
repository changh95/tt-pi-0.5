# PI0 Model for Tenstorrent

PI0 (Physical Intelligence Zero) is a vision-language-action model for robotics
that combines a vision encoder, language model, and action expert for end-to-end
robot control.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              PI0 Model                                  │
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
│  │  │                        │    │                        │        │ │
│  │  │  Q_vlm ──┐             │    │  Q_exp ──┐             │        │ │
│  │  │  K_vlm ──┼─► SHARED ◄──┼────┼─ K_exp   │             │        │ │
│  │  │  V_vlm ──┘   ATTN      │    │  V_exp ──┘             │        │ │
│  │  │                        │    │                        │        │ │
│  │  │  MLP_vlm               │    │  MLP_exp               │        │ │
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
- **Flow Matching**: Iterative denoising from pure noise to actions over 10 steps
- **Dual Experts**: VLM (2B) processes images+language, Expert (300M) processes actions

## Directory Structure

```
pi0/
├── common/                     # Shared configs and utilities
│   ├── configs.py              # Model configurations
│   ├── weight_loader.py        # Checkpoint loading
│   ├── pi05_host.py            # pi0.5 megakernel host inputs (im2col, masks, RoPE rows)
│   └── utils.py                # Common utilities
├── reference/                  # PyTorch reference implementation
│   ├── torch_pi0_model.py      # Main PI0 model
│   ├── torch_paligemma.py      # PaliGemma backbone
│   ├── torch_siglip.py         # SigLIP vision tower
│   ├── torch_gemma.py          # Gemma attention/MLP
│   ├── torch_prefix.py         # Prefix embedding
│   ├── torch_suffix.py         # Suffix embedding
│   └── torch_denoise.py        # Denoising logic
├── tt/                         # TTNN implementation
│   ├── ttnn_pi0_model.py       # Main PI0 model (TTNN); pi05=True delegates to ttnn_pi05_model
│   ├── ttnn_pi05_model.py      # pi0.5: the whole sample_actions as three device ops
│   ├── megakernel/             # pi0.5 megakernel: kernels + host (see megakernel/README.md)
│   ├── ttnn_paligemma.py       # PaliGemma backbone (TTNN)
│   ├── ttnn_siglip.py          # SigLIP vision tower (TTNN)
│   ├── ttnn_gemma.py           # Gemma attention/MLP (TTNN)
│   ├── ttnn_prefix.py          # Prefix embedding (TTNN)
│   ├── ttnn_suffix.py          # Suffix embedding (TTNN)
│   └── ttnn_common.py          # Common TTNN utilities
├── tests/
│   ├── pcc/                    # PCC (accuracy) tests
│   ├── perf/                   # Performance benchmarks and e2e (2CQ + Trace)
│   ├── demo/                   # Demo scripts with ALOHA/LIBERO datasets
│   └── download_pretrained_weights.py
└── weights/                    # Pretrained checkpoints
    └── pi0_base/               # Base model checkpoint
```

## Quick Start

### 1. Environment Setup

```bash
# Set required environment variables
export TT_METAL_HOME=/path/to/tt-metal
export PYTHONPATH=$TT_METAL_HOME
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

# Activate virtual environment
source $TT_METAL_HOME/python_env/bin/activate
```

### 2. Download Pretrained Weights

The model requires pretrained weights to run. Download them using the provided script:

```bash
# Automatic download (requires gdown)
python $TT_METAL_HOME/models/experimental/pi0/tests/download_pretrained_weights.py

# Or with custom output directory
python $TT_METAL_HOME/models/experimental/pi0/tests/download_pretrained_weights.py \
    --output-dir /custom/path/weights
```

**Manual Download (if automatic download fails):**

1. Open: https://drive.google.com/drive/folders/1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN
2. Download the folder
3. Extract to: `$TT_METAL_HOME/models/experimental/pi0/weights/`

**Alternative: Using command-line tools:**

```bash
# Using gdown
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1qfY0EBGh_-6Zz-omKPQW6nBcc1Cp2_WN \
    -O $TT_METAL_HOME/models/experimental/pi0/weights/

# Using rclone (works with private folders)
rclone config  # Setup Google Drive remote named 'gdrive'
rclone copy gdrive:pi0_base $TT_METAL_HOME/models/experimental/pi0/weights/pi0_base
```

After download, verify the structure:
```
$TT_METAL_HOME/models/experimental/pi0/weights/
└── pi0_base/
    ├── model.safetensors
    └── config.json
```

## Running Tests

### PCC Tests (Accuracy Validation)

PCC (Pearson Correlation Coefficient) tests compare TTNN outputs against PyTorch reference.

**Full Model PCC Test:**

```bash
# Using pytest
pytest models/experimental/pi0/tests/pcc/test_pcc_ttnn_pi0_model.py -v
```

**Code Flow (what gets tested):**

```
PI0ModelTTNN.sample_actions()
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
│       └─► GemmaBlockTTNN.forward() × 18 layers (VLM blocks)
│           ├─► rms_norm_ttnn()
│           ├─► GemmaAttentionTTNN.forward()
│           │   ├─► ttnn.linear() for fused QKV
│           │   ├─► ttnn.experimental.rotary_embedding() for RoPE
│           │   └─► ttnn.transformer.scaled_dot_product_attention()
│           │
│           └─► GemmaMLPTTNN.forward()
│               ├─► ttnn.linear() for gate_proj, up_proj
│               ├─► ttnn.gelu()
│               └─► ttnn.linear() for down_proj
│
├─► [DENOISING LOOP × 10 steps]
│   │
│   ├─► self.embed_suffix(state, x_t, timestep)
│   │   └─► SuffixEmbeddingTTNN.embed_suffix()
│   │       └─► fuse_action_time MLP + action_embed + state_embed
│   │
│   └─► self.backbone.forward_expert(suffix_embs, past_key_values=prefix_kv_cache)
│       └─► PaliGemmaBackboneTTNN.forward_expert()
│           └─► GemmaBlockTTNN.forward() × 18 layers (Expert blocks)
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

**Test with Random vs Pretrained Weights:**

```bash
# Run with pretrained weights only (full validation)
pytest models/experimental/pi0/tests/pcc/test_pcc_suffix.py -v -k "pretrained_weight_true"

# Run with random weights only (fast CI)
pytest models/experimental/pi0/tests/pcc/test_pcc_suffix.py -v -k "pretrained_weight_false"
```

### Performance Tests (Benchmarking)

```bash
# Full model performance test
pytest models/experimental/pi0/tests/perf/test_perf_ttnn_pi0_model.py -v -s

# Direct execution
python models/experimental/pi0/tests/perf/test_perf_ttnn_pi0_model.py
```

### Performance Test (end-to-end (2CQ + Trace))
```bash
# Direct execution
pytest models/experimental/pi0/tests/perf/test_perf_e2e.py
```

## Demo Scripts

Demo scripts visualize model inference on robotics datasets.

**Extract Sample Images (required first):**

```bash
# ImageIO python library plugin PyAv is needed to extract images from videos
python -m ensurepip --upgrade && python -m pip install imageio[pyav]

# Extract ALOHA simulation samples (downloads from HuggingFace)
python models/experimental/pi0/tests/demo/extract_aloha_samples.py

# Extract LIBERO samples (downloads from HuggingFace)
python models/experimental/pi0/tests/demo/extract_libero_samples.py
```

This creates sample images in `tests/demo/sample_images/`:
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
# ALOHA simulation demo
python models/experimental/pi0/tests/demo/run_aloha_sim_demo.py

# LIBERO demo
python models/experimental/pi0/tests/demo/run_libero_demo.py

# Visualize results
python models/experimental/pi0/tests/demo/visualize_demo.py
```

## Troubleshooting

### `Checkpoint not found`

Download weights using the script:
```bash
python models/experimental/pi0/tests/download_pretrained_weights.py
```

## pi0.5 (`PI0ModelConfig(pi05=True)`)

With `pi05=True`, `PI0ModelTTNN` runs the whole pi0.5 `sample_actions` as **three device operations** (four with 3 or
4 cameras): persistent
`ttnn.generic_op` programs on 110 cores of a Blackhole chip (`tt/ttnn_pi05_model.py`, design in
[`tt/megakernel/README.md`](tt/megakernel/README.md)):

* VISION: SigLIP and the projector on the cameras (one program per group of <= 2 cameras);
* PREFIX: the language embedding and the VLM prefill;
* EXPERT: the N-step x 18-layer action expert with the time MLP / adaRMS conditioning, the action projections and
  the Euler steps.

Each call writes the request into fixed device buffers and replays a Metal trace of the three ops. pi0
(`pi05=False`) is unchanged.

Semantics follow openpi's pi0.5:

* the prompt's pad keys are masked;
* the prefix is bidirectional;
* the action tokens are placed at positions `n_valid + [0, H)`;
* the time embedding conditions every expert norm (adaRMS).

The torch reference (`reference/`) implements the same for `pi05=True`.

```python
import torch, ttnn
from models.experimental.pi0.common.configs import PI0ModelConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
from models.experimental.pi0.tt.ttnn_pi05_model import open_pi05_device

device = open_pi05_device(0)  # the profile's dispatch cores and the 64 KiB worker-L1 cut (PI05_DEVICE_PARAMS)
model = PI0ModelTTNN(PI0ModelConfig(action_horizon=50, pi05=True), PI0WeightLoader("lerobot/pi05_base"), device)
actions = model.sample_actions(
    images,      # 2 x [1, 3, 224, 224] in [-1, 1]
    img_masks,   # 2 x [1] True (or None)
    lang_tokens, # [1, 224] ids, right-padded (a pi0.5 policy that uses the state tokenizes it into the prompt)
    lang_masks,  # [1, 224] True on the real tokens (None -> tokens != 0)
    None,        # state: unused by pi0.5
    noise=noise, # [1, 50, 32] (None -> a fixed default drawn at construction)
)                # -> torch.float32 [1, 50, 32]
```

The first call compiles the kernels and captures the trace (a few seconds); later calls replay it.

**Supported:**

* batch 1, 1-4 cameras of 224 x 224 (`num_cameras`), 1..16 denoising steps (`num_denoising_steps`);
* an action horizon of 1..64 and a right-padded prompt of up to 224 real tokens (each request runs in the smallest of
  the 32 / 64 / 128 / 224-token prompt buckets that holds it; `prompt_bucket=` overrides it);
* masked cameras are refused: pass only the real cameras (a model built for that count);
* a single Blackhole chip, opened with `open_pi05_device()` (in pytest: `PI05_DEVICE_PARAMS` as the `device_params` of
  the device fixture), in one of two profiles chosen with `PI05_DISPATCH`: `eth` (default, "non-scalable": ethernet
  dispatch, 12 x 10 workers; needs a tt-metal runtime with Blackhole ethernet dispatch, PR #57142) or `tensix`
  ("scalable": Tensix dispatch, 11 x 10 workers, any runtime).

Anything else (another batch size, camera count, horizon, prompt length or step count, a device without the
worker-L1 cut, a mesh) raises an error that names the reason.

**Results.** Blackhole p150a, tt-metal main, host clock, median of 60. A call = host inputs + upload + replay +
readback.

| | upstream `pi05=True` before this change | pi0.5 megakernel |
|---|---|---|
| lerobot/pi05_base loads | no: `PI0Config.from_json` TypeError on the lerobot `config.json` | yes |
| time conditioning (time MLP, adaRMS) | none: dropped by the loader and the graph | yes |
| 2 cameras | TT_FATAL (shard height, `ttnn_gemma.py`) at every 2-camera shape | yes |
| PCC vs fp32 torch reference with openpi semantics | 0.050 (1 camera, 32 tokens, H 10, the only runnable case) | base, 6 padded prompts: min 0.9873, mean 0.9974 |
| PCC7 vs openpi GPU policy (8 LIBERO records) | - | mean 0.999977, min 0.999949 |
| device ops per call | one per stock ttnn op (not counted) | 3 |
| latency, base (2 x 224^2, 224 tokens, H 50) | does not run | 55.1 ms per call, 54.0 ms per replay |
| latency, LIBERO (2 x 224^2, 32 tokens, H 10) | does not run; 1 camera: 139.3 ms per call, 138.6 ms per replay | 52.5 ms per call, 51.2 ms per replay |
| latency, other presets (2 cameras, N 10) | does not run | 52.0-55.4 ms per call over the 32 / 64 / 128 / 224-token buckets x 32 / 64 action rows |
| 3 cameras (N 10; PCC vs fp32, 8 presets x 6 prompts) | does not run | min 0.9975, every mean >= 0.9994; PCC7 vs openpi (3-camera LIBERO records, right wrist = left wrist) mean 0.99998; 69.4-71.3 ms per call at 32 action rows, 74.3-76.5 ms at 64 |
| 4 cameras (N 10; PCC vs fp32, 8 presets x 6 prompts) | does not run | min 0.9975, every mean >= 0.9993; PCC7 vs openpi (3-camera records + a copy of the base camera) mean 0.99998; 85.9-89.0 ms per call at 32 action rows, 88.1-96.6 ms at 64 |

**Known limitation (one input-sensitive trajectory).** Over the full matrix (32 presets x 1..10 steps, 6 padded
prompts each, vs the fp32 torch reference) the gate "PCC min >= 0.95, mean >= 0.98" passes 314 of 320 (preset, steps)
sets. The 6 failures are one input: 2 cameras, a full 224-token prompt, 64 action rows, matrix seed 5, at 5-10 steps.
The openpi GPU policy in bf16 also fails there from 6 steps on, but the megakernel is lower at every step count:

| steps | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| openpi GPU bf16 vs fp32 | 0.966 | 0.936 | 0.933 | 0.901 | 0.828 | 0.895 |
| megakernel vs fp32 | 0.928 | 0.870 | 0.845 | 0.803 | 0.813 | 0.763 |

The other five prompts of that preset stay >= 0.9978 at every step count. Rounding the reference's own K / V caches to
bf16 or bfp8 leaves it at >= 0.99994, and raising the VLM matmul fidelity does not help, so this trajectory amplifies
the prefix arithmetic error of any reduced-precision implementation; the megakernel's output departs further than
openpi's bf16 there. Closed-loop LIBERO is the end-to-end check.

**Known limitation (expert oracle at one action row).** A4 compares the device output with an fp32 expert loop fed the
device's own K / V caches (so it isolates the action expert). Over the matrix it passes on 286 of the 288 (preset,
steps) sets with 2..10 steps; both misses have one action row: 1 camera / 64-token bucket / 5 steps (0.998746) and
2 cameras / 128-token bucket / 3 steps (0.997803), against a 0.999 bar. At 1 step it is recorded, not gated: 21 of the
96 (preset, horizon) values are below 0.999, the lowest 0.993544 (1 camera, 32-token bucket, 1 action row). The
residual is the expert's reduced-precision arithmetic (bf16 activations, HiFi3 matmuls); every expert matmul at HiFi4
lifts the two 2..10-step cells to 0.999595 and 0.999807, but its speed depends on the shape (replay vs HiFi3, 10
steps unless noted: 1 camera L32 S32 1 step -0.11 ms, 2 cameras L64 S32 -1.27 ms, 4 cameras L224 S32 -2.46 ms,
3 cameras L224 S32 +0.58 ms, 4 cameras L224 S64 +1.60 ms), so it is not adopted: every expert matmul stays at HiFi3.
A per-preset expert fidelity is a follow-up for the next release.

**Tests.** Device tests skip when the weights are absent. Weights come from `PI05_BASE_WEIGHTS` /
`PI05_LIBERO_WEIGHTS` (a directory or a HuggingFace id). The openpi records come from `PI05_OPENPI_GOLDEN`.

```bash
# CPU only: kernel-header / core-map / CB invariants, weight-arena orders, the decomposition vs the plain math,
# attention inputs vs openpi, request checks, device refusals
pytest models/experimental/pi0/tests/pcc/test_pi05_megakernel_host.py

# device: PCC vs the torch reference (base) and vs openpi (LIBERO), mask live / exact, replays, three ops per call
pytest models/experimental/pi0/tests/pcc/test_pcc_pi05_megakernel.py

# device: latency at both shapes
pytest models/experimental/pi0/tests/perf/test_perf_pi05_megakernel.py

# no device: kernel-config ring footprint of each program (mock cluster)
TT_METAL_CACHE=$(mktemp -d) python -m models.experimental.pi0.tt.megakernel.size_check
```

## Model Specifications

| Component | Details |
|-----------|---------|
| Vision Encoder | SigLIP (27 transformer blocks, 1152 hidden dim) |
| VLM Backbone | Gemma 2B (18 transformer blocks) |
| Action Expert | Gemma 300M (18 transformer blocks) |
| Image Size | 224×224 |
| Action Dimension | 32 |
| Action Horizon | 50 |

## License

SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
SPDX-License-Identifier: Apache-2.0
