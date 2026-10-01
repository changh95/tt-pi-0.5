# π0.5 Model for Tenstorrent

π0.5 (Physical Intelligence 0.5) is a vision-language-action (VLA) model for
robotics that combines a vision encoder, language model, and action expert for
end-to-end robot control. This repository is a port of π0.5 to Tenstorrent
hardware via TTNN, derived from `lerobot/pi05_base`, running on a single
Blackhole p150a.

Since 2026-10-01 the default inference path (`PI05_MEGAKERNEL=whole`) is the **whole-model megakernel**:
every call of `PI0ModelTTNN.sample_actions_fused` runs the model as **one fused op** on the device, a single
persistent `ttnn.generic_op` (a `ProgramDescriptor` with custom kernels) on all 110 worker cores, captured in a
Metal trace whose replay holds exactly that one op ([`docs/megakernel/`](docs/megakernel/DESIGN.md), §11):

| Part of one call | Where it runs | Device ops per replay |
|---|---|---:|
| image resize / normalisation, im2col of the patches, prompt building and tokenisation, the mask / RoPE rows, the initial noise, the host->device copies of these inputs, the readback | host (input / output formatting; no learned parameter, no model arithmetic) | 0 |
| SigLIP on both cameras (patch embedding + 27 layers + post-LN), the projector, the language embedding, the Gemma-2B VLM prefill (18 layers) writing 18 bf8 K / V caches, the action in-projection, 10 Euler steps x 18 Gemma-300M expert layers (adaRMS, GQA attention over the prefix K/V with the padding mask and the offset RoPE, GeGLU MLP, gated residuals), the action out-projection | **ONE persistent `ttnn.generic_op`**: `tt/megakernel/pe_program.py` (`WholeMegakernel`) with the kernels `tt/megakernel/kernels_p2/whole_{brisc,ncrisc,trisc}.cpp` (the prefix engine `pe_*.hpp` + the expert loop `kernels/mk_*`), weights streamed from per-core DRAM arenas, activations DRAM-staged between the prefix's 314 in-kernel ops and L1-resident in the expert loop | **1** |

No part of the default path is a stock TT-NN op on the device. The traced TT-NN paths still exist only behind
the comparator knobs:

| `PI05_MEGAKERNEL` | What runs | Device ops per replay | Role |
|---|---|---:|---|
| `whole` (default; also the value when unset) | the whole model as ONE persistent generic_op | 1 | served / shipped path |
| `expert` | phase 1: SigLIP / VLM prefix as traced stock TT-NN ops (891), the expert loop as ONE persistent generic_op | 892 | comparator only |
| `off` | the path shipped before 2026-09-30: stock TT-NN ops plus three custom `generic_op` programs (fused expert attention `tt/ttnn_fused_attn.py`, adaRMS row rsqrt and GeGLU `tt/ttnn_fused_norm.py`) | 2,551 | comparator / oracle only |

The megakernels need the device opened with the 64 KiB worker-L1 cut (`common/device_open.py`:
`open_pi05_device` / `device_kwargs`; `worker_l1_size=1395712`, which leaves the 136,192 B kernel-config ring the
program needs); the model refuses a device without it by name. `whole` serves batch 1, one chip, 2 cameras, bf8 K/V
caches and 10 steps; any other configuration requested together with it refuses at build / server start. On a
multi-chip mesh the UNSET default resolves to `off` (the megakernels are single-p150a programs). The SigLIP / VLM
TT-NN modules are still constructed under `whole` (their weights are uploaded) but are never enqueued.

Since 2026-09-29 the graph applies openpi's attention semantics: right-padded prompt tokens are
masked out of every query, and the action tokens are rotated at positions `n_valid_prefix + [0, H)`.
Before that fix, both the port and its torch reference attended the pad tokens and rotated the
action tokens at `[0, H)`, so the PCC tests could not see either bug. Details and raw results:
[`docs/FUSED_FIX_2026-09-29.md`](docs/FUSED_FIX_2026-09-29.md).

## Results

Measured 2026-10-01 on one Blackhole p150a (AICLK 1350 MHz before and after every benchmark) with tt-metal `main` @
[`668c2907575`](https://github.com/tenstorrent/tt-metal/commit/668c290757550588d0ce46b180c344a462a2aaf5)
(v0.79.0-dev20260914), batch 1, 10 flow-matching steps, branch `megakernel-2026-09-29` at the integration commit
(whole-model kernel digest `4aa02cdf21ed0c94`). "default" = `PI05_MEGAKERNEL` UNSET (resolves to `whole`); `expert`
and `off` = the comparator knobs, measured in the same session with the arms alternated across processes. Raw files:
`docs/megakernel/integrate_p2/results/` and the record in `docs/megakernel/JOURNAL.md` ("integrate-p2"); the
phase-2 exit gates and their independent verification are in the same journal ("PHASE-2 EXIT GATE TABLE",
"verify-p2-r0").

### What runs on the device

| | default (`whole`) | `expert` | `off` |
|---|---:|---:|---:|
| device ops per trace replay (tracy, per replay session) | **1** (GenericOp, 110 cores; 21 / 21 sessions per shape) | 892 (891 stock + 1 GenericOp) | 2,551 |
| device ops a request issues outside the trace (9 requests, prompt lengths 1 / 128 / 224) | **0** | | |
| device time per replay, served shape (profiler; `whole`: the op's duration, median of 21 replays; `expert` / `off`: sum of the ops' durations, median of 3 / 2 profiled replay sessions) | **53.94 ms** | 69.49 ms | 82.30 ms |
| device time per replay, LIBERO shape | **51.26 ms** | 64.34 ms | 75.10 ms |

Inside the one op (host-clock marginals inside the real model, `L_*.json`): a SigLIP layer 354.2 us (served) /
352.1 us (LIBERO), a VLM layer 1,618 / 1,526 us, the whole prefix (SigLIP x2 + projector + embedding + VLM -> K/V)
38.1 / 35.7 ms per run. Kernel-config ring footprint (offline mock-cluster compile into an empty cache, readelf +
descriptor-counted args): 128,636 B served / 126,492 B LIBERO of the 136,192 B ring (gate 131,072 B).

### Accuracy

| Check | default (`whole`) | `expert` | `off` |
|---|---:|---:|---:|
| vs the openpi GPU golden: `lerobot/pi05_libero`, 8 real LIBERO observations, prompt right-padded to 32 tokens, H = 10, PCC over the 7 action dims | mean **0.999976**, min **0.999955** | mean 0.999884, min 0.999778 | mean 0.999839, min 0.999712 |
| Whole call vs the fp32 torch reference of the whole model on the same inputs, served shape, 32 seeds (prompts with 1-224 real tokens) | mean **0.99851**, min 0.98882; closer than `off` on **32 / 32** seeds (smallest margin +0.00121) | mean 0.97666, min 0.80383 | mean 0.97034, min 0.76397 |
| Same, LIBERO shape, the 8 golden observations | closer than `off` and `expert` on 8 / 8 | | |
| Per-layer K / V (18 layers x K, V, 8 inputs) vs the fp32 reference's own VLM cache | closer than the TT-NN caches on 288 / 288 per shape; min PCC 0.99171 served / 0.99905 LIBERO | (= `off`, bitwise) | min PCC 0.92868 / 0.99060 |
| Prompt-length edge cases vs fp32 (served n 1 / 224 / 128 / 150 inside the 32 seeds; LIBERO n 32 / 1, 4 inputs) | closer than `off` on every one | | |
| Ten calls after other prompts, ten raw trace replays, output buffer poisoned between calls | bit-identical | bit-identical | bit-identical |
| Alternating prompts and a shape switch (20 calls vs fresh-model outputs; `tests/megakernel/verify_alternating.py`) | 20 / 20 bit-identical; `all_ok` | | |
| 20 consecutive processes x 31 calls (prompt lengths 1 / 128 / 224) | 20 / 20, no hang, one output digest | | |

`expert` is closer to fp32 than `whole` on one of the 32 seeds (seed 707: 0.99769 vs 0.99522); that seed's PCC is
sensitive to small numerical changes (DESIGN.md §11.4). The fp32 comparison on random inputs is dominated by the
prefix, which is why the gate is "at least as close as the shipped path", not a fixed floor (DESIGN.md §7, amended
2026-09-30 and 2026-10-01).

### Latency

Host wall time per call (upload + trace replay + readback), median of 30 calls per process, two rounds; trace replay
= `execute_trace` alone (`integrate_p2/results/G_speed.json`):

| Shape | default: per call | default: replay | `expert`: per call | `off`: per call |
|---|---:|---:|---:|---:|
| Served shape: 2 x 224² images, 224 tokens, H = 50 | **55.87 / 55.95 ms** | 54.10 / 54.11 ms | 70.84 / 70.77 ms | 84.17 / 84.06 ms |
| LIBERO shape: 2 x 224² images, 32 tokens, H = 10 | **53.10 / 53.11 ms** | 51.21 / 51.24 ms | 65.73 / 65.78 ms | 76.97 / 76.96 ms |

Every difference is more than 80 times 2 x the MAD-based standard error of the median. 50 actions / 55.9 ms = 894
actions/s (in-process call median).

Served over HTTP by `server/app.py` (served shape, 30 warm requests per server, servers alternated default / `expert` /
default / `expert`, each confirmed by `/info` `megakernel.backend`; `integrate_p2/results/S_*`): `timing_ms.inference`
median **55.97 / 55.91 ms** default vs 70.84 / 70.87 ms `expert`; `timing_ms.total` 57.6 / 57.4 vs 72.3 / 72.4 ms; client
wall 59.0 / 58.6 vs 73.5 / 73.9 ms. `server/smoke_test.py` passes on every server, and a mask probe through the batcher
shows the request's own `lang_masks` reach the model (prompts `[2, 0]` and `[2]` give different actions, repeats are
identical). The startup refusals of the default for `PI05_NUM_IMAGES=1`, `PI05_BATCH_SIZES=1,2` and `PI05_KV_DTYPE=bf16`
fire before the device is opened.

### LIBERO closed loop

`lerobot/pi05_libero` @ `a217bfd3` with openpi's `pi05_libero` norm stats, libero_spatial
(10 tasks x official init states 0-9), openpi's evaluation loop (5 of each 10-action chunk
executed), served through openpi's websocket protocol by the default path (2026-10-01; every episode is stamped
`megakernel=whole`, kernel digest `4aa02cdf21ed0c94`, code `7f32fcc`):

| Policy | Device | Success |
|---|---|---:|
| this port, default (`whole`: one fused op per call) | p150a | **99 / 100** (0 errors, 0 timeouts; t9/i4 hit the step cap) |
| same, paired subset (init states 0-4) | p150a | 49 / 50 |
| phase-1 expert megakernel (`expert`, 2026-09-30) | p150a | 99 / 100 (paired 49 / 50) |
| previous path (`off`, 2026-09-29) | p150a | 98 / 100 (paired 48 / 50) |
| openpi `PI0Pytorch`, same weights and client (init states 0-4) | RTX 5090 | 50 / 50 |

Server-side policy latency: median **53.6 ms** per call (p10 53.3, p90 54.1; 2,189 calls), vs 66.4 ms for `expert`
and 77.6 ms for `off`. The open-loop golden check through the same LIBERO wrapper gave PCC7 mean 0.999976, min
0.999955, deterministic. The LIBERO server wrapper and the client are not part of this repository; the summaries are
in `docs/megakernel/integrate_p2/results/libero/`.

## What the fused graph does

With the default `PI05_MEGAKERNEL=whole`, the trace holds ONE op, `WholeMegakernel.run` (`tt/megakernel/`):
`pe_geometry.py` (shapes and the core map of the prefix engine, parsed from `kernels_p2/pe_common.hpp`),
`pe_host.py` (the prefix parameters from the checkpoint, norm affines folded into the consuming matmuls, the
weight arenas and the RoPE / mask tables), `pe_program.py` (the `ProgramDescriptor` of the whole program: the
prefix engine runs a fixed list of 314 ops -- patch embed, 27 SigLIP layers, post-LN, projector, embedding, 18 VLM
layers -- with one global barrier per op, then hands the 18 K / V caches to the phase-1 expert loop inside the same
kernel launch), `pe_size_check.py` (offline mock-cluster compile + kernel-config ring footprint); the expert loop's
own pieces are `geometry.py`, `arena.py`, `host_model.py` and `program.py` with `kernels/mk_*`. Numerics: bfp8
weights except the VLM qkv and the expert o / down projections (bf16), SigLIP / VLM matmuls at HiFi2 with fp32
accumulation, fp32 residual streams, bf16 activations between ops, bfp8 K / V caches; the expert loop keeps phase 1's
fidelities (DESIGN.md §4.10). DESIGN.md §11.2 / §11.4 list every choice and the arms that measured it.

The rest of this section describes the traced TT-NN graph (stock TT-NN ops plus the three custom `generic_op` programs
of `tt/ttnn_fused_attn.py` / `tt/ttnn_fused_norm.py`) that the comparator knobs (`expert` for its stock-op prefix,
`off` for everything) still run; it is not on the default path.

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
`common/fused_config.py`). Under `whole` only `PI05_MEGAKERNEL`, `PI05_TRACE` and `PI05_TRACE_REGION_SIZE` affect the
device computation; the others configure the traced TT-NN graph of the comparator knobs:

| Env | Default | Meaning |
|-----|---------|---------|
| `PI05_MEGAKERNEL` | `whole` | `whole` = the whole model as ONE persistent generic_op per call (the served path); `expert` = phase 1, only the expert loop is one generic_op, the prefix is traced stock ops (comparator only); `off` = the previous path: stock TT-NN ops plus the 3 custom programs (fused attention, row_rsqrt, geglu_rc), 1,660 ops after the last K/V-cache write (comparator only) |
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
│   │   └── megakernel/             # the megakernels: pe_geometry / pe_host / pe_program / pe_size_check (whole model,
│   │                               #   kernels_p2/whole_{brisc,ncrisc,trisc}.cpp + pe_*.hpp) and geometry / arena /
│   │                               #   host_model / program / size_check (the expert loop, kernels/mk_*)
│   ├── server/                     # app.py (FastAPI, served by tt-model-manager), smoke_test.py
│   └── tests/
│       ├── test_fused_host.py      # Torch-only proofs of the reformulations and the attention inputs (no device)
│       ├── test_server_masks.py    # CPU: every server path passes the request's own language mask
│       ├── megakernel/             # CPU tests of the megakernel host side + device tools (bring-up, gates, soak)
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

# Device: the same tests on the comparator paths
PI05_MEGAKERNEL=expert pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_fused.py -v -s -k libero
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
- **`PI05_MEGAKERNEL=whole refused: the device was opened without the 64 KiB worker-L1 cut`**: open the device
  with `common/device_open.py` (`open_pi05_device(fused)` or `ttnn.open_device(**device_kwargs(fused))`), or set
  `PI05_MEGAKERNEL=off` for the comparator path. Other `refused:` messages name the knob the kernels do not compile
  (`PI05_KV_DTYPE=bf16`, `PI05_NUM_STEPS` != 10, batch > 1, a mesh, `PI05_NUM_IMAGES` != 2 under `whole`).
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
re-measured. The p150a row is the current served number (`integrate_p2/results/S_whole_123520.lat.json`: 55.97 ms; the second default server gave 55.91).

| setting | ms | vs p150a |
|---|---:|---|
| p150a, default path = the whole-model megakernel (served `timing_ms.inference`, 2026-10-01) | 56.0 | — |
| RTX 5090 fp32 strict | 144.1 | p150a 2.57× faster |
| RTX 5090 bf16 autocast | 121.5 | p150a 2.17× faster |
| RTX 5090 fp16 autocast | 123.7 | p150a 2.21× faster |
| RTX 5090 bf16 weights resident (eager) | 99.9 | p150a 1.79× faster |
| RTX 5090 bf16 weights + whole-request `torch.compile` | 46.6 | GPU 1.20× |

Methodology: same host, the torch reference (same weights and preprocessing as the served path) run
eagerly in PyTorch 2.11 cu128 (fp32 weights + `torch.autocast` unless stated; no TensorRT), batch 1,
medians of 50 iterations after warm-up, H2D/D2H included. p150a power was not measured, so no
efficiency comparison is made. The p150a is faster than every eager GPU row; the GPU needs resident bf16 weights and a compiled whole-request graph to be faster (1.20×). Full table: [`GPU_COMPARISON.md`](GPU_COMPARISON.md).
