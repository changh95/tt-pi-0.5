# models.experimental.pi0_5 — π0.5 on one Tenstorrent Blackhole p150a

π0.5 (`lerobot/pi05_base`: SigLIP + Gemma-2B VLM + Gemma-300M flow-matching action expert) on one p150a:
`PI0ModelTTNN.sample_actions_fused(images, lang_tokens, noise=None, lang_masks=None)` returns a `[B, H, 32]` chunk of
normalised actions. By default (`PI05_MEGAKERNEL=whole`, since 2026-10-01) the whole model -- SigLIP on both
cameras, projector, language embedding, VLM prefill writing the K / V caches, the 10-step x 18-layer expert loop,
action in / out projections and the Euler updates -- runs as ONE persistent `ttnn.generic_op` (custom kernels on 110
cores, `tt/megakernel/`), captured in a Metal trace whose replay holds exactly that one device op. The host does only
input / output formatting (image normalisation, im2col, tokenisation, mask / RoPE rows, noise, copies, readback).
`PI05_MEGAKERNEL=expert` (traced stock-op prefix + the expert loop as one op) and `PI05_MEGAKERNEL=off` (the earlier
traced stock-op graph) are kept only as comparator knobs.

Attention follows openpi: right-padded prompt tokens are masked out of every query
(`lang_masks=None` means `tokens != 0`), and the action tokens are rotated at positions
`n_valid_prefix + [0, H)`.

Validated on tt-metal `main` @ `668c2907575` (v0.79.0-dev20260914), one p150a, 2026-10-01 (default path):

| | |
|---|---:|
| served shape (2 × 224², 224 tokens, H = 50), per call / trace replay | 55.9 / 54.1 ms |
| LIBERO shape (2 × 224², 32 tokens, H = 10), per call / trace replay | 53.1 / 51.2 ms |
| device ops per trace replay | 1 |
| PCC vs the openpi GPU golden (`pi05_libero`, 8 LIBERO observations, 7 action dims) | mean 0.999976, min 0.999955 |
| LIBERO-spatial closed loop (`pi05_libero`, 10 tasks × 10 init states) | 99 / 100 |

Layout: `common/` (configs, `FusedConfig` knobs, torch-side graph inputs, weight loader),
`reference/` (the torch oracle), `tt/` (the megakernels under `tt/megakernel/`, the comparator TT-NN graph and its
`generic_op` kernels under `tt/kernels/`),
`server/` (the tt-model-manager HTTP app and its smoke test), `tests/` (host proofs, PCC, perf).
The mesh / tensor-parallel code (`TT_MESH_SHAPE`, `tt/ttnn_ccl.py`) is kept but was not re-validated after
the mask / RoPE fix.

Full results, commands and the knob table: the repository `README.md`, `docs/megakernel/` and `docs/FUSED_FIX_2026-09-29.md`
at [github.com/changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5).
