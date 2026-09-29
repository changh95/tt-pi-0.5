# models.experimental.pi0_5 — π0.5 on one Tenstorrent Blackhole p150a

π0.5 (`lerobot/pi05_base`: SigLIP + Gemma-2B VLM + Gemma-300M flow-matching action expert) as one
fused / traced TT-NN graph: `PI0ModelTTNN.sample_actions_fused(images, lang_tokens, noise=None, lang_masks=None)`
returns a `[B, H, 32]` chunk of normalised actions. The whole device graph (SigLIP, VLM prefill,
10 expert steps) is captured in one Metal trace on the first call and replayed afterwards.

Attention follows openpi: right-padded prompt tokens are masked out of every query
(`lang_masks=None` means `tokens != 0`), and the action tokens are rotated at positions
`n_valid_prefix + [0, H)`.

Validated on tt-metal `main` @ `668c2907575` (v0.79.0-dev20260914), one p150a, 2026-09-29:

| | |
|---|---:|
| served shape (2 × 224², 224 tokens, H = 50), per call / trace replay | 84.2 / 82.8 ms |
| LIBERO shape (2 × 224², 32 tokens, H = 10), per call / trace replay | 76.9 / 75.6 ms |
| PCC vs the openpi GPU golden (`pi05_libero`, 8 LIBERO observations, 7 action dims) | mean 0.99984, min 0.99971 |
| LIBERO-spatial closed loop (`pi05_libero`, 10 tasks × 10 init states) | 98 / 100 |

Layout: `common/` (configs, `FusedConfig` knobs, torch-side graph inputs, weight loader),
`reference/` (the torch oracle), `tt/` (the TT-NN graph and the `generic_op` kernels under `tt/kernels/`),
`server/` (the tt-model-manager HTTP app and its smoke test), `tests/` (host proofs, PCC, perf).
The mesh / tensor-parallel code (`TT_MESH_SHAPE`, `tt/ttnn_ccl.py`) is kept but was not re-validated after
the mask / RoPE fix.

Full results, commands and the knob table: the repository `README.md` and `docs/FUSED_FIX_2026-09-29.md`
at [github.com/changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5).
