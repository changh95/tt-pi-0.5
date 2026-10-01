Phase 2 of the pi0.5 megakernel on one Blackhole p150a (tt-metal `668c2907575`): the WHOLE model is one fused op per call.

**What changed.** `PI0ModelTTNN.sample_actions_fused` now runs SigLIP on both cameras (patch embed + 27 layers + post-LN), the projector, the language embedding, the Gemma-2B VLM prefill writing the 18 bf8 K/V caches, the 10-step x 18-layer action-expert loop, the action in/out projections and the Euler updates as ONE persistent `ttnn.generic_op` on 110 cores (`models/experimental/pi0_5/tt/megakernel/pe_program.py` `WholeMegakernel`, kernels `tt/megakernel/kernels_p2/whole_{brisc,ncrisc,trisc}.cpp` + `pe_*.hpp`, which include phase 1's `kernels/mk_*`). The Metal trace replay holds exactly that one device op; the host only formats inputs and outputs (image normalisation, im2col, tokenisation, mask / RoPE rows, noise, copies, readback).

- `PI05_MEGAKERNEL=whole` is the default (unset / empty). `expert` (phase 1: traced stock-op prefix + the expert loop as one op, 892 ops) and `off` (stock TT-NN ops plus the 3 custom programs fused attention / row_rsqrt / geglu_rc, 2,551 ops) stay as comparator knobs. `whole` refuses by name: a mesh, batch > 1, bf16 K/V, step counts other than 10, `PI05_NUM_IMAGES` != 2, and a device opened without the 64 KiB worker-L1 cut.
- Kernel-config ring footprint 128,636 B (served) / 126,492 B (LIBERO) of 136,192 B (offline mock-cluster compile, gate 131,072 B).

**Measured** (2026-10-01, AICLK 1350 MHz, final code commit; raw files in `docs/megakernel/integrate_p2/results/`):

| | default (`whole`) | `expert` (phase 1) | `off` |
|---|---:|---:|---:|
| device ops per trace replay | 1 | 892 | 2,551 |
| device time per replay, served / LIBERO shape (profiler) | 53.94 / 51.26 ms | 69.49 / 64.34 ms | 82.30 / 75.10 ms |
| per call, served shape (224 tok, H=50), two rounds | 55.87 / 55.95 ms | 70.84 / 70.77 ms | 84.17 / 84.06 ms |
| per call, LIBERO shape (32 tok, H=10), two rounds | 53.10 / 53.11 ms | 65.73 / 65.78 ms | 76.97 / 76.96 ms |
| served over HTTP, `timing_ms.inference` median, two servers | 55.97 / 55.91 ms | 70.84 / 70.87 ms | |
| openpi golden PCC7, 8 LIBERO obs (mean / min) | 0.999976 / 0.999955 | 0.999884 / 0.999778 | 0.999839 / 0.999712 |
| whole call vs the fp32 whole-model torch reference, 32 seeds | mean 0.99851, closer than `off` on 32/32 | mean 0.97666 | mean 0.97034 |
| LIBERO-spatial closed loop, 10 tasks x 10 inits | 99/100 at 53.6 ms/call | 99/100 at 66.4 ms/call | 98/100 at 77.6 ms/call |

- Replays, the alternating-prompt / shape-switch test and output-buffer poisoning: bit-identical. 20/20 consecutive processes ran with no hang and gave one output digest. `tests/pcc/test_pcc_pi05_fused.py` passes both cases under the default (the base case's 0.95 floor now passes: seed 3 0.98882).
- Gates amended by the user (DESIGN.md §7): 2026-09-30 per seed at least as close as the shipped path to the fp32 reference; 2026-10-01 per-layer K/V at least as close to the fp32 host decomposition as the TT-NN caches (288/288 per shape).
- Known: seed 707 is the one of 32 seeds where `expert` is closer to fp32 than `whole` (0.99769 vs 0.99522; `whole` is still closer than `off`). The SigLIP layer's host-clock time is 354.2 us against a 355 us go line.
- Docs: label fixes carried from the HF card fix (profiler basis per path, `off` is not "all stock ops", the 09-29 served figures from the shipped image's cycle 2, `S_struct.json` refiled).

The journal, design and the independent verification are in `docs/megakernel/` (`JOURNAL.md`, `DESIGN.md`, `verify_p2_r0/`, `integrate_p2/`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
