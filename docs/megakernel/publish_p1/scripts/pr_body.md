Phase 1 of the pi0.5 megakernel on one Blackhole p150a (tt-metal `668c2907575`).

**What changed.** The whole 10-step x 18-layer action-expert loop, together with the action in/out projections and the Euler updates, now runs as ONE persistent `ttnn.generic_op` program on 110 cores (`models/experimental/pi0_5/tt/megakernel/`: host program + `kernels/mk_{brisc,ncrisc,trisc}.cpp`, `mk_defs.hpp`, `mk_dm.hpp`). It replaces 1,660 stock/custom ops. The op sits inside the existing Metal trace, after the prefix. The prefix (SigLIP + Gemma-2B VLM prefill + K/V cache writes) is still 891 traced stock TT-NN ops and is the phase-2 target.

- `PI05_MEGAKERNEL=expert` is the default (unset/empty). `PI05_MEGAKERNEL=off` keeps the previous path as the comparator. The megakernel refuses by name: a multi-chip mesh, bf16 K/V caches, step counts other than 10, and a device opened without the 64 KiB worker-L1 cut (`common/device_open.device_kwargs`).
- server: the batcher passes each request's own language mask; `/info` reports the megakernel backend and kernel digest.

**Measured** (2026-09-30, AICLK 1350 MHz; raw files in `docs/megakernel/integrate_p1/results/`):

| | default (megakernel) | off (previous) |
|---|---:|---:|
| openpi golden PCC7, 8 LIBERO obs (mean / min) | 0.999884 / 0.999778 | 0.999839 / 0.999712 |
| expert vs fp32 expert-loop oracle fed the device's K/V, 22 seeds | mean 0.99974, closer on 22/22 | mean 0.99596 |
| per call, served shape (224 tok, H=50) | 70.7 ms | 84.3 ms |
| per call, LIBERO shape (32 tok, H=10) | 65.8 ms | 76.7 ms |
| expert-loop device time, served / LIBERO | 17.49 / 16.07 ms (one op) | 30.31 / 26.83 ms |
| served over HTTP, `timing_ms.inference` median | 70.93 / 70.88 ms | 84.04 / 84.10 ms |
| LIBERO-spatial closed loop, 10 tasks x 10 inits | 99/100 at 66.4 ms/call | 98/100 at 77.6 ms/call |

- Replays, the alternating-prompt / shape-switch test, and output-buffer poisoning: bit-identical. 20/20 consecutive processes ran with no hang and gave one output digest.
- The phase-1 base gate was amended by the user on 2026-09-30 (DESIGN.md §7): the megakernel must be at least as close as the shipped path to the fp32 expert-loop oracle.
- Still failing, on `main` too: `tests/pcc/test_pcc_pi05_fused.py -k base` fails its own 0.95 whole-call floor on seed 3 in both paths (default 0.94833, off 0.93811).

The journal, design, and the two independent verifications are in `docs/megakernel/` (`JOURNAL.md`, `DESIGN.md`, `verify_p1_r0/`, `verify_p1_r1/`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
