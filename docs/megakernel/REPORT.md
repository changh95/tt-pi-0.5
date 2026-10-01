# pi0.5 on one Blackhole p150a as a megakernel: final report (2026-10-01)

The request had three parts: (1) run the expert loop as a megakernel, (2) run the whole model as ONE fused op (a
persistent custom-kernel `ttnn.generic_op` program, not a Metal trace of stock ops), and (3) ship it to GitHub and HF.
**All three are done.** This report was written by the final completeness critic. Every number below comes from a file
named next to it. The critic's own evidence is in `docs/megakernel/final_critic/`.

## Released state

| item | value | evidence |
|---|---|---|
| GitHub `changh95/tt-pi-0.5` main | `821e8c528dfffa0d1d6e73ad6abf181a749b39d9` (PR #3 merge) | `git ls-remote origin` 2026-10-01 13:35 KST |
| HF `changh95/pi05-base-p150` head | `990e22b54012d7a80055d9e9762676b3e49bdbe5` (175 files; parent cf08fb95) | `HfApi.list_repo_commits`, 13:37 KST |
| image | `tt-model/pi05-base-p150:6fb244df57ff` (`sha256:6fb244df57ff8d20...`), tt-metal 668c2907 | `publish_p2/results/pkg-pi05-mkp2.log`, `final_critic/hold.log` |
| default backend | `PI05_MEGAKERNEL=whole`, kernel digest `4aa02cdf21ed0c94`, source commit 821e8c5 | `final_critic/info_profiled.json` |

## Before / after

Base shape: 2 x 224^2, 224 tokens, H = 50, 10 steps. LIBERO shape: 32 tokens, H = 10.

| metric | before (`off`, shipped 09-29) | phase 1 (`expert`) | phase 2 (`whole`, shipped now) | source |
|---|---:|---:|---:|---|
| device ops per trace replay | 2,551 | 892 | **1** (GenericOp, 110 cores) | `integrate_p2/results/S_struct.json`; pulled image: `final_critic/FC_prof_pulled.json` |
| in-process call, base (ms, median of 30) | 84.17 (09-29 fix: 84.17, n 60) | 70.84 (09-30: 70.68) | **55.87** | `integrate_p2/results/G_speed.json` base_r1; `docs/fused_fix_2026-09-29/fix2_base.json`; `integrate_p1/results/A_default_base.json` |
| trace replay, base (ms) | 82.82 | 69.56 | **54.10** | `G_speed.json` base_r1 |
| device time per replay, base (ms, profiler) | 82.30 (sum of ops) | 69.49 (sum of ops) | **53.94** (one op; 53.96 in the pulled image) | `S_struct.json`; `final_critic/FC_prof_pulled.json` |
| in-process call, LIBERO (ms) | 76.97 | 65.73 | **53.10** | `G_speed.json` libero_r1 |
| served by the shipped image, `timing_ms.inference` (median of 100) | 84.05 (image 672900e23919, c2) | 70.81 (image fe0d2e3d68a7, c2) | **55.84** (p90 56.13; total 57.13) | `publish_p2/results/bench-c2-mkp2-*.json`; `publish_p1/results/bench-c2-mkp1-*.json`; GPU_COMPARISON 09-29 |
| openpi golden PCC7 (8 obs), mean / min | 0.999839 / 0.999712 | 0.999884 / 0.999778 | **0.999976 / 0.999955** | `integrate_p2/results/A_*_libero_r1.json` |
| whole model vs fp32 torch reference (32 seeds), mean / min | 0.97034 / 0.76397 | 0.97666 / 0.80383 | **0.99851 / 0.98882** (closer than off 32/32, than expert 31/32) | `integrate_p2/results/G_gates_r1.json` |
| LIBERO-spatial closed loop (100 episodes) | 98/100 at 77.6 ms/call | 99/100 at 66.4 ms/call | **99/100 at 53.6 ms/call** (paired init 0-4: 49/50 vs GPU 50/50) | `integrate_p2/results/libero/tt_spatial.jsonl` (recomputed); HF `demo/libero_eval.json` |
| kernel-config ring | - | - | 128,636 B of 136,192 B (gate 131,072) | `integrate_p2/results/M_size_base.json` |

The shipped image went from 84.05 to 55.84 ms served (1.51x faster). Device time per replay went from 82.30 to
53.94 ms.

## What the final critic checked (2026-10-01 13:35-13:45 KST)

1. **One program per call in the pulled shipped image, under the profiler.**
   - **Setup.** I ran the image that tt-model had pulled from HF (id `sha256:6fb244df57ff...`; shipcheck-p2 showed
     its code is identical to HF `code/`) with the device profiler on. Settings: `TT_METAL_DEVICE_PROFILER=1`,
     `TT_METAL_PROFILER_TRACE_TRACKING=1`, C++ post-processing, and a private `TT_METAL_CACHE`.
   - **Container spec.** Devices, binds, env, entrypoint and the uvicorn command were copied from the container that
     `tt-model serve` had just started (`final_critic/run_cmd.txt`, `mkrun.py`). The code was unmodified and nothing
     was imported from the host.
   - **Request path.** The run used the real HTTP server: lifespan warm-up (compile + capture), then 12 `/predict`
     requests over 3 prompts of different lengths, then SIGTERM. Shutdown went through the lifespan, and
     `close_device` dumped the profiler data. One with-device hold, card reset afterwards (exit 0).
   - **Result** (`final_critic/FC_prof_pulled.json`, from `cpp_device_perf_report.csv.gz`):
     - 14 replay sessions of trace 0 (2 warm-up + 12 requests). **Each session is exactly 1 program on 110 cores**,
       and it is the same program every time (one global call id).
     - **0 device programs outside the trace after the first replay.** All 2,160 non-session rows come before the
       first replay (model construction and the first call).
     - Device kernel time per replay: median 53.96 ms (53.90-54.07), against 53.94 in `S_struct.json`.
     - `/info`: backend whole, digest 4aa02cdf21ed0c94, commit 821e8c5.
     - Actions were identical across repeats of each prompt. Prompt 0's head is [-0.0708, -0.1553], the card's example.
     - With the profiler on, the request median was about 57.3 ms. That includes profiler overhead and is not a speed
       claim.
2. **Card numbers match the files.**

   | card figure | value | source file |
   |---|---|---|
   | served: median / p90 / total / client | 55.84 / 56.13 / 57.13 / 58.52 | `bench-c2-mkp2` |
   | previous image | 70.81 / 72.16 | `publish_p1` bench-c2 |
   | replay (whole / expert / off) | 54.1 / 69.6 / 82.8 | `G_speed` |
   | device time (whole / expert / off) | 53.94 / 69.49 / 82.30 | `S_struct` |
   | PCC7, seed gate, K/V 288/288 | as in the table above | `A_*_libero_r1`, `G_gates_r1` |
   | ring | 128,636 B | `M_size_base` |

3. **LIBERO matches the JSONL.**
   - Recomputed from `integrate_p2/results/libero/tt_spatial.jsonl`, which is byte-identical to the
     `gr00t/libero_eval/pi05/megakernel-p2` copy.
   - 100 unique episodes, 99 successes, 0 errors. The one failure is t9/i4 at the 230-step cap. Init states 0-4:
     49/50.
   - All 100 episodes are stamped `megakernel=whole` and `mk_digest=4aa02cdf21ed0c94`.
   - HF `demo/libero_eval.json` agrees: 99, 0.99, 53.6 ms, paired 49/50 vs 50/50, golden 0.999976 / 0.999955.
4. **The README video plays in headless Chromium** (`final_critic/pw_fc.py`, Playwright chromium-1243).
   - The page has 1 `<video>`. After a seek to 12 s and play: readyState 4, no media error, playing, t 16.0 of 29.13 s,
     960x1080, src `resolve/main/demo/pi05_libero_spatial.mp4`.
   - The captured frame (`final_critic/video_frame.png`) shows the current caption ("the whole-model megakernel (one
     fused op per call)", "server side: median 53.6 ms / call").
   - The page text contains 55.84 and the one-fused-op wording.
5. **No stale claims.**
   - I grepped the HF card, SERVING.md, tt-model.yaml, the demo README, the GitHub README and the package README for:
     "median of 60", "all stock", "SigLIP tower and the VLM prefill run as traced", 17.49, expert-oracle rows, an
     expert default, and 70.8 / 84.0 quoted as current.
   - Nothing stale was found. The only hits are correct statements about `expert` as a comparator.
   - The card's "What runs where" table puts every model stage inside the one generic_op. The host only formats
     inputs and outputs.
6. **NOTES_FOR_SHIP_P2.md items 1-5 are applied.**
   1. Replay is "median of 30".
   2. The profiler basis is given per path (21 sessions; 3 / 2).
   3. `off` is described as "stock TT-NN ops plus 3 custom programs" in the card, SERVING.md and tt-model.yaml.
   4. `integrate_p1/results/S_struct.json` has been moved (23fc3dd).
   5. The previous image's figures come from cycle 2 (70.81 / 72.16; 09-29: 84.05 / 85.30).
7. **The gate amendments are recorded in DESIGN.md** (§7): the 2026-09-30 per-seed gate, the 2026-10-01 P2-3 K/V
   clause, and `whole` as the default (§0 / §4.12).

## Open problems (not gates)

- **Seed 707 is fragile.** It is the one seed of 32 where `expert` is closer to fp32 than `whole` (0.99769 vs
  0.99522; both beat `off`, 0.99175). It has ranged from 0.975 to 0.997 across near-identical builds, and faster
  numerics experiments failed it. It limits further numeric shortcuts.
- **The SigLIP layer is close to its go line.** It measures 354.2 us (host clock, `L_base.json`) against the 355 us
  line.
- **`whole` still pays for the old prefix modules.**
  - The TT-NN prefix modules are still built and hold device DRAM, even though they are never enqueued.
  - Each request still copies host data into their unused inputs (im2col / tokens).
  - Removing them would save boot time and memory, not device time.
- **Scope of the megakernel.** It is single-chip, batch 1, 2 cameras and 10 steps. On a mesh, an unset
  `PI05_MEGAKERNEL` falls back to `off`. The mesh / TP / batch > 1 paths exist only on `off` and have not been
  re-validated since the mask / RoPE fix.
- **The GPU rows are old.** The RTX 5090 rows are the 2026-09-14 pre-fix reference runs and were not re-measured. The
  best whole-request `torch.compile` GPU row (46.6 ms) is still faster than the p150a.
- **Kernel sources were edited while a hold had the card**, once in phase 1 (that soak was rerun) and once on
  2026-09-30 at about 20:36 (kernels_p2). The integrate-p2 re-run checked source and ELF md5s before and after every
  hold, so the shipped results are not affected.
- **There are two served-latency figures.** The GitHub README quotes the in-process served A/B (55.97 ms); the HF card
  quotes the image bench (55.84 ms). Both are labelled.
