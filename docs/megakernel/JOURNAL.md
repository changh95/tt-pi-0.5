# pi0.5 megakernel journal (p150a)

Single source of continuity between sessions. Numbers come from files (paths given), never from memory.
Goal (user's words): ALL of the model's ops become ONE fused op -- a persistent program of custom kernels
(ttnn.generic_op + ProgramDescriptor), like the GR00T DiT megakernel. A trace of stock ops is NOT a megakernel.

Branch: megakernel-2026-09-29 (from main c8d4234). tt-metal: /home/deepgadget/experiments/gr00t/tt-metal @ 668c2907575.

## 2026-09-30 00:03:54 KST -- session start, phase "profile"

- Created branch megakernel-2026-09-29 from main c8d4234.
- Task: op-by-op device profile of the shipped path (PI0ModelTTNN.sample_actions_fused, traced) at base (2x224^2,
  224 tokens, H=50) and LIBERO (32 tokens, H=10) shapes -> docs/megakernel/PROFILE.md.
- Method chosen: tracy op profiler (python -m tracy -r -p -v), one process per shape. Per process: warm+capture,
  one EAGER pass of the same graph with signposts pushed/popped by wrappers around every stage function (labels
  every op), then N traced replays each synchronised and drained with ReadDeviceProfiler. Traced ops are mapped to
  labels by index against the eager op sequence (op codes must match 1:1, else refuse). Instance-count check per
  replay (truncation rule from memory note profiler-zone-log-collision). Unprofiled replay wall clock in a second
  sub-command (clock) for the profiler overhead.
2026-09-30 00:06:38 launching base profiled run
2026-09-30 00:13:44 base profile + clock done; libero profile failed on ENOSPC (disk 100%: my base run left a 3.7 GB tracy_ops_times.csv); cleaned, rerunning libero with fresh cache

## 2026-09-30 00:20:22 KST -- profile phase DONE -> docs/megakernel/PROFILE.md

Files: docs/megakernel/profile/{ops_perf_results,run,clock,summary}_{base,libero}_20260930.*, tables_20260930.md.
Tool: models/experimental/pi0_5/tests/perf/profile_pi05_ops.py (run / clock / summarise / report).

Measured (all from the files above):
- Unprofiled: base 84.21 ms per call / 82.81 replay; LIBERO 76.77 / 75.64 (reproduces the shipped numbers). aiclk 1350.
- Device span (profiled) base 83.86 ms = SigLIP 10.77 + VLM 41.54 + expert 31.26 + rest 0.28. LIBERO 76.75 = 10.77 +
  37.85 + 27.85 + 0.26. 2551 ops per replay at both shapes, launch gaps 1.59 / 1.66 ms total (median 0.53 us per op).
- Expert per layer 173.1 us (base) / 153.8 (LIBERO); per step 3.143 / 2.801 ms. Expert streams 437.1 MB per step (o_proj and
  down are bf16) = 4.37 GB per request; matmuls reach 261 GB/s during their kernels, 140 GB/s over the stage. Stream
  floor 9.42 ms at 464 GB/s. Non-matmul ops 14.1 ms of the 31.3 (fused_attn 43 us, 2-core adaRMS rms_norm 14.2 us per layer).
- VLM: MLP chunking (256 rows) re-streams the MLP weights per chunk: 5.63 GB streamed vs 1.99 GB unique. The down
  projection runs at 43.9 TFLOP/s (7.2 % LoFi peak), 19.5 ms per base request. LIBERO's 3rd chunk has 32 rows and costs
  8.0 ms per request for 32 rows. The GELU is a separate unary (3.74 ms base).
- SigLIP: 10.77 ms; fc1/fc2 at 59/52 TFLOP/s HiFi2; layernorm + create/concat heads 3.1 ms of pure movement.

Decisions / lessons:
- FW START is taken early on cores the previous op does not use, so FW durations overlap (their sum is 136.9 ms against
  an 83.9 ms span). Per-op time = end-to-end increment = kernel + launch gap. Any future profile must use the same rule.
- tracy writes a 3.7 GB .logs/tracy_ops_times.csv per base run. The disk was at 100 % and the first LIBERO run died with
  ENOSPC in a kernel build (rerun with a fresh TT_METAL_CACHE, OK). Delete .logs/*.csv after every run.

Open / next (the megakernel itself, not started):
- Step 2 of the user's plan comes after this. Design inputs from PROFILE.md: the expert is fixed-cost plus weight-stream
  bound (floor 9.4 ms bf16 / ~8.0 ms if o_proj/down went bfp8 -- needs a PCC check); the VLM MLP has large measured
  excess (re-streamed weights, a 7 %-peak down projection); SigLIP's movement ops.
- Not yet read in depth: GR00T mk_k4*/k5/f0*/BACKBONE_FUSION_PLAN docs (required before writing kernels).

## 2026-09-30 00:46:10 KST -- phase "design" DONE -> docs/megakernel/DESIGN.md (v1)

Read before designing: GR00T mk_k1 / k4 / k4b / k4c / k5 / f0 / f0c summaries, BACKBONE_FUSION_PLAN.md head, the listed
memory notes, docs/history/MEGAKERNEL_p300.md, the shipped fused attention / norm-fold code. No device time, no kernels.

Design decisions (details and arithmetic in DESIGN.md; numbers reproducible with docs/megakernel/design/mk_design_calc.py):
- Phase 1 = one persistent generic_op for the 10 x 18 expert loop. Weights stream from a per-core consumption-ordered DRAM
  arena (per-step adaRMS folds on BOTH norms, so the in0 of qkv and up|gate is the raw residual; r = row rsqrt computed
  by the hub during its multicast). Core map: 64 Q producers + 16 K/V producers (qkv), 80 attention units (8 heads x 2 q
  row tiles x 5 key chunks), 64 MLP cores (2-D up|gate / down with a row-local h exchange and a column reduce), 32 x
  owners (fp32 residual, o_proj), hubs (10,9) / (10,8). 3 hub rounds + 4 group-local exchanges per layer.
- Predicted expert loop: base 13.90 ms bottom-up / 18.91 expected (x1.36 GR00T K4b realisation) / 23.36 first build
  (x1.68) vs 31.26 today; LIBERO 10.05 / 13.66 / 16.88 vs 27.85. Whole call base 66.7 / 71.7 / 76.1 ms vs 84.21.
- Phase 2 = SigLIP + projector + embedding + VLM (2-D bands, weights read once per column and column-multicast, no
  per-owner in0 exchange) + the phase-1 loop in ONE program with phase overlays; predicted whole call base 42.4 / 57.1 /
  72.3 ms, LIBERO 32.9 / 44.3 / 56.0. The VLM layer is the uncertain term (1.20-2.15 ms vs TTNN 2.437): WP-P2-1 is a go/no-go.
- Pre-registered gates in DESIGN.md §7 (task gates + per-WP gates).

Open / next:
- Step 2 starts at WP-P1-0 (scaffolding, host injectivity tests, mock compile, shipped path under the 64 KiB cut).
- A-values (unmeasured exchange costs R2/R3/R6/R7, disjoint-rectangle multicast concurrency) are measured by WP-P1-2
  before any layer code.
- Open questions for the user (DESIGN.md §8): larger L1 cut for the phase-2 program only if its binary cannot fit
  136,192 B; batch > 1 is out of the gates.
