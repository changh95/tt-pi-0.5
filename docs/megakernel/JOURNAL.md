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

## 2026-09-30 00:58:14 KST -- review "kernel feasibility" of DESIGN.md v1 (reviewer agent, no device time)

Verdict: NOT ACCEPTED as is; 3 blocking spec defects, all cheap to fix before WP-P1-0 encodes the tables.
- B1 cb_in0 (DESIGN.md:282) holds bf16 x/x_mid AND bfp8 ctx under one CB id, contradicting :223 (one format per CB).
  Fix: one CBDescriptor with two format_descriptors (two ids, one buffer; tt-metal program_descriptors.hpp:75), which also
  retires the "not demonstrated" aliasing risk of :482. A separate ctx CB instead would cost 139,264 B > the 117,170 headroom.
- B2 weight rings (:223-226) are smaller than one op's share (w8 87,040 < ug 139,264 per core) and the stream is ONE in-order
  cursor, so DRAM idles during attention and Wd cannot prefetch. Fluid model docs/megakernel/design/review_stream_sim.py
  (validated: unbounded rings reproduce 76.8 / 55.2): layer base 82.4 us (84.0 at 414 GB/s) vs 76.8 claimed, LIBERO 67.5
  (69.1) vs 55.4 claimed. w8 = 139,264 (+52,224 B) restores 76.8 / 55.2 (58.7 at 414).
- B3 R3 merge (:195) is a "tree of depth 3" but cb_part / part_arrive exist only on the merger (:297, :330), and the cost counts
  one part send for three levels (1.5 + 3 x 3.5 = 12.0; three sends -> 15.0); GR00T's 3.5 us merge had 4 O tiles, here 8.
Non-blocking items (numbers, sources): see the review result returned to the orchestrator.

## 2026-09-30 01:13:32 KST -- phase "design-revise" DONE -> docs/megakernel/DESIGN.md v2

Input: the two v1 reviews (kernel feasibility R-K: B1 cb_in0 formats, B2 rings, B3 merge + 10 items; goal / integration
R-G: B1 no structural gate, B2 cb_in0 + 9 items). All resolved, none rejected; DESIGN.md §10 is the item-by-item table.
Arithmetic: docs/megakernel/design/mk_design_calc_v2.py -> mk_design_calc_v2.out (imports the reviewer's
review_stream_sim.py, now committed). No device time.

Key changes (numbers from mk_design_calc_v2.out):
- cb_in0 = ONE tensor-backed CBDescriptor with two format descriptors, ids 0 bf16 / 29 bfp8, 139,264 B. Evidence read in
  tt-metal: circular_buffer_config.cpp:65-100 binds the buffer then processes each CBFormatDescriptor; nanobind exposes
  format_descriptors read-write. Device proof + 2 named fallbacks in WP-P1-0.
- cb_w8 16 x 8,704 = 139,264 B (>= the up|gate share). Layer time = the reviewer's per-core ring-constrained timeline:
  base 76.8 / LIBERO 61.1 us at 414-464 GB/s (v1 rings: 84.1 / 76.9 at 414). The stream no longer binds at either shape.
- R3 = flat dh-split merge (4 slice mergers per (h, r), 2 dh tiles each), priced per tile-op at T_E = 0.125 us (A, from
  GR00T's 3.5 us / 28 tile-ops): base 9.0 us, LIBERO 10.75. RoPE moved to the Q / K producers (pair exchange d ^ 4); KL
  multicasts only after the 8 Q column multicasts (kl_qdone). Attention priced at the 1.87 us/key-tile slope.
- Chain base 76.8 us (unchanged by offsetting corrections), LIBERO 52.3 -> 61.1 (no ad-hoc LIBERO scaling any more).
  Expert loop base 13.91 / 18.92 / 23.37 ms, LIBERO 11.08 / 15.07 / 18.62; whole call base 66.7 / 71.7 / 76.2, LIBERO
  59.8 / 63.8 / 67.4.
- L1: union 1,101,952 B base / 790,656 LIBERO; KV co-tenant 78,336 B per core (whole pages per bank); allocatable
  1,371,136 (l1_small 24,576); headroom 170,848 base. CB ids 30 of 64 (Blackhole).
- Sync words: cumulative counters keyed on (what, slot, peer), generation in the value; full table in §4.8; explicit ready
  credits for every leader multicast incl. non-source receivers.
- Goal lens: structural exit gates (phase 1: one device op after the VLM; phase 2: exactly one device op per replay),
  PI05_MEGAKERNEL knob (default off until phase-2 exit), model stamp asserted in test_pcc_pi05_fused.py, refusals (shape,
  B>1, mesh/dp/pipeline, no L1 cut), replay L1 guard, server WPs P1-6 / P2-5, LIBERO speed gates binding, prompt-length
  edge cases, size <= 128 KB gates in P1-3/4/5. Phase 1 is intermediate; phase-2 failure = needs_user.
- Phase 2: pad rows never enter the resident KV region (25 / 18 key-tile chunking stands); "upper" column is not a bound;
  GR00T-realised range 2,522-5,079 us per VLM layer straddles TTNN 2,437 -> P2-1 decides.

Open / next: WP-P1-0. Open question for the user (DESIGN.md §8): batch-1-only megakernel server acceptable?

## 2026-09-30 01:55:20 KST -- session mk1-r0-s0: phase-1 implementation v1 (kernels + host), offline

Read first: JOURNAL.md, DESIGN.md v2, GR00T allgather.hpp / weight_stream.hpp / row_layernorm.hpp (ln_llk fidelity
wrappers), mk_k4 / f0 summaries, the memory notes named in the task. No device time yet in this session.

Implementation v1 (models/experimental/pi0_5/tt/megakernel/): one generic_op over the whole 11x10 grid, three kernels
(kernels/mk_ncrisc.cpp weight streams on NoC 0, mk_brisc.cpp every exchange on NoC 1, mk_trisc.cpp all arithmetic),
constants shared through kernels/mk_defs.hpp which geometry.py parses (one source of truth). Deviations from DESIGN.md
§4.2-4.8, each chosen to cut the number of distinct mechanisms for a first correct build (speed levers left for later):
- qkv on 40 "pair" cores (x in 0..9, y in 0..3; each owns dh tiles j and j+4 of one Q head / K / V): RoPE is local, no
  pair exchange (R2 pair step removed); 2x the per-core qkv compute (~4 us instead of ~2).
- attention: 80 units (base) / 48 (LIBERO) as designed, but ONE merger per (head, row) (the kc = 0 unit) that folds the
  NCH parts with diag(w_i / L) HiFi4 matmuls (D_i packed fp32) instead of the dh-split merge; m / l travel as full tiles.
- ctx travels as bf16 (not bfp8): CB_IN0 is 64*RT bf16 pages; no two-format CB needed (L1 union allows it).
- adaRMS r is a FULL tile (row value in every column) from H0 (sum of x^2 by fp32 DST mul-accumulate, then @ (1/1024)
  tile, + eps, rsqrt); every epilogue (r*acc + c, gates, RoPE, GeGLU) runs on the SFPU in fp32 DST with row-broadcast
  constant tiles streamed per (step, layer) in a third ring (CB_WC, one 8-tile page per layer).
- landing CBs are used in full-capacity cycles only (every pointer at the base between transactions), so remote writers
  address a peer's CB base (identical layout on every core). Multicast rounds carry explicit ready credits; the
  point-to-point deposits rely on the layer's dependency chain (argument per buffer in mk_brisc.cpp header).
- rings never let a waited region straddle the ring end (W8 16 pages: waits of 8 / 2 pages; W16 8 pages: page-by-page;
  H0's per-step W16 stream padded to 16 pages).
Host checks (CPU): host_model.loop_decomposed (the kernel's decomposition: folds, chunked flash + diag merge, 2-D MLP,
fp32 residual, folded out-proj) vs loop_reference with the real pi05_base expert weights and random prefix K/V:
fp64 PCC 1.0000000000000002, max |diff| 5.1e-14 (base shape); fp32 PCC 0.9999984 (base, round-off amplified over 180
layers) / 0.99999999996 (LIBERO). Scratch: scratchpad/mk/host_check{,64}.{py,log}.
Mock-cluster compile (size_check.py, private TT_METAL_CACHE): base 69,992 B, LIBERO 68,696 B incl. args / CB configs
(brisc 18,192, ncrisc 2,224, trisc0 21,568, trisc1 18,384, trisc2 7,424) vs the 128 KB gate -> docs/megakernel/impl/
size_check_v1_*.json. CB union per core (geometry.cb_union_bytes): base 1,156,768 B, LIBERO 851,616 B (to be checked against measured free L1).
Next: first device bring-up (debug stop after 1 generation, watcher on), then 18, then 180 generations.

## 2026-09-30 02:56:10 KST -- phase 1 on the device: bring-up, integration, gates (session mk1-r0-s0)

Results dir: docs/megakernel/impl/results/ (copies of the scratchpad JSONs named below).

Bring-up (tests/megakernel/mk_bringup.py: megakernel alone, real pi05_base expert weights, synthetic bf8 prefix K/V,
the fused graph's real mask / RoPE inputs, host oracle = host_model.loop_decomposed on the device's exact inputs):
- b1 (01:56, 1 generation, watcher): HUNG 900 s, killed + reset. Signature (watcher dumps #16..#713, identical): every
  compute core BRISC at XFLG (waiting for H0's x flag); H0 BRISC CWFW (cb_wait_front CB_ROUT), H0 TRISC0 UPAD,
  TRISC1 MWDD, TRISC2 K. DPRINT trace (b2, PI05_MK_TRACE=1, TT_METAL_DPRINT_CORES=(10,9)): H0 NCRISC had delivered 8
  pages; H0 math and pack reached "inproj_nb 0", the unpacker never did -> it waited on CB_SCR16, which the BRISC had
  produced once (noise) and the TRISC packer then produced: the TRISC pack side keeps a LOCAL copy of the CB's
  received count, so its push rewrote the shared count to a value the unpacker had already consumed. Rule adopted:
  every CB has ONE producer RISC and ONE consumer RISC per core (H0's noise -> CB_Q; H0's x_new -> CB_PART / CB_HG,
  which the BRISC copies into its own CB_IN0); pinned by test_cpu_one_producer_per_cb_on_h0.
- b3 (1 generation): x after layer 0 vs host fp32 PCC 0.99992 (every owner column >= 0.99985), finite, 1760 ms first
  launch (compile). b4 (18 generations = step 0 + tail): x_t PCC 0.999992. b5 (180): runs 18.36 / 18.11 ms untraced
  wall, 3 launches bit-identical; x_t PCC 0.981 vs the fp32 host on SYNTHETIC random K/V (chaotic amplification; the
  real-input gates below are the measure).

Integration (PI05_MEGAKERNEL=expert; common/device_open.py opens with worker_l1_size 1,395,712; the model builds the
host params from the checkpoint, one ExpertMegakernel per shape sharing the 4.6 GB of weight arenas; the traced graph
is the ttnn prefix + ONE generic_op). All with bfp8 matmuls at LoFi unless noted; HiFi2 became the default at 02:38.
- LIBERO golden (L1.json, LoFi): PCC7 mean 0.99988, min 0.99978 (gate >= 0.9995 / 0.999: PASS); traced; ten replays
  bit-identical; call 65.59 ms median / replay 64.36 (shipped 76.77 / 75.64): PASS < 76.77.
- base padded prompts (B1.json LoFi, B2.json HiFi2) vs the fixed torch reference, seeds of test_pcc_pi05_fused.py
  (n_real 40, 97, 12, 150, 201, 224): shipped (B0.json, same session) 0.99843 0.98397 0.93811 0.99549 0.98017 0.98793;
  megakernel LoFi 0.99814 0.97545 0.93818 0.99843 0.98777 0.98659; HiFi2 0.99917 0.97190 0.94833 0.99850 0.99356
  0.98810. Gate "per seed within 0.005 of the current path": FAILS on seed 2 (n_real 97) in both arms (-0.0085 LoFi,
  -0.0121 HiFi2); all other seeds pass. Mask live (PCC -0.68 vs unmasked), pad ids invisible bit-identical, default
  mask bit-identical, ten replays bit-identical; call 70.66 ms (LoFi) / 70.75 (HiFi2) vs shipped 84.18: PASS < 84.2.
- V1.json (both paths in ONE process under the cut, 18 seeds, HiFi2): mean PCC vs reference megakernel 0.98253,
  shipped 0.97956; megakernel better on 14/18 seeds, worse by > 0.005 on seed 2 (-0.0121) and seed 109 (-0.0062);
  megakernel vs shipped device outputs 0.9924..0.9997. The shipped path under the 64 KiB cut: identical per-seed
  values to B0 and call 83.75 ms (P1-0 gate <= 84.2 + 1 %: PASS).
- Structural gate (profiler op lists, structural_{base,libero}.json, 4 replays each): 892 ops per replay (shipped
  2551); the ONLY op after the VLM's last K/V-cache update is the megakernel GenericOp (mk_trisc.cpp) on 110 cores,
  0 ops after it: PASS. Its device kernel time: base 17.48-17.51 ms (gate < 31.26: PASS), LIBERO 16.05-16.10 ms
  (gate < 27.85: PASS). Kernel-config footprint 70.7 KB (gate <= 128 KB: PASS).
- Alternating prompts / shape switch (tests/megakernel/verify_alternating.py, alternating_A1.json): every call
  bit-identical to its fresh-model reference at both shapes, shape switch clean, golden PCC7 mean 0.999884: PASS.
- CPU: tests/megakernel/test_cpu_mk.py 10 passed; tests/test_fused_host.py 23 passed;
  tests/pcc/test_reference_vs_openpi.py 1 passed.
Decision: PI05_MK_FID8 default hifi2 (bfp8 matmuls at HiFi2): no measurable time cost (70.75 vs 70.66 ms), mean PCC up.
- 03:17 soak 1 (soak_base_run1.log): 20 consecutive base processes (build + warm-up + capture + 31 calls, timeout
  600 s each), 0 hangs, every process's 31 calls identical; 19 share digest 7f070cd96b1e0d17; run 6 (02:57:45) has
  41ed0707c8702edd because I edited kernels/** at 02:57 while the soak held the card (a process-rule violation: its JIT
  compiled the half-edited fp32-score variant). The soak is therefore NOT a clean 20 and must be redone on the final
  kernel; the edit was reverted at 02:57:35 and later reapplied off-card as the A/B below.
- fp32 attention scores A/B (V2.json, L2.json; S kept fp32, exact SFPU S - m, fp32 m): base mean vs reference
  0.98247 (HiFi2 without: 0.98253), seed 2 0.97366; LIBERO PCC7 0.999886 / 0.999777; +0.55 ms (71.30 vs 70.75).
  No measurable gain -> REVERTED (kernels back to d82788e + HiFi2 default).
- Expert-only oracle (O1.json, tests/megakernel/mk_expert_oracle.py): host_model.loop_reference (fp32 checkpoint
  weights) fed with the DEVICE's prefix K/V caches, mask, RoPE tables and noise of the shipped run (the ttnn prefix
  is shared and deterministic), 18 seeds. vs this oracle: megakernel mean 0.99983 (min 0.99941), shipped 0.99737
  (min 0.98843); the megakernel is closer on 18/18 seeds (seed 2: 0.99978 vs 0.99810). The oracle itself vs the fp32
  torch reference: seed 2 0.97553 (shipped 0.98397, megakernel 0.97190), seed 3 0.95243, seed 109 0.97855.
  => The "per seed within 0.005 of the current path vs the torch reference" gate fails on seed 2 for the megakernel
  AND would fail for a perfect fp32 expert (-0.0084): the reference distance on those seeds is set by the device
  prefix (SigLIP / VLM / bf8 caches); the shipped expert's own error happens to offset it on seed 2. I do not
  redefine the gate; it is reported as failing with this evidence (needs the user's ruling).
- Edge cases (E1.json base n_lang 1 / 224 / 128 / 150, E2.json LIBERO 32 / 1; seeds 11..): megakernel minus shipped vs
  the torch reference +0.0028 +0.0035 +0.0062 +0.0012 | +0.0001 +0.0002 (gate "no worse": PASS all six); vs the
  expert oracle the megakernel is closer on all six (>= 0.99991 base, 0.99998 LIBERO).
- Soak 2 (soak_base_run2.log, 03:21-03:34, kernel md5s verified unchanged before and after): 20/20 consecutive base
  processes rc=0, one output digest 7f070cd96b1e0d17, every process's 31 calls identical: PASS (no hang in 20).
- Server (P1-6): fastapi / uvicorn are not in the tt-metal venv; installed into scratchpad/pylib (not the GR00T tree).
  server/app.py opens through common/device_open.py and refuses PI05_MEGAKERNEL with B > 1 / mesh / dp at startup with
  a log line before any device open (server_refusals.txt; pipeline was already refused by load_config). Served
  (served_S_{expert,off}.json, same session, 30 POSTs of the smoke payload): megakernel inference median 70.72 ms
  (total 72.10), shipped 84.07 (total 85.45): PASS < 84.02; smoke_test PASS both; /info carries backend + program.
- Replay L1 guard (l1_guard_G1.json): PI0ModelTTNN records the L1 allocator signature at capture and raises before
  execute_trace when it changed; raised on a 1-tile L1 allocation, replay after the free bit-identical, cost 35 us
  per call (< 0.1 ms: on in production, PI05_MK_L1_GUARD=0 disables). Free L1 at capture: largest free block
  1,253,888 B per bank vs the CB union 1,156,768 B (base, 97,120 B headroom) / 851,616 B (LIBERO).
- P1-4 (mk_vs_shipped_golden_VG.json): megakernel vs shipped x_0 on the 8 golden LIBERO observations 0.99987..0.99992
  (gate >= 0.999: PASS); PCC7 vs openpi higher than the shipped path on 8/8.

### Phase-1 exit gate table (03:41; files in docs/megakernel/impl/results/)
| gate | result | evidence |
|---|---|---|
| structural: ONE device op after the VLM, covering loop + action io | PASS (892 ops/replay, 0 after the megakernel) | structural_{base,libero}.json |
| stamp asserted in test_pcc_pi05_fused.py | PASS (backend "expert" from the object) | L1.json / B2.json "stamp" |
| PCC7 vs openpi mean >= 0.9995, min >= 0.999 | PASS 0.99988 / 0.99978 | L1.json, alternating_A1.json D |
| base per seed within 0.005 of the current path vs torch ref | FAIL on seed 2 (0.97190 vs 0.98397); 5/6 pass | B0.json, B2.json, O1.json (a perfect fp32 expert also fails: 0.97553) |
| ten replays bit-identical | PASS both shapes | L1.json, B2.json |
| alternating prompts / shape switch | PASS (all bit-identical to fresh refs) | alternating_A1.json |
| no hang in 20 consecutive runs | PASS (soak 2) | soak_base_run2.log |
| prompt-length edge cases no worse than current | PASS 6/6 | E1.json, E2.json |
| existing suites green | test_fused_host 23/23, test_reference_vs_openpi 1/1, mk CPU 10/10, test_pcc libero PASS; test_pcc base RED on its own min-0.95 gate for BOTH paths (megakernel 0.948, shipped 0.938, seed 3) | B0.json, B2.json |
| expert loop device time < 31.26 / 27.85 ms | PASS 17.50 / 16.08 ms | structural_*.json |
| whole call < 84.2 / 76.77 ms | PASS base 70.75 ms (B2.json, final config), LIBERO 65.76 ms (L3.json, final config; PCC7 0.999884 / 0.999778, traced, 10 replays bit-identical) | B2.json, L3.json |
| size <= 128 KB | PASS 70,696 / 71,032 B | size_check (impl/size_check_v1_*.json is the first build) |
| shipped path under the 64 KiB cut | PASS (identical PCCs, 83.75 ms) | V1.json |
| server P1-6 | PASS (smoke, 70.72 ms served, stamp, refusals) | served_*.json, server_refusals.txt |

Open for the user (not redefined by me): the base per-seed gate fails on seed 2 and the base pytest's own min-0.95
floor is red for both paths; O1.json shows the megakernel's expert loop is closer to an fp32 oracle than the shipped
one on 18/18 seeds, and that a perfect expert would fail the same gate. Next after a ruling: phase 2 (WP-P2-0: the
whole sample_actions binary / L1 overlay; the phase-1 program is 70.7 KB of the 128 KB budget).
Known deviations from DESIGN v2 in the build (speed levers, not correctness): pair cores instead of Q/K/V producers +
pair exchange, single merger per (head, row), bf16 ctx (no two-format CB), no dual-NoC hub multicast; WP P1-1 / P1-2
harness gates were not run (the integrated loop meets the speed gates with 44 % / 42 % margin).

## 2026-09-30 13:57:28 KST -- verify-p1-r0: independent verification of phase 1 (verifier session, no code changed)

Everything re-run by the verifier's own scripts (docs/megakernel/verify_p1_r0/scripts/, results in ../results/), HEAD
e8cbc78, private TT_METAL_CACHE, both arms in the same session 13:07-13:56. The PI05_MEGAKERNEL env var picked the arm.
Source md5s (tt/**/*.cpp,hpp,py) were the same at 13:05 (checksums_start.txt), before the soak (13:37) and after it (13:56).
- Structural (tracy, S_verifier.json; P_*.json): expert: 21 traced sessions, 892 ops each, 36 KV-cache ops, and exactly
  ONE op after the last one: GenericOp mk_trisc.cpp on 110 cores. No session was missing a device duration.
  Off (shipped): 2551 ops, 1660 after the last KV op (matmul / layernorm / fused_attn / row_rsqrt / geglu_rc ...). The prefix is the same 891 ops
  in both arms, so the A/B arms are different programs. Megakernel kernel time median 17.503 ms base / 16.070 LIBERO
  (20 replays + first; gates < 31.26 / 27.85). Binaries from the CSV: base 18,812+2,300+21,612+18,484+7,692 B (<= 128 KB).
- Amended base gate (O_verifier.json). The oracle was RE-DERIVED independently of host_model: the torch reference's own
  denoising.sample_actions / forward_expert (fp32), fed each arm's own device prefix K/V (bf8 caches read back) and the device noise.
  Positive control: fed the reference's own VLM cache, it reproduces the reference bit-exactly (seeds 1, 2).
  20 seeds (6 of test_pcc + 14 new, n_real incl. 1 / 224 / 128 / 150). Result: megakernel closer to the oracle on 20/20. Mean 0.99980
  (min 0.99942) vs shipped 0.99649 (min 0.98784). Seed 2: 0.99979 vs 0.99806. Prefix K/V bit-identical across the arms on all 20.
  vs the full fp32 reference: mean mk 0.97350, shipped 0.96853. The base pytest's own min-0.95 floor stays red for BOTH arms (seed 3: mk 0.94833,
  shipped 0.93811; pre-existing on main).
- openpi golden PCC7 (A_expert_libero.json): mean 0.999884, min 0.999778 (shipped same session 0.999839 / 0.999712): PASS.
- Replays: 10 re-upload calls + 10 raw execute_trace, bit-identical to the first call, both shapes and both arms. The 20 profiled
  replays were also identical.
- Alternating / shape switch vs fresh-model references (ALT_expert.json, own design): 18/18 calls bit-identical across base
  A B C D x2 (n_real 12/150/1/224), then LIBERO E F E F F E, then base again D A C B. The positive control was max PCC between refs 0.952.
  The shipped arm also passed 18/18. The named tests/megakernel/verify_alternating.py also passed (OVERALL all_ok=True, ALT_named_expert.log).
- Soak (soak_results.txt, hold4.log): 20/20 consecutive processes rc=0, each doing warm-up + capture + 31 calls that alternate two
  prompts. Every run gave all_equal=true and the same digest c9a66e4da14f7563, about 40 s each. No hangs.
- Latency with a clock witness (A_*.json): per call base 70.97 ms vs shipped 84.13, LIBERO 65.71 vs 76.91. Replay 69.54 / 64.38 vs
  82.83 / 75.63. The witness is time.time and monotonic over the 30-call loop (= the perf_counter sum to within 0.1 ms), plus a 10 s sustained replay count
  (144 vs 121 replays at base) and tt-smi aiclk 1350 MHz throughout both arms.
- CPU suites: test_fused_host + test_cpu_mk + test_reference_vs_openpi 34 passed (cpu_suites.txt).
Not re-run by the verifier: served latency (P1-6), LIBERO n_lang 32 / 1 edge cases, the replay L1 guard.
Open: the 2026-09-30 gate amendment is NOT yet recorded in DESIGN.md §7 (still the old "within 0.005" text).
Verdict: phase 1 ACCEPTED on the amended gates.

## 2026-09-30 14:14:22 KST -- fix-p1-r0: blocking review defects fixed, gate amendment recorded, unverified gates re-run

Code commit 6b8632c (kernels unchanged: kernel_digest 328761c8a1ce3fd9, same as the verified build). Results in
docs/megakernel/fix_p1_r0/results/.
- BLOCKING 1 (PI05_KV_DTYPE=bf16 not refused; bf16 pages landed at the bfp8 stride and overran CB_KV): fixed.
  geometry.megakernel_refusal(kv_dtype, num_steps) is called in PI0ModelTTNN._init_megakernel before any parameter
  build and in the server startup refusals; ExpertMegakernel.program() also refuses unless there are 18 layers of
  bfloat8_b (K, V) caches.
- BLOCKING 2 (num steps != 10 not refused; the one-dt guard passes 5 / 16 / 20): fixed. Same refusal function, plus
  ExpertMegakernel.__init__ refuses len(params.dts) != G.N_STEPS (parsed from mk_defs.hpp) before any upload.
  Server evidence (server_refusals.txt, under the device lock, no device opened): PI05_KV_DTYPE=bf16,
  PI05_NUM_STEPS=20 and =5 each log "refused at startup" and exit rc=3.
- CPU tests added to tests/megakernel/test_cpu_mk.py: test_cpu_refuses_bf16_kv_and_other_step_counts (every entry:
  pure check, model init, ExpertMegakernel step check, program() dtype / layer-count check);
  test_cpu_merge_dst_budget_and_host_invariants; test_cpu_decomposition_equals_reference_loop[base, libero]
  (loop_decomposed vs loop_reference over the whole 10 x 18 loop, synthetic weights, padded prompt: PCC
  0.99999999999997 / relmax 2.46e-7 base, 2.32e-7 LIBERO; the positive control that drops chunk 0's keys from the
  decomposed arm only gives relmax 0.043 / 0.046, PCC 0.99921 / 0.99911). CPU suites: 38 passed (cpu_suites.txt).
- Non-blocking items fixed (host side only): Shape.check now requires NCH <= 6 (merge: DST 0 + weights 1..NCH + temp 7;
  test with an NCH = 7 shape). check_roles asserts that mergers and units with y < 8 are MLP cores (K/V prefetch
  path). It also asserts that the multicast destination counts parsed from mk_brisc.cpp (rx 80, rm 64, ro 32, rk 8*RT,
  row 7, ra 109, col NU-1) equal area minus an in-rectangle sender, for every row and column instance, and that
  x_round_receivers equals the counts derived from the roles (80 / 72). The positive control is a doctored literal,
  which gives a mismatch. The geometry.py and arena.py docstrings now cite test_cpu_mk.py.
- DESIGN.md §7: the user's 2026-09-30 amendment (phase 1 and phase 2 text, date, reason, acceptance evidence);
  the old "within 0.005" text is struck through and points at the amendment.
- Re-run on device at 6b8632c (hold.log, 14:05-14:12, WITH_DEVICE_RESET_AFTER=1, private TT_METAL_CACHE):
  - Digest regression (soak_one.log): 31 calls all_equal, digest c9a66e4da14f7563. This is the verifier's soak digest,
    so the outputs did not change.
  - LIBERO n_lang 32 / 1 edge cases (E2.json): mk vs ref 0.99960 / 0.99948, shipped 0.99949 / 0.99930. No worse: PASS.
    Values are identical to impl/results/E2.json. Call 65.71 vs 76.90 ms.
  - Replay L1 guard (G.json), both shapes: it raised on a 1-tile L1 allocation, the replay after the free was
    bit-identical, and the guard costs 38.1 / 38.7 us. Largest free block 1,253,888 B vs CB union 1,156,768 / 851,616 B.
  - Served P1-6 (S_*.lat.json), arms alternated expert / off / expert / off. Inference median 70.93 and 70.78 ms vs
    shipped 83.94 and 84.10 ms (gate < 84.02: PASS). smoke_test PASS on all 4. /info reports backend expert with
    kernel_digest 328761c8a1ce3fd9, and off with program None.
Open (not changed by me):
- tests/pcc/test_pcc_pi05_fused.py base still fails its own 0.95 per-observation minimum on seed 3 for BOTH arms
  (mk 0.94833, shipped 0.93811; this was already true on main). That floor is the fp32-torch-reference distance that
  the amended gate replaced. Changing the test's floor is the user's call.
- The mk_brisc.cpp:8 header still cites a nonexistent tests/megakernel/test_cpu_mk_protocol.py, and the per-buffer
  argument for the 7 uncredited deposits is not written anywhere. A comment edit changes kernel_digest, so this is
  deferred to the next kernel change.
- Other non-blocking items: owner x/r landing ordered only by causality; mixed-width matmuls with no static guard;
  CB_RTOK pushed without a reserve; debug x dump on every launch.
Next: phase 2 (WP P2-0).

## 2026-09-30 15:17:00 KST -- verify-p1-r1: second independent verification of phase 1 (verifier session, no code changed)

HEAD 51a9ba3 (kernel_digest 328761c8a1ce3fd9). New scripts (docs/megakernel/verify_p1_r1/scripts/, not verify_p1_r0's), results in
../results/. Four holds 14:21-15:15, both arms in the same session. PI05_MEGAKERNEL picked the arm and the device was opened by
common/device_open.py. Each arm had its OWN TT_METAL_CACHE. md5 of every tt/ and common/ source (sums_*.txt) was the same
(ALL 1eee2dfc...) at the start and end of holds 1-2 and before and after the soak.
- A/B arms differ (kernels_{expert,off}.txt): the expert cache compiled mk_brisc / mk_ncrisc / mk_trisc and none of the shipped
  expert's generic_op kernels (compute / reader / writer) or minimal-matmul kernels. The off cache compiled no mk_* kernel.
  From the profiles: the prefix is the same 891 op codes in both arms, then 1 op (expert) vs 1660 ops (off).
- Structural (tracy, S_struct.json): expert has 21 replay sessions per shape, 892 ops each, 36 UpdateKVCache ops, and exactly 1 op
  after the last one. That op is a GenericOp on 110 cores, with compute mk_trisc.cpp and data movement mk_brisc.cpp + mk_ncrisc.cpp,
  and one program hash per shape. No device durations were missing. Kernel time median 17.495 ms base (17.459-17.533), 16.079 LIBERO
  (gates < 31.26 / 27.85). Off: 2551 ops, sum of post-cache kernel durations 30.32 / 26.83 ms.
- Size (size_check_r1.json, mock-cluster compile): 70,728 B base / 71,064 B LIBERO (<= 128 KB). Real-device ELFs in my cache
  have the same text+data as the mock ELFs of the same hash. The profiler build is slightly larger (brisc 18,812 vs 18,732).
- Amended base gate (O_r1.json, 22 seeds). These are the 6 of test_pcc plus 16 NEW seeds with n_lang on and around the 32-token
  tiles (1, 2, 31, 32, 33, 63, 64, 65, 96, 128, 129, 159, 191, 192, 223, 224). The oracle is re-derived (scripts/oracle.py):
  my own 10-step Euler loop around the reference's fp32 velocity function, with the mask and positions built independently.
  It is fed each arm's own device bf8 prefix K/V (read back) and the device noise.
  Controls: (1) fed the reference's own VLM cache, it reproduces ref.sample_actions (PCC 1 - 5e-14, max|d| 5.2e-7; the
  residual is dt rounding). (2) K/V and noise are bit-identical across the arms on 22/22 seeds, and the oracles are equal.
  (3) Negative control: an oracle fed another seed's K/V scores mk 0.93 / 0.88 / -0.08, against 0.9997 for the right K/V.
  Result: the megakernel is closer on 22/22 seeds. Mean 0.99974 (min 0.99916) vs shipped 0.99596 (min 0.98575).
  The smallest margin is +4.3e-5 (seed 708, n 65). Seed 2: 0.99979 vs 0.99806.
  vs the full fp32 reference: mean mk 0.97535 vs shipped 0.96921, mk closer on 19/22.
- openpi golden PCC7 (A_expert_libero.json): mean 0.999884, min 0.999778 (shipped 0.999839 / 0.999712): PASS.
- Replays: 10 calls after all other prompts and 10 raw execute_trace were bit-identical to the first call, in both arms and both
  shapes; the 20 profiled replays were too. Output poisoning: I wrote 7.0 into the trace's output buffer (the read-back was all
  7.0), then called with another prompt. The output was bit-identical to that prompt's earlier output with no 7.0 left, so the
  traced program writes the output.
- Alternating / shape switch (ALT_{expert,off}.json, own design): base A B C D D2 A D2 D C B, where D / D2 have the same length
  and different content, so the attention inputs are not rewritten between them. Then LIBERO E F E F F E, then base B D2 A C.
  All 20 calls were bit-identical to fresh-model references. The fresh refs were also bit-identical to the other process's
  (d_arm) outputs for the same seeds. The refs differ pairwise (max PCC 0.952). The named repo test
  tests/megakernel/verify_alternating.py under expert: OVERALL all_ok=True. The task's scratchpad copy (09-29) passed with off.
- Soak (soak_results.txt): 20/20 consecutive processes, genuine rc=0. Each did warm-up + capture + 30 calls cycling n_lang 1 / 128 /
  224. All had all_equal and finite output, the same digest 099ce3592052e5c4, and took 36-40 s each. No hang.
- Latency with a clock witness (A_*.json). Per call median: base 70.95 ms (MAD-based se 0.04) vs shipped 84.14; LIBERO 65.76 vs
  76.90. Replay: 69.54 / 64.39 vs 82.80 / 75.63. For the 30-call loop, time.time_ns and monotonic_ns agree with the perf_counter
  sum to within 1 ms. A fixed 10 s time.time window gave 144 vs 121 replays (base) and 156 vs 133 (LIBERO). tt-smi aiclk was 1350
  before and after every bench.
- Suites: CPU 38 passed (cpu_suites.txt). tests/pcc/test_pcc_pi05_fused.py: libero PASS in both arms. Base FAILS its own min-0.95
  fp32-reference floor in BOTH arms on seed 3 (mk 0.94833, shipped 0.93811), which is pre-existing; the per-seed values are identical
  to O_r1.json.
Findings (non-blocking):
- DESIGN §4.12 refusal (d), "device not opened with the 64 KiB cut", is not implemented in the model. The scratchpad
  verify_alternating.py opens the device without the cut. Under expert it failed cleanly (no hang, 36 s) inside tt-metal with
  TT_FATAL "Program size (70752) too large for kernel config buffer (70656)". That is a 96 B margin by accident, not a named
  refusal (ALT_named_scratch_expert_nocut.log).
- The `echo "$(date) ... rc=$?"` idiom reports the rc of `date` (always 0). The rc columns of hold1/2/4.log here, and
  verify_p1_r0's hold1-3.log, are therefore not exit codes; every result above is confirmed from its output file instead. The soak
  used `rc=$?` on its own line, so its rc values are genuine.
Verdict: phase 1 ACCEPTED on the amended gates (all re-run gates pass). Open for the user: the base pytest floor (see above).

## 2026-09-30 15:24:09 KST -- integrate-p1: phase-1 expert megakernel becomes the DEFAULT path (session start)

User request relayed with this task: "a. 일단 HF에 update. 이후 다음 phase 작업 진행" (update HF first, then the next phase).
This session = integrate-p1 (default switch, docs, gate re-run, LIBERO closed loop); the HF push itself is the Ship phase.
Code commit 488fe36 (kernels unchanged):
- FusedConfig.megakernel default "expert" (env unset / empty / dataclass default). resolved(n>1) turns only the UNSET
  default off on a multi-chip mesh (no megakernel exists there); an explicit PI05_MEGAKERNEL=expert still refuses on a
  mesh. PI05_MEGAKERNEL=off = the previous shipped path, kept as the comparator knob.
- DESIGN §4.12 refusal (d) implemented (verify_p1_r1 finding): PI0ModelTTNN.megakernel_device_refusal names a device
  opened without the 64 KiB worker-L1 cut (L1 + L1_SMALL total per bank > 1,395,712) before any upload.
- server: _Batcher / _DPRouter carry each request's own lang mask into sample_actions_fused (the batcher serves EVERY
  request, batch 1 included, and used to drop it -> tokens != 0 fallback); the batch / DP warm-ups pass the mask too.
- tests that opened the device directly (perf, rollout_libero, action_divergence) now use device_kwargs (the cut).
- CPU: tests/test_server_masks.py (5), test_cpu_mk new default/resolution/refusal-(d) tests; 43 passed with
  test_fused_host (fastapi from scratchpad/pylib).
Device gate plan (scripts copied from verify_p1_r1 into scratchpad/ip1, arms "default" = PI05_MEGAKERNEL UNSET and
"off" = PI05_MEGAKERNEL=off; the fp32 torch reference outputs ref_base.pt are reused from vp1r1: reference/ unchanged
since aa7bf50).

## 2026-09-30 16:08:26 KST -- integrate-p1 (resumed session): state found after the interrupted session

- The previous integrate-p1 process died during hold2 (tracy profile, started 15:51:31; no HOLD END in the guard ledger;
  card probed healthy 16:06:43). Lock free at resume, no device process running.
- hold1 at cfe9fa2 (15:40-15:50, reset-after, source md5 ALL 403253dd... identical at hold start/end) is COMPLETE and kept:
  scratchpad/ip1/out/{NOCUT,A_*,ALT_*,O_ip1}.json (out_first/ = an earlier hold1 at 488fe36, superseded by the cfe9fa2 code change).
- hold2 (profiles) is INCOMPLETE (only P_default_base.json, no ops CSV): discarded and re-run. hold3 (soak) and hold4
  (pytest, served) never started: run now. Working-tree edits of README.md / DESIGN.md are doc-only (sums unaffected).
