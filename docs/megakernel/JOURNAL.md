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
| whole call < 84.2 / 76.77 ms | PASS 70.75 / 65.59 ms (LoFi) - 66.08 (HiFi2 fp32-S arm) | B2.json, L1.json, L2.json |
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
