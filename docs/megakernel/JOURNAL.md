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

## 2026-09-30 16:49:56 KST -- integrate-p1: gate re-run results (holds 1-4), arms "default" = PI05_MEGAKERNEL UNSET, "off" = comparator

Code: hold1 at cfe9fa2, holds 2-4 at a46beb5 (a46beb5 only changed server/app.py docstring + /info text). md5 of every
tt/ + common/ source = ALL 403253dd516ba74bd937e6ac4a329009 at the start and end of every hold (sums_*.txt), kernel_digest
328761c8a1ce3fd9 (the verify_p1_r1 digest). Results copied to docs/megakernel/integrate_p1/results/ (from scratchpad/ip1/out/).
- Refusal (d) (NOCUT.json, NOCUT_final.json): a device opened without the cut is refused by name before weight conversion
  (model_built_without_cut False, error names the cut); with the cut, no refusal.
- A/B arms differ: the default cache compiled mk_brisc / mk_ncrisc / mk_trisc and not the off path's compute / reader / writer /
  dm_in0_sender / dm_in1_sender_out; the off cache no mk_* (kernels_{default,off}.txt).
- Structural (tracy, integrate_p1/results/S_struct.json; until 2026-10-01 misfiled under hold4_serve_invalid/, moved by
  publish-p2): default 21 replay sessions per shape, 892 ops each, 36 UpdateKVCache, exactly 1 op after
  the last one = GenericOp on 110 cores (mk_trisc / mk_brisc + mk_ncrisc), one program hash per shape, no missing durations.
  Megakernel kernel time median 17.490 ms base (17.448-17.521), 16.071 ms LIBERO (16.046-16.128). Off: 2551 ops, 1660 after the
  last cache write, post-cache device sum 30.31 / 26.83 ms. The 891-op prefix sequence is identical in both arms (both shapes).
- Amended base gate (O_ip1.json, 22 seeds, oracle fed each arm's own device K/V): default closer on 22/22; mean 0.99974
  (min 0.99916) vs off 0.99596 (min 0.98575); smallest margin +4.3e-5 (seed 708). Controls as verify_p1_r1 (K/V and noise
  bit-identical across arms 22/22; oracle reproduces ref.sample_actions to PCC 1-5e-14; wrong-K/V negative control 0.93/0.88/-0.08).
- openpi golden PCC7 (A_default_libero.json): mean 0.999884, min 0.999778 (off 0.999839 / 0.999712). PASS.
- Replays / poisoning (A_*.json): ten calls and ten raw execute_trace bit-identical to the first in both arms and shapes; the
  20 profiled replays too (P_*.json). Output buffer poisoned with 7.0 -> next call equals that prompt's earlier output, no 7.0.
- Alternating / shape switch (ALT_{default,off}.json): 20/20 calls bit-identical to fresh-model references, refs pairwise
  distinct (max PCC 0.952), fresh refs equal to the other process's outputs. Named repo test verify_alternating.py under the
  default: all_ok True (ALT_named_repo_default.json).
- Soak (hold3.log): 20/20 processes rc=0, each 31 calls cycling n_lang 1 / 128 / 224, all_equal, finite, digest
  099ce3592052e5c4 (= verify_p1_r1's), 40-43 s each. No hang.
- Latency (A_*.json, aiclk 1350 before/after each bench): per call median base 70.68 ms vs off 84.26; LIBERO 65.84 vs 76.72.
  Replay 69.54 / 64.41 vs 82.84 / 75.65. 10 s time.time window: 144 vs 121 (base), 156 vs 133 (LIBERO) replays.
- tests/pcc/test_pcc_pi05_fused.py (pytest_pcc_{default,off}.log): libero PASS in both arms; base FAILS its own 0.95 min
  fp32-reference floor on seed 3 in BOTH arms (default 0.94833, off 0.93811), pre-existing, values identical to O_ip1.json.
- CPU suites: 44 passed (cpu_suites.txt, 15:54); test_server_masks.py 5 passed again at a46beb5.
- Served A/B in hold4 is INVALID and discarded (moved to scratchpad/ip1/out/hold4_serve_invalid/): the loop's `kill $P`
  hit the subshell of the `arm` shell function, not uvicorn, so the first (default) server kept port 20000; all 4 "arms"
  measured it (every /info says backend expert) and the three later servers blocked in UMD "Waiting for lock
  CHIP_IN_USE_0_PCIe held by PID 889382" and were left holding /dev/tenstorrent/0 after the hold (guard: "orphaned device
  users"). I killed them by pid at 16:48 (card then probed ok). Only the first default server's numbers are genuine
  (S_default_164515: smoke PASS, inference median 70.97 ms, mask probe pass). Re-run as hold5.sh: setsid per server,
  process-group stop, port + process table confirmed empty, /info backend must equal the arm before measuring.

## 2026-09-30 17:13:30 KST -- integrate-p1: served A/B, LIBERO closed loop, demo (session end)

- hold5 (16:49-16:53) was a second invalid served attempt: GNU `timeout` makes itself a process-group leader, so the
  setsid group kill missed uvicorn. Its arm check caught it (arm 2 /info said expert, measurements skipped). Only its
  genuine arms are kept: default_164949 (inference median 70.855 ms) and off_165209 (83.92 ms). I killed leftover
  servers by pid at 16:51 and 16:53. The eval hold's probe at 16:53:52 was ok.
- hold6 (17:05-17:08, hold6.sh: stops every uvicorn of the app by pid, then confirms the process table and port are
  empty; /info backend must equal the arm): alternated default / off / default / off, each arm confirmed by /info.
  timing_ms.inference median 70.925 / 70.875 ms (default) vs 84.04 / 84.10 ms (off); total 72.31 / 72.28 vs 85.53 / 85.50;
  client wall 73.50 / 73.47 vs 86.85 / 86.84 (S_*_1705*..1708*.lat.json). Served gate < 84.02: PASS. smoke_test PASS on
  all 4. Mask probe on all 4 (S_*.mask.json), through the batcher at batch 1: X = [2, 0] vs Y = [2] differ (maxabs 0.85),
  and X repeated is identical. So the request's own mask reaches the model; with the old tokens != 0 fallback X and Y
  would give identical outputs. No uvicorn left after the hold.
- LIBERO closed loop with the default (/home/deepgadget/experiments/gr00t/libero_eval/pi05/megakernel-p1/, tools/ =
  copies of ../fused/tools; the policy opens the device via common/device_open.device_kwargs, PI05_MEGAKERNEL is unset
  in run_fused.sh, and the stamp includes backend + mk_digest; the server lifetime is capped at 1800 s):
  - Open-loop golden through the wrapper (openloop_pcc_mkp1.json): mean PCC7 0.999884, min 0.999778, deterministic,
    host preprocessing ok, steady call 65.75 ms.
  - libero_spatial 10 tasks x inits 0-9 (tt_spatial_summary.json): **99/100**, 0 errors, 0 timeouts, 0 missing; the one
    failure is t9/i4 at the step cap. Paired inits 0-4: 49/50 vs GPU 50/50 (only_gpu [9,4]); inits 5-9: 50/50. Previous
    fused path: 98/100. ACCEPTED (>= 80 and paired >= GPU - 10 pts). Server policy latency median 66.4 ms (p10 66.1,
    p90 66.7, 2,130 calls) vs 77.6 on 09-29. Every episode carries the stamp megakernel=expert, backend=expert,
    mk_digest=328761c8a1ce3fd9, code=a46beb54a24f+dirty. The dirty files are README / DESIGN / JOURNAL only.
  - Demo (demo/, manifest.json): 4 on-screen viewer recordings, t3/i0 95 steps, t7/i0 131, t0/i0 78, t5/i0 108, all
    SUCCESS, each equal to its eval episode's steps. Median 66.5-66.7 ms per call. Captioned clips + combined
    pi05_libero_spatial.mp4 (29.6 s) + poster. I LOOKED at the poster (t3 SUCCESS frame, the bowl is on the plate), the
    title card (99/100, 66.4 ms, "the phase-1 expert megakernel"), a t7 running frame and the end card. The captions
    are correct.
- Docs: README Results rewritten from these files (default vs off accuracy, latency, device time 17.49 / 16.07 vs 30.31 /
  26.83 ms, served 70.9 vs 84.0-84.1, LIBERO 99/100, RTX ratios recomputed against 70.9 ms). The architecture table marks
  which part is ONE persistent op (the expert loop) and which is still traced stock ops (the 891-op prefix). DESIGN §4.12
  amendment + status row. Results in docs/megakernel/integrate_p1/.
Ship-phase handoff (NOT done here: the staging package is outside the pi0.5 repo):
- /home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused/tt-model.yaml: an unset PI05_MEGAKERNEL already
  gives expert. Still, add `PI05_MEGAKERNEL: "expert"` to serve.env so the served path is explicit. Update
  PI05_SOURCE_COMMIT, the card numbers (84 ms -> 70.9 ms served, LIBERO 99/100 at 66.4 ms, the GPU ratios) and the
  "one fused, traced TT-NN graph" wording (the expert loop is one persistent generic_op; the prefix is traced stock ops).
  The verify glob `tt/kernels/*/*.cpp` (asserts 9) does not cover tt/megakernel/kernels/mk_*.cpp. Add a check that
  mk_{brisc,ncrisc,trisc}.cpp + mk_defs.hpp / mk_dm.hpp ship. Their tt_metal includes (api/compute/*, api/dataflow/*,
  api/debug/*) all exist under publish/tt-metal-668c2907575/tt_metal/hw/inc; I checked this at 16:1x.
- The container must open the device through device_kwargs. server/app.py does; the image's own smoke should read /info
  megakernel.backend == expert.
Open (unchanged): base pytest 0.95 floor fails in both arms on seed 3 (user's call); mk_brisc.cpp:8 stale test path
comment (deferred to the next kernel change); the non-blocking items listed by fix-p1-r0.
Next: Ship (HF update), then phase 2 (WP P2-0).

## 2026-09-30 17:16:31 KST -- publish-p1: Ship phase 1 (session start)

Branch HEAD 8812ec8 (integrate-p1 end; descends from e8cbc78). Plan: push branch, PR to main, merge commit (trailer in the
merge body); then set PI05_SOURCE_COMMIT = merge sha in the staging package, add PI05_MEGAKERNEL=expert to serve.env and a
megakernel include check to verify:, package ONCE on publish/tt-metal-668c2907575, boot + smoke + 100-request bench via
with-device.sh, ONE HF create_commit (parent pinned), then verify (sha256, code/ vs GitHub main, image pull-back, video).
Prior publish scripts: scratchpad/pub/ (copied to scratchpad/pub1/ for this round).

## 2026-09-30 17:24:24 KST -- publish-p1: merged, image built

- GitHub: branch pushed, PR #2 (https://github.com/changh95/tt-pi-0.5/pull/2) merged with a merge commit:
  main = f7f173bd61241f09b286e56a9eb4038f76ee0782 (parents c8d4234 + dafecda; message ends with the Co-Authored-By
  trailer). The local branch was fast-forwarded to f7f173b and pushed; work continues on megakernel-2026-09-29.
- Staging /home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused: code/models = git archive f7f173b models
  (diff -rq vs a fresh GitHub clone: identical). tt-model.yaml: serve.env PI05_MEGAKERNEL=expert,
  PI05_SOURCE_COMMIT=f7f173b...; new verify lines: the megakernel modules import; FusedConfig default = expert;
  kernels dir = exactly the five mk_* files, KERNELS exist, kernel_digest() == 328761c8a1ce3fd9; every quoted #include
  of all 14 port C++ sources resolves to a sibling or tt_metal/hw/inc. Host-tested on a fake /opt layout: pass; a
  negative control (bogus include appended to mk_dm.hpp) failed both the digest and the include check.
- Package (logs/pkg-pi05-mkp1.log, buildkit log copied to logs/pkg-pi05-mkp1-buildkit.log): rc 0, all verify lines ran
  in the image. Image tt-model/pi05-base-p150:fe0d2e3d68a7, digest
  sha256:fe0d2e3d68a752709a443cbe6b8e5aa5f9d914ab21f03631959cb81623aeeafe, tt_metal 668c2907 dirty=false,
  code_sha256 08b350f19c7a04e2... build/pi05-base-p150/code/models == GitHub main models (+ tt-metal's
  models/common/lightweightmodule.py, as before).
- Boot validation hold started (scratchpad/pub1/validate.log; 2 cycles, cold then warm; /info megakernel.backend must be expert).

## 2026-09-30 17:34:16 KST -- publish-p1: SHIPPED (session end)

Files: docs/megakernel/publish_p1/{scripts,results}/ (copies of scratchpad/pub1).
- Boot validation (results/validate.log, one with-device hold 17:24-17:27, WITH_DEVICE_RESET_AFTER=1, reset exit 0):
  image fe0d2e3d68a7, cycle 1 cold (package cache removed; serve 97.7 s, first warm-up 31.45 s) and cycle 2 warm (43.3 s,
  13.08 s). Both: /info megakernel.backend = expert, kernel_digest 328761c8a1ce3fd9, source.commit f7f173bd; the device was
  opened with worker_l1_size 1395712 (the cut); smoke_test PASS; 100 warm requests with identical actions.
  timing_ms.inference median 70.83 (c1) / **70.81 ms (c2)** p90 70.95; total 72.18 / 72.16; client wall 73.73 / 73.69
  (results/bench-c{1,2}-*.json). Card, README and GPU_COMPARISON use c2. Previous image: 84.02 / 85.35.
- HF changh95/pi05-base-p150: ONE create_commit **9f6b082bfdda94c1fdf319670764e250c660b369**, parent pinned 2900530f.
  148 adds (code/, image/ OCI blobs, tt_kernel_manifest.json, README, SERVING, GPU_COMPARISON, tt-model.yaml,
  requirements.lock, demo/{README.md, libero_eval.json, 4 clips, combined mp4, poster}); 33 deletes (only the superseded
  image blobs); the demo file names are unchanged, so the old demo content was replaced in place. The YAML front matter was
  kept (no outdated field). The card says which part is the ONE persistent op (the expert loop) and which is traced stock
  ops (the SigLIP / VLM prefix, 891 ops); every number comes from bench-c2, integrate_p1/results (O_ip1, A_default_base,
  profiler), or the megakernel-p1 LIBERO files. The rows for the 125.9 ms image and the old 0.938-0.998 fp32 range were removed.
- Verify (results/hf_verify.log): 151 files in the tree, none missing or unexpected, deleted blobs gone. sha256: 133 files
  downloaded + 15 LFS image blobs via lfs.sha256, 0 mismatches. code/: all 83 GitHub-main files byte-identical (HF-only
  = tt-metal's common/lightweightmodule.py). Image pull-back: removed local images, then "tt-model pull changh95/pi05-base-p150" docker-loaded sha256:fe0d2e3d68a752709a443cbe6b8e5aa5f9d914ab21f03631959cb81623aeeafe
  (= recorded digest). Served the pulled package by repo id (results/serve_pulled.log): megakernel expert, digest
  328761c8a1ce3fd9, smoke PASS, 70.97 ms. Headless Chromium (results/pw_video.log): the README video
  played (readyState 4, t 3.99 s of 29.63 s, no media error). The one failed request (replay.mp4, 404) comes from the HF
  page itself, not from the card.
- Disk: deleted the superseded docker image tt-model/pi05-base-p150:672900e23919 (3.02 GB; the 09-29 image, still on HF
  history at 2900530f). build/pi05-base-p150 was overwritten by the new package. 31 GB free after.
Open (unchanged): base pytest 0.95 floor on seed 3 (both arms, pre-existing); mk_brisc.cpp:8 stale comment.
Next: phase 2 (WP P2-0), whole model as one fused op, on this branch.

## 2026-09-30 17:41:59 KST -- shipcheck-p1: independent check of the phase-1 release

Files: scratchpad/shipcheck/ (fresh GitHub clone gh/, HF snapshot hf/ at 9f6b082b without image blobs, pull.log, serve_check.log, info.json, bench.json).
- GitHub main = f7f173bd (merge of PR #2). HF main = 9f6b082b. diff -rq gh/models vs hf/code/models: identical except HF-only
  models/common/lightweightmodule.py (byte-identical to tt-metal 668c2907's). models/ at a46beb5 == f7f173b (git diff empty).
- Pull: removed local image fe0d2e3d68a7 (17:38:50), tt-model pull re-loaded image id fe0d2e3d68a7 (rc 0, 17:40:20).
  Served the pulled package by repo id in one with-device hold (17:40:39-17:41:23): /info source.commit f7f173bd,
  megakernel.backend expert, kernel_digest 328761c8a1ce3fd9; HF code/ smoke_test PASS; 30 warm requests: inference median
  70.94 ms, total 72.5, identical actions; action head equals the card's example. Card says 70.81 (bench-c2): consistent.
- Structure (scratchpad/ip1/out/ops_default_base.csv.gz, last 892 rows): exactly 1 GenericOp (mk_trisc) and 891 stock ops
  (Matmul 298, BinaryNg 160, LayerNorm 90, ..., UpdateKVCache 36). "891 traced stock ops" is accurate.
- LIBERO recomputed from megakernel-p1/tt_spatial.jsonl: 100 unique episodes, 99 successes, 0 errors, failure t9/i4 at 230 steps
  (GPU 138), paired 0-4 49/50 vs GPU 50/50, 5-9 50/50, every line backend expert + mk_digest; server log 2,130 calls median
  66.4 / p10 66.1 / p90 66.7 / max 71.2. HF demo/libero_eval.json per_task equals the JSONL. Demo steps equal the eval (95/131/78/108).
  Previous run 98/100, paired 48/50, 77.6 ms: match fused/.
- Card numbers vs files: O_ip1 (22/22, 0.99974/0.99916, off 0.99596; whole-call 0.804-0.9994 mean 0.975, off 0.764-0.9987 mean 0.969,
  oracle 0.810-0.9995), A_* (69.5 / 82.8 replay), S_struct (17.49 / 30.31 ms, 1,660 ops), GPU ratios 1.41-2.03x, 1.52x: all match.
- Videos: the 5 HF mp4s are byte-identical to megakernel-p1/demo/, fully decode with ffmpeg 7.0.2 (h264 High, 960x1080, 30 fps;
  combined 29.63 s), moov atom at the front, HTTP HEAD 200 video/mp4. Looked at the title card, a t7 SUCCESS frame and the end card.
- Discrepancies (all wording / labels, no wrong result):
  1. "median of 60" for the in-process trace replay (HF card + tt-model.yaml long description, GPU_COMPARISON.md:13) and
     "median of 60 calls" (GitHub README Latency): A_*.json hold 30 calls and 30 replays.
  2. Card row "Device time ... (profiler, median of 21 replays)": true for the megakernel (n 21); the previous-path 30.31 ms is
     post_cache_device_ms_first (the first of 5 sessions), not a median of 21.
  3. tt-model.yaml:93 comment calls off "the previous all-stock-op expert path" and SERVING.md:53 "all-TT-NN": off also runs the
     3 custom generic_op programs (540 GenericOps after the last cache write). README / card say "stock / custom" correctly.
  4. Provenance: S_struct.json (the structural / profiler gate file) sits in integrate_p1/results/hold4_serve_invalid/, the
     folder of the discarded served run, while the JOURNAL cites it as a valid result. Valid content (hold2 profile), misfiled.
  5. Basis mix: the previous image's 84.02 / 85.35 are cycle 1 of the 09-29 bench (c2 was 84.05 / 85.30); current is c2. Negligible.

## 2026-09-30 18:03:49 KST -- session mk2-r0-s0: PHASE 2 (whole sample_actions as ONE generic_op), session start

State found: HEAD 925758f (phase 1 shipped; main = f7f173b). No phase-2 code, no phase-2 journal entries: the "session died"
note refers to this task's previous launch, which left nothing on disk (git status clean except untracked .omc/, generated/).
Read: JOURNAL (all), DESIGN §0-§10, phase-1 kernels (mk_{brisc,ncrisc,trisc}.cpp, mk_defs/mk_dm.hpp), program.py, arena.py,
geometry.py, the shipped prefix (ttnn_siglip.py, ttnn_paligemma.py, ttnn_gemma.py VLM paths), memory notes (k4 device facts incl.
the F2 "fused Qwen3 layer at 7-8 % of peak" addenda, operand-format rules, cb-pop, mock compile, noclone).
Plan for this session: implementation design v3 for the prefix engine (below), then WP-P2-0 (mock compile + size), then the
single-op matmul measurement (the go/no-go input), then WP-P2-1 (one VLM layer).

## 2026-09-30 18:59:40 KST -- mk2-r0-s0: prefix engine v1 on the device (layer-0 ops correct, one race fixed)

Implementation design v3 (to be written into DESIGN.md §11; deviations from §5.3 recorded there): the prefix is a fixed
sequence of 314 ops (patch, 27 x [LN1, qkv, attn, o, LN2, fc1, fc2], post-LN, projector, embedding, 17 x [RMS1, qkv, attn,
o, RMS2, gate|up, down], RMS1 + qkv of layer 17) with ALL activations DRAM-staged between ops and ONE global barrier per op
(hub (10, 9)). Matmuls: 8 bands (rows 0..7) x 11 columns; weights read once per column from a bank-striped arena by a
feeder (x, 8) and multicast down the column; in0 bands multicast along the row by a feeder (b, 9); mode R (resident in0
band, N-outer, K in DST) for every op but the VLM down (mode S: streamed K blocks, fp32 partials reloaded). CB ids 32..62
declared tiny and RE-POINTED per op into the phase-1 CB region (+ a tail) by every RISC; the NCRISC hands the TRISCs a
per-op descriptor (P_OPD, read_tile_value/mailbox) so no geometry code sits in the TRISC binaries. Kernels
tt/megakernel/kernels_p2/ (whole_{brisc,ncrisc,trisc}.cpp include the phase-1 mk_*.cpp unchanged as
mk_expert_kernel_main()); host pe_geometry.py / pe_host.py / pe_program.py; CPU tests tests/megakernel/test_cpu_pe.py
(8 passed incl. the real-weight host model vs the torch reference: SigLIP+projector PCC > 0.99999, VLM K/V > 0.99999).
- Size (mock compile, pe_size_check.py): whole program base 130,196 B (brisc 28,944, ncrisc 15,840, trisc0 35,680,
  trisc1 30,528, trisc2 14,688 + args 3,444 + CB cfg 1,008) <= 131,072 gate (p2/results/size_check_mock8.log). First
  build was 145,324 B; brought under by Os + noinline on control code, one fidelity-switched matmul K loop, and moving
  describe/layout out of the TRISCs (P_OPD).
- Device (tests/megakernel/pe_bringup.py, pe_debug.py; prefix-only program; results p2/results/b*.json): every layer-0
  op vs the host decomposition on the device's own inputs: patch 0.9999992, SigLIP LN 0.9999986, qkv 0.99998, attn
  0.99998, o 0.999997, fc1 0.99997, fc2 0.999994, whole SigLIP layer 0.99997 (rel-L2 0.0083); VLM RMS 0.999997, q/k/v
  0.99988, attn 0.99975, o 0.99994, gate|up 0.99979, down 0.99996 (b10.json). No hang in any run.
- Bugs found and fixed on the way (each localised by a device experiment, not guessed):
  1. VLM attention PCC 0.81: P_SS (scores) ring of 8 with chunks of 6 tiles straddled the ring end (pack_tile/unpack
     index past fifo_limit) -> P_SS is re-pointed per chunk to exactly n tiles (full-capacity cycles). SigLIP (chunks of
     4) had passed by luck.
  2. VLM down PCC 0.75, band-dependent (b7/b9 fits: middle K blocks missing on bands far from the feeders): SUMMED ring
     credits. A receiver that runs ahead covered for a laggard and the feeder overwrote a slot still in use (the F0
     fact 4 / DESIGN §3.2 rule I had read and still missed). Fix: one credit word per receiver, the feeder waits on the
     MINIMUM (PS_W_RDY0 + row, PS_I_RDY0 + column). Discriminating arms: flag barrier (no change), 1 in0 slot (partial),
     direct DRAM in0 (bands 0-1 still wrong -> the weight ring, not the in0 path).
  3. Watcher NoC sanitizer false positive on re-pointed CBs ("NOC transaction overflows a circular buffer"): run the
     watcher with TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1.
  Also adopted ttnn's Blackhole rule: flush the data multicast before the flag multicast (separate command buffers).
  Phase-1's mcast_round lacks that flush (latent; phase 1 unchanged).
- Device time per op (hub wall-clock stamps, median of 20 in-kernel reps, b11.json): SigLIP layer 0.607 ms (LN 0.086 x2,
  qkv 0.053, attn 0.147, o 0.043, fc1 0.093, fc2 0.099) vs TTNN 0.395 -> SLOWER, P2-2 gate (<= 0.355) not met yet;
  VLM layer 1.611 ms (RMS 0.098 x2, qkv 0.089, attn 0.335, o 0.076, gate|up 0.555, down 0.360) vs TTNN 2.437 -> under the
  P2-1 go bar 2.20 (its accuracy half, PCC vs the ttnn layer on real inputs, not yet run).
Next: whole prefix end to end (all 314 ops, all arenas) vs the host model and the ttnn caches; then whole-model
integration; speed work on SigLIP (norms: per-call inits; attention: single-chunk softmax; in0 fill overlap) and the
weight feeders (VLM matmuls look feeder-bound: gate|up 2.95 us per 34.8 KB page per column).

## 2026-09-30 21:06:06 KST -- mk2-r0-s0: WHOLE sample_actions as ONE generic_op works end to end (first numbers)

- Whole prefix on the engine (all 314 ops, p2/results/b18.json, base, the same inputs as the shipped ttnn prefix in the
  same process): 18-layer K / V vs the host fp32 decomposition min PCC 0.98589 (L0 K 0.99963, L17 V 0.98958); the SHIPPED
  caches vs the same fp32 host min 0.96562 (L17 V 0.96580): mine is closer to fp32 on every layer. Mine vs shipped min
  0.95975: the WP-P2-3 bar "vs the ttnn caches >= 0.999" is failed by the shipped path's own error (recorded as such).
  Prefix device time 44.2 ms per rep (5 reps) vs TTNN 52.42 ms: WP-P2-3 time bar PASS.
- Stage checks (b17.json): SigLIP after 27 layers 0.99984, post-LN 0.999999, projector rows 0.999997, embedding rows
  exact, pad rows 0, VLM layers 0-2 x 0.99994..0.99999, K 0.99989..0.99991.
- Hang on the way: the embedding op hung when it was the FIRST op of a launch (TRISC tilize after hw_startup; watcher
  signature b16: 28 language-item cores BRISC CWFW / NCRISC PNGO / TRISC0 UPTW / TRISC1 MWDD / TRISC2 K, 81 cores PBGW, hub
  PHUB). Fix: the NCRISC tilizes by word copies (face layout), the TRISC only scales (-1.4 KB TRISC code). One 30 min hold
  (b15, 19:38-20:08) was lost to it; single-op triage now runs with 150 s timeouts.
- PI05_MEGAKERNEL=whole in tt/ttnn_pi0_model.py (_whole_for, _pe_write_inputs; the trace holds wm.run only):
  * pe_whole_run.py (w19_*.json, one hold): whole 62.68 ms median vs expert 70.62 ms; 20 replays bit-identical; the first
    request again after the others identical. whole vs expert actions PCC 0.961 / 0.970 / 0.992 (random inputs).
  * tests/pcc/test_pcc_pi05_fused.py under whole (pcc_whole_{libero,base}.json): LIBERO openpi golden PCC7 mean 0.999959,
    min 0.999926 (phase 1 0.999884 / 0.999778; shipped 0.999839 / 0.999712), traced, ten replays identical, call
    58.71 ms (phase 1 65.8). Base vs the fp32 torch reference on the test's seeds: 0.99481 0.98146 0.99286 0.99760 0.98780
    0.99839 (shipped B0 0.99843 0.98397 0.93811 0.99549 0.98017 0.98793): closer on 4/6, WORSE on n40 (-0.0036) and n97
    (-0.0025): the amended per-seed gate is not met yet. Mask live (-0.62), pad ids invisible, default mask identical,
    replays identical; call 62.75 ms.
- Process note: I edited kernels_p2/*.hpp (#ifdef'd timing arms only) at ~20:36 while b19 held the card; its whole arm
  compiled at 20:41 from the edited sources (arms off by default, default code unchanged, but the rule was broken).
Next: precision (VLM matmuls HiFi2 A/B; the per-seed gate over >= 18 seeds), speed (norms 8 ms -> FPU stats + batched
apply; attention 9.7 ms; feeders), then the full gate set.

### 2026-09-30 21:55:13 P2 SigLIP single-chunk attention, HiFi2 VLM default, FPU norms, progressive in0, K/V multicast
- Commits the uncommitted work of the previous context: VLM matmuls HiFi2 by default (PE_VLM_LOFI arm = LoFi; 22-seed gate
  HiFi2 22/22 vs LoFi 18/22, docs/megakernel/p2/results/seeds_{hifi2,lofi}.json), FPU LN/RMS statistics, progressive in0
  (per-K-piece flags), K/V multicast by 4 feeders, TRISC arg trim.
- SigLIP attention: one chunk of all 8 key tiles, 96 items (16 heads x {3,3,2} q row tiles x 2 images), l shared through
  P_PL and normalised in DST (flash_part(..., single)). Mock size base 130,624 B / libero 129,584 B, gate 131,072
  (p2/results/mock19.json).
- Device (b29, p2/results/b29.json): sig0_attn PCC 0.9999885; SATTN 134 -> 89.6 us; SigLIP layer 494 -> 450.1 us
  (LN1 43.0, QKV 52.5, ATTN 89.6, O 40.2, LN2 43.0, FC1 92.4, FC2 89.2). VLM layer 1.726 ms unchanged.
- Whole prefix (p2/results/b29_prefix.json): 41.8 ms/rep (was 43.0); K/V vs host fp32 min PCC 0.994709.
- P2-2 still open: 450 us > 394.7 stop line. Next levers: fuse LN into the producer (O / FC2 epilogue emits the LN'd
  bf16 copy -> removes 2 x 43 us minus epilogue cost), QKV / FC1 / FC2 matmul efficiency.

### 2026-09-30 22:16:30 P2 norms: column groups + distributed statistics; size-check fix; BRISC reads the NCRISC's (Op, Lay)
- SIZE CHECK BUG (found 21:5x): pe_size_check picked the NEWEST ELF per RISC; a cached (unchanged) kernel kept its old
  mtime, so the base row could read the libero BRISC (mock19's base BRISC 28,320 B = mock18's libero BRISC). mock19's
  130,624 B was therefore wrong: the committed 0498289 state was ~131,440 B (over the 131,072 gate, under the 136,192
  ring). Fixed: only ELFs built by this compile count, missing one = error; the mock cache is now emptied before a run.
- BRISC no longer compiles describe / layout: the NCRISC publishes (Op, Lay) at P_SYNC words 48..63 and PS_OPK = k + 1;
  the BRISC waits and copies (-1.8 KB BRISC). P_SYNC 768 -> 1,024 B.
- Norm items are (row tile, column group): SigLIP 16 x 6, VLM mt x 4 (<= 96 <= 110 cores, one item per core). Three
  steps measured (bringup sig0/vlm0, 20 in-kernel reps):
  b30 redundant statistics (every group core reads the whole row): SLN 43.0 -> 56.3 us, VRMS 51 -> 83.6 us (DRAM:
      6 / 4 x the x reads) -- rejected.
  b31 one statistics core per row, (rstd, mu) sent to the row's cores: SLN 33.9 us, VRMS 50.8 us (the stats core's full
      row read + pass bound it) -- superseded.
  b32 every group core computes partial (sum x^2, row sum x) over its w tiles, all-to-all into slot g of P_R of the row's
      cores (noc_async_write + write barrier + PS_NR increments), each sums the ncg partials (HiFi4 matmuls with ONES;
      mu via the x32 row-sum trick): SLN 28.2 us, VRMS 40.0 us. PCC LN1 0.9999985, LN2 0.9999986, RMS1 0.9999971,
      RMS2 0.999997 (unchanged to 7 digits). Files p2/results/b30.json b31.json b32.json.
- Size base 130,212 B / libero 128,084 B (p2/results/mock25.json; cold paths use one out-of-line page read / write,
  BRISC common args cut at PA_BRISC_N = 141, the weight arenas are NCRISC-only).
- b32: SigLIP layer 421.9 us (LN 28.2 x 2, QKV 52.7, ATTN 90.1, O 40.4, FC1 92.6, FC2 90.0); VLM layer 1.702 ms;
  prefix 40.65 ms/rep (b32_prefix.json, K/V vs host fp32 min PCC 0.99497); whole call base 59.15 ms median of 20,
  replays identical, alternation identical (w32.json).
- P2-2 still open: 421.9 > 394.7 us.

### 2026-09-30 22:45:18 P2 SigLIP 421.9 -> 370.7 us: GELU / exp arithmetic, distributed staggered in0, per-role trace arm
- Localisation (files p2/results/arms3_*.json, tr*.json; tests/megakernel/pe_trace.py = timing arm PE_DBG_TRACE: per
  core BRISC / NCRISC wall-clock start and end of the k-th op):
  * GELU: arm PE_DBG_NO_GELU FC1 92.6 -> 52.3 us: the stock fp32-accurate gelu_tanh costs 40 us (~2,200 cycles/tile,
    serial with the matmul under dst_full_sync). x*sigmoid(2u) with exp_21f + approx recip: 84.0 us (b33).
  * in0: one feeder per band on row 9 delivers 11 GB/s each with eight reading (FC2 in0 reads alone 49.8 us, with one
    feeder alone 20.4 us = 27 GB/s, tr5); bigger transactions (one read per bank run, tr3) and bank-staggered issue order
    (tr4) changed nothing. Distributed in0 (column q of the band reads K piece q and multicasts it along the row,
    per-piece flags PS_IV0 + q, no credits in mode R) helped O / VO but made FC2 WORSE (89 -> 106 us, b34: all pieces
    arrive together, piece 0 as late as the last); staggered (source q starts after piece q - 1 landed) fixed it.
  * VLM matmuls: gate|up is compute + GEGLU bound (weights stream ~110 GB/s but are credit-gated).
- Now (b35, p2/results/b35.json): GELU = relu(x) - t q(t), degree-9 fit on [0, 4.25] (max abs err 1.3e-5 fp64,
  4.0e-5 fp32); softmax exp = exp_21f (bf16-accurate, P is bf16); in0 distributed + staggered.
  SigLIP layer 370.7 us (LN 27.9, QKV 47.6, ATTN 82.5, O 34.0, LN2 27.9, FC1 77.9, FC2 73.0); VLM layer 1.606 ms
  (RMS 39.9 x 2, QKV 94.9, ATTN 231.1, O 63.6, GU 671.5, DOWN 461.5). PCC unchanged: sig0 layer 0.9999719, fc1
  0.9999684, attn 0.9999885, vlm0 attn 0.9997526, gu 0.9999045; K/V vs host fp32 min 0.995117 (b35_prefix.json).
  Prefix 37.6 ms/rep; whole call base 55.99 ms (w35.json, 20 replays identical, alternation identical).
- P2-2: 370.7 us is under the 394.7 stop line, above the 355 go line. Size base 129,332 B (mock29.json).

### 2026-09-30 22:58:38 P2-2 GO: SigLIP layer 350.5 us (<= 355): folded norm affines, K / V all-gather
- arms4 (p2/results/arms4_*.json): FC1 no-GELU 46.8, empty-GELU call 48.1, polynomial GELU 77.4 us: the SFPU arithmetic
  itself is ~1.2 us/tile (~50 cycles per 32-lane row); two interleaved Horner chains (b36) changed nothing
  (throughput-, not latency-bound). VLM gate|up: empty-GELU 518 vs 670 us: GEGLU costs 152 us per VLM layer.
- Norm affines folded into the consuming matmuls on the host (pe_host.fold_norms: diag(g) W, b + beta W; SigLIP LN1 ->
  qkv, LN2 -> fc1, post-LN -> projector, VLM (1 + w) -> qkv / gate|up); the norms apply only (x - mu) rstd / x r and
  read no gamma / beta (b37): SLN 27.9 -> 23.5 us, VRMS 39.9 -> 30.9 us. PCC: LN1 0.9999985, qkv 0.9999686, fc1
  0.9999665, sig0 layer 0.9999806 (was 0.9999719), vlm0 layer 0.9988911 (was 0.9989235), K/V vs host fp32 min 0.995023
  (was 0.995117). The real-weight CPU test (PI05_SLOW_CPU=1) passes with the folded parameters.
- SigLIP attention K / V: the 3 item cores of a head each read a third (key tiles t = g mod 3) and write it into the
  other two (PS_KVX counts), 9.2 MB -> 3.1 MB of DRAM reads per op (b38): SATTN 81.8 -> 71.9 us.
- b38 (p2/results/b38.json): SigLIP layer 350.5 us (LN 23.4, QKV 47.4, ATTN 71.9, O 34.1, LN2 23.5, FC1 77.4, FC2
  72.8) -> P2-2 GO (<= 355). VLM layer 1.579 ms. Prefix 36.74 ms/rep (b38_prefix.json). Whole call base 55.14 ms
  (w38.json; 20 replays identical, alternation identical). Size base 128,548 B (mock33.json).
- Next: accuracy re-verification of the whole model (22-seed gate vs shipped, LIBERO golden) after the numerics
  changes (polynomial GELU, exp_21f softmax, folded norms), then the full exit gate set.

### 2026-09-30 23:42:11 P2 accuracy re-check after the numerics changes; bf16 VLM qkv; library exp; SigLIP 353.6 us
- 22-seed amended gate (pe_seed_gate.py; off = seeds_off_v2.pt, same session; files p2/results/seeds_*.json):
  | build | passes | whole mean | min margin |
  |---|---|---|---|
  | 21:28 HiFi2 (pre-session-2 numerics) | 22/22 | 0.99853 | +0.00123 |
  | c6aa39e (folded norms, poly GELU, exp_21f) | 21/22 | 0.99652 | -0.01370 (seed 707) |
  | + PE_EXP_STOCK | 22/22 | 0.99677 | +0.00086 |
  | + PE_GELU_STOCK | 22/22 | 0.99768 | +0.00080 |
  | + both stock | 22/22 | 0.99726 | +0.00120 |
  | bf16 VLM qkv weights (exp_21f) | 21/22 | 0.99740 | -0.01187 (707) |
  | bf16 qkv + PE_EXP_STOCK | 22/22 | 0.99838 | +0.00121 |
  | bf16 qkv + degree-4 2^f exp (2.7e-6) | 21/22 | 0.99707 | -0.01710 (707) |
  | final (bf16 qkv, library exp, b41) | 22/22 | 0.99838 | +0.00121 (707: 0.99522 vs 0.99175) |
  Seed 707 (n_lang 64) moves between 0.975 and 0.997 with numerically equivalent builds (an exp accurate to 2.7e-6
  still failed it): the whole-model PCC is chaotic at that seed (tiny changes flip bf16 roundings of P / activations).
  The library exp passed it in all three builds tested and costs 7-8 us per SigLIP attention and ~16 us per VLM
  attention; it is the default, the fast exps are arm PE_EXP_FAST.
- Error budget (CPU emulation, p2/results/emu/: ttnn's bfp8 packing reproduced bit-exactly by the emulator): VLM K/V
  rel error at layer 17 (V) 6.6 % with bfp8 weights, 2.7 % with bf16 weights, activations bf16 vs fp32 immaterial;
  qkv weights bf16 alone 4.95 %. SigLIP output 0.55 % (bfp8) vs device 1.8 % (HiFi2) / 1.2 % (arm PE_MM_HIFI4,
  acc4_*.json: prefix 37 -> 50 ms, not taken). bf16 attention scores / P cost little (0.55 -> 0.59 %).
  -> VLM qkv weights now bf16 (own arena per layer, PA_WV16): VQKV 95 -> 111 us, vlm0 layer PCC 0.99889 -> 0.99927.
- in0: sources read their pieces at once, only the multicasts are staggered (lag 1; arm PE_IN0_READ_STAGGER = old):
  SigLIP 359.2 -> 353.6 us with the library exp (arms6_*.json); lag 2 / 3 slower.
- b41 (p2/results/b41.json, b41_prefix.json, w41.json): SigLIP layer 353.6 us (LN 23.6, QKV 46.8, ATTN 80.5, O 33.9,
  LN2 23.4, FC1 76.6, FC2 69.0) -> P2-2 GO; VLM layer 1.612 ms; prefix 37.3 ms/rep; K/V vs host fp32 min 0.995379;
  whole call base 55.67 ms (20 replays identical, alternation identical). Size base 128,636 B (mock36.json).

### 2026-10-01 00:46:30 PHASE-2 EXIT GATE TABLE (build 4ed4178 + 24b30ed scripts; kernel_digest 4aa02cdf21ed0c94; files docs/megakernel/p2/gates/results/, scripts ../scripts/)
Holds A-G 23:46-00:45 (holds.log in the scratchpad; private TT_METAL_CACHE per arm; kernel md5s identical before/after the soak).
| gate | result | evidence |
|---|---|---|
| Structural: the replay holds exactly ONE device op | PASS: 21 / 21 traced replay sessions per shape = 1 op each, GenericOp whole_{trisc,brisc,ncrisc}.cpp on 110 cores; device time median 53.96 ms base / 51.26 ms LIBERO | S_struct_whole.json, ops_whole_*.csv.gz, P_whole_*.json (20 profiled replays bit-identical) |
| Stamp asserted in test_pcc_pi05_fused.py | PASS (backend "whole", digest 4aa02cdf21ed0c94) | g1_pcc_whole_*.json |
| PCC7 vs openpi golden mean >= 0.9995, min >= 0.999 | PASS 0.999976 / 0.999955 (phase 1 0.999884 / 0.999778, same session) | g1_pcc_whole_libero.json, A_*_libero_r*.json, ALT_named_whole.log [D] |
| Amended base gate: per seed at least as close as shipped to the fp32 whole-model reference, >= 18 seeds | PASS 22 / 22, mean 0.99838 vs shipped 0.96921, min margin +0.00121 (seed 707 +0.00347: fragile, see 23:40 entry) | seeds_final.json |
| Ten replays bit-identical | PASS both shapes (10 calls after other prompts + 10 raw execute_trace; output-poisoning check: the trace writes the output) | A_whole_*_r*.json, g1_pcc_whole_*.json |
| Alternating prompts / shape switch vs fresh-model refs | PASS 20 / 20 bit-identical (positive control: ref pairs max PCC 0.955); named verify_alternating.py OVERALL all_ok=True | ALT_whole.json, ALT_named_whole.log |
| No hang in 20 consecutive runs | PASS 20 / 20 processes rc 0, 31 calls each all equal, one digest 886f7341c1085571, 48.8-66.9 s | soak_results.txt, soak_hold.log, sums_soak_*.txt |
| Prompt-length edge cases no worse than shipped | PASS 6 / 6: base n 1 / 224 / 128 / 150: 0.99879 / 0.99989 / 0.99764 / 0.99981 vs 0.99604 / 0.96898 / 0.99289 / 0.99648; LIBERO n 32 / 1: 0.99996 / 0.99997 vs 0.99949 / 0.99930 | E_base.json, E_libero.json |
| Existing suites green (megakernel selected) | PASS: pytest test_pcc_pi05_fused.py under whole 2 passed (the base floor 0.95 now passes: seed 3 0.98882); test_fused_host + test_cpu_mk + test_cpu_pe (incl. real-weight) + test_reference_vs_openpi 47 passed; test_server_masks 5 passed | pytest_pcc_whole_tail.log, cpu_suites.txt, cpu_server_masks.txt |
| Speed: whole-call < phase 1, same session, alternated, > 2 x MAD se | PASS base 56.11 / 56.06 vs 71.05 / 70.86 ms (diff 14.9 / 14.8, 2 se 0.10 / 0.13); LIBERO 53.05 / 52.99 vs 65.71 / 65.63 (diff 12.7 / 12.6, 2 se 0.17 / 0.14); aiclk 1350 throughout | A_{whole,expert}_{base,libero}_r{1,2}.json |
| P2-0 size / CB ids / L1 | PASS: 128,636 B base / 126,492 LIBERO <= 131,072 (profiler build 129,004); CB ids 32..62 (+ phase 1's 0..31) <= 64; arena 1,156,192 B + tail below the largest free block 1,253,888 B | p2/results/mock36.json, mock_prof, A_whole_*.json megakernel_l1 |
| P2-1 VLM layer: PCC vs ttnn >= 0.9995, time <= 2.20 ms | PASS: PCC 0.99987; rel vs fp32 0.0056 vs ttnn's 0.0170; 1.612 ms | L_layer_vs_ttnn.json, p2/results/b41.json |
| P2-2 SigLIP layer: PCC vs ttnn >= 0.9995, rel <= ttnn + 20 %, time <= 355 us | PASS: PCC 0.99992; rel 0.0067 vs ttnn's 0.0133; 353.6 us | L_layer_vs_ttnn.json, b41.json |
| P2-3 stack time < 52.42 base / 48.72 LIBERO | PASS 37.3 / 35.7 ms | P23_*.json |
| P2-3 per-layer K/V vs the ttnn caches PCC >= 0.999 | **FAIL**: min 0.9642 base / 0.9607 LIBERO. The ttnn caches themselves are 0.9656 / 0.9612 from the fp32 host decomposition; the engine is 0.9954 / 0.9945, closer on 36 / 36 (layer, K / V) at both shapes (PCC and rel-L2). The bar measures the shipped path's error: no more accurate prefix can meet it. Needs the user's ruling. | P23_base.json, P23_libero.json |
| P2-5 server | PASS: served inference median 55.74 / 56.13 ms (whole) vs 70.86 / 70.76 (expert, P1-6 was 70.72); smoke PASS x4; /info backend whole + digest; mask probe pass; refusals at startup for PI05_KV_DTYPE=bf16, NUM_STEPS=20, BATCH_SIZES=1,2 (rc 3, no device open) | S_*.json, S_*.smoke.log, refuse_whole_*.log |
Not done: PI05_MEGAKERNEL default is still expert (P2-5 calls whole "the new default"; left for the ship decision / the P2-3 ruling).

## 2026-10-01 11:23:25 KST -- verify-p2-r0: independent verification of PHASE 2 (verifier session, no code changed)

HEAD 59732a7 (kernel_digest 4aa02cdf21ed0c94), own scripts docs/megakernel/verify_p2_r0/scripts/ (new seed set, own
fp32 oracle incl. its own VLM K/V, own size accounting, own host-clock layer timing), results ../results/. Holds 1-5
10:09-11:21 with WITH_DEVICE_RESET_AFTER=1, private caches per arm (ttcache_{whole,expert,off}[_prof], rbA, rbB).
Source md5 aggregate ALL 505df88a... identical at start (10:03), around every hold, before / after the soak, at the end.
whole / expert / off measured in the same session; the arm came from PI05_MEGAKERNEL, the device was opened by device_kwargs.
- Structural (tracy, S_struct.json): whole has 21 / 21 replay sessions per shape = 1 op each. That op is a GenericOp on
  110 cores with whole_trisc / whole_brisc + whole_ncrisc, one program hash per shape, no missing durations. Device time
  median 53.95 ms base (53.83-54.08) / 51.27 LIBERO. The request path (d_prof3, 9 sample_actions_fused calls alternating
  n 1 / 128 / 224) gives 10 sessions x 1 GenericOp and ZERO device ops outside the trace after the capture. Arms differ:
  expert has 892 ops per replay (1 after the 36th KV write = mk_*), off has 2551 (1660 after the last KV write), and
  the expert and off prefix op codes are equal. Caches: whole compiled whole_* (no mk_* dir), expert compiled mk_*, off neither.
- Golden PCC7 (A_whole_libero_r1.json): mean 0.999976, min 0.999955. Same session: expert 0.999884 / 0.999778,
  off 0.999839 / 0.999712. PASS.
- Amended per-seed gate vs the fp32 whole-model torch reference (c_refs.py; positive control = my written-out pipeline
  equals ref.sample_actions bit-exactly on 2 inputs per shape), 32 seeds (the r1 22 + 10 new 801-810, three at n 64):
  whole closer than off on 32 / 32. Mean 0.99851 vs off 0.97034 (expert 0.97666). Margin vs off: min +0.00121 (seed 703),
  median +0.0101, max +0.235. Seed 707: whole 0.995222, off 0.991753 (+0.00347), expert 0.997693 (-0.00247; the ONLY
  seed of 32 where whole is less close than the phase-1 expert path). LIBERO 8 / 8 closer than off and than expert.
- Seed 707 rebuild sensitivity: two EMPTY caches rbA / rbB compiled the same source. ELF loadable content was identical
  (objcopy md5), and the outputs of 706 / 707 / 708 / 809 / 810 were bit-identical across rbA, rbB and the main cache
  (707 = 0.995222 in all three). It does not move between rebuilds; the 0.975-0.997 spread came from source changes only.
- P2-3 amended layer gate (oracle = the fp32 reference's own VLM cache, valid prefix rows; 8 base seeds + 8 LIBERO
  records): whole closer than the ttnn caches on 288 / 288 (layer, K/V, input) per shape, in both PCC and rel-L2.
  Whole min PCC is 0.99171 base / 0.99905 LIBERO, ttnn 0.92868 / 0.99060. The expert K/V equal off bitwise. PASS.
- Replays: 10 calls after other prompts + 10 raw execute_trace were bit-identical (every arm and shape). The output
  poison check passed (the trace writes the output). Profiled replays were identical too.
- Alternating / shape switch (ALT_whole.json): 20 / 20 calls bit-identical to fresh-model refs, and the refs are
  bit-identical to the d_arm process outputs. Positive control: ref pairs max PCC 0.955. The named repo
  verify_alternating.py under whole gave OVERALL all_ok=True.
- Soak (hold3.log): 20 / 20 processes rc 0, 31 calls each, all_equal, finite, one digest 886f7341c1085571 (the
  implementer's soak digest), 50.9-54.4 s. Source md5s and compiled-ELF md5s identical before / after.
- Per-layer times from the HOST clock (d_layers.py, prefix-only variant inside the real model, (T_27 - T_1) / (26 x 10 reps)):
  SigLIP 353.1 us base / 353.2 us LIBERO (go line 355: PASS, 1.9 us margin). VLM 1619 / 1526 us (<= 2200: PASS).
  Prefix stack 37.9 / 35.8 ms per rep (< 52.42 / 48.72: PASS). The whole-model output is unchanged after these launches.
- Latency (A_*_r{1,2}.json, two rounds, arms alternated, aiclk 1350 before / after every bench). Call median whole vs
  expert vs off: base 55.80 / 70.77 / 84.17 (r1) and 56.02 / 70.73 / 84.06 (r2); LIBERO 52.95 / 65.70 / 76.86 and
  53.08 / 65.80 / 76.96. whole - expert is 14.7-15.0 ms base and 12.7 ms LIBERO, against 2 x MAD-se of 0.10-0.16. Clock witnesses: time_ns,
  monotonic and the perf_counter sum agree to 1 ms; a 10 s window gave 185 / 144 / 121 replays at base and 196 / 156 / 133 at LIBERO.
- Size (m_size.py, mock cluster, EMPTY cache, readelf + descriptor-counted args): base 124,992 B binaries + 2,572 args
  + 1,008 CB + 64 sem = 128,636 B; LIBERO 126,492 B (<= 131,072: PASS). The mock ELFs have the SAME kernel hashes and
  objcopy md5s as the real-device ELFs of hold 1, so the dummy-weight mock build is the shipped binary.
- Edge cases: base n 1 / 224 / 128 / 150 are inside the seed gate (margins +0.144, +0.012 / +0.019, +0.235, +0.004).
  LIBERO n 32 / 1 (EDGE_libero.json, 4 inputs): whole closer than off on 4 / 4.
- Suites: pytest test_pcc_pi05_fused.py under whole gave 2 passed. CPU (PI05_SLOW_CPU=1): 47 passed, plus server masks 5 passed.
- Incident (mine): at 10:27 the tracy raw logs of hold 2 (18 GB in total; 5 GB for the off arm alone) filled the SHARED
  root filesystem (100 %, 55 MB free). The off-base profile failed, and so did the alt tests that came after it (rc 120,
  no device fault; the guard reset the card). I deleted the raw logs after extracting the CSVs and reran those steps
  in hold 2b (off profiled with 1 replay). Any other process writing to / around 10:27 may have hit ENOSPC.
Open (not gates; must be done before shipping): DESIGN.md §7 P2-3 row still has the old "vs the ttnn caches >= 0.999"
text, and the 2026-10-01 amendment is not recorded. PI05_MEGAKERNEL is not yet "whole" by default
(fused_config.py still says "whole: phase 2, not built").
Verdict: phase 2 ACCEPTED on the amended gates (every re-run gate passes).

## 2026-10-01 11:29:22 KST -- integrate-p2: the whole-model megakernel becomes the DEFAULT path (session start)

User request relayed with this task: "응, 둘 다 yes. 검증 후 HF 배포까지 진행해" (yes to both [the 2026-10-01 rulings]; verify, then
ship to HF). This session = integrate-p2 (default switch, docs, gate re-run on the final commit, LIBERO closed loop + demo);
the HF push is the Ship phase.
Code commit 0f19387 (kernels unchanged; models/ diff 59732a7..6e752ba empty):
- FusedConfig.megakernel default "whole" (env unset / empty / dataclass default); resolved(n>1) still turns only the
  UNSET default off on a mesh. expert (phase 1) and off (previous shipped path) are comparator knobs only.
- server: docstring; new startup refusal for whole with PI05_NUM_IMAGES != 2 (the kernels are built for 2 cameras).
- lang_masks: already carried through every server path since integrate-p1 (488fe36): run_inference, _Batcher._loop
  (batch 1 included), _DPRouter -> _Batcher, the batch / DP warm-ups (grep of every sample_actions_fused call in
  server/app.py: all pass lang_masks); tests/test_server_masks.py 5 passed. The served mask probe is re-run below.
- CPU (scratchpad/ip2/out/cpu_suites.txt, PI05_SLOW_CPU=1): test_fused_host + test_cpu_mk + test_cpu_pe +
  test_reference_vs_openpi + test_server_masks: 52 passed.
Device gate plan (scratchpad/ip2 = copies of verify_p2_r0/scripts; arm "whole" = PI05_MEGAKERNEL UNSET, files keep the
resolved name; expert / off explicit; fp32 whole-model references ref_{base,libero}.pt reused from vp2r0: reference/ and
weights unchanged): hold1 arms x shapes (32 base seeds + 8 LIBERO), hold2 profiles (raw tracy logs deleted per run: disk
25 GB free), hold3 alternation + edge + pytest, hold4 soak 20, hold5 layer times + latency r2, hold6 served A/B + refusals.

## 2026-10-01 12:40:57 KST -- integrate-p2: gate re-run on the final code commit (holds 1-6, 11:29-12:40)

Code 0f19387 (+ ce9806a docs). Source md5 aggregate (tt/ + common/) ALL 6ffe696f1c7e8e2d796609ce49689132 at the start
and the end of every hold (sums_*.txt). Arm "whole" = PI05_MEGAKERNEL UNSET (every A_whole_*.json records env
PI05_MEGAKERNEL None and stamp backend whole, digest 4aa02cdf21ed0c94). Each hold WITH_DEVICE_RESET_AFTER=1 + timeout
1800, private caches scratchpad/ip2/ttcache_*. Files: docs/megakernel/integrate_p2/results/ (scripts ../scripts/ =
verify_p2_r0's with the arm runner of env.sh, plus c_prof3.py, c_lat.py, hold6.sh). No hang, every rc 0.
| gate | result | evidence |
|---|---|---|
| default = the verified build | the default's outputs are bit-identical to verify-p2-r0's explicit whole on all 32 base seeds + 8 LIBERO records (digests equal; expert / off also identical to their vp2r0 runs) | A_*_r1.json vs vp2r0 |
| Structural: ONE op per replay | PASS 21 / 21 sessions per shape = 1 GenericOp (whole_trisc / whole_brisc + whole_ncrisc), 110 cores, one program hash per shape; device 53.94 ms base (53.80-54.08) / 51.26 LIBERO (51.17-51.38); request path (9 calls n 1/128/224): 10 sessions x 1 op, 0 device ops outside the trace; arms differ: expert 892 ops (1 after the 36th KV write), off 2551 (1660 after), prefix op codes expert == off | S_struct.json, S_prof3.json, ops_*.csv.gz, P_*.json |
| openpi golden PCC7 | PASS mean 0.999976, min 0.999955 (expert 0.999884 / 0.999778, off 0.999839 / 0.999712) | A_*_libero_r1.json |
| Amended per-seed gate vs the fp32 whole-model reference | PASS 32 / 32 closer than off; mean 0.99851 (min 0.98882) vs off 0.97034 (min 0.76397), expert 0.97666; margin vs off min +0.00121 (seed 703), median +0.0101; vs expert 31 / 32 (seed 707 whole 0.99522 vs expert 0.99769, off 0.99175); LIBERO 8 / 8 | G_gates_r1.json |
| P2-3 amended (K/V vs fp32) | PASS 288 / 288 per shape (PCC and rel-L2); whole min 0.99171 / 0.99905 vs ttnn 0.92868 / 0.99060; expert K/V == off bitwise | G_gates_r1.json |
| Ten replays / poisoning | PASS every arm and shape (10 calls, 10 raw execute_trace, poison check, after-bench equal) | A_*_r1.json |
| Alternating / shape switch | PASS 20 / 20 bit-identical vs fresh-model refs (ref pairs max PCC 0.955); repo verify_alternating.py under the default all_ok=True | ALT_whole.json, ALT_named_whole.json |
| 20-run soak | PASS 20 / 20 rc 0, 31 calls each all_equal, finite, one digest 886f7341c1085571 (= verify-p2-r0's), 49.9-63.2 s; source md5s and compiled whole_* / mk_* ELF md5s identical before / after | hold4.log, sums_soak_*.txt, kelf_soak_*.txt |
| Edge cases | PASS: base n 1/224/128/150 are seeds inside the 32; LIBERO n 32 / 1 closer than off on 4 / 4 | EDGE_libero.json |
| Suites | PASS: pytest test_pcc_pi05_fused.py with PI05_MEGAKERNEL unset 2 passed (stamp whole); CPU 52 passed | pytest_pcc_default.log, cpu_suites.txt |
| Speed (whole < phase 1, alternated, > 2 x MAD se) | PASS: call median base 55.87 / 55.95 vs expert 70.84 / 70.77 vs off 84.17 / 84.06 ms (diff vs expert 14.97 / 14.82, 2 se 0.13 / 0.15); LIBERO 53.10 / 53.11 vs 65.73 / 65.78 vs 76.97 / 76.96 (diff 12.63 / 12.67, 2 se 0.11 / 0.14); replay 54.10 / 51.21 ms; aiclk 1350 before / after every bench | G_speed.json |
| P2-1 / P2-2 / P2-3 times (host clock, in the real model) | PASS: SigLIP layer 354.2 us base / 352.1 LIBERO (go line 355; base margin 0.8 us, verify-p2-r0 had 353.1); VLM 1618 / 1526 us (<= 2200); prefix 38.06 / 35.74 ms (< 52.42 / 48.72) | L_base.json, L_libero.json |
| Size | PASS 128,636 B base / 126,492 B LIBERO (empty mock cache; identical to verify-p2-r0) | M_size_*.json |
| Served A/B (P2-5) | PASS: inference median whole 55.97 / 55.91 vs expert 70.84 / 70.87 ms (P1-6 70.72); total 57.57 / 57.42 vs 72.29 / 72.37; smoke PASS x4; /info backend = arm on every server; mask probe pass x4 (batcher, batch 1: [2, 0] vs [2] differ by 0.837, repeat identical); no uvicorn left after each stop | S_*.json, S_*.smoke.log |
| Startup refusals of the default | PI05_NUM_IMAGES=1 (new), PI05_BATCH_SIZES=1,2, PI05_KV_DTYPE=bf16: rc 3, "PI05_MEGAKERNEL=whole refused at startup: ...", no "Opening device" line | refuse_default_*.log |

## 2026-10-01 13:01:24 KST -- integrate-p2: LIBERO closed loop, demo, docs (session end)

- LIBERO with the default (/home/deepgadget/experiments/gr00t/libero_eval/pi05/megakernel-p2/; tools/ = copies of
  ../megakernel-p1/tools (themselves ../fused/tools + device_kwargs + stamp), paths / captions changed, run_fused.sh unsets
  PI05_MEGAKERNEL, run_all.sh adds WITH_DEVICE_RESET_AFTER=1; make_manifest.py new; copies in integrate_p2/results/libero/):
  - Open-loop golden through the wrapper (openloop_pcc_mkp2.json): PCC7 mean 0.999976, min 0.999955, deterministic,
    host preprocessing ok, steady call 53.5 ms; stamp megakernel=whole.
  - libero_spatial 10 tasks x inits 0-9 (tt_spatial_summary.json): **99/100**, 0 errors, 0 timeouts, 0 missing; the one
    failure is t9/i4 at the step cap (230 steps; also the only failure of the phase-1 run). Paired inits 0-4: 49/50 vs GPU
    50/50 (only_gpu [9, 4]); inits 5-9: 50/50. ACCEPTED (>= 80 and paired >= GPU - 10 pts). Server policy latency median
    53.6 ms (p10 53.3, p90 54.1, max 61.4, 2,189 calls) vs 66.4 (expert, 09-30) and 77.6 (off, 09-29). All 100 episodes
    stamped backend=whole, mk_digest=4aa02cdf21ed0c94, code=7f32fccbe6b4 (clean tree). 20 episodes have the same steps and
    outcome as the phase-1 run.
  - Demo (demo/, manifest.json): 4 on-screen viewer recordings t3/i0 96 steps, t7/i0 134, t0/i0 116, t5/i0 114, all
    SUCCESS, each equal to its eval episode's steps; median 54.0-54.1 ms per call. Captioned clips + combined
    pi05_libero_spatial.mp4 (29.13 s) + poster. I LOOKED at the poster (t3 SUCCESS frame, the bowl on the plate, caption
    "Blackhole p150a — the whole-model megakernel (one fused op per call)"), the title card (99/100, 53.6 ms, "SigLIP +
    VLM + expert loop = ONE persistent generic_op per call"), a t7 running frame (12 s) and the end card (27.5 s). Correct.
    Note: render needs /usr/bin/python3 (the tt-metal venv's imageio has no ffmpeg plugin); this host has no ffprobe
    (the manifest probes with ffmpeg -i).
- Docs: README (intro table: what is the ONE persistent op, what stays on the host, the comparator knobs and their op
  counts; Results rewritten from integrate_p2/results + the LIBERO files; GPU ratios recomputed against served 55.97 ms;
  architecture, knob table, tree, troubleshooting), the package README, GPU_COMPARISON.md (2026-10-01 section), DESIGN
  §0 status rows, §4.12 and §7 / §11.3 amendments (2026-10-01).
Ship-phase handoff (NOT done here: the staging package /home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused
is outside the pi0.5 repo):
- tt-model.yaml serve.env: PI05_MEGAKERNEL "expert" -> "whole" (an unset value now resolves to whole as well; keep it
  explicit); PI05_SOURCE_COMMIT = the new main merge sha; header comments (lines 4-5, 46, 93) and the card text (70.8 ms,
  "the SigLIP tower and the VLM prefill run as traced TT-NN ops", 17.49 ms expert device time, the 22-seed expert-oracle
  rows) describe phase 1 and must be rewritten from integrate_p2/results (+ the image's own bench).
- verify: lines: FusedConfig default assert -> 'whole' (expert / off still selectable); the kernels-dir assert lists only
  mk_* (still true for tt/megakernel/kernels) but nothing checks tt/megakernel/kernels_p2/ (9 files: pe_{brisc,common,defs,
  dm,ncrisc,trisc}.hpp, whole_{brisc,ncrisc,trisc}.cpp) or pe_program.kernel_digest2() == '4aa02cdf21ed0c94'; the
  all-sources include check asserts len(ks) == 14, now 23 (checked on the repo tree at 7f32fcc: 23 sources, 69 quoted
  includes, all resolve to a sibling or to tt_metal/hw/inc; kernels_p2 includes ../kernels/mk_*.cpp and
  internal/circular_buffer_interface.h). The image's smoke must read /info megakernel.backend == whole.
- Apply NOTES_FOR_SHIP_P2.md items 1-5 when the card is rewritten.
Open (unchanged, not gates): seed 707 is the one of 32 seeds where expert is closer to fp32 than whole (0.99769 vs
0.99522; whole still closer than off); the SigLIP layer's host-clock margin under the 355 us go line is small (354.2 us
base here, 353.1 in verify-p2-r0); under whole the per-request copies to the unused ttnn prefix inputs (im2col / tokens)
still happen (host-side, small); the ttnn prefix modules are built and hold device DRAM though never enqueued.

## 2026-10-01 13:05:37 KST -- publish-p2: Ship phase 2, the whole-model megakernel (session start)

Branch HEAD 54da605 (integrate-p2 end). models/ diff 0f19387..54da605: package README only (kernels and Python unchanged
since the gate re-run; kernel_digest2 4aa02cdf21ed0c94). Plan: GitHub doc fixes of NOTES_FOR_SHIP_P2.md (items 1-5),
push, PR, merge commit (trailer in the merge body); staging package = git archive of the merge's models/, tt-model.yaml
serve.env PI05_MEGAKERNEL=whole + PI05_SOURCE_COMMIT = merge sha, verify: lines for kernels_p2 + kernel_digest2 + the 23-source
include check; package ONCE on publish/tt-metal-668c2907575; boot + smoke + 100-request bench (2 cycles) via with-device.sh;
ONE HF create_commit (parent pinned cf08fb95); verify (sha256, code/ vs GitHub main, image pull-back, video).
NOTES items applied to the GitHub docs in this session:
- 1 (replay "median of 60"): no phase-1/2 occurrence left in the GitHub docs; GPU_COMPARISON's 09-29 "median of 60" is correct
  (docs/fused_fix_2026-09-29/fix2_base.json latency.runs = 60).
- 2 (profiler basis): README "device time per replay" row now names the basis per path: whole median of 21 replays,
  expert / off median of 3 / 2 profiled replay sessions (integrate_p2/results/S_struct.json n_sessions).
- 3 ("off" = all stock): README (comparator section, knob table) and the package README now say stock TT-NN ops plus the
  3 custom programs (fused attention, row_rsqrt, geglu_rc).
- 4: integrate_p1/results/hold4_serve_invalid/S_struct.json moved (git mv) to integrate_p1/results/S_struct.json; the
  integrate-p1 JOURNAL citation names the new path.
- 5: GPU_COMPARISON 09-29 section quoted cycle 1 of the FIRST, unshipped build (image ca3d23378236: 84.0 / 85.3 / 86.9 ms,
  scratchpad pub/logs/bench-c1-r0-*); now cycle 2 of the shipped image 672900e23919 (bench-c2-r1-20260929-165458.json:
  84.05 / 85.30, p10 83.88, p90 84.26, client 86.75); ratios unchanged at 2 decimals.

## 2026-10-01 13:26:33 KST -- publish-p2: SHIPPED (session end)

Files: docs/megakernel/publish_p2/{scripts,results}/ (copies of scratchpad/pub2).
- GitHub: docs commit 23fc3dd (the NOTES_FOR_SHIP_P2 items, see the session-start entry), branch pushed, PR #3
  (https://github.com/changh95/tt-pi-0.5/pull/3) merged with a merge commit: main = 821e8c528dfffa0d1d6e73ad6abf181a749b39d9
  (parents f7f173b + 23fc3dd; the message ends with the Co-Authored-By trailer). The local branch was fast-forwarded to
  821e8c5 and pushed; work continues on megakernel-2026-09-29.
- Staging /home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused: code/models = git archive 821e8c5 models (107
  files, file list identical, no __pycache__). tt-model.yaml (scripts/yaml_edit.py): header comments describe the whole-model
  op; serve.env PI05_MEGAKERNEL=whole (comment: expert / off are comparators, off = stock ops + 3 custom programs, 2,551
  ops), PI05_SOURCE_COMMIT=821e8c5...; verify: pe_* imports + WholeMegakernel; FusedConfig default ({} and '') == whole,
  expert / off selectable; kernels_p2 = exactly the nine files, KERNELS2 exist, 14 KERNEL_SOURCES2, kernel_digest2() ==
  4aa02cdf21ed0c94 (phase-1 kernels line kept: 328761c8a1ce3fd9); include check over all 23 port C++ sources (69 quoted
  includes) resolving to a sibling path (incl. ../kernels/mk_*.cpp) or tt_metal/hw/inc. Host-run on a fake /opt layout
  (scripts/run_verify.py): all pass except the 3 server.app imports (fastapi not in the host venv); negative controls: a bogus
  include appended to kernels_p2/pe_dm.hpp failed both the digest (f5517f40e1d88012) and the include check; removing
  whole_ncrisc.cpp failed the file-list and include-count lines.
- Package (results/pkg-pi05-mkp2.log, buildkit log .gz): rc 0, 13:09-13:14, all verify lines ran in the image (#45 DONE).
  Image tt-model/pi05-base-p150:6fb244df57ff, digest sha256:6fb244df57ff8d20da139e37a3cf1fb38a4d68d5381240e0dd64052e033b53b4,
  tt_metal 668c2907 dirty=false, code_sha256 8016f1493a3ee10f...; build code/models == GitHub main models (+ lightweightmodule.py).
- Boot validation (results/validate.log, one with-device hold 13:14:55-13:18:18, WITH_DEVICE_RESET_AFTER=1, reset exit 0):
  cycle 1 cold (package cache removed; serve 75.9 s, first warm-up 23.6 s) and cycle 2 warm (57.2 s, 21.7 s). Both: /info
  megakernel.backend whole, kernel_digest 4aa02cdf21ed0c94, source.commit 821e8c5; smoke_test PASS; 100 warm requests,
  identical actions. timing_ms.inference median 56.08 (c1) / **55.84 ms (c2)**, p90 56.13; total 57.39 / 57.13; client wall
  59.23 / 58.52 (results/bench-c{1,2}-mkp2-*.json). Card / GPU_COMPARISON use c2; previous image fe0d2e3d68a7 c2 = 70.81 / 72.16.
- HF changh95/pi05-base-p150: ONE create_commit **990e22b54012d7a80055d9e9762676b3e49bdbe5**, parent pinned cf08fb95.
  172 adds (code/ incl. the 24 new kernels_p2 / pe_* / test files, image/ OCI blobs, tt_kernel_manifest.json, README,
  SERVING, GPU_COMPARISON, tt-model.yaml, requirements.lock (the image's own lock; package versions moved, e.g. fastapi
  0.141.1 -> 0.142.2), demo/{README.md, libero_eval.json, 4 clips, combined mp4, poster}); 35 deletes (only superseded
  image blobs); the demo file names are unchanged, so the old demo content was replaced in place. YAML front matter kept
  (no outdated field). The card says what is the ONE persistent op and what the host does; every number comes from
  bench-c2, integrate_p2/results (S_struct, A_*_r1, G_gates_r1, M_size_base), p2/gates/results/g1_pcc_whole_base.json
  (padding), or the megakernel-p2 LIBERO files. NOTES items kept on the card: replay "median of 30", profiler basis per path
  (21 sessions / 3 / 2), off = stock ops + 3 custom programs, previous-image figures from the same cycle (c2). SERVING.md: boot
  times re-measured; its misordered tt-metal paragraph fixed.
- Verify (results/hf_verify.log): 175 files in the tree, none missing / unexpected, deleted blobs gone. sha256: 157 files
  downloaded + 15 LFS image blobs via lfs.sha256, 0 mismatches. code/: all 107 GitHub-main files (fresh clone at 821e8c5)
  byte-identical (HF-only = tt-metal's common/lightweightmodule.py). Image pull-back (results/pull.log): removed the local
  image, tt-model pull docker-loaded sha256:6fb244df57ff8d20da139e37a3cf1fb38a4d68d5381240e0dd64052e033b53b4 (= recorded
  digest). Served the pulled package by repo id (results/serve_pulled.log, hold 13:2x, reset exit 0): /info whole,
  4aa02cdf21ed0c94, source 821e8c5; smoke PASS; 30 warm requests inference median 56.06 ms. Headless Chromium
  (results/pw_video.log): the README video played (readyState 4, t 3.99 s of 29.13 s = the new combined clip, no media
  error; the replay.mp4 404 comes from the HF page itself, as on 09-30).
- Disk: deleted the superseded docker image tt-model/pi05-base-p150:fe0d2e3d68a7 (3.02 GB, the 09-30 phase-1 image; still in
  HF history at 9f6b082b / cf08fb95). build/pi05-base-p150 was overwritten by the new package. 16 GB free after.
Open (unchanged, not gates): seed 707 (expert closer than whole: 0.99769 vs 0.99522); SigLIP layer host-clock margin under
355 us is small (354.2 us); the GitHub README quotes the in-process served A/B (55.97 ms), the HF card the image bench (55.84 ms).

## 2026-10-01 13:35:15 KST -- shipcheck-p2: independent check of the p2 release (HF 990e22b5, GitHub main 821e8c5)

Files: docs/megakernel/shipcheck_p2/ (serve_check.sh, serve_check.log, info.json). Result: all checks hold, no discrepancy.
- GitHub main (fresh clone) = 821e8c5; HF snapshot @ 990e22b5 code/models vs clone models: identical except HF-only
  models/common/lightweightmodule.py (tt-metal file, expected).
- Image: removed local tt-model/pi05-base-p150:6fb244df57ff (13:29), tt-model pull re-loaded image id sha256:6fb244df57ff8d20...
  (= recorded digest). /opt/tt-metal/models/experimental/pi0_5 + models/common copied out of the pulled image: identical to HF code/.
- Served the pulled package (with-device, WITH_DEVICE_RESET_AFTER=1, timeout 1200, 13:31-13:33, reset exit 0): /info backend
  whole, kernel_digest 4aa02cdf21ed0c94, source 821e8c5; env PI05_MEGAKERNEL=whole; HF code/ smoke_test PASS; 30 warm requests
  (after 5) inference median 55.87 ms (55.61-56.39), identical actions, head [-0.0708,-0.1553,0.2969,0.1006] = card example.
- Card numbers re-read from files: bench-c2 55.84/p90 56.13/57.13/58.52; previous image publish_p1 bench-c2 70.81/72.16; replay
  A_*_base_r1 54.10/69.56/82.82; S_struct 53.94 (21) / 69.49 (892 ops, 3) / 82.30 (2551, 2); golden A_*_libero_r1 0.999976/0.999955,
  0.999884/0.999778, 0.999839/0.999712; G_gates_r1 seeds 32/32 vs off, 31/32 vs expert, means/mins match, seed 707 0.99522/0.99769/0.99175;
  p23 288/288 both shapes, mins 0.99171/0.92868, 0.99905/0.99060; M_size_base 128,636 B (ring 136,192, gate 131,072); padding
  g1 pad_ids_invisible_bit_identical; ALT_whole 20/20; soak 20/20 digest 886f7341c1085571; GPU ratios recomputed.
- LIBERO recomputed from tt_spatial.jsonl (= the gr00t/libero_eval copy): 100 unique episodes, 99 success, 0 errors, fail t9/i4 at 230
  steps; init 0-4 49/50; GPU ref_gpu_spatial.jsonl 50/50; server log 2,189 calls median 53.6 p10 53.3 p90 54.1 max 61.4; every
  episode stamp megakernel=whole mk_digest 4aa02cdf21ed0c94; history rows (off 98/100 48/50 77.6, expert 99/100 49/50 66.4) match files.
- Demo: the 5 HF mp4 + poster are byte-identical to libero_eval/pi05/megakernel-p2/demo; all decode fully with cv2 (h264 960x1080
  30 fps; combined 874 frames = 29.13 s); resolve URLs return 200 (mp4 1,558,428 B). recorded_runs.jsonl: 4 runs, whole, success.
- Stale-claim grep (card, SERVING, GPU_COMPARISON, demo README, GitHub README / GPU_COMPARISON / package README): no traced stock-op
  stage is called a megakernel. Only "megakernel" tokens in the 09-14 GPU section are historical file paths (reports/megakernel/...,
  logs/publish-megakernel/...) in a section marked "as recorded". Wording note (not a number error): "same benchmark cycle" for the
  previous image means cycle 2 of its own 09-30 validation, not the same session.

## 2026-10-01 13:43:40 KST -- final-critic: completeness check + REPORT.md

Files: docs/megakernel/final_critic/ (prof_hold.sh, mkrun.py, analyze.py, pw_fc.py, hold.log, FC_prof_pulled.json,
cpp_device_perf_report.csv.gz, requests.json, info_*.json, video_frame.png). Report: docs/megakernel/REPORT.md.
- Pulled image 6fb244df57ff served unmodified under the device profiler (container spec copied from tt-model serve's
  container; private TT_METAL_CACHE; one hold 13:38:15-13:41:25, WITH_DEVICE_RESET_AFTER=1, reset exit 0; the guard's
  "probe hung/failed" wording on the RESET line is its fixed message for the after-job reset, rc=0): 14 replay sessions
  (2 warm-up + 12 HTTP requests, 3 prompts) x exactly 1 program on 110 cores, one program id, 0 device programs after the
  first replay outside the trace; device kernel median 53.96 ms (53.90-54.07; S_struct 53.94). Actions identical per prompt.
- Card numbers, LIBERO (recomputed from tt_spatial.jsonl: 99/100, 49/50, all whole + digest), HF video in headless
  Chromium (plays, t 16.0 s / 29.13, frame shows the whole-model caption), stale-claim grep, NOTES items 1-5: all hold.
- No new open problems; the carried list is in REPORT.md.
