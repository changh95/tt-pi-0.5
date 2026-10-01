# pi0.5 as a true megakernel on one p150a: design v2 (2026-09-30)

Goal (the user's words, not reinterpreted): **all of the model's ops become ONE fused op**, a persistent program of custom
kernels (`ttnn.generic_op` + `ProgramDescriptor`, like the GR00T DiT megakernel). A Metal trace of stock ttnn ops, or
"fewer launches", is not a megakernel. Two stages in the user's order:

- **Phase 1**: the whole 10-step action-expert denoise loop (18 Gemma-300M layers x 10 steps, adaRMS, GQA 8 q heads + 1
  KV head at head_dim 256 over the prefix KV cache with the padding mask and the offset RoPE, GeGLU MLP, gated residuals,
  Euler step, action in/out projections) as ONE persistent generic_op: in-kernel layer and step loops, weights streamed
  from DRAM, activations in L1.
- **Phase 2**: SigLIP x2 + projector + language embedding + the VLM 18-layer prefill (which produces the prefix KV
  cache) + the phase-1 expert loop, all ONE fused op for the whole `sample_actions`.

**Phase 1 is an intermediate stage, not the deliverable.** The deliverable is phase 2: the traced replay holds exactly one
device op (§4.12, §7 structural gates). If the phase-2 exit gates of §7 cannot be met, the result is status `needs_user`
with the measurements, never "ship phase 1 as the megakernel".

Every number below carries its source. **M** = measured (file named), **D** = derived arithmetic from measured inputs,
**A** = assumed (not measured on this card for this shape; each A has a work package that measures it and a gate).
Nothing here has run on the device yet. v2 resolves the two reviews of v1 (ac2e6ea); §10 records every item and how it
was resolved. The v2 arithmetic is `docs/megakernel/design/mk_design_calc_v2.py` (output `mk_design_calc_v2.out`); the
stream timeline uses the review's fluid model `docs/megakernel/design/review_stream_sim.py`.

## 0. Status

| item | status | notes |
|---|---|---|
| profile of the shipped path (`PROFILE.md`) | DONE | b93723c |
| this design (`DESIGN.md`) | v2 | v1 (ac2e6ea) reviewed twice, not accepted; v2 resolves every item (§10). The gates of §7 are fixed from v2 on |
| WP-P1-0 scaffolding + host checks + mock compile + device-open / L1 facts | DONE (2026-09-30) | two-format CB not needed (bf16 ctx); L1 guard + free-L1 measured; JOURNAL |
| WP-P1-1 weight arena + stream harness | arena DONE; harness gate not run | the integrated loop meets the speed gates |
| WP-P1-2 exchange microbenchmarks (the A-values of §4.3, both shapes) | NOT RUN | superseded for phase 1 by the end-to-end result; still useful for phase 2 |
| WP-P1-3 one expert layer in-kernel | DONE (bring-up b3: layer 0 PCC 0.99992 vs host fp32) | |
| WP-P1-4 whole 10-step loop, standalone | DONE (b5; P1-4 gate vs shipped 0.99987+ on 8 golden obs) | |
| WP-P1-5 integration into `sample_actions_fused` + all gates | DONE except the base per-seed gate (seed 2) | JOURNAL phase-1 exit gate table |
| WP-P1-6 server integration of phase 1 | DONE (served 70.72 ms) | |
| integrate-p1: phase 1 as the DEFAULT path (`off` = comparator), server mask plumbing, refusal (d), gate re-run, LIBERO closed loop | DONE 2026-09-30 (gates re-run PASS; served 70.9 vs 84.0 ms; LIBERO 99/100 at 66.4 ms/call) | JOURNAL "integrate-p1"; `integrate_p1/results/` |
| WP-P2-0 phase-2 binary size + L1 overlay proof (no card) | DONE (2026-09-30) | 128,636 B base / 126,492 B LIBERO <= 131,072 (p2/results/mock36.json; verify_p2_r0 M_size_*.json) |
| WP-P2-1 VLM layer prototype (go / no-go on timing) | DONE: GO (1.612 ms <= 2.20) | PCC 0.99987 vs the ttnn layer (p2/gates/results/L_layer_vs_ttnn.json, p2/results/b41.json) |
| WP-P2-2 SigLIP layer prototype | DONE: GO (353.6 us <= 355) | PCC 0.99992 vs the ttnn layer; host-clock 353.1 us (verify_p2_r0 L_base.json) |
| WP-P2-3 whole prefix in-kernel (SigLIP + projector + embed + VLM -> KV) | DONE on the 2026-10-01 amended clause | stack 37.3 / 35.7 ms; K/V closer to fp32 than the ttnn caches 36 / 36 per shape (P23_*.json), 288 / 288 (verify_p2_r0 G_gates_r1.json) |
| WP-P2-4 whole `sample_actions` as one program + all gates | DONE 2026-10-01 (accepted on the amended gates) | JOURNAL "PHASE-2 EXIT GATE TABLE" and "verify-p2-r0" |
| WP-P2-5 server integration of phase 2 | DONE (served 55.7 ms vs expert 70.9) | p2/gates/results/S_*.json |
| integrate-p2: `whole` as the DEFAULT path (`expert` / `off` = comparators), gate re-run on the final commit, LIBERO closed loop | DONE 2026-10-01 (every gate re-run PASS; served 55.97 / 55.91 vs expert 70.84 / 70.87 ms; LIBERO 99/100 at 53.6 ms/call) | JOURNAL "integrate-p2"; `integrate_p2/results/` |

## 1. Inputs and hard limits

- Target: lerobot/pi05_base (served shape: 2 x 224^2 cameras, 224 tokens, H = 50 -> 64 suffix rows, prefix P = 736) and
  lerobot/pi05_libero (32 tokens, H = 10 -> 32 suffix rows, P = 544). Batch 1 (§4.12, §8 open question 2).
- tt-metal `/home/deepgadget/experiments/gr00t/tt-metal` @ 668c2907575 (read only). All new code lives in the pi0.5 repo
  under `models/experimental/pi0_5/tt/megakernel/` (GR00T code is copied in, never imported, §6.1).
- **Kernel-config ring: 136,192 B per program** (`1,531,904 - worker_l1_size`, worker_l1_size 1,395,712 = the 64 KiB
  cut; `mk_k5_summary.md` §2.2). It holds the binaries of the largest kernel group, the runtime args and 16 B per CB id
  up to the highest id used (`UINT32_WORDS_PER_LOCAL_CIRCULAR_BUFFER_CONFIG = 4`, tt-metal
  `circular_buffer_constants.h:40`; `dispatch.cpp:326` sizes the local CB config as `max_local_end_index x 16 B`). Checked by an offline mock-cluster compile + size check before any device run.
- **64 KiB worker-L1 cut** -> allocatable L1 per core **1,371,136 B** = 1,395,712 - l1_small_size 24,576 (pi0.5 opens the
  device with `l1_small_size=24576`, `tests/pcc/test_pcc_pi05_fused.py:68`; GR00T's 1,362,944 assumed 32,768). The CB
  ceiling is an **address** per core range, not a sum (memory note `megakernel-k4-device-facts` addendum): WP-P1-0
  measures the minimum contiguous free L1 above the CB base at the generic_op's position in the trace.
- **CB ids: 64 on Blackhole** at this tt-metal (`circular_buffer_constants.h:38`, `hal.hpp:531`: 32 only for WORMHOLE_B0).
- Grid 11 x 10 = 110 worker cores, aiclk 1350 MHz (M: `profile/clock_*_20260930.json`).
- DRAM stream rates (M, `mk_k1_summary.md` §0): **464 GB/s bf16, 414 GB/s bfp8** direct-mode arena reads; K1's matmul-only
  block chain delivered 479 GB/s bf16 / 383 GB/s bfp8 effective.

## 2. What must be beaten (from `PROFILE.md`, all M)

| shape | whole call | replay | SigLIP | VLM | expert (10 x 18) | action io | expert per layer |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | **84.21 ms** | 82.81 | 10.767 | 41.536 | **31.263** | 0.167 | 173.1 us |
| LIBERO | **76.77** | 75.64 | 10.765 | 37.853 | **27.845** | 0.163 | 153.8 us |

The stage columns are profiled device times; the whole-call column is unprofiled. The current expert layer (base, M,
`PROFILE.md` per-expert-layer table): adaRMS `rms_norm` on 2 cores 14.2 us, qkv 9.5, fused attention (16 cores) 43.0,
typecast 2.0, o_proj dit (22 cores) 20.6, row_rsqrt (2 cores) 7.1, up|gate 26.7, geglu_rc 7.6, down dit (22 cores) 36.3,
plus 9 launch gaps. It streams 24.28 MB per layer (D: qkv 2,560 bfp8 tiles, o_proj 2,048 bf16, up|gate 8,192 bfp8, down
4,096 bf16), 437.1 MB per step, 4.371 GB per request; the matmuls stream at 261 GB/s while active and 140 GB/s over the
stage. The replay holds 2551 device ops at both shapes (`PROFILE.md`).

## 3. What the prior attempts teach, and how this design uses it

### 3.1 Why the p300 chained-matmul PoC lost (`docs/history/MEGAKERNEL_p300.md`)

The PoC chained 18 stages of `[64,1024] x [1024,1024]` in one program with the **weights resident in L1** (32 cores,
bf8 W). Per stage (M, p300): compute ~3 us, but the activation exchange (gather 124 KB into one sender + one 128 KB
multicast, back to back on one core's links) cost >= 7 us, so the best variant (v3) ran 10.8-11.4 us per stage against
9.2 us for `ttnn.linear`, which already pays the same exchange inside its op. 32 concurrent multicasters serialised
(v2: 137.6 us); one sender per grid row multicasting to the whole grid was slower than one sender (v5: 15.2 us). The
lesson it wrote down: *no in-program fusion of matmul stages beats the launch-per-op baseline; the megakernel budget
must go to the ops whose cost is not data movement.*

Three things made that PoC unwinnable, and this design removes each:

1. **Nothing to hide the exchange behind.** With resident weights a stage is 3 us of compute plus >= 7 us of exchange,
   in series. Here the weights stream from DRAM (they cannot be resident: 18.4-24.3 MB per layer x 18 layers >> 150 MB of
   L1). Each core's NCRISC walks its own consumption-ordered share of every layer and every step into two rings, each at
   least as large as the core's largest single-op share in that format (§4.4). A ring therefore never blocks the stream
   in the middle of an op, and the next op's share streams in while the current op's exchanges run. The per-core
   ring-constrained timeline (§4.11) shows the stream is not the bound at either shape: the layer is chain-bound at
   76.8 us (base) and 61.1 us (LIBERO) against a 58.7 us stream at 414 GB/s. (v1 claimed "DRAM never idles"; with v1's
   87 KB bfp8 ring that was false, see §10 B2.)
2. **The exchange unit was the matmul stage.** Here it is the layer: 3 hub all-gather rounds per layer (x, ctx, x_mid)
   plus the group-local exchanges of R2, R3, R6 and R7, against 9 launches today and GR00T's 11 hub rounds per DiT block. No exchange is an
   all-to-all of multicasts: every one-to-many step is ONE multicaster per rectangle (gather-to-leader by unicast, then
   one multicast), and **concurrent multicasts go to disjoint rectangles only** (§4.6). This respects both the p300 rule
   (one multicaster per program region) and the GR00T in0 law (~1.8 us per other owner when L owners multicast into a
   band, `mk_f0c_summary.md` §4.3).
3. **The stage it fused was already efficient in ttnn.** The time this design removes is exactly the non-matmul time
   the p300 note pointed at: 14.1 of the 31.3 ms expert stage (M) is not matmul — the 2-core adaRMS norm, 43 us of
   attention on 16 cores, two small custom programs, a typecast, 9 fixed kernel costs per layer — and the o_proj/down
   dits run on only 22 cores at 203/231 GB/s.

### 3.2 GR00T DiT megakernel (same card, same tt-metal) — what is reused

- **Measured round costs** (`mk_k4b_summary.md` §3.6, bfp8, warm self block 105 us): hub round (96 x 2 KB gather ->
  192 KB multicast, dual NoC) ~6 us on top of compute; attn_out round 10.3 us = Wo 3.7 + round; Q unicast 1.5 us; attention
  part send 1.5 us + two merges ~7 us (a composition estimate in §3.5, not a separate measurement; each part there was
  2 rows x 2 dh tiles); ff2 chunk rounds are latency-bound at ~5.3 us quiet / ~11 us in the steady state, one chunk in
  flight per hub (`mk-chunk-rounds-latency-bound` memory note).
- **Measured compute rate**: QKV `mk_P2_mm` 5.5 us for 192 tile-matmuls per core (2 rows x 2 N tiles x 48 K) at HiFi2 ->
  **T_MM = 0.029 us per tile-matmul** (D). Used for every matmul-compute estimate below (conservative for LoFi ops).
- **Eltwise / SFPU tile-op rate** (new in v2, A): **T_E = 0.125 us per tile-op**, from GR00T's 3.5 us per merge divided by
  its 28 tile-ops (2 rows x (max 1 + sub 2 + exp 2 + l update 3) + 4 O tiles x 3). The shipped fused attention bounds it
  below (1.72 us per key tile for ~20 tile-ops = ~0.09 us). Every non-matmul compute term below is priced by counting
  tile-ops at T_E; WP-P1-2 measures T_E on the merge and reduce microbenchmarks.
- **Realisation ratio**: the plan estimated the bfp8 block at 77 us; K4 measured 129.3 (x1.68), K4b 105 (x1.36)
  (`mk_k4_summary.md` §0, `mk_k4b_summary.md` §0). Every bottom-up time below is also quoted x1.36 ("expected") and
  x1.68 ("first build"). These ratios come from a stream-bound DiT; they are used for phase 1 (same regime) and NOT as
  a bound for the compute-bound phase-2 prefill (§5.4).
- **Protocol**: round-counter flags (absolute values, `noc_semaphore_set_multicast` of a local word), receiver credits
  announced ahead of the hub (parity-split counters), dual-NoC hub multicast (NCRISC half + BRISC half + flag after the
  NCRISC flush), hub landing ring, per-member absolute credits taking the MINIMUM (a summed credit is wrong from ring depth
  3, F0 fact 4), every sync word on its own 16 B line (F0 fact 6), BootBarrier after zeroing (F0 fact 5; semaphores
  are re-initialised on every enqueue including trace replay).
- **Device facts that shape the code**: `pack_tile<true>` whenever an output index is used (F0 fact 1); no DST group
  across a ring wrap, ring slots a multiple of the group (F0 fact 3); `dst_full_sync_en` gives 8 fp32 DST tiles (F0
  fact 8); a CB in `UnpackToDestFp32` must not be an FPU operand (F0 fact 2); PACK-side GELU init re-issued before every
  GELU pack when MATH-side SFPU ops exist (K4 fact 3); `noinline, noclone` on every multiply-called TRISC helper (K4c:
  -18.3 KB); two same-width FPU operands only (tensix-operand-format-rules); `cb_pop_front` is not consumption
  (a slot producer waits for the NEXT pop); `invalidate_l1_cache()` + compiler clobber before reading runtime args
  and polled words (rule 27).
- **Integration**: K5 ran the DiT megakernel as one generic_op inside a trace, with in-trace transient L1 residents
  (allocated before the op, freed after) and traced == untraced bit for bit (`mk_k5_summary.md` §2.2). The untraced
  launch of a 45-49-tensor generic_op costs ~3 ms of host dispatch, so the megakernel must always be traced.

### 3.3 GR00T fused prefill (the warning for phase 2)

`BACKBONE_FUSION_PLAN.md` (head) and the memory note `fused-backbone-e2e-slower-than-ttnn-n16`: the fused
Qwen3/SigLIP/VL-SA stages ended **slower** than TTNN end to end (0.844x / 0.66x / 0.77x). Causes measured there: the in0
band exchange at ~1.8 us per other owner per matmul round (~38 us per matmul at L = 22), weight delivery limited by a
unicast forward (129-206 GB/s against 464), concurrent band multicasts costing 148-168 us per layer until staggered,
GELU epilogues at ~1.5 us per output tile on the PACK thread, a per-block launch floor, exchanges at 26 % of the layer,
and matmuls realised at 7-15 % of peak inside the fused layer against 39-55 % in the F0 harness. Phase 2 is therefore
budgeted by a measured prototype (WP-P2-1 is a go/no-go), and its geometry (§5.3) avoids the per-owner in0 exchange by
construction: one in0 multicaster per band, weights read once per column and multicast down the column.

## 4. Phase 1: the expert denoise loop as one persistent program

### 4.1 What the kernel computes (per step s, layer l)

Constants per (s, l) are precomputed on the host at load time (the adaRMS modulations depend only on the fixed
timestep schedule, as today's `_precomputed_block_mods`):

- adaRMS fold (today's `tt/ttnn_fused_norm.py`, extended to the attention side):
  `rms(x)*(1+a)+b @ W = r ⊙ (x @ W') + c` with `W' = diag(1+a) W` (bfp8, per step), `c = b @ W`, `r = rsqrt(mean(x^2)+eps)`
  per row. Both norms of every layer are folded: `Wqkv'[s,l]`, `c_qkv[s,l]`, `Wug'[s,l]`, `c_ug[s,l]`. The in0 of qkv and
  up|gate is therefore the **raw** residual x: no norm pass sits before a multicast. `r` is a per-row scalar computed by
  the hub while it multicasts x (§4.3).
- `q,k,v = r_in ⊙ (x @ Wqkv') + c_qkv`; RoPE on q and on the suffix k at positions `n_valid + [0, S)` (tables are inputs,
  §4.5; 1/sqrt(256) folded into the q tables, the rotate-half sign into the sin tables, as today). head_dim 256 = 8 dh
  tiles and rotate-half pairs dh tile d with d ± 4, so RoPE never needs an intra-tile rotation.
- attention of the S suffix rows over [prefix keys (from the KV cache) | suffix keys], additive key mask (prompt pads and
  the suffix tile-padding rows), 8 q heads sharing 1 KV head.
- `x_mid = x + g_attn[s,l] ⊙ (ctx @ Wo)`; `u|g = r_post ⊙ (x_mid @ Wug') + c_ug`; `h = u * gelu_tanh(g)`;
  `x = x_mid + g_mlp[s,l] ⊙ (h @ Wd)`.
- per step tail: final adaRMS folded into `Wout'[s]` / `c_out[s]` (includes the out-proj bias): `v = r_f ⊙ (x @ Wout') +
  c_out`; Euler `x_t += dt * v` (dt = -1/10) in fp32; next step's input `x = x_t @ W_in + b_in`.

Weights in DRAM: `Wqkv'` and `Wug'` exist once per step (10 copies; D: 180 x (2.785 + 8.913) MB = 2.106 GB), `Wo`, `Wd`
once (18 x (4.194 + 8.389) MB = 0.226 GB). Today's path already keeps the 10 up|gate copies (1.6 GB).

### 4.2 Core map (logical coordinates; virtual coordinates are read from the device, never hand-computed)

```
 y\x   0    1    2    3    4    5    6    7  |  8    9  | 10
  0   T1   T1   T1   T1   T1   T1   T1   T1  | KV   KV  | spare
  1   T1   T1   ...                       T1  | KV   KV  | spare
  2   T2   ...                            T2  | KV   KV  | spare
  3   T2   ...                            T2  | KV   KV  | spare
  4   T3   ...                            T3  | KV   KV  | spare
  5   T3   ...                            T3  | KV   KV  | spare
  6   T3   ...                            T3  | KV   KV  | spare
  7   T3   ...                            T3  | KV   KV  | spare
  8   T4   ...                            T4  | KL   sp  | H1 (ctx hub, 10,8)
  9   T4   ...                            T4  | sp   sp  | H0 (x hub, 10,9)
```

| role | cores | what they do |
|---|---|---|
| **Q producers** | (h, d), h = x in 0..7, d = y in 0..7 (64) | qkv: Q column tile (head h, dh tile d), both row tiles; **RoPE of their own tile** after a pair exchange with (h, d ^ 4) |
| **K / V producers** "KV" | (8, d) K tile d, (9, d) V tile d (16) | qkv: suffix K / V column tile d; K producers RoPE their tile after a pair exchange with (8, d ^ 4) |
| **KV leader** "KL" | (8, 8) | gathers the roped K_s and V_s (32 tiles), multicasts them to the last-chunk attention cores **after** the 8 Q column multicasts have completed (§4.6) |
| **attention units** | (h, y), y = 2*kc + r (80 at base: 8 heads x 2 q row tiles x 5 key chunks) | one key chunk of one (head, q row tile): S, mask, softmax statistics, O part |
| **slice mergers** | units kc = 0..3 of each (h, r) (64 at base) | merger kc owns dh tiles [2kc, 2kc+2) of (h, r): folds the other units' (m, l, O-slice) parts into its own, finalises, sends a 2-tile ctx slice to H1 |
| **Q leaders** | (h, 8) | gather roped Q[h] (8 x 2 tiles) from (h, 0..7), multicast to column h (10 cores) |
| **MLP cores** (kg, ng) | (x = ng, y = kg), 0..7 x 0..7 (64) | up|gate (4 N tiles = 2 u + 2 g), GeGLU, down partial (16 K x 4 N) |
| **row leaders** | (0, kg) | gather the row's h slice, multicast it along row kg |
| **x owners** O_n | n = 0..31 at (x = n // 4, y = 4 + n % 4) (32, the T3 rows) | own x[:, n] (fp32), o_proj column n, gated residuals |
| **hubs** | H0 (10, 9): x and x_mid rounds + the per-step tail; H1 (10, 8): ctx round | gather -> landing ring -> dual-NoC multicast |

Core types: T1 = Q producer + MLP + attention kc 0 + slice merger; T2 = Q + MLP + attention kc 1 + slice merger; T3 = Q +
MLP + attention kc 2/3 + slice merger + x owner + o_proj; T4 = attention kc 4 only (holds the suffix keys; row 8 are the
Q leaders); KV = K/V producers. At LIBERO (1 q row tile, 18 key tiles) the attention units are 8 heads x 6 chunks of 3 key
tiles = 48 cores (rows 0..5, y = kc; the suffix chunk is kc 5), slice mergers are kc 0..3 (rows 0..3) with 5 remote
parts each, rows 6..9 carry no attention work, and the Q multicast rectangle is (h, 0..5). The role tables are host-built
runtime tables (one compiled program per shape, as the trace is per shape today), and a host test asserts every table
is injective and every rectangle is what the role needs.

### 4.3 Layer schedule (base; the LIBERO column uses the same A-values unscaled)

T_MM = 0.029 us per tile-matmul, T_E = 0.125 us per eltwise tile-op, T_KEY = 1.87 us per key tile (the slope of the
shipped fused attention between its two measured shapes, (43.00 - 29.90) / (25 - 18), `PROFILE.md` :223 / :401; the
per-core work at both shapes is one q row tile). No round cost is scaled down for LIBERO's smaller payloads: no
measurement supports a scaling, so the base A-value is used and WP-P1-2 measures both shapes.

| # | phase | cores | data movement | base us | LIBERO us |
|---|---|---|---|---:|---:|
| R1 | **x round**: owners send x tiles to H0 (32 x 2 tiles bf16 = 128 KB landing); H0 multicasts x to rectangle (0..9, 0..7), dual NoC; H0's TRISC computes `r_in` during the multicast and multicasts the 2 `r` tiles after it | 32 -> H0 -> 80 | 128 KB | A 6.5 | A 6.5 |
| C1 | **qkv**: 1 N tile x rows x 32 K per core, fp32 acc, epilogue `r ⊙ acc + c` | 80 | weights in ring | D 1.86 | D 0.93 |
| R2 | **RoPE + Q / K / V distribution**: Q / K producers swap raw tiles with partner d ^ 4 (unicast, 2 tiles) and RoPE their own tile (3 tile-ops per row tile); Q producers unicast roped Q tiles to leader (h, 8), which multicasts Q[h] (32 KB) to column h; K/V producers unicast to KL, which multicasts K_s | V_s (64 KB) to rectangle (0..7, 8..9) **after** the Q multicasts (§4.6) | column-local | 8 x 32 KB + 64 KB | A 1.0 pair + D 0.75 RoPE + A 3.5 = 5.25 | 4.88 |
| C2 | **attention chunk**: S = q K^T over 5 key tiles (K/V chunk prefetched from the cache, §4.5), mask, row max / exp / row sum, O = P V (8 dh tiles) -> part (m, l, O) | 80 | — | 5 x T_KEY + A 1.0 = 10.36 | 3 x T_KEY + 1.0 = 6.61 |
| R3 | **dh-split flat merge**: each unit sends 2-dh-tile O slices + (m, l) to the 4 slice mergers of its (h, r) (12,288 B per slice); each slice merger folds its 4 (LIBERO 5) remote parts into its own in fixed kc order, 14 tile-ops per fold, then finalises | column-local | per unit 3-4 x 12 KB | A 2.0 send + 4 x 14 x T_E = 9.00 | 2.0 + 5 x 14 x T_E = 10.75 |
| R4 | **ctx round**: 64 slice mergers (LIBERO 32) send 2 bfp8 tiles each to H1; H1 multicasts ctx (128 bfp8 tiles, 136 KB) to the 32 owners (0..7, 4..7) | 64 -> H1 -> 32 | 136 KB | A 6.5 | A 6.5 |
| C3 | **o_proj + gated residual** on owner n: 64 K x 1 N x rows, `x_mid = x + g_attn ⊙ acc` (fp32 residual) | 32 | weights in ring | D 3.71 + A 0.5 = 4.21 | 2.36 |
| R5 | **x_mid round** (as R1, H0; `r_post` computed on H0) to the 64 MLP cores (0..7, 0..7) | 32 -> H0 -> 64 | 128 KB | A 6.5 | A 6.5 |
| C4 | **up|gate** 4 N x rows x 32 K, epilogue `r ⊙ acc + c`; **GeGLU** `h = u * gelu(g)` (2 g tiles per row tile, PACK GELU 1.5 us per tile, F0) | 64 | weights in ring | D 7.42 + A 6.0 = 13.42 | 3.71 + 3.0 = 6.71 |
| R6 | **h row exchange**: row kg (8 cores) gathers its h slice at row leader (0, kg) (7 x 8 KB unicast), leader multicasts 64 KB along row kg | 8 rows in parallel | 8 x 64 KB | A 4.0 | A 4.0 |
| C5 | **down partial** 16 K x 4 N x rows | 64 | weights in ring | D 3.71 | D 1.86 |
| R7 | **reduce to owners**: column ng (8 cores) sends each partial tile (fp32) to its owner in rows 4..7 of the same column; the owner sums 8 in fp32 (copy_tile -> SFPU binary: 7 landed copies + 7 adds per row tile) and applies `x = x_mid + g_mlp ⊙ sum` (1 mul + 1 add), 16 tile-ops per row tile | column-local | 7 x rows fp32 tiles per owner | A 1.5 + 32 x T_E = 5.50 | 1.5 + 16 x T_E = 3.50 |

Chain per layer (sum, `mk_design_calc_v2.out`): **base 76.8 us, LIBERO 61.1 us** (v1: 76.8 / 52.3; §10 lists what moved).

The KL path (pair exchange + RoPE on the K producers, 32-tile gather at KL, 64 KB multicast after the Q multicasts) is
off the critical path by construction: a T4 unit's chunk starts with its **prefix** key tiles (base: tiles 20..22 of
chunk 4, 3 x 1.87 = 5.6 us; LIBERO: tiles 15..16 of chunk 5, 3.7 us) and needs K_s | V_s only for its last 2 (LIBERO 1)
key tiles. WP-P1-2 gates that K_s | V_s land before a T4 unit finishes its prefix tiles.

Per-step tail on H0, replacing layer 0's R1 gather->multicast gap: after layer 17 the owners send x to H0 as for R1; H0
computes `r_f` (3 us), `v` (64 tile-matmuls, 2 us), the fp32 Euler update (0.5 us), the next input `x_t @ W_in + b_in`
(64 tile-matmuls, 2 us) and multicasts it as the next step's layer-0 x round; the owners copy their two tiles of that
multicast into their fp32 residual. A 8 us per step. At the last step H0 writes `x_0` to the output tensor instead.

### 4.4 Weight streaming plan

- **Arena**: one DRAM buffer per shape, packed on the host so that each core's NCRISC reads ONE contiguous stream in
  consumption order over the whole request: for s in 0..9, for l in 0..17, its share of `Wqkv'[s,l]`, then (owners)
  `Wo[l]`, then `Wug'[s,l]`, then `Wd[l]`, each followed by the per-(s,l) constant tiles that core needs (`c` rows,
  gates). Transactions are K-blocks of 8 tiles (8.7 KB bfp8 / 16 KB bf16), striped over the 8 banks as GR00T's direct
  mode does (`arena.py`, `weight_stream.hpp`). Interleaved stock tensors would give 1-2 KB transactions for these
  one-column shares, deep in the read knee (196 GB/s at 6 KB, 362 GB/s by 16 KB, `mk_f0c_summary.md` §2 / memory
  `device-hang-triage-rules` item 26), so the arena is required here (unlike F0's backbone finding).
- **Per-core shares per layer** (D): qkv 34,816 B (80 cores); o_proj 131,072 B bf16 (32 owners); up|gate 139,264 B
  (64); down 131,072 B bf16 (64). A T3 core streams 436 KB per layer, a T1/T2 core 305 KB, a KV core 35 KB.
- **Rings** (v2): two per compute core, one per data format: `cb_w8` (bfp8, **16 slots x 8,704 B = 139,264 B**) and `cb_w16`
  (bf16, 8 slots x 16,384 B = 131,072 B). The NCRISC fills them in stream order under per-slot credits; the TRISC
  consumes. **Sizing rule: each ring holds at least the core's largest single-op share in its format** (w8 >= the up|gate
  share 139,264; w16 >= the Wo / Wd share 131,072). Consequences, per core: the qkv share of layer l+1 and the up|gate
  share of layer l never wait for each other beyond one op; on T3 the Wd share enters w16 only after C3 has consumed Wo
  (w16 holds one bf16 op), which the timeline of §4.11 includes. v1's 87,040 B w8 ring was smaller than the up|gate
  share, so the one in-order stream blocked mid-op and Wd could not prefetch (§10 B2).
- **Rate targets**: the layer needs 24.28 MB / 76.8 us = **316 GB/s** aggregate at base (D) and 24.28 / 61.1 = 397 GB/s at
  LIBERO. The stream floor is 55.4 us per layer at 414 bfp8 / 464 bf16 (D: 11.70 MB / 414 + 12.58 MB / 464) and 58.7 us at
  a single 414 GB/s. Pass target for WP-P1-1: >= 400 GB/s sustained over the 180-layer stream with the real shares.
- bf16 o_proj/down stay (today's precision rule, `PI05_FUSED_RESIDUAL=bf16`). All-bfp8 would cut the stream to 44.4 us
  per layer (D); neither shape is stream-bound in v2, so it is not a lever for phase 1.

### 4.5 Runtime inputs: what enters per request, and why the op stays trace-replayable

The generic_op's runtime args hold only addresses and shape constants, fixed per shape. Everything that changes per
request is data in fixed-address tensors that the host rewrites before `execute_trace` (as today:
`copy_host_to_device_tensor`, only when the prefix validity changed):

| input | tensor (today's object) | read by | when |
|---|---|---|---|
| prefix K/V, 18 layers | the backbone-owned caches `[1,1,cache_len,256]` bf8, L1 interleaved (`ttnn_paligemma.allocate_kv_caches`; cache_len 800 base / 640 LIBERO, `fused_host.kv_cache_plan`), written by the ttnn VLM ops earlier in the same trace | attention cores, their chunk rows (5 key tiles x 8 dh x K,V = 87,040 B per layer), on NoC 1, prefetched one layer ahead into a double buffer | every layer |
| key mask row `[1,1,32,P+S]` bf16 (`exp_mask`) | persistent DRAM input | attention cores: their chunk's tiles | once per launch (prologue) |
| RoPE tables cosq/sinq/cosk/sink `[1,1,S,256]` | persistent L1 inputs (`attn_in["tables"]`) | Q producers (q tables, own dh tile, own rows), K producers (k tables) | prologue |
| noise x_T `[1,S,32]` | persistent input (`_fused_in_noise`) | H0 | prologue |
| output x_0 `[1,S,32]` | persistent output (the trace's output tensor) | written by H0 | end |

n_valid enters only through the mask and the RoPE rows, so the program never changes with prompt length. The suffix
K/V are never written to the cache (as today). The in0 landing buffer (`cb_in0`, §4.7) and the hub landing rings are
**transient in-trace L1 residents** (sharded L1 tensors allocated right before the generic_op, freed right after; K5
pattern), never returned to the caller (memory `megakernel-buffers-must-not-escape`). They are tensor-backed so that
their L1 address is known to the host when it builds the runtime args (the hubs and leaders write into them remotely).

### 4.6 Exchange and barrier protocol (reused from GR00T; nothing new in kind)

- **RISC roles** on a compute core: NCRISC (NoC 0) = weight stream only; BRISC (NoC 1) = all exchanges, KV chunk
  prefetch, table loads, in a non-blocking service loop (K4c: a blocking BRISC made the ff2 rounds 6 % slower);
  TRISC = compute. On the hubs: NCRISC + BRISC split every multicast in halves on the two NoCs, TRISC computes r / tail.
- **Generations.** Every exchange instance is identified by the layer generation `g = s*18 + l` (0..179) and its kind.
  **Every sync word is a cumulative counter**, keyed injectively on (what, landing slot, peer) with the generation in its
  VALUE, never in its address (memory rule 3, "count generations where the slot is chosen"; 180 generations x 3 hub kinds
  = 540 instances could never have one word each in 128 words). A word never resets inside a launch, so a stale read
  can only under-report. Two forms:
  - *per-peer word* (one writer): value after generation g is `g + 1`; waited as `>= g + 1`.
  - *summed arrival word* (n writers depositing disjoint bytes of one landing slot, one atomic each): value `n*(g + 1)`;
    waited as `>= n*(g + 1)`. It is valid only when no writer can deposit generation g+1 before the waiter has passed g;
    every summed word below is paired with a per-member ready credit that enforces exactly that, and the host test
    checks the pairing.
  - *credits*: per member, absolute, the waiter takes the MINIMUM over members (F0 fact 4); a producer reuses a slot
    only after the consumer's NEXT pop (`cb-pop-is-not-consumption`).
- **Hub rounds** = `allgather.hpp` `GatherSender` / `McastSender` (K4b form): receivers announce ready credits for round
  (k, g) once they have finished reading the previous occupant of the landing buffer; the hub waits for all sources in its
  landing ring, multicasts, then publishes `hub_flag[k] = g + 1`. The ready point of each landing is fixed: on T3,
  `cb_in0` holds x (R1) -> ctx (R4) -> x_mid (R5) -> next x; the H1 ready credit for ctx is issued after C1 has consumed x,
  the H0 credit for x_mid after C3 has consumed ctx, the next-layer x credit after C4 has consumed x_mid.
- **Leader exchanges** (R2, R6): unicast writes into the leader's source-attributed landing slots + one atomic per
  source into the leader's summed arrival word; the leader waits, multicasts, publishes its flag. **Every receiver of a
  leader multicast, including receivers that are not sources** (the Q-column cores (h, 9); the 16 T4 cores of KL), announces
  a per-member ready credit (`qlead_ready`, `kl_ready`, `row_ready`) after its last read of the previous occupant; the
  leader waits for the minimum before multicasting. The causality chain (a receiver's q read precedes its part send,
  which precedes the next layer's Q) also orders these, which is why the credits never sit on the critical path; the
  credit makes the ordering local and checkable instead of transitive. Destinations that are also sources release their
  landing slot before blocking on the round (the self-membership rule).
- **KL ordering**: KL's rectangle (0..7, 8..9) intersects all 8 Q column rectangles (h, 0..9). KL therefore waits on
  `kl_qdone` (one atomic from each Q leader after its multicast write barrier) before it multicasts; at no time do two
  multicasts with intersecting rectangles run.
- **Merges / reduces** (R3, R7): source-attributed landing slots (part slot index = sender kc on each slice merger;
  reduce slot = sender kg on each owner), one per-peer arrival word per slot, folded in fixed slot order (bit-identical
  across replays), and a per-peer ready credit back to each sender before its slot is reused.
- **Start**: every L1 sync word zeroed, then one BootBarrier (one atomic per core to H0, one multicast back).
- **Multicast hygiene**: NCRISC multicasts pass rectangles end-first; loopback when the sender is inside the rectangle
  (Q leaders, row leaders), `num_dests` excluding the sender otherwise (KL, hubs); sync atomics stay on the unicast VC
  (`transaction-count-is-not-the-cost-model-vc-contention-is`); concurrent multicasts go to disjoint rectangles only
  (the 8 Q columns, the 8 MLP rows), and those are staggered by index when WP-P1-2 shows contention.
- **Hang diagnosability**: one unique waypoint per blocking call site inside every poll loop; every fixed-size kernel
  array has a host-side mirror check; a debug word per core holds (s, l, phase) for post-mortem reads.

### 4.7 Per-core L1 budget (bytes; bf16 tile 2,048, bfp8 1,088, fp32 4,096; `mk_design_calc_v2.out`)

| CB (id) | base arithmetic | base | LIBERO | T1 | T2 | T3 | T4 | KV |
|---|---|---:|---:|:-:|:-:|:-:|:-:|:-:|
| cb_in0 (0 bf16 / 29 bfp8): x / x_mid (bf16) and ctx (bfp8) landing, ONE tensor-backed CBDescriptor with two format descriptors | max(64 bf16, 128 bfp8) | 139,264 | 69,632 | x | x | x | | x |
| cb_w8 (1) | 16 x 8,704 | 139,264 | 139,264 | x | x | x | | x |
| cb_w16 (2) | 8 x 16,384 | 131,072 | 131,072 | x | x | x | | |
| cb_acc (3) fp32 | 8 | 32,768 | 16,384 | x | x | x | | x |
| cb_out (4) bf16: qkv output, then O-part staging (8 dh) | 8 | 16,384 | 16,384 | x | x | x | x | x |
| cb_q (5) roped Q[h] landing | 16 bf16 | 32,768 | 16,384 | x | x | x | x | |
| cb_rpart (6) RoPE partner tiles | 2 bf16 | 4,096 | 2,048 | x | x | x | | x |
| cb_rtmp (7) RoPE scratch | 2 bf16 | 4,096 | 2,048 | x | x | x | | x |
| cb_rope (8) cos/sin, own dh tile, own rows | 4 bf16 | 8,192 | 4,096 | x | x | x | | x |
| cb_kv (9) chunk double buffer | 2 x 5 x 8 x 2 bfp8 | 174,080 | 104,448 | x | x | x | x | |
| cb_ksuf (10) roped K_s, V_s | 32 bf16 | 65,536 | 32,768 | | | | x | |
| cb_mask (11) | 5 bf16 | 10,240 | 6,144 | x | x | x | x | |
| cb_s (12) scores / fold ping-pong | 5 fp32 | 20,480 | 12,288 | x | x | x | x | |
| cb_oacc (13) | 8 fp32 | 32,768 | 32,768 | x | x | x | x | |
| cb_ml (14) m, l (+ ping-pong) | 4 fp32 | 16,384 | 16,384 | x | x | x | x | |
| cb_part (15) slice-merger landing | 4 (LIBERO 5) x (2 bf16 + 2 fp32) | 49,152 | 61,440 | x | x | x | | |
| cb_ctx (16) ctx slice | 2 bfp8 | 2,176 | 2,176 | x | x | x | | |
| cb_r (17) r tiles | 2 bf16 | 4,096 | 2,048 | x | x | x | | x |
| cb_bias (18) c / gate tiles | 6 bf16 | 12,288 | 12,288 | x | x | x | | x |
| cb_ug (19) | 4 fp32 | 16,384 | 8,192 | x | x | x | | |
| cb_h (20) own h | 4 bf16 | 8,192 | 4,096 | x | x | x | | |
| cb_hg (21) row h slice (down in0) | 32 bf16 | 65,536 | 32,768 | x | x | x | | |
| cb_dpart (22) down partials | 8 fp32 | 32,768 | 16,384 | x | x | x | | |
| cb_red (23) owner landing | 7 x 2 fp32 | 57,344 | 28,672 | | | x | | |
| cb_x (24) fp32 residual | 2 fp32 | 8,192 | 4,096 | | | x | | |
| cb_xs (25) send staging | 2 bf16 | 4,096 | 2,048 | | | x | | |
| cb_const (26) ones / scaler / zero / ident | 4 bf16 | 8,192 | 8,192 | x | x | x | x | x |
| cb_fence (27) | 1 fp32 | 4,096 | 4,096 | x | x | x | x | x |
| cb_sync (28) raw sync words, 16 B stride | 128 words | 2,048 | 2,048 | x | x | x | x | x |

- **cb_in0 carries two formats through two ids on one buffer** (§10 B1): one `CBDescriptor` backed by the transient
  sharded tensor, `format_descriptors = [(0, bf16, page 2,048), (29, bfp8, page 1,088)]`, total 139,264 B (68 bf16 pages
  = 128 bfp8 pages; LIBERO 69,632 = 34 = 64). tt-metal builds every format descriptor of a descriptor after binding its
  backing buffer (`circular_buffer_config.cpp:65-100`: the globally allocated address is set, then each
  `CBFormatDescriptor` sets its own id's format and page size, requiring `total_size % page_size == 0`); the Python
  binding exposes `format_descriptors` read-write (`ttnn-nanobind/program_descriptors.cpp`), so the second descriptor is
  appended to the one `cb_descriptor_from_sharded_tensor(0, t)` returns. WP-P1-0 proves it on the device (program
  build accepted, a tiny kernel writes id 0 as bf16 and reads id 29 as bfp8 at the same address). Fallbacks, in order:
  (a) two descriptors `cb_descriptor_from_sharded_tensor(0, t)` and `(29, t)` at `address_offset = 0` (the documented
  several-CBs-per-tensor form); (b) GR00T's demonstrated non-tensor-backed aliased pair (`bb_descriptors.py:846-878`,
  `test_mk_mm2d.py:511 test_aliased_cb_accepted`) with the address read back from the built program. Ordering on the
  shared bytes is §4.6's hub ready points.
- Union of all CBs laid out identically on every compute core (the conservative case): **1,101,952 B at base, 790,656 B at
  LIBERO** (v1: 1,154,560; the changes are +52,224 w8, -49,152 part, -6,528 ctx, -49,152 RoPE moved to the producers).
- Co-tenants at launch time: the 36 L1-interleaved KV caches. An interleaved buffer takes the same address range on
  every bank, in whole pages: 200 tiles per cache at base (cache_len 800; LIBERO 640 -> 160 tiles) -> ceil(200 / 110) =
  2 pages x 1,088 B x 36 = **78,336 B per core** at both shapes (v1's 71,214 was a byte average). Plus the L1 RoPE tables
  and noise (< 20 KB, per-bank whole pages).
- Headroom: 1,371,136 - 78,336 - 20,000 - 1,101,952 = **170,848 B** per core at base (LIBERO 482,144). Because the CB
  ceiling is an address, WP-P1-0 measures the real bound: the minimum over the compute cores of the contiguous free L1
  above the CB base at the generic_op's position in the trace, and the host L1 test compares the union against that
  measurement, not against a sum.
- H0 / H1: landing ring 2 x 139,264 (H0 id 0 bf16, H1 id 29 bfp8) + r / tail scratch 32,768 + x_t fp32 8,192 + W_in
  65,536 + b_in tiles 65,536 + Wout' stream slot 65,536 + r / c_out / noise / const / sync 38,912 = **555,008 B**.

### 4.8 CB-id and sync-word map

CB ids are the (id) column of §4.7: ids 0..28 plus the bfp8 alias id 29 = **30 of the 64 Blackhole ids**; they are kept
dense from 0 because each id up to the highest one used costs 16 B of the kernel-config ring. Hub-only buffers reuse
compute ids on the disjoint hub range with the same data format (landing ring = id 0 / 29, r = 17, x_t fp32 = 24,
W_in / Wout' = 2). Format-bound constraints: `cb_acc`, `cb_x`, `cb_red`, `cb_dpart` carry `UnpackToDestFp32` and are
never FPU operands (fp32 adds go copy_tile -> SFPU binary).

Sync words (each a cumulative counter on its own 16 B line in `cb_sync`, or in the hub landing ring header; form per §4.6):

| word | lives on | written by | form | target after generation g |
|---|---|---|---|---|
| `hub_arrive[k]` k = x, x_mid (H0), ctx (H1) | hub | each source, 1 atomic | summed (paired with `hub_ready`) | n_src x (g+1) |
| `hub_ready[k][parity]` | hub | each receiver | per member, min (GR00T parity split) | g+1 |
| `hub_flag[k]` | each receiver | hub multicast | per peer | g+1 |
| `pair_arrive`, `pair_ready` | Q / K producer | RoPE partner d ^ 4 | per peer | g+1 |
| `qlead_arrive` | Q leader (h, 8) | 8 Q producers | summed (paired with `qlead_ready`) | 8 (g+1) |
| `qlead_ready[m]` m = 0..9 | Q leader | each column receiver incl. (h, 9) | per member, min | g+1 |
| `qlead_flag` | column receivers | Q leader multicast | per peer | g+1 |
| `kl_arrive` | KL | 16 K / V producers | summed (paired with `kl_ready`) | 16 (g+1) |
| `kl_qdone` | KL | 8 Q leaders after their multicast barrier | summed | 8 (g+1) |
| `kl_ready[m]` m = 0..15 | KL | each T4 receiver | per member, min | g+1 |
| `kl_flag` | T4 cores | KL multicast | per peer | g+1 |
| `part_arrive[src]` src = remote kc | slice merger | that unit | per peer | g+1 |
| `part_ready[j]` j = slice merger | unit | slice merger j | per peer | g+1 |
| `row_arrive` | row leader | 7 row members | summed (paired with `row_ready`) | 7 (g+1) |
| `row_ready[m]` m = 0..7 | row leader | each row member | per member, min | g+1 |
| `row_flag` | row members | row leader multicast | per peer | g+1 |
| `red_arrive[kg]` | owner | MLP core (ng, kg) | per peer | g+1 |
| `red_ready[owner]` | MLP core | each of its 4 owners | per peer | g+1 |
| `boot` | all | BootBarrier (the only tt-metal semaphore) | — | — |

The heaviest core (KL: 1 + 1 + 16 + 1, or a T1 row-leader merger: 3 hub flags + 2 pair + 1 qlead_flag + 4 part_ready +
4 part_arrive + 1 + 8 + 1 row + 4 red_ready) needs < 40 of the 128 words. A host test enumerates every (what, slot,
peer) tuple over all roles at both shapes and asserts: injectivity of the word addresses; every summed word is paired
with a per-member ready credit covering all its writers; no core publishes through a word it also waits on; every
landing buffer written by a peer has a ready credit (or, for `cb_in0`, the §4.6 ready point).

### 4.9 Kernel-config ring estimate (136,192 B)

The closest measured binary is GR00T K4c's DiT compute-core program: 110,256 B of ELF text + data (brisc 5,076, ncrisc
17,824, trisc0 36,704, trisc1 31,616, trisc2 19,036), 114,368 B with the 4,096 B config allowance, i.e. 21,824 B of
headroom (M, `mk_k4c_summary.md` §0). It contains distributed LayerNorm, flash SDPA with partial merge, stream matmul
with a GELU epilogue, tail matmuls and Euler. Phase 1 needs the same primitive set minus the compute-core LayerNorm
(the stats move to the hubs, a separate kernel group with its own binary) plus RoPE, the r/c epilogue, GeGLU and fp32
reduce adds. **Estimate: 100-118 KB (A, no basis beyond that analogy)**; hub group 60-90 KB (A). The union compute group
(T1..T4 + KV roles) is larger than any GR00T group, and K4b's first build of a comparable feature set was 150,304 B
(> 136,192), so the estimate is not relied on: **"program size <= 128 KB by mock compile" is an explicit pass gate of
P1-3, P1-4 and P1-5** (P1-0 only compiles an empty kernel), counting the runtime args (the role tables are runtime args
and live in the ring) and 16 B per CB id. Rules from day one: `noinline, noclone` on every helper called from more than
one site, runtime CB ids and tile counts (not compile-time constants) for anything with several call sites, bring-up dump
knobs compiled out of production builds. Checked by the copied `size_check.py` mock-cluster compile on every kernel change.
If the union group does not fit, the lever is role-specialised kernel groups (T4 and KV cores run no MLP code) before any
code reduction that changes numerics.

### 4.10 Precision plan (phase 1)

| quantity | shipped path (today) | megakernel | effect |
|---|---|---|---|
| Wqkv, Wug | bfp8, LoFi | bfp8 (per-step folds `diag(1+a)W`), LoFi, fp32 acc | new: the qkv-side fold (measured PCC 0.999 vs unfolded on p300, `ttnn_fused_norm.py` comment); gated in WP-P1-3 |
| Wo, Wd | bf16, HiFi2 | bf16, HiFi2, fp32 acc | same |
| residual stream | bf16 | **fp32 on the owners**, bf16 on the wire | more precise |
| r (row rsqrt) | bf16 tile from RowRsqrt (HiFi4, fp32 acc) | fp32 compute on the hub, bf16 tile on the wire | same |
| RoPE | in the fused attention kernel, bf16 | on the Q / K producers, bf16 tables, fp32 DST | same values (rotate-half is tile-aligned) |
| K/V cache | bf8 | bf8 (read from the same caches) | same |
| attention | one pass over 25 key tiles, HiFi2 | 5-chunk flash with (m, l, O) parts; O parts bf16, merge in fp32, fixed fold order | new rounding of O parts (<= 1 bf16 ulp); gated |
| ctx into o_proj | bf8 then typecast to bf16 | bfp8 | same values |
| h into down | bf16 | bf16 | same |
| down K-split | one fp32 DST accumulation over K = 128 tiles | 8 partials of 16 K tiles, summed in fp32 | fp32 reordering only |
| GELU | tanh form | tanh form (exact, not `fast_and_approx`: GR00T's approximation failed 9/12 golden taps) | same |
| Euler state | bf16 (dit op) | fp32 on H0 | more precise |

### 4.11 Time prediction, phase 1 (per-core ring-constrained timeline)

The layer time is no longer `max(chain, stream)`. It is the steady-state layer time of the fluid model
`review_stream_sim.run`: each core class (T1/T2, T3, KV) streams its items in consumption order into its two rings (v2
sizes) under an aggregate DRAM cap shared by the unblocked cores, and consumes only in its compute phases (§4.3 phase
list). With v1's rings the model gives base 84.1 us at 414 GB/s, i.e. the v1 claim of 76.8 was optimistic (§10 B2).

| | base | LIBERO |
|---|---:|---:|
| chain per layer (sum of §4.3; D from M + A) | 76.8 us | 61.1 us |
| ring-constrained timeline at 414 / 440 / 464 GB/s aggregate (v2 rings) | 76.8 / 76.8 / 76.8 | 61.1 / 61.1 / 61.1 |
| same with v1 rings, 414 GB/s | 84.1 | 76.9 |
| step = 18 x layer + 8 us tail | 1,391.1 us | 1,108.2 us |
| **expert loop, bottom-up** | **13.91 ms** | **11.08 ms** |
| expected (x1.36, GR00T K4b realisation) | **18.92 ms** | **15.07 ms** |
| first build (x1.68, GR00T K4) | 23.37 ms | 18.62 ms |
| today (M, profiled) | 31.26 + 0.17 action io | 27.85 + 0.16 |
| **whole call** = today (unprofiled) - expert + action-io stage (profiled) + expert loop | 66.7 / **71.7** / 76.2 ms | 59.8 / **63.8** / 67.4 ms |

The whole-call row subtracts profiled stage times (31.43 / 28.01 ms, which carry a share of the ~1 ms profiler overhead)
from unprofiled calls (84.21 / 76.77), so it mixes bases by up to ~0.4 ms; it is a prediction only, and the gates of §7
measure the whole call directly. Break-even: the design misses the phase-1 speed gate only if the realised layer is
> 2.25x (base) / 2.52x (LIBERO) its bottom-up timeline. If WP-P1-2 finds the 2-D down exchange (R6 + R7, 9.5 us) not
reachable, the fallback is GR00T's measured ff2 chunk ring (4 chunks of 128 KB at 5.3-11 us each, `mk_k4b_summary.md`
§3.4): chain 88-111 us, expert 16.0-20.1 ms bottom-up, still under the gate. The whole-call figures assume the ttnn
SigLIP/VLM ops keep their time under the 64 KiB cut (WP-P1-0 measures it).

### 4.12 Selection, stamps, refusals and the shape contract (applies to both phases)

- **Knob**: `FusedConfig.megakernel` from `PI05_MEGAKERNEL` in {`off`, `expert`, `whole`}, read once at model build.
  ~~Default `off` until the phase-2 exit (P2-4 + P2-5) passes; it then becomes `whole`. `expert` exists for bring-up and
  as the phase-1 comparator; it is never the served default and is never described as "the megakernel".~~
  **Amended 2026-09-30 (integrate-p1)**: after phase 1 was accepted on the amended gates, the user asked to update the
  Hugging Face package first and then continue with the next phase ("a. 일단 HF에 update. 이후 다음 phase 작업 진행"); the
  integrate-p1 task made `expert` the default (`FusedConfig`, the server, the package serve env) and kept `off` (the
  previous shipped path) only as the comparator. It is always described as "the phase-1 expert megakernel" with the
  prefix labelled as traced stock ops, never as "the megakernel" of the whole model. Phase 2 stays the deliverable;
  on its exit the default becomes `whole`. **Amended 2026-10-01 (integrate-p2)**: phase 2 was accepted on the amended
  gates and the user ruled (2026-10-01, binding) that `PI05_MEGAKERNEL=whole` becomes the DEFAULT (`FusedConfig`, the
  server, the package serve env); `expert` (phase 1) and `off` (the previous shipped path) stay selectable only as
  comparator knobs. The UNSET default resolves to `off` on a multi-chip mesh
  (`FusedConfig.resolved`); an explicit `expert` there refuses. Refusal (d) is implemented
  (`PI0ModelTTNN.megakernel_device_refusal`, checked at the start of `__init__`; device evidence
  `integrate_p1/results/NOCUT.json`).
- **Stamp**: the constructed model carries `model.megakernel_backend` (`off` / `expert` / `whole`) and
  `model.megakernel_program` (the descriptor hash and kernel source list of the generic_op captured in the trace).
  `tests/pcc/test_pcc_pi05_fused.py`, the megakernel device tests and the server smoke test assert the stamp read from
  the constructed object (not from the env), so a silent fall-through to today's path cannot pass (memory notes
  `direct-construction-gates-miss-the-policy-path`, `measured-configuration-must-be-selectable-by-the-shipping-path`).
- **Refusals** (a `RuntimeError` at model build or server start naming the knob; never a silent fallback): megakernel
  selected with (a) a shape other than (P, S) = (736, 64) or (544, 32); (b) batch > 1 or more than one batch size in
  `PI05_BATCH_SIZES`; (c) `TT_MESH_SHAPE` other than 1x1, or `PI05_LAYOUT` dp / pipeline (the megakernel is single-chip);
  (d) a device not opened with the 64 KiB worker-L1 cut. Each refusal has a CPU test. A general (P, S) rule for the chunk
  and role tables is not in scope; the refusal is the contract.
- **Semantic precondition**: prefix RoPE positions 0..P-1 equal openpi's `cumsum(valid) - 1` only for a right-padded
  prompt with every camera valid. The host refusal at `tt/ttnn_pi0_model.py:283-286` (right padding) stays on the
  megakernel path, and the megakernel input builder refuses a masked camera.
- **Replay L1 safety**: tt-metal checks static CBs against the lowest occupied L1 address only when a program is
  enqueued (`program.cpp:2141-2149`, `validate_circular_buffer_region`); an `execute_trace` replay does not re-check, and
  the megakernel's static CBs and transient residents cover most of L1. Two rules: (1) by construction, a megakernel
  process prepares exactly one shape (refusal (b)), so no other shape's persistent L1 inputs exist; (2) a host guard
  records the L1 allocation state at the end of capture (`ttnn.get_memory_view(device, L1)`: allocated bytes and the block
  table) and, before every replay, asserts that it is unchanged; any L1 allocation after capture raises before
  `execute_trace`. WP-P1-0 measures the guard's cost; if it exceeds 0.1 ms it runs in tests and debug builds only, and rule
  (1) plus a test carries production.
- **What stays outside the fused op** (input / output formatting, no learned parameter and no model arithmetic): image
  resize and pixel normalisation, im2col of the patches (a rearrangement; the patch-embed matmul is inside), prompt
  building and tokenisation (the state is discretised into the prompt text), the mask / RoPE row construction
  (`fused_host.attention_inputs`), noise sampling, the host->device copies of these inputs, and the output slice to H rows
  and un-normalisation of the actions. In phase 1 the SigLIP / VLM ttnn ops are also outside (that is why phase 1 is not
  the deliverable).

## 5. Phase 2: the whole `sample_actions` as one program

### 5.1 Structure

One generic_op, one launch, phases separated by global barriers (BootBarrier form, A ~4 us each, 4 transitions):

1. **SigLIP x2** (27 layers, 512 rows = 2 images x 256 patches) from the im2col'ed images;
2. **projector** (1152 -> 2048) + **language embedding** (in-kernel gather of 224 rows of the DRAM table by token id,
   tilize, x sqrt(2048)) + assembly into the prefix residual (736 rows);
3. **VLM prefill** (18 Gemma-2B layers; the last one KV-only; no final norm), which writes every layer's roped K and V
   for rows 0..P-1 straight into an L1-resident KV region laid out exactly as phase 1 reads it;
4. **expert loop** = the phase-1 schedule unchanged, reading the resident KV region instead of the ttnn caches.

The VLM pads its rows to a tile-band multiple (base 736 -> 768, LIBERO 544 -> 576, §5.3). **The pad rows are never
written to the resident KV region**: the region holds exactly P rows (base 23 key tiles, LIBERO 17), so phase 1 keeps
its 23 + 2 = 25 = 5 x 5 (base) and 17 + 1 = 18 = 6 x 3 (LIBERO) chunking unchanged. Pad query rows are computed and
discarded (not stored anywhere). The VLM's own attention reads keys 0..P-1 only from the same region, with the key mask
row `[1, P]` = the per-key bias of `reference/torch_pi0_model.prefix_attention_inputs` (0 for valid, MASK_NEG for pad
keys; the prefix is bidirectional, so one row serves every valid query); a host test compares it with `vlm_mask[b, 0,
i, :]` for every valid query row i (today's `vlm_mask` is `[B,1,P,P]`, `fused_host.py:247`).

All four share one binary per core group (§5.6) and one L1 plan with phase overlays (§5.5). Nothing is returned to the
host between phases.

### 5.2 Inputs (all fixed-address tensors; per request only their contents change)

im2col images `[2, 256, 608]` bf16 (today's `_fused_in_im2col`), token ids `[224]` uint32, the VLM key mask row `[1, P]`
(§5.1), the prefix RoPE tables (positions 0..P-1, constant per shape under the §4.12 precondition), the expert mask +
RoPE rows (§4.5), noise; output x_0. The embedding gather computes DRAM addresses from token contents in-kernel, so it
is trace-safe.

### 5.3 VLM geometry: 2-D bands, weights read once per column and multicast down it

Base: pad 736 -> 768 rows = 24 row tiles; **R = 8 bands** (grid rows 0..7) x **rt = 3 row tiles**; **C = 11 columns** (88
cores). LIBERO: 544 -> 576 = 18 row tiles, R = 9 x 2 (grid rows 0..8) or R = 6 x 3. Pad rows are discarded queries and
never become keys (§5.1).

- **qkv** (N = 80 tiles, head-aligned): column c < 8 = head c's Q (8 tiles), c = 8 K, c = 9 V; core (b, c) computes
  [band b rows] x [its 8 N tiles]; the band's normalised x (3 x 64 tiles) is multicast along the band by ONE band leader;
  weights of column c are read once from DRAM by one core of the column and multicast down it (8 cores).
- **attention** on core (b, c < 8): head c, query rows of band b, all P keys: K / V (roped K) are written by columns 8 / 9
  into the resident KV region (rows 0..P-1 only) and **pulled** by the attention cores in chunks on NoC 1 (a 1 KV head is
  shared by all 8 heads, so no per-head copy exists). Per core 3 q row tiles x 23 key tiles.
- **o_proj**: K-split by head, aligned with where ctx already is: core (b, c) computes ctx[b, head c] @ Wo[head c rows]
  -> partial [b, 64 N]; N-chunked reduce to the band's owners (b, j) in fp32, pipelined with the partial compute.
- **gate_up**: core (b, c) computes [band b] x [column c's N slice] K-outer in N sub-passes (acc 3 x 12 tiles fp32),
  in0 = the band's normalised x resident in L1, weights column-multicast; GeGLU in the epilogue; h stays on the core.
- **down**: K-split aligned with gate_up's N split (core (b, c) owns h[b, K slice c]), weights column-multicast,
  partial [b, N chunk j] computed chunk by chunk and sent to owner (b, j), who sums C partials in fp32 and adds the
  residual. No global all-gather of h (24 MB) and no per-owner in0 band exchange exist anywhere in the layer.
- **norms**: per-band row statistics = one band all-reduce of partial sums of squares (3 tiles per core), each owner
  normalises its slice, band leader gathers + multicasts the normalised x (GR00T's LN round, 15.3 us incl. compute, M).

SigLIP uses the same 2-D band machinery (R = 8 bands x 2 row tiles, images split 4 + 4 bands; hidden 1152 = 36 tiles;
qkv already padded to 16 heads x 96 in today's weights, `512x1152x4608` in PROFILE.md; LayerNorm with bias; GELU-tanh fc1
epilogue; attention per image, block-diagonal, 16 heads over 8 attention columns).

### 5.4 Time prediction, phase 2 (per layer, base) — a range for the go / no-go, not a bound

VLM layer, 162.07 GFLOP (D). **Lower** = every op at the best rate measured on this card for its shape, with the removable
excess gone (weights read once, GELU fused), on the §5.3 geometry (88 cores):

| op | lower us | source | at F0-harness rates (NOT an upper bound) |
|---|---:|---|---:|
| qkv | 32.9 | M TTNN 234 TFLOP/s | 58.0 (133 T, F0 llm_qkv_s512) |
| o_proj | 26.2 | M TTNN 235 T | 46.4 (133 T) |
| gate + up, unchunked | 588.0 | M TTNN 168 T (chunked; the weights are now read once) | 1,050.9 (94 T, F0 gate_up) |
| GELU epilogue, 24 x 512 / 88 = 139.6 tiles per core | 81.0 | M GR00T F1 encoder GELU 0.58 us / tile | 209.4 (1.5 us / tile, F0 PACK) |
| up * gelu | 14.0 | A 0.1 us / tile | 14.0 |
| down | 200.0 | M-reported isolated mcast2d probe `[736,16384]x[16384,2048]` 0.20 ms (`common/fused_config.py` docstring, 2026-09-13, not re-measured) | 422.2 (117 T, F0 down) |
| attention | 159.6 | M TTNN SDPA 139.6 + A 20 in-kernel RoPE / heads | 159.6 |
| 2 norms | 30.6 | M GR00T LN round 15.3 each | 30.6 |
| exchanges | 84.7 | A 10 % of matmul (lever 1: ring depth hides in0, F0c) | 190.0 (5 x 38 us, F0c law at L = 22) |
| **layer** | **1,217** | | 2,181 |

**What GR00T actually realised** inside its fused layers (7-15 % of peak for the matmuls, exchanges 26 % of the layer,
§3.3) gives, on the same terms (`mk_design_calc_v2.out`): 2,522 us per layer at 15 % and 5,079 us at 7 % — both above the
TTNN layer (2,437 us, M). At 7 % gate+up alone is 2,319 us. The VLM stack then spans lower 20.8 ms to 43.0 ms (15 %) and
86.4 ms (7 %), against TTNN's 41.54. The x1.36 factor used in v1 for a "point" came from a stream-bound DiT and is not
carried to this compute-bound prefill. **The phase-2 VLM cost is therefore unknown within a range that straddles TTNN;
WP-P2-1's measured go / no-go decides it**, and nothing in this section is a bound.

SigLIP layer lower bound: qkv 39.5 (M) + o 17.5 (M) + fc1 36.2 (F0 fc1 without epilogue, 140.6 T) + GELU 22 tiles x 0.58 =
12.5 + fc2 52.5 (F0 fc2 97.1 T) + attention 30.1 (M SDPA 25.1 + A 5) + 2 LN 30.6 + exchanges 14.6 (A 10 %) = **234 us**
(TTNN 394.7, M); tower incl. patch embed / post-LN (M 94 us): lower 6.40 ms; above it the same caveat applies (the GR00T
fused towers came in slower than their TTNN), so the gate decides.

| whole call, base | low (all lower bounds) | what decides |
|---|---:|---|
| phase 2 = SigLIP + projector/embed (0.12, M) + VLM + transitions 0.05 + expert (§4.11) + host (1.40, M) | 42.7 ms | P2-1 / P2-2 / P2-3 measurements |
| phase 1 (§4.11) | 66.7 / 71.7 / 76.2 ms | P1-5 |

Phase 2 beats phase 1 iff fused SigLIP + projector/embed + VLM < today's 52.42 ms (base) / 48.72 ms (LIBERO), the stage
times they replace.

### 5.5 Phase-2 L1 plan (overlays)

A CB has one address, size and id set for the whole program, so every phase's buffers must coexist or alias. Plan:
CB ids are assigned by **function** across phases (in0, w8, w16, acc, out, q, kv, s, oacc, ml, part, red, x, sync, ...),
sized at the max over phases, and phase-disjoint buffers **alias one buffer through several format descriptors** — the
same mechanism as §4.7's `cb_in0` (one `CBDescriptor`, one id per format), which WP-P1-0 proves for a tensor-backed
buffer. For non-tensor-backed buffers the mechanism is already demonstrated in the GR00T tree (`bb_descriptors.py:846-878`,
`test_mk_mm2d.py:511 test_aliased_cb_accepted`); v1's claim that it was undemonstrated was wrong (§10). The resident KV
region and the owners' residual slices are never aliased.

Resident KV region: a design-owned sharded layout holding rows 0..P-1 only, 36 x 23 x 8 = 6,624 bfp8 tiles at base,
ceil(6,624 / 110) = 61 tiles = **66,368 B per core** (LIBERO 4,896 tiles, 48,960 B).

VLM MLP phase on core (b, c), base, the tightest phase: band in0 resident 3 x 64 bf16 = 393,216 + weight landing ring
2 x (8 K x 12 N) bfp8 = 208,896 + acc 3 x 12 fp32 = 147,456 + h slice 3 x 47 bf16 = 288,768 + residual slice 3 x 6 fp32 =
73,728 + reduce landing 18 fp32 = 73,728 + resident KV 66,368 + misc 40,000 = **1,292,160 B** of 1,371,136 (78,976 B
headroom). If the WP-P2-0 descriptor build does not close under the limit, the levers in order: h in bfp8 (-135 KB, PCC
check), in0 re-multicast per N pass instead of resident (-393 KB, +in0 traffic), acc 3 x 8.

The union of phase-1 and phase-2 CB ids must stay within the 64 Blackhole ids (phase 1 uses 30); it is a WP-P2-0 host
check, with ids kept dense for the ring cost.

### 5.6 Ring budget and how to split kernels while staying one program

The ring holds the five binaries of the largest kernel group (`program.cpp` `finalize_offsets`, `mk_k4c_summary.md` §2), so
code for every phase a core runs must fit at once. Estimate if the phases were written as separate specialised kernels:
phase-1 set (100-118 KB, A) + distributed LayerNorm / RMS stats + 2-D band matmul with column-multicast weights + tilize +
patch embed + SigLIP attention variant: **115-145 KB (A)**, i.e. possibly over 136,192. The plan to stay under it in ONE
program and ONE launch:

1. **Primitive interpreter**: the compute kernel is a loop over a per-core op table (runtime data in DRAM/L1: op kind,
   CB ids, tile counts, flags). The primitives are shared by all four phases: one K-outer matmul (1-D is R = 1), one
   row-stats routine (RMS and LN), one norm-apply, one RoPE, one flash-chunk routine (runtime dh, key tiles, mask), one
   merge, GELU-tanh, eltwise add/mul, fp32 reduce, tilize, Euler. Code grows with the primitive count, not with layers,
   models or phases.
2. **Role-specialised kernel groups**: code only some cores run lives only in their group's binary (the hub group:
   embedding gather + tilize, patch-embed staging, Euler, stats; the compute group: the primitives). The ring must fit
   the largest group only.
3. **Per-RISC balance**: table construction, mask expansion and address math move to BRISC/NCRISC (built `-Os`).
4. **If 1-3 do not reach <= 128 KB (8 KB margin) in the WP-P2-0 mock compile: stop and ask the user** (status
   needs_user) with the measured per-RISC sizes. The two remaining options both need their say-so: a larger worker-L1
   cut, which is a **process-wide device-open option** (`worker_l1_size` is fixed per `open_device`, so it shrinks L1 for
   every program in the process, including the ttnn ops of any comparator path in the same process; phase 2 itself has
   no ttnn co-tenants), or two launches (which the goal excludes).

### 5.7 Precision plan (phase 2)

Same weights and fidelities as today: SigLIP bfp8 HiFi2 (patch embed bf16), VLM bfp8 LoFi, fp32 accumulation everywhere,
exact GELU-tanh, K/V written as bf8 (today's `kv_dtype`). New numerics to gate: the K-split reductions of o_proj / down
(fp32 partials, reordered), the 2-D flash attention merges, fp32 residual slices (more precise). Accuracy is judged per
layer on real weights and real inputs against the ttnn layer in the same run (rel-L2 as well as PCC, because SigLIP's
absmax jumps 12.3 -> 1765.9 at block 10 and PCC is discontinuous across a magnitude regime change), then end to end.

## 6. Implementation notes

### 6.1 Files (new, in this repo)

`models/experimental/pi0_5/tt/megakernel/`: `core_map.py` (roles, rectangles; the device's logical->virtual table read at
run time), `geometry.py` (role tables, sync-word tables, host injectivity / pairing / bound checks), `arena.py` (per-core
consumption-order packer with the per-step folds), `descriptors.py` (CB / semaphore specs incl. multi-format descriptors +
materialise with an explicit compute config incl. `dst_full_sync_en`), `expert_program.py` (build, bind inputs, launch,
trace, stamp), `l1_guard.py` (the §4.12 replay guard), `size_check.py`; `kernels/mk_kernel.cpp` and `kernels/ops/*.hpp`
copied and adapted from GR00T (`allgather.hpp`, `exchange.hpp`, `weight_stream.hpp`, `block_matmul.hpp`,
`block_matmul2d.hpp`, `head_sdpa.hpp`, `pack_fence.hpp`, later `dist_layernorm.hpp`, `rope_rows*.hpp`) plus the four
deepseek `unified_kernels` headers they include, copied so that every kernel `#include` resolves inside this repo (memory
`feedback-source-code-cpp-includes`). Plain `ttnn.ProgramDescriptor` / `KernelDescriptor` (no GR00T Python import). Tests
under `models/experimental/pi0_5/tests/megakernel/` (`test_cpu_*` host tests, device tests always through
`bin/with-device.sh`).

Existing files touched: `common/fused_config.py` (the `megakernel` knob), `tt/ttnn_pi0_model.py` (selection, stamp,
refusals), and **one device-open helper** `common/device_open.py::open_pi05_device(fused_cfg, ...)` that passes
`worker_l1_size = 1,395,712` when the megakernel is selected, used by all seven open sites today:
`server/app.py:694-700`, `tests/pcc/test_pcc_pi05_fused.py:66-71`, `tests/perf/test_perf_pi05_fused.py:86-89`,
`tests/pcc/test_rollout_libero.py:300-303`, `tests/pcc/test_action_divergence_lerobot.py:394-397`,
`tests/perf/profile_pi05_ops.py:56-64`, `tests/pcc/test_pcc_pi05_mesh.py:79` (the mesh sites keep today's path and
refuse the megakernel, §4.12). `server/app.py`: /info backend strings (`_graph_string`, :925-928) report the megakernel and
its stamp; warm-up and capture order unchanged for B = 1; B > 1 / mesh / dp / pipeline with the megakernel selected refuse
at startup with a log line.

### 6.2 Bring-up rules (from the memory notes; binding)

Mock-cluster compile + size check before every device run of a changed kernel; `WITH_DEVICE_RESET_AFTER=1` and a timeout
(`--timeout-method=thread`) on every first run; private `TT_METAL_CACHE`; never edit `kernels/**` while a device job may
run; screen new compute with constant non-zero inputs (zero is a fixed point of the operand-format faults); bit-identity is
checked on whole tensors; a compile-out knob is not a measurement until the default build passes the accuracy gate; delete
the profiler zone-location logs after kernel edits; profile zones sampled (e.g. steps 0 and 9 only) so 180 layer
iterations do not overflow the device profiler buffers; record hang signatures (waypoints, cores, how it presented).

## 7. Work packages and pre-registered pass gates

Gates below are fixed as of v2. "Speed" gates use unprofiled medians of 30 calls unless stated; kernel device time uses
the profiler's kernel duration of the megakernel op, median of >= 20 traced replays. "Size" = the mock-compile ring
footprint of the largest kernel group incl. runtime args and CB-id config.

| WP | deliverable | pass gate | device |
|---|---|---|---|
| P1-0 | scaffolding (§6.1), host tests (role / sync-word injectivity and pairing, L1 budgets at both shapes, refusals, mask / RoPE tables vs `reference/torch_pi0_model.prefix_attention_inputs` incl. the `n_valid + [0,H)` rows), empty persistent kernel mock-compiled; on the device: the shipped path under the 64 KiB cut, the two-format tensor-backed `cb_in0` descriptor, the free-L1 measurement, the L1 guard | CPU tests green; size check runs; shipped path under the cut: PCC7 unchanged (mean >= 0.9995, min >= 0.999) and per call <= 84.2 ms + 1 %; the two-format CB is accepted and a tiny kernel reads bfp8 through id 29 what it wrote as bf16 through id 0 (or a named fallback of §4.7 is); minimum contiguous free L1 above the CB base at the op's trace position >= the §4.7 union at both shapes; the guard raises on an L1 allocation after capture | ~15 min |
| P1-1 | arena + stream-only kernel over the real 10 x 18 shares, v2 rings | sustained >= 400 GB/s aggregate; read-back checksum exact; 10 launches identical | ~20 min |
| P1-2 | exchange microbenchmarks under a concurrent full weight stream, **at base and LIBERO**: x / ctx hub rounds; pair exchange + RoPE; Q column leader x 8 columns; KV leader ordered after Q (lands before a T4 unit's prefix tiles finish); dh-split merge incl. the fp32 folds (and T_E); row h exchange x 8 rows; column reduce x 8 **incl. the owner's fp32 sums and gate** | each measured cost <= 1.5x its §4.3 A-value at both shapes; else re-plan that exchange (named fallbacks: ff2 chunk ring for R6/R7; merger-only tree for R3) and re-run the §4.11 timeline before P1-3 | ~30 min |
| P1-3 | one expert layer in-kernel (one step), real weights | layer output vs the current ttnn layer on the same input: PCC >= 0.9995 and rel-L2 <= the ttnn layer's own rel-L2 vs fp32 torch + 20 %; constant-input screen exact; 10 launches bit-identical; **size <= 128 KB** | ~30 min |
| P1-4 | 18 layers x 10 steps + in/out proj + Euler, standalone (inputs = the ttnn VLM's caches) | x_0 PCC vs the current path >= 0.999 per observation (8 golden obs); 10 launches bit-identical; kernel time < 31.26 ms base and < 27.85 ms LIBERO; **size <= 128 KB** | ~30 min |
| P1-5 | integrated into `sample_actions_fused` (one generic_op in the trace), `PI05_MEGAKERNEL=expert` | **the phase-1 exit gates below**; **size <= 128 KB** | several holds |
| P1-6 | server with `PI05_MEGAKERNEL=expert` | `server/smoke_test.py` green; served median < 84.02 ms at base; /info reports the backend and stamp; B > 1, mesh, dp and pipeline configurations refuse at startup with a log line (tested) | ~20 min |
| P2-0 | phase-2 binary (primitive interpreter) mock-compiled; L1 overlay descriptor built on the host | compute group <= 128 KB; the multi-format aliasing used by the overlay is accepted by the program build and a tiny kernel reads/writes through two aliases correctly; union of CB ids <= 64; L1 plan <= the P1-0-measured free L1 per core | ~10 min |
| P2-1 | one VLM layer in-kernel (736 rows, real weights) | PCC >= 0.9995 vs the ttnn layer (same run); **time <= 2.20 ms (go); 2.20-2.44 ms: report and ask; > 2.44 ms (slower than TTNN): stop, needs_user** | ~30 min |
| P2-2 | one SigLIP layer in-kernel | PCC >= 0.9995 and rel-L2 as in P1-3; time <= 355 us (go); > 394.7 us: stop, needs_user | ~30 min |
| P2-3 | whole prefix in-kernel -> resident KV (rows 0..P-1 only) | ~~per-layer K/V vs the ttnn caches PCC >= 0.999~~ -- REPLACED 2026-10-01 by the user: **per layer, K/V at least as close to the fp32 host decomposition as the ttnn caches, at both shapes** (see the amendment below); rel-L2 recorded; stack time < 52.42 ms base and < 48.72 ms LIBERO | ~30 min |
| P2-4 | whole `sample_actions` as one program, `PI05_MEGAKERNEL=whole` | **the phase-2 exit gates below** | several holds |
| P2-5 | server with `PI05_MEGAKERNEL=whole` (the new default) | as P1-6, with served median < the P1-6 served median; /info reports `whole` | ~20 min |

**Exit gates (both phases unless stated):**

- **Structural (the user's literal goal).** Measured with the profiler's per-replay op list, the same method and
  instance-count check as `PROFILE.md`. *Phase 1*: after the last VLM op the replay holds exactly ONE device op, the
  megakernel generic_op (identified by its program hash / kernel names), covering the whole 10 x 18 loop plus the action
  in/out projections; zero expert or action-io ttnn ops remain. *Phase 2*: the replay holds **exactly ONE device op**;
  only the host->device input copies of §4.12 happen around `execute_trace`. In both, the constructed model's stamp
  (§4.12) equals the selected backend, asserted in `tests/pcc/test_pcc_pi05_fused.py`.
- **Correctness** (with the megakernel selected): PCC7 vs the openpi golden (`libero_eval/pi05/openloop_golden.pt`, 8 obs)
  **mean >= 0.9995, min >= 0.999**; base-shape padded-prompt accuracy: **~~no worse than the current path vs the fixed torch reference on the same
  seeds (per seed within 0.005)~~ -- REPLACED 2026-09-30 by the user, see the amendment below**; **ten replays bit-identical**; the **alternating-prompt / shape-switch
  test** (`verify_alternating.py`) passes vs fresh-model references; **no hang in 20 consecutive runs** (a hang or timeout
  counts as a failure; one run = one process doing warm-up + capture + 30 calls, timeout 600 s). Prompt-length edge cases
  vs the torch reference, each no worse than the current path: base n_lang = 1, 224 (no pad key), 128 (n_valid 640 = the
  kc 3 / kc 4 boundary: chunk 4's prefix tiles fully masked, suffix keys valid) and 150 (the valid / pad boundary inside a
  key tile of chunk 4); LIBERO n_lang = 32 (no pad) and 1. The existing suites stay green with the megakernel selected:
  `tests/pcc/test_pcc_pi05_fused.py`, `tests/pcc/test_reference_vs_openpi.py`, `tests/test_fused_host.py`, and the
  megakernel CPU tests.
- **Gate amendment (user decision 2026-09-30, binding).** The base-shape gate "per seed within 0.005 of the current path
  vs the fp32 torch reference" is REPLACED by: *phase 1*: **per seed, the megakernel expert is at least as close as the
  shipped path to the fp32 expert-loop oracle fed the device's own prefix K/V (the O1.json method), over >= 18 seeds**;
  *phase 2*: **per seed, at least as close as the shipped path to the fp32 torch reference of the WHOLE model on the
  same inputs**. Reason: the old gate measured prefix noise (SigLIP / VLM / bf8 K/V cache) that no expert change can
  remove -- the fp32 expert oracle itself fails it on seed 2 (0.97553 vs the 0.98397 - 0.005 bar; impl/results/O1.json).
  Phase 1 was accepted on this amended gate (option (a)): megakernel closer to the oracle on 20/20 seeds, mean 0.99980 /
  min 0.99942 vs shipped 0.99649 / 0.98784 (verify_p1_r0/results/O_verifier.json). Every other gate in this section
  stands unchanged.
- **Gate amendment (user decision 2026-10-01, binding).** WP-P2-3's clause "per-layer K/V vs the ttnn (shipped) caches
  PCC >= 0.999" is REPLACED by: **per layer, K/V at least as close to the fp32 host decomposition as the ttnn caches, at
  both shapes**. Reason: the old clause measured the shipped path's own error -- the ttnn caches themselves are only
  0.9656 (base) / 0.9612 (LIBERO) PCC from the fp32 host decomposition, so no more accurate prefix could meet it. Measured
  by the implementer: the engine is closer on 36 / 36 (layer, K / V) at both shapes, engine min PCC 0.9954 / 0.9945 vs
  ttnn 0.9656 / 0.9612 (p2/gates/results/P23_{base,libero}.json); re-measured by verify-p2-r0 against the fp32 whole-model
  reference's own VLM cache on 8 inputs per shape: 288 / 288 per shape (verify_p2_r0/results/G_gates_r1.json). In the
  same ruling `PI05_MEGAKERNEL=whole` became the default (§4.12). All other phase-2 gates stand.
- **Speed**: *phase 1* expert-loop device time **< 31.26 ms base and < 27.85 ms LIBERO**, whole-call latency **< 84.2 ms
  base and < 76.77 ms LIBERO**; *phase 2* whole-call latency **< phase 1's at base and at LIBERO** (same shape, arms
  alternated across processes in one session, 30 calls each, difference larger than 2 x the MAD-based standard error of
  the median).
- **Deliverable rule**: phase 1 passing is not completion. If P2-1 stops (needs_user) or the phase-2 exit gates fail, the
  run returns `needs_user` with the measurements.

## 8. Risks (ranked) and open questions

1. **Exchange latency above the A-values** (R2, R3, R6, R7 are unmeasured shapes; T_E is an estimate; concurrent
   multicasts to disjoint rectangles are unmeasured on this card). Mitigation: P1-2 at both shapes before layer code,
   named fallbacks; the phase-1 gate has 2.25x (base) / 2.52x (LIBERO) of margin over the bottom-up timeline.
2. **Phase-2 VLM matmul efficiency** (GR00T's fused layer realised 7-15 % of peak; §5.4 shows that range straddles TTNN).
   Mitigation: P2-1 go/no-go against the TTNN layer; the geometry has no per-owner in0 exchange and reads each weight once.
3. **Kernel-config ring** for the phase-1 union group (no basis for the estimate; K4b's first comparable build was over)
   and for phase 2 (115-145 KB estimate). Mitigation: the size gate in P1-3/P1-4/P1-5/P2-0; role-specialised groups;
   §5.6; a needs_user stop rather than a silent second launch.
4. **L1 co-tenancy in phase 1**: the 64 KiB cut may make ttnn SigLIP/VLM programs clash (GR00T K5 moved its adapter to DRAM
   for this); the CB ceiling is an address. Measured in P1-0; mitigation: ttnn intermediates to DRAM where needed, KV
   caches stay L1.
5. **Sync-word aliasing / credit off-by-one hangs** (GR00T faults 15-19). Mitigation: cumulative counters keyed on
   (what, slot, peer), explicit ready credits for every landing, host injectivity and pairing tests, next-pop credits,
   unique waypoints, reset-after on first runs.
6. **Two-format tensor-backed CB** not accepted at runtime. Mitigation: the two named fallbacks of §4.7, decided in P1-0.
7. **Clock under sustained compute** (F0: 1,068-1,212 MHz at 125-140 W on compute-bound shapes). Phase 2's VLM may throttle;
   aiclk is sampled in every timing run.
8. **qkv-side adaRMS fold precision** (bfp8 of `diag(1+a)W`; p300 unit PCC 0.999 vs unfolded). Gate in P1-3; fallback:
   apply `(1+a)` to the landed x on the qkv cores (32 bcast multiplies, ~1 us) with the unfolded Wqkv.
9. **KV chunk pulls on NoC 1 contending with exchanges** (7 MB per layer across the attention cores at base). Mitigation:
   one-layer-ahead prefetch; fallback: one reader per chunk multicasting to its 16-core row pair.

Open questions for the user (none blocks phase 1):

1. If the phase-2 binary cannot be brought under 136,192 B (§5.6 step 4): may the process open the device with a larger
   worker-L1 cut (process-wide)?
2. The server's batching workers run B > 1 through today's fused path; this design is batch 1 and a megakernel process
   refuses B > 1 (§4.12). The geometry extends to B = 2 (rows = B x 64), but B > 1 is not in the gates. Is batch-1-only
   acceptable for the megakernel server, or is B > 1 required?

## 9. Sources

pi0.5: `docs/megakernel/PROFILE.md`, `docs/megakernel/profile/summary_{base,libero}_20260930.json`,
`docs/history/MEGAKERNEL_p300.md`, `docs/FUSED_FIX_2026-09-29.md`, `models/experimental/pi0_5/common/fused_config.py`,
`tt/ttnn_fused_attn.py`, `tt/ttnn_fused_norm.py`, `tt/ttnn_gemma.py`, `tt/ttnn_paligemma.py`, `tt/ttnn_pi0_model.py`,
`common/fused_host.py`, `reference/torch_pi0_model.py`, `server/app.py`.
GR00T (`/home/deepgadget/experiments/gr00t/tt-metal/models/experimental/gr00t/tests/tt/results/`): `mk_k1_summary.md` §0,
`mk_k4_summary.md` §0 §3 §5 §7.2 §8, `mk_k4b_summary.md` §0 §1 §3.4-3.6 §5 §6, `mk_k4c_summary.md` §0-§2 §6,
`mk_k5_summary.md` §0 §2, `mk_f0_summary.md` §0 §4 §6 §20, `mk_f0c_summary.md` §4.3; `docs/plan/BACKBONE_FUSION_PLAN.md`
(head); code `tt/megakernel/{core_map,descriptors,bb_descriptors}.py`, `tests/tt/test_mk_mm2d.py`, `kernels/ops/*.hpp`.
tt-metal @ 668c2907575: `tt_metal/api/tt-metalium/program_descriptors.hpp` (CBDescriptor / FormatDescriptors),
`tt_metal/impl/buffers/circular_buffer_config.cpp:65-100`, `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp`,
`tt_metal/api/tt-metalium/circular_buffer_constants.h:33-40`, `tt_metal/llrt/hal.hpp:531`,
`tt_metal/impl/program/program.cpp:2130-2149`, `ttnn/cpp/ttnn-nanobind/device.cpp:195-204` (MemoryView). Memory notes under
`~/.claude/projects/-home-deepgadget-experiments-gr00t/memory/` named inline. Arithmetic: v2
`docs/megakernel/design/mk_design_calc_v2.py` (+ `.out`) and `review_stream_sim.py`; v1 `mk_design_calc.py` is kept for
the record.

## 10. Review resolution (v1 ac2e6ea -> v2)

Two reviews of v1, neither accepted: **R-K** (kernel feasibility; fluid model `design/review_stream_sim.py`) and **R-G**
(goal / integration lens). Every item below is resolved in the text; none is rejected. Figures are from
`mk_design_calc_v2.out` unless another file is named.

| # | review item | resolution | where |
|---|---|---|---|
| B1 (R-K B1, R-G B2) | `cb_in0` holds bf16 x/x_mid and bfp8 ctx under one id; a separate ctx CB (+139,264 B) breaks the budget | One tensor-backed `CBDescriptor` with two format descriptors, ids 0 (bf16) and 29 (bfp8), 139,264 B. Evidence that tt-metal supports it: `circular_buffer_config.cpp:65-100` processes every format descriptor after binding the backing buffer; `format_descriptors` is read-write in the binding. R-G's point that GR00T's `bb_descriptors.py:855-860` skips aliasing for tensor-backed CBs is correct about that helper, not about tt-metal; the helper is not reused. Proven on the device in P1-0, with two named fallbacks (offset-0 double descriptor; GR00T's non-tensor-backed aliased pair). Ordering on the shared bytes = the §4.6 hub ready points. | §4.5, §4.6, §4.7, §7 P1-0, §8 risk 6 |
| B2 (R-K B2) | rings smaller than one op's share; one in-order stream blocks, so DRAM idles and Wd cannot prefetch; v1 "DRAM never idles" / "40-50 us look-ahead" false | `cb_w8` 87,040 -> 139,264 B (16 slots, +52,224 B); sizing rule "each ring >= the core's largest single-op share in its format". `max(chain, stream)` replaced by the per-core ring-constrained timeline (the reviewer's model with v2 rings and phases): base 76.8 / LIBERO 61.1 us at 414-464 GB/s; v1 rings give 84.1 / 76.9 at 414. The false claims are removed from §3.1 and §4.4. | §3.1, §4.4, §4.7, §4.11 |
| B3 (R-K B3) | "tree of depth 3" merge has no landing slots / arrival words at intermediate nodes; cost counts one send for three levels; 3.5 us per merge is for a 4-tile GR00T part, here 8 dh tiles + m, l | Chosen: a **flat, dh-split** merge. Units kc 0..3 of each (h, r) are slice mergers owning 2 dh tiles each; every unit sends its 2-tile O slices + (m, l) (12,288 B) to the other slice mergers; each folds 4 (LIBERO 5) parts in fixed order. Priced per O tile: 14 tile-ops per fold (1 row x (8 stat ops) + 2 O tiles x 3) at T_E = 0.125 us (GR00T's 3.5 us / 28 tile-ops): base 2.0 + 4 x 1.75 = 9.0 us, LIBERO 10.75 us (priced the same way, a merger-only depth-3 tree is 3 x (1.5 send + 32 tile-ops x T_E) = 16.5 us and a merger-only flat merge 1.5 + 4 x 4.0 = 17.5 us). L1: `cb_part` = 4 x 12,288 = 49,152 B on T1/T2/T3 at base, 5 x 12,288 = 61,440 B at LIBERO (listed, not "halved"). Sync: `part_arrive[src]` per slice merger + `part_ready` back. R4 becomes a 64-source (LIBERO 32) hub gather of 2-tile slices. | §4.2, §4.3, §4.7, §4.8 |
| R-K 4 | KL multicast rectangle intersects all 8 Q column rectangles | KL multicasts only after `kl_qdone` (8 atomics from the Q leaders after their multicast barriers). The RoPE of K_s moved to the K producers (pair exchange), so KL only gathers and multicasts; the KL path is hidden behind the T4 units' prefix key tiles (5.6 us base, 3.7 us LIBERO), with a P1-2 gate. q RoPE likewise moved to the Q producers (priced in R2). | §4.2, §4.3, §4.6, §7 P1-2 |
| R-K 5 | "one arrival word per (slot, generation)" cannot be literal (540 generations, 128 words); Q-column receivers (h, 9) are not sources | All sync words are cumulative counters keyed on (what, slot, peer), generation in the value; two forms (per-peer, summed-with-paired-ready) defined; full word table with targets; explicit per-member ready credits for every leader multicast incl. non-source receivers (`qlead_ready`, `kl_ready`, `row_ready`), with the causality argument recorded as the reason they are off the critical path. Host test asserts injectivity, pairing and ready coverage. | §4.6, §4.8 |
| R-K 6, R-G 3 | KV co-tenant is 78,336 B per core (whole pages per bank), not 71,214; allocatable is 1,371,136 with l1_small 24,576; the CB ceiling is an address | Adopted: 2 pages x 1,088 x 36 = 78,336 B (both shapes: 200 / 160 tiles per cache); allocatable 1,371,136 B; headroom 170,848 B (base) after all v2 changes; P1-0 measures the minimum contiguous free L1 above the CB base at the op's trace position and the L1 gate uses that measurement. Phase-2 resident KV recomputed for its own layout (66,368 B). | §1, §4.7, §5.5, §7 P1-0 |
| R-K 7 | Blackhole allows 64 CB ids, not 32 | Corrected (`circular_buffer_constants.h:38`, `hal.hpp:531`); ids kept dense because each costs 16 B of the ring. | §1, §4.8, §5.5, §7 P2-0 |
| R-K 8 | attention priced with the per-key average 1.72, not the slope 1.87; the 1.0 us term has no source | T_KEY = 1.87 (slope). The RoPE of q moved out of C2 (to the producers, priced by tile-ops in R2); C2 = 5 x 1.87 + A 1.0 (mask, finalise, part pack), gated in P1-2. Base C2 10.36 us, LIBERO 6.61. | §4.3 |
| R-K 9 | R7's fp32 sums and gate are unmeasured compute inside the 5.0 us | Priced: 16 tile-ops per row tile at T_E + A 1.5 transfer = 5.5 us base / 3.5 LIBERO; the P1-2 column-reduce benchmark includes the fp32 sums and gate. | §4.3, §7 P1-2 |
| R-K 10 | LIBERO A-values are ad-hoc scalings; merge stays 12.0 while parts rise to 5 | No LIBERO scaling any more: base A-values unscaled for every round (conservative, labelled A), merge re-priced for 5 parts; P1-2 gates at both shapes. LIBERO chain 52.3 -> 61.1 us; the layer is now chain-bound at LIBERO too. | §4.3, §4.11, §7 P1-2 |
| R-K 11 | ring estimate has no basis; union group larger than any GR00T group; K4b's first build was 150,304 B | Estimate relabelled A with no basis; explicit "size <= 128 KB by mock compile" gate in P1-3, P1-4, P1-5 (and P2-0), counting runtime args (the role tables) and CB-id config. | §4.9, §7 |
| R-K 12 | phase-2 "upper" column is not a bound; x1.36 carried from a stream-bound DiT; exchanges 10 % vs GR00T's 26 %; GELU on 100 cores vs 88 | Column renamed "at F0-harness rates (NOT an upper bound)"; GELU 139.6 tiles per core on 88 cores (81.0 us lower); the "point" (x1.36) is dropped for phase 2; GR00T-realised figures added (15 %: 2,522 us, 7 %: 5,079 us per layer, 26 % exchanges), stating the range straddles TTNN and P2-1 decides. | §3.2, §5.4 |
| R-K 13 | whole-call arithmetic mixes profiled stage times with unprofiled calls | Stated in §4.11 (<= ~0.4 ms); the gates measure the whole call directly. | §2, §4.11 |
| R-G B1 | no gate proves "ALL ops are ONE fused op"; no stamp; no knob; phase 1 could be shipped as the megakernel | Structural exit gates (phase 1: one device op after the VLM for the whole loop + action io; phase 2: exactly one device op per replay), measured with PROFILE.md's per-replay op-list method; `PI05_MEGAKERNEL` knob with default and refusals; `megakernel_backend` / `megakernel_program` stamp asserted in `test_pcc_pi05_fused.py`; "phase 1 is intermediate; phase-2 failure = needs_user". | header, §4.12, §7 |
| R-G 3 (factual) | GR00T already demonstrates non-tensor-backed CB aliasing (`bb_descriptors.py:846-878`, `test_aliased_cb_accepted`) | Corrected; only the tensor-backed two-format descriptor is new, and P1-0 proves it. | §4.7, §5.5 |
| R-G 5 | trace replay does not re-validate CB regions; parked B > 1 shapes (L1 inputs) could corrupt / be corrupted | (1) a megakernel process prepares exactly one shape (B > 1 refused); (2) an L1 guard compares the allocator state (`ttnn.get_memory_view`) at capture end with the state before every replay and raises on any change; P1-0 tests the guard; P1-6 tests the B > 1 refusal (the server warm-up with `PI05_BATCH_SIZES=1,2` refuses at startup). | §4.12, §7 P1-0 / P1-6 |
| R-G 6 | phase 2 pads the prefix (768 / 576) but claims the phase-1 schedule unchanged; VLM key mask for pad rows undefined | Pad rows are never written to the resident KV region (rows 0..P-1 only), so the 25 / 18 key-tile chunkings stand; pad queries are discarded; the VLM key mask is the `[1, P]` per-key row of `prefix_attention_inputs`, host-tested against `vlm_mask`. §5.3 now says "all P keys" consistently. | §5.1, §5.2, §5.3 |
| R-G 7 | no server work package; the 64 KiB cut is absent at all 7 device-open sites; /info, mesh / B > 1 refusals, served latency missing | WP-P1-6 and WP-P2-5 with gates (smoke test, served median < 84.02 ms then < the phase-1 served median, /info stamp, refusals tested); one `open_pi05_device` helper used by the 7 sites listed with line numbers. | §6.1, §7 |
| R-G 8 | test-plan gaps on the required semantics | Exit gates add base n_lang 1 / 224 / 128 (kc 3 / kc 4 boundary) / 150 (boundary inside a chunk-4 key tile), LIBERO n_lang 32 / 1; the existing suites stay green with the megakernel selected; P1-0 host test of the mask / RoPE tables against `prefix_attention_inputs` (cumsum - 1 and `n_valid + [0,H)`). | §7 P1-0, exit gates |
| R-G 9 | LIBERO speed not gated | Binding: phase 1 expert < 27.85 ms and whole call < 76.77 ms at LIBERO; phase 2 < phase 1 at LIBERO too; P2-3 stack < 48.72 ms at LIBERO. | §7 |
| R-G 10 | shape contract covers only two shapes; RoPE precondition unstated | Explicit refusal for any other (P, S), tested; the precondition (right padding, every camera valid) stated, the `ttnn_pi0_model.py:283-286` refusal kept, a masked-camera refusal added. | §4.12 |
| R-G 11 | larger L1 cut is process-wide, not per program; host-side transforms outside the fused op unlisted | §5.6 step 4 and open question 1 reworded (process-wide, and what it costs a comparator path); §4.12 lists every host-side transform and why it is input / output formatting. | §4.12, §5.6, §8 |

What did not change, and why: the phase-1 base chain stays 76.8 us. The corrections that raise it (attention slope +0.8,
RoPE priced +1.75, reduce compute +0.5) are offset by the flat dh-split merge (9.0 us, against v1's unsupported 12.0 and
the 16.5 us a merger-only tree costs when priced per O tile the same way). The base expert-loop prediction (13.91 / 18.92
/ 23.37 ms) and the whole-call prediction (66.7 / 71.7 / 76.2 ms) are therefore within 0.1 ms of v1. LIBERO moves from
10.05 / 13.66 ms to 11.08 / 15.07 ms, still under today's 27.85 ms with 2.52x of margin.

## 11. Phase-2 implementation design v3 (2026-09-30, session mk2-r0-s0) -- what was built, and the deviations from §5

The whole `sample_actions` is ONE generic_op whose three kernels (`tt/megakernel/kernels_p2/whole_{brisc,ncrisc,trisc}.cpp`)
run the prefix engine and then call the phase-1 expert kernel (`../kernels/mk_*.cpp`, included byte-identical and renamed
`mk_expert_kernel_main`). `PE_WHOLE=0` builds the prefix-only program used for bring-up.

### 11.1 Execution model (deviation from §5.3: DRAM-staged ops instead of L1-resident bands)
- A fixed sequence of 314 ops (`kernels_p2/pe_defs.hpp`, `pe_common.hpp describe`): patch; 27 x [LN1, qkv, attn, o, LN2,
  fc1, fc2]; post-LN; projector; embedding; 17 x [RMS1, qkv, attn, o, RMS2, gate|up, down]; RMS1 + qkv of layer 17.
- Every activation lives in DRAM scratch between ops; consecutive ops are separated by ONE global barrier (every core
  arrives after its writes are acknowledged and its NCRISC is done; the hub (10, 9) multicasts go). Reason: one
  synchronisation mechanism orders every buffer reuse, and each op can be run and checked alone (bring-up by op range).
  Cost: 314 barriers (~3 us each) and the DRAM round trips of the activations (priced in §11.4).
- Matmuls: 8 bands (grid rows 0..7) x 11 columns. Weights are read ONCE per column from a bank-striped arena by a
  feeder core (x, 8) and multicast down the column; in0 bands are read by a feeder (b, 9) and multicast along row b.
  Mode R (every op but the VLM down): in0 band resident, N-outer, the whole K accumulated in DST. Mode S (VLM down,
  K = 512 tiles): in0 streamed in K blocks, K-outer, fp32 partials reloaded (UnpackToDestFp32).
- Rings: per-page flag = (op << 16 | page + 1); credits are one word PER RECEIVER and the feeder waits on the minimum
  (a summed credit is wrong for ring depth > 1: device-reproduced on the down op). Data multicast flushed before the
  flag multicast (Blackhole command-buffer ordering).
- CB ids 32..62 are declared tiny on the host and re-pointed per op by every RISC into the arena = phase-1 CB region +
  a tail CB (host descriptor order fixes contiguity; the kernel reports lo / hi in a diagnostics tensor). A CB used
  with variable page counts is re-pointed to full-capacity cycles (P_SS per attention chunk).
- The NCRISC computes each op's geometry and hands the TRISCs one descriptor page per op (P_OPD, read_tile_value /
  mailbox): no geometry code in the TRISC binaries, and a TRISC can never start an op before its barrier.
### 11.2 Numerics (as §5.7, with these differences; revised 2026-09-30 evening, see §11.4)
fp32 residual streams (SigLIP and VLM) instead of bf16 / bf8; ~~LN / RMS statistics in exact fp32 on the SFPU~~ (now
§11.4: FPU statistics, fp32 partials exchanged); bf16 q and normalised activations; K / V bfp8 into the expert caches
(as today); h of the VLM MLP bfp8 (as today); ~~VLM matmuls LoFi~~ VLM matmuls HiFi2 with fp32 accumulation (LoFi
failed 4 of 22 seeds of the amended gate), SigLIP HiFi2; q scale folded into the RoPE tables (VLM, exact 1/16) and into
Wq / bq (SigLIP, before the bfp8 rounding).
### 11.3 Gates
§7 with the 2026-09-30 amendment (phase 2: per seed vs the fp32 whole-model reference) and the 2026-10-01 amendment (P2-3 K/V vs fp32, not vs the ttnn caches). WP-P2-1's comparison "vs the ttnn layer (same run)" is run on real
activations; per-op checks against the host decomposition on the device's own inputs are recorded in addition.

### 11.4 Revisions after the first end-to-end build (2026-09-30 21:30 - 23:40; JOURNAL.md has every number)
- Norms: items are (row tile, column group) (SigLIP 16 x 6, VLM mt x 4, one item per core). Every item core computes
  partial (sum x^2, row sum x) over its group on the FPU (accumulating ELWMUL, HiFi4 matmul with ONES), writes the fp32
  pair into slot g of P_R on the row's item cores (noc_async_write, write barrier, PS_NR increments) and reduces the
  ncg partials itself: var = E[x^2] - mu^2. The affine parts are FOLDED on the host into the consuming matmul
  (pe_host.fold_norms: diag(g) W, b + beta W; SigLIP LN1 -> qkv, LN2 -> fc1, post-LN -> projector, VLM (1 + w) ->
  qkv and gate|up), so the norm applies (x - mu) rstd / x r only.
- in0 of mode R: distributed. The band's compute core in column q reads K piece q (all rp rows) from DRAM and
  multicasts it along its row, flag PS_IV0 + q ((op << 16) | 1); the multicast of piece q waits for piece q - 1 (the
  reads do not). No credits (the arena is free after the op's go). The row-9 feeders only serve mode S (VLM down).
  Reason: one reader per band capped the 8 bands at ~90 GB/s together (row-9 links).
- SigLIP attention: one key chunk (8 tiles) per q row tile, 96 items = (image, head, row group {3, 3, 2}), one per
  core; the 3 item cores of a head each read a third of its K / V and write it into the other two (PS_KVX).
- VLM K / V for attention: 4 feeders (7..10, 9) read quarters of the L1 caches and multicast to every core.
- GELU: relu(x) - t q(t), t = min(|x|, 4.25), degree-9 fit of the tanh form (max abs err 4e-5 in fp32); the library
  gelu_tanh costs ~2,200 cycles per tile serial with the matmul (dst_full_sync). Softmax exp: the library fp32 exp
  (faster SFPU exps: arm PE_EXP_FAST; they failed seed 707 of the base gate in three builds).
- VLM qkv weights bf16 (own arena per layer, PA_WV16): bfp8 weights are the largest prefix K / V error term (CPU
  emulation, p2/results/emu/); other VLM and all SigLIP weights stay bfp8.
- The BRISC takes the op's (Op, Lay) from the NCRISC (P_SYNC words 48..63, PS_OPK) instead of computing them.
- Timing / precision arms (PI05_PE_DEFINES): PE_DBG_TRACE=<k> (per-role wall-clock marks of the k-th op, read by
  tests/megakernel/pe_trace.py), PE_DBG_NO_GELU, PE_DBG_GELU_EMPTY, PE_DBG_IN0_NOREAD / NOMCAST / BIGREAD / ONLY0,
  PE_IN0_LAG, PE_IN0_READ_STAGGER, PE_MM_HIFI4, PE_GELU_STOCK, PE_EXP_FAST, PE_VLM_LOFI, plus the §11.1 ones.
