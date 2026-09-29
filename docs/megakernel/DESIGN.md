# pi0.5 as a true megakernel on one p150a: design (2026-09-30)

Goal (the user's words, not reinterpreted): **all of the model's ops become ONE fused op**, a persistent program of custom
kernels (`ttnn.generic_op` + `ProgramDescriptor`, like the GR00T DiT megakernel). A Metal trace of stock ttnn ops, or
"fewer launches", is not a megakernel. Two stages in the user's order:

- **Phase 1**: the whole 10-step action-expert denoise loop (18 Gemma-300M layers x 10 steps, adaRMS, GQA 8 q heads + 1
  KV head at head_dim 256 over the prefix KV cache with the padding mask and the offset RoPE, GeGLU MLP, gated residuals,
  Euler step) as ONE persistent generic_op: in-kernel layer and step loops, weights streamed from DRAM, activations in L1.
- **Phase 2**: SigLIP x2 + projector + language embedding + the VLM 18-layer prefill (which produces the prefix KV
  cache) + the phase-1 expert loop, all ONE fused op for the whole `sample_actions`.

Every number below carries its source. **M** = measured (file named), **D** = derived arithmetic from measured inputs,
**A** = assumed (not measured on this card for this shape; each A has a work package that measures it). Nothing here
has run on the device yet.

## 0. Status

| item | status | notes |
|---|---|---|
| profile of the shipped path (`PROFILE.md`) | DONE | b93723c |
| this design (`DESIGN.md`) | DONE (v1) | reviewed by nobody yet; the pre-registered gates of §7 are fixed from this version on |
| WP-P1-0 scaffolding + host checks + mock compile | NOT STARTED | first thing of step 2 |
| WP-P1-1 weight arena + stream harness | NOT STARTED | |
| WP-P1-2 exchange microbenchmarks (the A-values of §4.11) | NOT STARTED | decides the down/merge geometry before any layer code |
| WP-P1-3 one expert layer in-kernel | NOT STARTED | |
| WP-P1-4 whole 10-step loop, standalone | NOT STARTED | |
| WP-P1-5 integration into `sample_actions_fused` + all gates | NOT STARTED | phase-1 exit |
| WP-P2-0 phase-2 binary size + L1 overlay proof (no card) | NOT STARTED | |
| WP-P2-1 VLM layer prototype (go / no-go on timing) | NOT STARTED | GR00T's fused prefill ended slower than TTNN, so this is gated by measurement |
| WP-P2-2 SigLIP layer prototype | NOT STARTED | |
| WP-P2-3 whole prefix in-kernel (SigLIP + projector + embed + VLM -> KV) | NOT STARTED | |
| WP-P2-4 whole `sample_actions` as one program + all gates | NOT STARTED | phase-2 exit |

## 1. Inputs and hard limits

- Target: lerobot/pi05_base (served shape: 2 x 224^2 cameras, 224 tokens, H = 50 -> 64 suffix rows, prefix P = 736) and
  lerobot/pi05_libero (32 tokens, H = 10 -> 32 suffix rows, P = 544). Batch 1 (§8 open question 2).
- tt-metal `/home/deepgadget/experiments/gr00t/tt-metal` @ 668c2907575 (read only). All new code lives in the pi0.5 repo
  under `models/experimental/pi0_5/tt/megakernel/` (GR00T code is copied in, never imported, §6.1).
- **Kernel-config ring: 136,192 B per program** (`1,531,904 - worker_l1_size`, worker_l1_size 1,395,712 = the 64 KiB
  cut; `mk_k5_summary.md` §2.2). Checked by an offline mock-cluster compile + size check before any device run.
- **64 KiB worker-L1 cut** -> allocatable L1 per core 1,362,944 B (`mk_k4b_summary.md` §5; memory note
  `megakernel-k4-device-facts` addendum "Budget: 1,362,944 B per core").
- Grid 11 x 10 = 110 worker cores, aiclk 1350 MHz (M: `profile/clock_*_20260930.json`).
- DRAM stream rates (M, `mk_k1_summary.md` §0): **464 GB/s bf16, 414 GB/s bfp8** direct-mode arena reads; K1's matmul-only
  block chain delivered 479 GB/s bf16 / 383 GB/s bfp8 effective.

## 2. What must be beaten (from `PROFILE.md`, all M)

| shape | whole call | replay | SigLIP | VLM | expert (10 x 18) | action io | expert per layer |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | **84.21 ms** | 82.81 | 10.767 | 41.536 | **31.263** | 0.167 | 173.1 us |
| LIBERO | 76.77 | 75.64 | 10.765 | 37.853 | **27.845** | 0.163 | 153.8 us |

The current expert layer (base, M, `PROFILE.md` per-expert-layer table): adaRMS `rms_norm` on 2 cores 14.2 us, qkv 9.5,
fused attention (16 cores) 43.0, typecast 2.0, o_proj dit (22 cores) 20.6, row_rsqrt (2 cores) 7.1, up|gate 26.7,
geglu_rc 7.6, down dit (22 cores) 36.3, plus 9 launch gaps. It streams 24.28 MB per layer (D: qkv 2,560 bfp8 tiles,
o_proj 2,048 bf16, up|gate 8,192 bfp8, down 4,096 bf16), 437.1 MB per step, 4.371 GB per request; the matmuls stream at
261 GB/s while active and 140 GB/s over the stage.

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
   L1) and the stream is the clock: each core's NCRISC walks its own consumption-ordered share of every layer and every
   step into rings that hold most of a layer's share (§4.4), so DRAM never idles while exchanges run. At LIBERO the
   layer is stream-bound (§4.11): the exchanges are fully hidden.
2. **The exchange unit was the matmul stage.** Here it is the layer: 3 hub all-gather rounds per layer (x, ctx, x_mid)
   plus 4 group-local exchanges, against 9 launches today and GR00T's 11 hub rounds per DiT block. No exchange is an
   all-to-all of multicasts: every one-to-many step is ONE multicaster per rectangle (gather-to-leader by unicast, then
   one multicast), which respects both the p300 rule (one multicaster per program region) and the GR00T in0 law
   (~1.8 us per other owner when L owners multicast into a band, `mk_f0c_summary.md` §4.3).
3. **The stage it fused was already efficient in ttnn.** The time this design removes is exactly the non-matmul time
   the p300 note pointed at: 14.1 of the 31.3 ms expert stage (M) is not matmul — the 2-core adaRMS norm, 43 us of
   attention on 16 cores, two small custom programs, a typecast, 9 fixed kernel costs per layer — and the o_proj/down
   dits run on only 22 cores at 203/231 GB/s.

### 3.2 GR00T DiT megakernel (same card, same tt-metal) — what is reused

- **Measured round costs** (`mk_k4b_summary.md` §3.6, bfp8, warm self block 105 us): hub round (96 x 2 KB gather ->
  192 KB multicast, dual NoC) ~6 us on top of compute; attn_out round 10.3 us = Wo 3.7 + round; Q unicast 1.5 us; attention
  part send 1.5 us + two merges ~7 us; ff2 chunk rounds are latency-bound at ~5.3 us quiet / ~11 us in the steady state,
  one chunk in flight per hub (`mk-chunk-rounds-latency-bound` memory note).
- **Measured compute rate**: QKV `mk_P2_mm` 5.5 us for 192 tile-matmuls per core (2 rows x 2 N tiles x 48 K) at HiFi2 ->
  **0.029 us per tile-matmul** (D). Used for every matmul-compute estimate below (conservative for LoFi ops).
- **Realisation ratio**: the plan estimated the bfp8 block at 77 us; K4 measured 129.3 (x1.68), K4b 105 (x1.36)
  (`mk_k4_summary.md` §0, `mk_k4b_summary.md` §0). Every bottom-up time below is also quoted x1.36 ("expected") and
  x1.68 ("first build").
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

`BACKBONE_FUSION_PLAN.md` (head) and the memory note `fused-backbone-e2e-slower-than-ttnn-all-versions`: the fused
Qwen3/SigLIP/VL-SA stages ended **slower** than TTNN end to end (0.844x / 0.66x / 0.77x). Causes measured there: the in0
band exchange at ~1.8 us per other owner per matmul round (~38 us per matmul at L = 22), weight delivery limited by a
unicast forward (129-206 GB/s against 464), concurrent band multicasts costing 148-168 us per layer until staggered,
GELU epilogues at ~1.5 us per output tile on the PACK thread, a per-block launch floor, and matmuls realised at 7-15 %
of peak inside the fused layer against 39-55 % in the F0 harness. Phase 2 is therefore budgeted by a measured prototype
(WP-P2-1 is a go/no-go), and its geometry (§5.3) avoids the per-owner in0 exchange by construction: one in0 multicaster
per band, weights read once per column and multicast down the column.

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
  §4.5; 1/sqrt(256) folded into the q tables, the rotate-half sign into the sin tables, as today).
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
  0   T1   T1   T1   T1   T1   T1   T1   T1  | KV   KV  | H1 (spare)
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
| **Q producers** | (h, d), h = x in 0..7, d = y in 0..7 (64) | qkv: Q column tile (head h, dh tile d), both row tiles |
| **K / V producers** "KV" | (8, d) K tile d, (9, d) V tile d (16) | qkv: suffix K / V column tile d |
| **KV leader** "KL" | (8, 8) | gathers K_s / V_s (32 tiles), RoPEs K_s, multicasts to the last-chunk attention cores |
| **attention units** | (h, y), y = 2*kc + r (80 at base: 8 heads x 2 q row tiles x 5 key chunks) | one key chunk of one (head, q row tile) |
| **mergers** | (h, r), r = y in 0..1 (16) | fold the 4 remote (m, l, O) parts of unit (h, r) |
| **Q leaders** | (h, 8) | gather Q[h] (8 x 2 tiles) from (h, 0..7), multicast to column h (10 cores) |
| **MLP cores** (kg, ng) | (x = ng, y = kg), 0..7 x 0..7 (64) | up|gate (4 N tiles = 2 u + 2 g), GeGLU, down partial (16 K x 4 N) |
| **x owners** O_n | n = 0..31 at (x = n // 4, y = 4 + n % 4) (32, the T3 rows) | own x[:, n] (fp32), o_proj column n, gated residuals |
| **hubs** | H0 (10, 9): x and x_mid rounds + the per-step tail; H1 (10, 8): ctx round | gather -> landing ring -> dual-NoC multicast |

Core types: T1 = Q producer + MLP + attention kc 0 + merger; T2 = Q + MLP + attention kc 1; T3 = Q + MLP + attention
kc 2/3 + x owner + o_proj; T4 = attention kc 4 only (holds the suffix keys); KV = K/V producers. At LIBERO (1 q row tile,
18 key tiles) the attention units are 8 heads x 6 chunks of 3 key tiles = 48 cores (rows 0..5; the suffix chunk is kc 5);
rows 6..9 carry no attention work. The role tables are host-built runtime tables (one compiled program per shape, as the
trace is per shape today), and a host test asserts every table is injective and every rectangle is what the role needs.

### 4.3 Layer schedule (base; LIBERO halves the rows)

| # | phase | cores | data movement | critical-path cost |
|---|---|---|---|---|
| R1 | **x round**: owners send x tiles to H0 (32 x 2 tiles bf16 = 128 KB landing); H0 multicasts x (128 KB) to rectangle (0..9, 0..7), dual NoC; H0's TRISC computes `r_in` (64 tiles squared + row sums + rsqrt, fp32) during the multicast and multicasts the 2 `r` tiles after it | 32 -> H0 -> 80 | 128 KB | A 6.5 us (GR00T hub round, 192 KB) |
| C1 | **qkv**: 1 N tile x 2 rows x 32 K per core, fp32 acc, epilogue `r ⊙ acc + c` | 80 | weights in ring | D 64 x 0.029 = 1.9 us |
| R2 | **Q / K / V distribution**: (h, 0..7) unicast Q tiles to Q leader (h, 8), which multicasts Q[h] (16 tiles, 32 KB) to column h; K/V producers unicast to KL, which RoPEs K_s and multicasts K_s|V_s (32 tiles, 64 KB) to rectangle (0..7, 8..9) | column-local | 8 x 32 KB + 64 KB | A 3.5 us |
| C2 | **attention chunk**: RoPE q (own row, 8 tiles), S = q K^T over 5 key tiles (K/V chunk prefetched from the cache, §4.5), mask, row max / exp / row sum, O = P V (8 dh tiles) -> partial (m, l, O) | 80 | — | D 1.0 + 5 x 1.72 = 9.6 us (current kernel: 43.0 us / 25 key tiles, M) |
| R3 | **merge**: parts of (h, r) from (h, 2kc + r) to merger (h, r), tree of depth 3 | column-local | 4 x (8 bf16 + 2 fp32 tiles) | A 1.5 + 3 x 3.5 = 12.0 us (GR00T part send + merge per level) |
| R4 | **ctx round**: 16 mergers send ctx (8 bfp8 tiles each) to H1; H1 multicasts ctx (128 bfp8 tiles, 136 KB) to the 32 owners (0..7, 4..7) | 16 -> H1 -> 32 | 136 KB | A 6.5 us |
| C3 | **o_proj + gated residual** on owner n: 64 K x 1 N x 2 rows, `x_mid = x + g_attn ⊙ acc` (fp32 residual) | 32 | weights in ring | D 128 x 0.029 + 0.5 = 4.2 us |
| R5 | **x_mid round** (as R1, H0; `r_post` computed on H0) | 32 -> H0 -> 64 | 128 KB | A 6.5 us |
| C4 | **up|gate** 4 N x 2 rows x 32 K, epilogue `r ⊙ acc + c`; **GeGLU** `h = u * gelu(g)` (2 output tiles per row) | 64 | weights in ring | D 256 x 0.029 = 7.4 us + A 4 x 1.5 = 6.0 us (F0 PACK GELU per tile) |
| R6 | **h group exchange**: row kg (8 cores) gathers its h slice at the row leader (7 x 8 KB unicast), leader multicasts 64 KB to row kg | 8 rows in parallel | 8 x 64 KB | A 4.0 us |
| C5 | **down partial** 16 K x 4 N x 2 rows | 64 | weights in ring | D 128 x 0.029 = 3.7 us |
| R7 | **reduce to owners**: column ng (8 cores) sends each partial tile (fp32) to its owner in rows 4..7 of the same column; owner sums 8 in fp32, `x = x_mid + g_mlp ⊙ sum` | column-local | 7 x 2 fp32 tiles per owner | A 5.0 us |

Chain per layer (base): 6.5 + 1.9 + 3.5 + 9.6 + 12.0 + 6.5 + 4.2 + 6.5 + 13.4 + 4.0 + 3.7 + 5.0 = **76.8 us** (D from the
M and A inputs above). LIBERO: **52.3 us** (§4.11).

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
- **Rings**: two per compute core because a CB has one data format: `cb_w8` (bfp8, 10 slots x 8,704 B = 87,040 B) and
  `cb_w16` (bf16, 8 slots x 16,384 B = 131,072 B). The NCRISC fills them in stream order under per-slot credits; the
  TRISC consumes. A ring covers 45-70 % of a core's per-layer share, i.e. ~40-50 us of look-ahead, which spans the
  attention and exchange phases in which the TRISC does not consume weights.
- **Rate targets**: the layer needs 24.28 MB / 76.8 us = **316 GB/s** aggregate at base (D) to stay chain-bound (below
  the 464 / 414 GB/s K1 rates, M); at LIBERO the stream is the bound: 28.3 us (bfp8 part, 11.70 MB at 414) + 27.1 us
  (bf16 part, 12.58 MB at 464) = **55.4 us per layer** (D). Pass target for WP-P1-1: >= 400 GB/s sustained over the
  180-layer stream with the real shares.
- bf16 o_proj/down stay (today's precision rule, `PI05_FUSED_RESIDUAL=bf16`). All-bfp8 would cut the stream to 44.4 us
  per layer (D); it helps only LIBERO (chain 52.3 us), so it is a later lever behind a PCC check.

### 4.5 Runtime inputs: what enters per request, and why the op stays trace-replayable

The generic_op's runtime args hold only addresses and shape constants, fixed per shape. Everything that changes per
request is data in fixed-address tensors that the host rewrites before `execute_trace` (as today:
`copy_host_to_device_tensor`, only when the prefix validity changed):

| input | tensor (today's object) | read by | when |
|---|---|---|---|
| prefix K/V, 18 layers | the backbone-owned caches `[1,1,cache_len,256]` bf8, L1 interleaved (`ttnn_paligemma.allocate_kv_caches`), written by the ttnn VLM ops earlier in the same trace | attention cores, their chunk rows (5 key tiles x 8 dh x K,V = 87,040 B per layer), on NoC 1, prefetched one layer ahead into a double buffer | every layer |
| key mask row `[1,1,32,P+S]` bf16 (`exp_mask`) | persistent DRAM input | attention cores: their chunk's 5 tiles | once per launch (prologue) |
| RoPE tables cosq/sinq/cosk/sink `[1,1,S,256]` | persistent L1 inputs (`attn_in["tables"]`) | attention cores (own q row), KL (k) | prologue |
| noise x_T `[1,S,32]` | persistent input (`_fused_in_noise`) | H0 | prologue |
| output x_0 `[1,S,32]` | persistent output (the trace's output tensor) | written by H0 | end |

n_valid enters only through the mask and the RoPE rows, so the program never changes with prompt length. The suffix
K/V are never written to the cache (as today). The in0 landing arena and the hub landing rings are **transient
in-trace L1 residents** (allocated right before the generic_op, freed right after; K5 pattern), never returned to the
caller (memory `megakernel-buffers-must-not-escape`).

### 4.6 Exchange and barrier protocol (reused from GR00T; nothing new in kind)

- **RISC roles** on a compute core: NCRISC (NoC 0) = weight stream only; BRISC (NoC 1) = all exchanges, KV chunk
  prefetch, table loads, in a non-blocking service loop (K4c: a blocking BRISC made the ff2 rounds 6 % slower);
  TRISC = compute. On the hubs: NCRISC + BRISC split every multicast in halves on the two NoCs, TRISC computes r / tail.
- **Hub rounds** = `allgather.hpp` `GatherSender` / `McastSender` (K4b form): receivers announce ready credits before
  sending into the round; the hub waits for all sources in its landing ring, multicasts, then publishes the round
  counter. Round index `g = (s*18 + l)*3 + k` (k = x, ctx, x_mid) is the flag value (absolute, monotone), so no reset
  race exists and a stale word can only under-report.
- **Leader exchanges** (R2, R6): unicast writes into the leader's landing slot + one counted atomic per source into the
  leader's arrival word for generation g; the leader waits for `n_src` arrivals, multicasts, publishes g. Destinations
  that are also sources release their landing slot before blocking on the round (the self-membership rule).
- **Merges / reduces** (R3, R7): source-attributed slots (part slot index = sender kc; reduce slot = sender kg), one
  arrival word per (slot, generation), so no two (what, generation, peer) tuples share a word — asserted on the host
  for every instance before any device time (`sync-words-must-name-what-they-guard`).
- **Credits** for every reused slot are absolute per member and taken as the minimum; a producer reuses a slot only
  after the consumer's NEXT pop (`cb-pop-is-not-consumption`).
- **Start**: every L1 sync word zeroed, then one BootBarrier (one atomic per core to H0, one multicast back).
- **Multicast hygiene**: NCRISC multicasts pass rectangles end-first; loopback when the sender is inside the rectangle
  (Q leaders, row leaders), `num_dests` excluding the sender otherwise; sync atomics stay on the unicast VC
  (`transaction-count-is-not-the-cost-model-vc-contention-is`); concurrent multicasts go to disjoint rectangles only,
  and the per-column / per-row ones are staggered by index when WP-P1-2 shows contention.
- **Hang diagnosability**: one unique waypoint per blocking call site inside every poll loop; every fixed-size kernel
  array has a host-side mirror check; a debug word per core holds (s, l, phase) for post-mortem reads.

### 4.7 Per-core L1 budget (base; bytes; bf16 tile 2,048, bfp8 1,088, fp32 4,096)

| CB (id) | size arithmetic | bytes | T1 | T2 | T3 | T4 | KV |
|---|---|---:|:-:|:-:|:-:|:-:|:-:|
| cb_in0 (0) x / x_mid / ctx landing, tensor-backed | max(64 bf16, 128 bfp8) tiles | 139,264 | x | x | x | | x |
| cb_w8 (1) | 10 x 8,704 | 87,040 | x | x | x | | x |
| cb_w16 (2) | 8 x 16,384 | 131,072 | x | x | x | | |
| cb_acc (3) fp32 | 8 tiles | 32,768 | x | x | x | | x |
| cb_out (4) bf16 | 8 tiles | 16,384 | x | x | x | | x |
| cb_q (5) Q[h] landing | 16 bf16 | 32,768 | x | x | x | x | |
| cb_qr (6) roped q | 8 bf16 | 16,384 | x | x | x | x | |
| cb_rtmp (7) RoPE scratch | 8 bf16 | 16,384 | x | x | x | x | |
| cb_rope (8) cos/sin own row | 16 bf16 | 32,768 | x | x | x | x | |
| cb_kv (9) chunk double buffer | 2 x 5 x 8 x 2 bfp8 | 174,080 | x | x | x | x | |
| cb_ksuf (10) K_s, V_s | 32 bf16 | 65,536 | | | | x | |
| cb_mask (11) | 5 bf16 | 10,240 | x | x | x | x | |
| cb_s (12) scores | 5 fp32 | 20,480 | x | x | x | x | |
| cb_oacc (13) | 8 fp32 | 32,768 | x | x | x | x | |
| cb_ml (14) m, l | 4 fp32 | 16,384 | x | x | x | x | |
| cb_part (15) merger landing | 4 x (8 bf16 + 2 fp32) | 98,304 | x | | | | |
| cb_ctx (16) | 8 bfp8 | 8,704 | x | | | | |
| cb_r (17) r tiles | 2 bf16 | 4,096 | x | x | x | | x |
| cb_bias (18) c / gate tiles | 6 bf16 | 12,288 | x | x | x | | x |
| cb_ug (19) | 4 fp32 | 16,384 | x | x | x | | |
| cb_h (20) own h | 4 bf16 | 8,192 | x | x | x | | |
| cb_hg (21) row h slice (down in0) | 32 bf16 | 65,536 | x | x | x | | |
| cb_dpart (22) down partials | 8 fp32 | 32,768 | x | x | x | | |
| cb_red (23) owner landing | 7 x 2 fp32 | 57,344 | | | x | | |
| cb_x (24) fp32 residual | 2 fp32 | 8,192 | | | x | | |
| cb_xs (25) send staging | 2 bf16 | 4,096 | | | x | | |
| cb_const (26) ones / scaler / zero / ident | 4 bf16 | 8,192 | x | x | x | x | x |
| cb_fence (27) | 1 fp32 | 4,096 | x | x | x | x | x |
| cb_sync (28) raw sync words, 16 B stride | 128 words | 2,048 | x | x | x | x | x |

- Union of all CBs laid out identically on every compute core (the conservative case: static CB offsets are the same
  wherever a CB exists): **1,154,560 B**. Per type (sum of its rows): T1 1,019,392, T2 912,384, T3 982,016, T4 432,128.
- Co-tenants at launch time (phase 1 runs next to ttnn tensors): the 36 L1-interleaved KV caches, 7,833,600 B spread
  over 110 banks = 71,214 B per core (D), plus the L1 RoPE tables and noise (< 20 KB).
- Headroom: 1,362,944 - 71,214 - 20,000 - 1,154,560 = **117,170 B** per core in the worst (union) layout.
- H0 / H1: landing ring 2 x 139,264 + r / tail scratch 32,768 + x_t fp32 8,192 + W_in 65,536 + b_in tiles 65,536 +
  Wout' stream slot 65,536 + r / c_out / noise / const / sync 38,912 = **555,008 B**.
- LIBERO: every row-dependent CB halves; union ~0.9 MB.

### 4.8 CB-id and sync-word map

CB ids are the (id) column of §4.7 (29 of 32 ids; 29-31 spare for debug). Hub-only ids reuse compute ids on the
disjoint hub range with the same data format where a format-bound id is shared (landing ring = id 0 format, r = 17,
x_t fp32 = 24, W_in / Wout' = 2). Format-bound constraints: `cb_acc`, `cb_x`, `cb_red`, `cb_dpart` carry
`UnpackToDestFp32` and are never FPU operands (fp32 adds go copy_tile -> SFPU binary).

Sync words (each on its own 16 B line in `cb_sync` or the hub landing ring header), each encoding (what, generation,
peer): `hub_flag[k]` (round counter, k = x/ctx/x_mid), `hub_ready[k][parity]` (receiver credits), `qlead_arrive` +
`qlead_flag` per column, `kvlead_arrive` + `kvlead_flag`, `part_arrive[kc]` per merger, `row_arrive` + `row_flag` per
MLP row, `red_arrive[kg]` per owner, `kv_prefetch_done[buf]` (local BRISC -> TRISC via CB push, no word),
`boot`. Only `boot` uses a tt-metal semaphore; everything else is a zeroed L1 word. A host test enumerates all
(role, slot, generation, peer) tuples for both shapes and asserts injectivity and that no sender publishes through a
word it also waits on.

### 4.9 Kernel-config ring estimate (136,192 B)

The closest measured binary is GR00T K4c's DiT compute-core program: 110,256 B of ELF text + data (brisc 5,076, ncrisc
17,824, trisc0 36,704, trisc1 31,616, trisc2 19,036), 114,368 B with the 4,096 B config allowance, i.e. 21,824 B of
headroom (M, `mk_k4c_summary.md` §0). It contains distributed LayerNorm, flash SDPA with partial merge, stream matmul
with a GELU epilogue, tail matmuls and Euler. Phase 1 needs the same primitive set minus the compute-core LayerNorm
(the stats move to the hubs, which are a separate kernel group with their own binary) plus RoPE, the r/c epilogue,
GeGLU and fp32 reduce adds. **Estimate: 100-118 KB of the 136,192 B (A, GR00T ±7 %)**; hub group 60-90 KB (A). Rules
from day one: `noinline, noclone` on every helper called from more than one site, runtime CB ids and tile counts (not
compile-time constants) for anything with several call sites, bring-up dump knobs compiled out of production builds.
Checked by the copied `size_check.py` mock-cluster compile on every kernel change (gate in every WP).

### 4.10 Precision plan (phase 1)

| quantity | shipped path (today) | megakernel | effect |
|---|---|---|---|
| Wqkv, Wug | bfp8, LoFi | bfp8 (per-step folds `diag(1+a)W`), LoFi, fp32 acc | new: the qkv-side fold (measured PCC 0.999 vs unfolded on p300, `ttnn_fused_norm.py` comment); gated in WP-P1-3 |
| Wo, Wd | bf16, HiFi2 | bf16, HiFi2, fp32 acc | same |
| residual stream | bf16 | **fp32 on the owners**, bf16 on the wire | more precise |
| r (row rsqrt) | bf16 tile from RowRsqrt (HiFi4, fp32 acc) | fp32 compute on the hub, bf16 tile on the wire | same |
| K/V cache | bf8 | bf8 (read from the same caches) | same |
| attention | one pass over 25 key tiles, HiFi2 | 5-chunk flash with (m, l, O) merge; O parts bf16, merge in fp32 | new rounding of O parts (<= 1 bf16 ulp); gated |
| ctx into o_proj | bf8 then typecast to bf16 | bfp8 | same values |
| h into down | bf16 | bf16 | same |
| down K-split | one fp32 DST accumulation over K = 128 tiles | 8 partials of 16 K tiles, summed in fp32 | fp32 reordering only |
| GELU | tanh form | tanh form (exact, not `fast_and_approx`: GR00T's approximation failed 9/12 golden taps) | same |
| Euler state | bf16 (dit op) | fp32 on H0 | more precise |

### 4.11 Time prediction, phase 1

Inputs: GR00T measured round costs and rates (§3.2, M), the current fused attention's cost per key tile (M), assumed
exchange costs marked A in §4.3 (each is a WP-P1-2 measurement).

| | base | LIBERO |
|---|---:|---:|
| chain per layer (sum of §4.3; D from M + A) | 76.8 us | 52.3 us |
| weight stream per layer at 414/464 GB/s (D) | 55.4 us | 55.4 us |
| layer time = max (bound) | 76.8 (chain) | 55.4 (stream) |
| step = 18 x layer + 8 us tail | 1,390.5 us | 1,004.7 us |
| **expert loop, bottom-up** | **13.90 ms** | **10.05 ms** |
| expected (x1.36, GR00T K4b realisation) | **18.91 ms** | **13.66 ms** |
| first build (x1.68, GR00T K4) | 23.36 ms | 16.88 ms |
| today (M) | 31.26 + 0.17 action io | 27.85 + 0.16 |
| **whole call** = today - 31.43 (- 28.01) + expert loop | 66.7 / **71.7** / 76.1 ms | 58.8 / **62.4** / 65.6 ms |

Break-even: the current layer costs 173.1 us (M); the design misses the phase-1 speed gate only if the realised layer is
> 2.25x its bottom-up chain. If WP-P1-2 finds the 2-D down exchange (R6 + R7, 9 us assumed) is not reachable, the
fallback is GR00T's measured ff2 chunk ring (4 chunks of 128 KB at 5.3-11 us each, `mk_k4b_summary.md` §3.4): chain
90-113 us, expert 16.3-20.4 ms bottom-up — still under the gate. The whole-call figures assume the ttnn SigLIP/VLM ops
keep their time under the 64 KiB cut (WP-P1-0 measures it).

## 5. Phase 2: the whole `sample_actions` as one program

### 5.1 Structure

One generic_op, one launch, phases separated by global barriers (BootBarrier form, A ~4 us each, 4 transitions):

1. **SigLIP x2** (27 layers, 512 rows = 2 images x 256 patches) from the im2col'ed images;
2. **projector** (1152 -> 2048) + **language embedding** (in-kernel gather of 224 rows of the DRAM table by token id,
   tilize, x sqrt(2048)) + assembly into the prefix residual (736 rows);
3. **VLM prefill** (18 Gemma-2B layers; the last one KV-only; no final norm), which writes every layer's roped K and V
   straight into an L1-resident KV region laid out exactly as phase 1 reads it;
4. **expert loop** = the phase-1 schedule unchanged, reading the resident KV region instead of the ttnn caches.

All four share one binary per core group (§5.6) and one L1 plan with phase overlays (§5.5). Nothing is returned to the
host between phases.

### 5.2 Inputs (all fixed-address tensors; per request only their contents change)

im2col images `[2, 256, 608]` bf16 (today's `_fused_in_im2col`), token ids `[224]` uint32, the VLM key mask row (pad keys
masked, prefix bidirectional), the prefix RoPE tables (positions 0..P-1, constant per shape), the expert mask + RoPE
rows (§4.5), noise; output x_0. The embedding gather computes DRAM addresses from token contents in-kernel, so it is
trace-safe.

### 5.3 VLM geometry: 2-D bands, weights read once per column and multicast down it

Base: pad 736 -> 768 rows = 24 row tiles; **R = 8 bands** (grid rows 0..7) x **rt = 3 row tiles**; **C = 10-11 columns**.
LIBERO: 544 -> 576 = 18 row tiles, R = 9 x 2 (grid rows 0..8) or R = 6 x 3. Pad rows are masked keys and discarded queries.

- **qkv** (N = 80 tiles, head-aligned): column c < 8 = head c's Q (8 tiles), c = 8 K, c = 9 V; core (b, c) computes
  [band b rows] x [its 8 N tiles]; the band's normalised x (3 x 64 tiles) is multicast along the band by ONE band leader;
  weights of column c are read once from DRAM by one core of the column and multicast down it (8 cores).
- **attention** on core (b, c < 8): head c, query rows of band b, all keys: K / V (roped K) are written by columns 8 / 9
  into the resident KV region and **pulled** by the attention cores in chunks on NoC 1 (a 1 KV head is shared by all 8
  heads, so no per-head copy exists). Per core 3 q row tiles x 23 key tiles.
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

### 5.4 Time prediction, phase 2 (per layer, base)

VLM layer, 162.07 GFLOP (D). **Lower** = every op at the best rate measured on this card for its shape, with the removable
excess gone (weights read once, GELU fused):

| op | lower bound us | source | upper us (F0-harness rates, exposed exchange) |
|---|---:|---|---:|
| qkv | 32.9 | M TTNN 234 TFLOP/s | 58.0 (133 T, F0 llm_qkv_s512) |
| o_proj | 26.2 | M TTNN 235 T | 46.4 (133 T) |
| gate + up, unchunked | 588.0 | M TTNN 168 T (chunked; the weights are now read once) | 1,050.9 (94 T, F0 gate_up) |
| GELU epilogue, 118 tiles per core | 68.3 | M GR00T F1 encoder GELU 0.58 us / tile | 176.6 (1.5 us / tile, F0 PACK) |
| up * gelu | 11.8 | A 0.1 us / tile | 11.8 |
| down | 200.0 | M-reported isolated mcast2d probe `[736,16384]x[16384,2048]` 0.20 ms (`common/fused_config.py` docstring, 2026-09-13, not re-measured) | 422.2 (117 T, F0 down) |
| attention | 159.6 | M TTNN SDPA 139.6 + A 20 in-kernel RoPE / heads | 159.6 |
| 2 norms | 30.6 | M GR00T LN round 15.3 each | 30.6 |
| exchanges | 84.7 | A 10 % of matmul (lever 1: ring depth hides in0, F0c) | 190.0 (5 x 38 us, F0c law at L = 22) |
| **layer** | **1,202** | | **2,146** |

Point = lower x1.36 = **1,635 us** per layer (TTNN today 2,437, M). Stack (17 full + KV-only): lower 20.5 / **point 27.9** /
upper 36.6 ms (TTNN 41.54). LIBERO scaled by FLOPs (544/736): 15.2 / **20.6** / 27.1 ms (TTNN 37.85).

SigLIP layer lower bound: qkv 39.5 (M) + o 17.5 (M) + fc1 36.2 (F0 fc1 without epilogue, 140.6 T) + GELU 22 tiles x 0.58 =
12.5 + fc2 52.5 (F0 fc2 97.1 T) + attention 30.1 (M SDPA 25.1 + A 5) + 2 LN 30.6 + exchanges 14.6 (A 10 %) = **234 us**
(TTNN 394.7, M); tower incl. patch embed / post-LN (M 94 us): lower 6.40 / **point 8.70** ms; the upper bound is taken as
today's 10.77 ms (the GR00T fused towers came in slower than their TTNN, so the gate, not the estimate, decides).

| whole call | base | LIBERO |
|---|---:|---:|
| phase 2 = SigLIP + projector/embed (0.12, M) + VLM + transitions 0.05 + expert (§4.11) + host (1.40 / 1.13, M) | low 42.4 / **point 57.1** / high 72.3 ms | 32.9 / **44.3** / 56.0 ms |
| phase 1 (§4.11) | 66.7 / **71.7** / 76.1 ms | 58.8 / **62.4** / 65.6 ms |

Phase 2 beats phase 1 iff fused SigLIP + projector/embed + VLM < today's 52.42 ms (base); the high column gives 47.5 ms.

### 5.5 Phase-2 L1 plan (overlays)

A CB has one address, size and format for the whole program, so every phase's buffers must coexist or alias. Plan:
CB ids are assigned by **function** across phases (in0, w8, w16, acc, out, q, kv, s, oacc, ml, part, red, x, sync, ...),
sized at the max over phases, and phase-disjoint buffers of the same format **alias one transient L1 tensor** through
tensor-backed CB descriptors (`cb_descriptor_from_sharded_tensor` on the same buffer). The resident KV region (71,214 B per
core at base) and the owners' residual slices are never aliased.

VLM MLP phase on core (b, c), base, the tightest phase: band in0 resident 3 x 64 bf16 = 393,216 + weight landing ring
2 x (8 K x 12 N) bfp8 = 208,896 + acc 3 x 12 fp32 = 147,456 + h slice 3 x 47 bf16 = 288,768 + residual slice 3 x 6 fp32 =
73,728 + reduce landing 18 fp32 = 73,728 + resident KV 71,214 + misc 40,000 = **1,297,006 B** of 1,362,944 (65,938 B
headroom). If the WP-P2-0 descriptor build does not close under the limit, the levers in order: h in bfp8 (-135 KB, PCC
check), in0 re-multicast per N pass instead of resident (-393 KB, +in0 traffic), acc 3 x 8.

Two facts this plan relies on are **not yet demonstrated** in the GR00T tree and are WP-P2-0 gates: several CB ids backed by
one tensor, and the 32-id limit holding for the union of phase-1 and phase-2 ids (phase 1 alone uses 29).

### 5.6 Ring budget and how to split kernels while staying one program

The ring holds the five binaries of the largest kernel group (`program.cpp` `finalize_offsets`, `mk_k4c_summary.md` §2), so
code for every phase a core runs must fit at once. Estimate if the phases were written as separate specialised kernels:
phase-1 set (100-118 KB) + distributed LayerNorm / RMS stats + 2-D band matmul with column-multicast weights + tilize +
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
   needs_user) with the measured per-RISC sizes. The two remaining options both need their say-so: a larger worker-L1 cut
   for the phase-2 program only (each KiB of cut is a KiB of ring; phase 2 has no ttnn co-tenants), or two launches (which
   the goal excludes).

### 5.7 Precision plan (phase 2)

Same weights and fidelities as today: SigLIP bfp8 HiFi2 (patch embed bf16), VLM bfp8 LoFi, fp32 accumulation everywhere,
exact GELU-tanh, K/V written as bf8 (today's `kv_dtype`). New numerics to gate: the K-split reductions of o_proj / down
(fp32 partials, reordered), the 2-D flash attention merges, fp32 residual slices (more precise). Accuracy is judged per
layer on real weights and real inputs against the ttnn layer in the same run (rel-L2 as well as PCC, because SigLIP's
absmax jumps 12.3 -> 1765.9 at block 10 and PCC is discontinuous across a magnitude regime change), then end to end.

## 6. Implementation notes

### 6.1 Files (new, in this repo)

`models/experimental/pi0_5/tt/megakernel/`: `core_map.py` (roles, rectangles; the device's logical->virtual table read at
run time), `geometry.py` (role tables, sync-word tables, host injectivity / bound checks), `arena.py` (per-core
consumption-order packer with the per-step folds), `descriptors.py` (CB / semaphore specs + materialise with an explicit
compute config incl. `dst_full_sync_en`), `expert_program.py` (build, bind inputs, launch, trace), `size_check.py`;
`kernels/mk_kernel.cpp` and `kernels/ops/*.hpp` copied and adapted from GR00T (`allgather.hpp`, `exchange.hpp`,
`weight_stream.hpp`, `block_matmul.hpp`, `block_matmul2d.hpp`, `head_sdpa.hpp`, `pack_fence.hpp`, later
`dist_layernorm.hpp`, `rope_rows*.hpp`) plus the four deepseek `unified_kernels` headers they include, copied so that every
kernel `#include` resolves inside this repo (memory `feedback-source-code-cpp-includes`). Plain `ttnn.ProgramDescriptor` /
`KernelDescriptor` (no GR00T Python import). Tests under `models/experimental/pi0_5/tests/megakernel/` (`test_cpu_*` host
tests, device tests always through `bin/with-device.sh`).

### 6.2 Bring-up rules (from the memory notes; binding)

Mock-cluster compile + size check before every device run of a changed kernel; `WITH_DEVICE_RESET_AFTER=1` and a timeout
(`--timeout-method=thread`) on every first run; private `TT_METAL_CACHE`; never edit `kernels/**` while a device job may
run; screen new compute with constant non-zero inputs (zero is a fixed point of the operand-format faults); bit-identity is
checked on whole tensors; a compile-out knob is not a measurement until the default build passes the accuracy gate; delete
the profiler zone-location logs after kernel edits; profile zones sampled (e.g. steps 0 and 9 only) so 180 layer
iterations do not overflow the device profiler buffers; record hang signatures (waypoints, cores, how it presented).

## 7. Work packages and pre-registered pass gates

Gates below are fixed as of this version. "Speed" gates use unprofiled medians of 30 calls unless stated; kernel device
time uses the profiler's kernel duration of the megakernel op, median of >= 20 traced replays.

| WP | deliverable | pass gate | device |
|---|---|---|---|
| P1-0 | scaffolding (§6.1), host tests for role / sync-word injectivity and L1 budgets, empty persistent kernel mock-compiled; the shipped path run on a device opened with the 64 KiB cut | CPU tests green; size check runs; shipped path under the cut: PCC7 unchanged (mean >= 0.9995, min >= 0.999) and per call <= 84.2 ms + 1 % | ~10 min |
| P1-1 | arena + stream-only kernel over the real 10 x 18 shares | sustained >= 400 GB/s aggregate; read-back checksum exact; 10 launches identical | ~20 min |
| P1-2 | exchange microbenchmarks under a concurrent full weight stream: x / ctx hub rounds, Q column leader x 8 columns, KV leader, merge tree, row h exchange x 8 rows, column reduce x 8 | each measured cost <= 1.5x its A-value in §4.3; else re-plan that exchange (named fallback: ff2 chunk ring for R6/R7; merge variant for R3) before P1-3 | ~30 min |
| P1-3 | one expert layer in-kernel (one step), real weights | layer output vs the current ttnn layer on the same input: PCC >= 0.9995 and rel-L2 <= the ttnn layer's own rel-L2 vs fp32 torch + 20 %; constant-input screen exact; 10 launches bit-identical | ~30 min |
| P1-4 | 18 layers x 10 steps + in/out proj + Euler, standalone (inputs = the ttnn VLM's caches) | x_0 PCC vs the current path >= 0.999 per observation (8 golden obs); 10 launches bit-identical; kernel time < 31.26 ms base and < 27.85 ms LIBERO | ~30 min |
| P1-5 | integrated into `sample_actions_fused` (one generic_op in the trace) | **the phase-1 exit gates below** | several holds |
| P2-0 | phase-2 binary (primitive interpreter) mock-compiled; L1 overlay descriptor built on the host | compute group <= 128 KB estimate; aliasing CBs accepted by the program build and a tiny kernel reads/writes through two aliases correctly; union of CB ids <= 32; L1 plan <= 1,362,944 B per core | ~10 min |
| P2-1 | one VLM layer in-kernel (736 rows, real weights) | PCC >= 0.9995 vs the ttnn layer (same run); **time <= 2.20 ms (go); 2.20-2.44 ms: report and ask; > 2.44 ms (slower than TTNN): stop, needs_user** | ~30 min |
| P2-2 | one SigLIP layer in-kernel | PCC >= 0.9995 and rel-L2 as in P1-3; time <= 355 us (go); > 394.7 us: stop, needs_user | ~30 min |
| P2-3 | whole prefix in-kernel -> resident KV | per-layer K/V vs the ttnn caches PCC >= 0.999 and rel-L2 recorded; stack time < 52.42 ms base | ~30 min |
| P2-4 | whole `sample_actions` as one program | **the phase-2 exit gates below** | several holds |

**Exit gates (both phases; as set by the task):**

- Correctness: PCC7 vs the openpi golden (`libero_eval/pi05/openloop_golden.pt`, 8 obs) **mean >= 0.9995, min >= 0.999**;
  base-shape padded-prompt PCC vs the fixed torch reference **no worse than the current path on the same seeds (per seed
  within 0.005)**; **ten replays bit-identical**; the **alternating-prompt / shape-switch test** passes vs fresh-model
  references; **no hang in 20 consecutive runs** (a hang or timeout counts as a failure; one run = one process doing
  warm-up + capture + 30 calls, timeout 600 s).
- Speed: **phase 1** expert-loop device time **< 31.26 ms** (PROFILE.md expert stage, base; LIBERO reported against 27.85)
  and whole-call latency **< 84.2 ms** at base; **phase 2** whole-call latency **< phase 1's** (same shape, arms alternated
  across processes in one session, 30 calls each, difference larger than 2 x the MAD-based standard error of the median).

## 8. Risks (ranked) and open questions

1. **Exchange latency above the A-values** (R2, R3, R6, R7 are unmeasured shapes; concurrent multicasts to disjoint
   rectangles are unmeasured on this card). Mitigation: P1-2 before layer code, named fallbacks; the phase-1 gate has 2.25x
   of margin over the bottom-up chain.
2. **Phase-2 VLM matmul efficiency** (GR00T's fused layer realised 7-15 % of peak). Mitigation: P2-1 go/no-go against the
   TTNN layer; the geometry has no per-owner in0 exchange and reads each weight once.
3. **Kernel-config ring** for phase 2 (115-145 KB estimate). Mitigation: §5.6; a needs_user stop rather than a silent
   second launch.
4. **L1 co-tenancy in phase 1**: the 64 KiB cut may make ttnn SigLIP/VLM programs clash (GR00T K5 moved its adapter to DRAM
   for this). Measured in P1-0; mitigation: ttnn intermediates to DRAM where needed, KV caches stay L1.
5. **Sync-word aliasing / credit off-by-one hangs** (GR00T faults 15-19). Mitigation: host injectivity tests, next-pop
   credits, unique waypoints, reset-after on first runs.
6. **Clock under sustained compute** (F0: 1,068-1,212 MHz at 125-140 W on compute-bound shapes). Phase 2's VLM may throttle;
   aiclk is sampled in every timing run.
7. **qkv-side adaRMS fold precision** (bfp8 of `diag(1+a)W`; p300 unit PCC 0.999 vs unfolded). Gate in P1-3; fallback:
   apply `(1+a)` to the landed x on the qkv cores (32 bcast multiplies, ~1 us) with the unfolded Wqkv.
8. **KV chunk pulls on NoC 1 contending with exchanges** (7 MB per layer across the attention cores at base). Mitigation:
   one-layer-ahead prefetch; fallback: one reader per chunk multicasting to its 16-core row pair.

Open questions for the user (none blocks phase 1):

1. If the phase-2 binary cannot be brought under 136,192 B (§5.6 step 4): may the phase-2 program open the device with a
   larger worker-L1 cut?
2. The server's batching workers run B > 1 through today's fused path; this design is batch 1 (rows = 64 / 32). The
   geometry extends to B = 2 (rows = B x 64, per-request masks / RoPE / KV already per request), but B > 1 is not in the
   gates. Until told otherwise, B > 1 would keep today's path explicitly (a visible branch, never a silent fallback).

## 9. Sources

pi0.5: `docs/megakernel/PROFILE.md`, `docs/megakernel/profile/summary_{base,libero}_20260930.json`,
`docs/history/MEGAKERNEL_p300.md`, `docs/FUSED_FIX_2026-09-29.md`, `models/experimental/pi0_5/common/fused_config.py`,
`tt/ttnn_fused_attn.py`, `tt/ttnn_fused_norm.py`, `tt/ttnn_gemma.py`, `tt/ttnn_paligemma.py`, `common/fused_host.py`.
GR00T (`/home/deepgadget/experiments/gr00t/tt-metal/models/experimental/gr00t/tests/tt/results/`): `mk_k1_summary.md` §0,
`mk_k4_summary.md` §0 §3 §5 §7.2 §8, `mk_k4b_summary.md` §0 §1 §3.4-3.6 §5 §6, `mk_k4c_summary.md` §0-§2 §6,
`mk_k5_summary.md` §0 §2, `mk_f0_summary.md` §0 §4 §6 §20, `mk_f0c_summary.md` §4.3; `docs/plan/BACKBONE_FUSION_PLAN.md`
(head); code `tt/megakernel/{core_map,descriptors}.py`, `kernels/ops/*.hpp`. Memory notes under
`~/.claude/projects/-home-deepgadget-experiments-gr00t/memory/` named inline. Arithmetic:
`docs/megakernel/design/mk_design_calc.py` (output `mk_design_calc.out`; its 16-slot single ring predates the two-ring split of §4.7, whose sums are in the §4.7 text).
