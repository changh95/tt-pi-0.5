# v2 step 3 — an expert megakernel for pi0.5 (design notes, in progress)

## Why
The Gemma-300M action expert runs 18 blocks x 10 denoising steps on a 64-row suffix (one request). Measured on one
Blackhole p300 chip, in-trace, per block (B = 1):

| op (launch) | us | op (launch) | us |
|---|---:|---|---:|
| adaRMS `rms_norm` x2 | 14.7 each | `nlp_create_qkv_heads` | 24.5 |
| qkv `linear` (mcast1d) | 11 | `rotary_embedding` q / k | 12 / 11.5 |
| K cache write (fused rotary->cache) + V `fill_cache` | ~10 | `scaled_dot_product_attention` (8 heads, 786 keys) | 51 |
| `nlp_concat_heads` | 20 | o_proj + gated residual (fused dit) | 22 |
| up|gate `linear` (mcast1d) | 28.5 | `geglu` | 23 |
| down + gated residual (fused dit) | 39 | | |

Sum ~275 us / block -> 4.95 ms / step -> **49.5 ms** for the 10 steps, ~65% of the v1 request (76 ms). The work is
tiny (0.4 TFLOP and ~0.3 GB of weights per step): every one of these ops costs 10-25 us of fixed overhead
(dispatch, kernel start, per-op core barrier, DRAM round trips of 64-row activations) and only a few us of math.
Batching amortises it (B = 4 costs 1.57x B = 1) but a single robot sees the full 49.5 ms. No cheaper op-level
variant exists (2026-09-17 sweep: concat/split heads, sharded/fused rms_norm, GeGLU vs gelu+mul, batched-matmul q,
matmul-based attention all measured equal or slower).

## Target
ONE program per expert block (or per denoising step) that keeps the 64 x 1024 hidden state in L1, streams each
block's weights once (or keeps them resident: 300 MB bf8 over two chips fits their L1), and replaces the 14 launches
by phases separated by core barriers:

1. adaRMS (scale/shift precomputed per step) + qkv matmul (1D multicast of the normalised rows, N split over cores)
2. RoPE(q, k) + K/V write into the per-request cache (SFPU rotate-half on the owning cores)
3. attention: 8 heads x 64 queries x 786 keys (one head per core group; QK^T, row softmax, PV in L1)
4. o_proj + gated residual (row-parallel, cross-core reduce via multicast/semaphores)
5. adaRMS + up|gate matmul + GeGLU
6. down + gated residual

Per-phase cost floor is what the prototype below measures; the expectation is 8-12 us per phase -> ~60 us per block
-> ~11 ms per request for the expert (4.5x), i.e. a single-robot request of ~36 ms with the TP=4 prefix.

## Prototype plan (`ttnn.generic_op`, no C++ build)
1. Launch-floor measurement: trivial 1-tile programs in a trace on 1 / 8 / 110 cores (`bench/launch_floor_bench.py`).
2. Chained-matmul stage kernel: L stages of [64,1024] x [1024,1024] in one program, weights resident in L1 per core
   (N split), activation multicast from a sender core per stage, DRAM writeback + semaphore barrier per stage.
   Measures the per-stage floor with real data movement.
3. Grow phase by phase (norm, RoPE, attention, GeGLU) against the torch reference of one expert block.

## Measurements (2026-09-17, one p300 chip, in-trace)

**DRAM bandwidth** (`bench/dram_bw_bench.py`): read-only `sum` over 68 MB bf8: 308 GB/s; `clone` 68 MB (read+write):
419 GB/s of traffic. Expert matmuls with bf8 weights in DRAM, `ttnn.linear` mcast1d on the full grid, M = 64:

| matmul | weights | us | GB/s of weights |
|---|---:|---:|---:|
| qkv [1024 x 2560] | 2.7 MB | 11.8 | 236 |
| o_proj [2048 x 1024] | 2.1 MB | 16.8 | 133 |
| up\|gate [1024 x 8192] | 8.5 MB | 28.6 | 312 |
| down [4096 x 1024] | 4.2 MB | 29.9 | 149 |

So the wide matmul is already DRAM-bound; the N = 1024 ones (o_proj, down) use only 32 columns of cores and stream
at half rate. Per block the expert weights are 17.5 MB (bf8 incl. exponents): 18 blocks x 10 steps = 3.1 GB per
request, i.e. **~8 ms per request at 400 GB/s is the floor of a single-chip expert with streamed weights**
(today 49.5 ms). Keeping weights L1-resident needs > 2 chips (300 MB bf8) and per-phase cross-chip exchange
(measured all_reduce [64,1024] 17 us on the ring), which costs more than streaming: the single-chip streamed design
is the right target.

**Chained-matmul prototypes** (`tt/kernels/mk_poc*`, `bench/megakernel_poc*.py`; 18 stages of [64,1024] x [1024,1024]
in ONE `ttnn.generic_op` program on 32 cores, weights resident in L1, bf8 W / bf16 activations):

| variant | per stage | notes |
|---|---:|---|
| 18 x `ttnn.linear` (mcast1d, weights from DRAM) | 9.21 us | baseline, PCC 0.99752 vs torch |
| PoC v1: one sender core reads the [64,1024] activation from DRAM and loopback-multicasts it, all cores write outputs to DRAM, 3-semaphore barrier | 12.47 us | PCC 0.99696; the single-core DRAM read (128 KB) + writeback dominates |
| PoC v2: no DRAM between stages — every core multicasts its 2 output tiles into every core's CB (all-gather in L1), monotone semaphores | 137.6 us | correct (PCC 0.99696, identical over eager/trace runs) but 32 concurrent multicasters serialise on the NoC (path reserve); 1 core 1.2 us, 2 cores 2.4 us per stage |
| PoC v3: gather-to-one-sender over unicast (2 tiles per core), then ONE loopback multicast of the 128 KB block, the gather doubles as the barrier (1 semaphore each way) | 11.4 us | PCC 0.99696; with `matmul_block` instead of per-tile `matmul_tiles` 10.8 us; bf8 activations 8.65 us (ttnn.linear bf8 chain 7.9 us) |
| PoC v3 with compute skipped (movement + sync only) | 7.8 us bf16 / 6.5 us bf8 | the single sender's inbound gather (124 KB) and outbound multicast (128 KB) run back to back on one core's NoC links; compute is only ~3 us (2.2 us bf8) |
| PoC v5: one sender per grid row (4 senders x 32 KB gather + 32 KB multicast), chunk-ordered pushes so compute overlaps the later chunks, double-buffered input block (no global barrier) | 15.2 us | correct, but slower: 13.3 us with compute skipped (bf8: 13.4 / 12.2); 8 row senders (4x8 grid) 18.8 us. Concurrent multicasts serialise on this NoC, so ONE multicaster per program is the rule |

**Bisection of the v3 stage** (32 cores, bf16, compute skipped unless noted): sync only (semaphore multicast + CB
round trip, no data) 4.4 us; + gather 4.4 us (hidden); + multicast of the 128 KB block 7.8 us (multicast moves ~38
GB/s: 3.4 us; bf8 2.1 us); full stage 10.7 us (compute 2.9 us, not overlapped). The activation exchange of a
64-row layer is therefore >= 7 us per stage on this NoC, which is what `ttnn.linear` already pays inside its 9 us.

**Conclusion for chained matmuls**: no in-program fusion of matmul stages beats the launch-per-op baseline on
Blackhole; the wide matmuls are DRAM-bound anyway. The megakernel budget must go to the ops whose cost is not
data movement: `nlp_create_qkv_heads` + `rotary_embedding` x2 + cache writes + SDPA + `nlp_concat_heads` =
129 us per block for ~1 MB of data (a fused attention program with one launch, RoPE, masked softmax and the
suffix K/V kept local, should run in 15-25 us), and the ~5 us launch floor of every remaining op.

Per-launch floor of tiny ttnn ops in a trace: `add`/`exp` [32,32] 5.6 us, `typecast` [64,1024] 4.8 us, i.e. ~5 us of
the 9-12 us of every expert op is launch overhead that fusion removes.

Lessons: a kernel on NCRISC (NoC 1) must pass multicast rectangles end-first (as the stock mcast matmul's in1 writer
does) or the multicast never completes; the sender inside a non-loopback multicast rectangle is excluded and
`num_dests` must not count it; monotone semaphores (`noc_semaphore_wait_min`) avoid the reset races of the
set/clear protocol; a hung generic_op wedges the chip (`tt-smi -r`).

## Fused expert attention (built 2026-09-17): `tt/ttnn_fused_attn.py`, kernels `tt/kernels/fused_attn/`

One `ttnn.generic_op` program on 16 cores (8 heads x 2 query tile-rows) replaces `nlp_create_qkv_heads`,
`rotary_embedding` (q and k), the two cache fills, `scaled_dot_product_attention` and `nlp_concat_heads`:

| | us (one chip, in-trace) | PCC vs fp32 torch |
|---|---:|---:|
| ttnn path (5 launches) | 128.8 | 0.99977 |
| fused program (1 launch) | **41.0** | 0.99981 |

Per core: the reader brings this core's q tiles, the KV head's k/v suffix tiles, the RoPE tables (1/sqrt(dh) folded
into the q tables, the rotate-half sign into the sin tables), a key-mask tile and a reduce scaler, then the K and V
PREFIX rows of the cache (200 KB each, L1 -> L1). Compute: RoPE as three tile passes (x*cos, rotate_half(x)*sin,
add), S = q K^T with `matmul_tiles(transpose=1)` over 25 key tiles (23 from the cache, 2 local), mask on the last
tile, row max / exp / row sum / reciprocal, P V over the same 25 tiles, 1/rowsum. The suffix K/V rows are never
written to the cache (nothing else reads them), so the two `fill_cache` launches disappear too. Enabled with
`PI05_EXPERT_ATTN=fused` (batch 1; other batches fall back to the ttnn ops). Per request this removes
18 x 10 x 88 us = **15.8 ms** from the expert's 49.5 ms.

Model level (same torch reference as v1): 1x4 TP mesh 76.1 -> **60-61 ms** per request (PCC 0.9986 / 0.9985), one
chip 122.7 -> **103-104 ms**; per layer the program agrees with the ttnn ops at PCC 0.9998; 2+2 disaggregated
pipeline at B = 1: 18.6 -> **27 req/s** (latency 103 -> 70 ms). Variants measured: whole K/V prefix in L1
(773 KB of CBs, 41.0 us; chunked pushes 38.4 us), 2 x 8-row rings (620 KB, 39.1 us), 3 x 4-row rings + P in place of
S + per-row k tables (~440 KB, 40.3 us; adopted: the larger ones clash with the L1 buffers on the disaggregated expert
chips, Metal's `validate_circular_buffer_region`). CB lesson: a Metal circular buffer never splits a block across its end, so a ring must be pushed in ONE fixed block
size that divides its capacity (a shorter last chunk overflowed into the next CB and hung the 1x4 mesh trace while the
single-chip test still passed). Integration lesson: the unit test used 64 rows exactly and the
integration computed the core rows from the unpadded 50 -> one core row; found by prefilling the output with a marker.

Next optimisations (not done): 32 cores with a key split and
a (max, sum) combine; the same program shape for the VLM-prefill attention is not needed (prefill is compute-bound).

## adaRMS fold + fused GeGLU (2026-09-17 evening): `tt/ttnn_fused_norm.py`, `tt/kernels/row_rsqrt`, `tt/kernels/geglu_rc`

`rms_norm(x) * scale + shift @ W = r * (x @ diag(scale) W) + shift @ W` with `r = rsqrt(mean(x^2) + eps)` per row.
The post-attention norm of every expert layer is folded into per-step up|gate weights (bf8, DRAM, 18 x 10 copies =
1.5 GB) and a bias row; `RowRsqrt` (one core per tile-row, 7.5 us) computes `r`, `GegluRC` (64 cores, 6 us) applies
`r` and the bias and does `up * gelu(gate)`. Unit check vs the ttnn path: PCC 0.9993 (fp32 torch 0.9996 vs 0.9996);
MLP half of a layer 57 -> 42 us (B = 1), 90 -> 70 us (B = 4). Model: 1x4 mesh 60 -> 57.5 ms, one chip 103 -> 101.5.
The attention-side fold (`PI05_EXPERT_NORM_FOLD_ATTN=1`, the fused attention applies `r`/`c` to q/k/v) is correct but
not faster (the `r` program costs what the norm cost), so it is off.

Two operational lessons: (1) the folded expert needs TWO eager compile passes before the trace capture, otherwise the
capture finds a VLM matmul program that is not in the program cache ("Cannot load new binaries during trace capture");
root cause not found, the second pass is cheap. (2) Every small interleaved L1 tensor costs a page on EVERY L1 bank:
per-block copies of the constant tiles (18 x 8 tiles) cost ~290 KB per core and made the single-chip VLM matmuls and
the disaggregated expert chips clash with their circular buffers; the constants are now shared per device.

Batched fused attention: the same program on `batch x 16` cores. Per layer 47 us at B = 2 (ttnn 142), 122 us at B = 4
(ttnn 181; the batch-4 caches live in DRAM and the 64 cores re-read them 16x per request). Model B = 2 40 ms per
request (was 50.5), B = 4 36.5 (was 39.9). Next: one multicast of each request's K/V prefix to its 16 cores instead of
16 reads, and the K-split down projection with the GeGLU prologue (2 more launches).

## Where it runs
Independent of the disaggregation: on the v1 4-chip layout the fast expert runs replicated after the TP=4 prefix
(~25 + ~11 ms); on the 2+2 layout the expert board becomes ~4x cheaper per request and the prefix board bounds the
throughput (~29 req/s at B = 1).
