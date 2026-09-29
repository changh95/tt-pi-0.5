# π0.5 on 2× Blackhole p300 (4 chips) — tensor-parallel prefix, replicated action expert, one Metal trace

> **Historical record (superseded).** This was the package README of `changh95/pi05-base-p300x2` (2× p300, 4 chips),
> whose code was imported into this repo on 2026-09-29 (commit `ad95d97`) and then fixed (padding mask, action-token
> RoPE) and pruned: the disaggregated prefix / expert pipeline and the unfused path described below were removed.
> The numbers below predate the fix and were measured on p300 chips, not on the p150a. The mesh / tensor-parallel
> code is still in `tt/`, but was not re-validated after the fix. Current single-p150a numbers: the repo `README.md`.

> **2026-09-29 (branch `fused-fix-2026-09-29` of tt-pi-0.5):** this tree now serves the single-chip p150a port.
> The fused graph applies openpi's prompt padding mask and places the action tokens at RoPE positions
> `n_valid + [0, H)` (both were missing); the unfused path and the disaggregated 2+2 pipeline
> (`tt/ttnn_disagg.py`, `PI05_LAYOUT=pipeline`, described below) were removed. The numbers below are the p300x2
> measurements taken BEFORE that fix. p150a results: `docs/FUSED_FIX_2026-09-29.md` at the repo root.

Physical Intelligence's **π0.5** vision-language-action policy (`lerobot/pi05_base`: SigLIP-so400m + Gemma-2B
VLM + Gemma-300M flow-matching action expert) running on **four Tenstorrent Blackhole chips** (two p300 boards,
one 1×4 Ethernet ring) through TT-NN. Two 224×224 camera images + a ≤224-token prompt in, a 50-step chunk of
32-dim normalised actions out; batch 1; the whole device graph (host im2col → SigLIP → VLM prefill → 10 fused
expert denoising steps) is captured in **one Metal trace** and replayed per request.

This directory is the single-chip fused port of `changh95/pi05-base-p150` extended to a MeshDevice:

| | single p300 chip (this tree, fused trace) | **2× p300, 4 chips (this tree)** |
|---|---:|---:|
| latency per 50-step action chunk (median, traced, host→device→host) | 122.7 ms | **76.1 ms** (median of 100, min 75.6; 657 actions/s) |
| PCC vs the fp32 torch reference, 2 random observations (2×224², 224 tokens, 10 steps) | 0.8995 / 0.9987 | **0.9988 / 0.9986** |
| traced sections (separate traces of the two halves) | — | prefix + VLM **25.4 ms** · expert ×10 **49.5 ms** (4.95 ms per denoising step) |
| PCC vs torch over 16 further random observations | — | min 0.9201 · mean 0.9846 · median 0.9899 · max 0.9989 (the p150a card's own 16: min 0.8865, mean 0.9840) |

The reference p150a card measured 125.9 ms for the same graph on one chip; one p300 chip (16 GB, 120 Tensix) is
122.7 ms with this tree (`PI05_MESH=1x1`), so the four chips give 1.6× end to end and 4.8× on the sharded prefix. Bit-exact repeats; every chip holds the same output (the host reads chip 0).

## How the four chips are used

`FusedConfig.tp` (`PI05_TP`, default 0 = the mesh size) shards the **prefix** over the chips; the **expert** is
replicated. Everything lives in `tt/ttnn_ccl.py` plus small `tp` branches in `ttnn_siglip.py`, `ttnn_paligemma.py`
and `ttnn_gemma.py`:

- **SigLIP (27 blocks, 16 heads):** heads 16 → 4 per chip (`wqkv` per chip = `[wq_i | wk_i | wv_i]`, biases likewise),
  `out_proj` row-parallel, `fc1` column-parallel (4304 zero-padded to 4352 = 4 × 34 tiles), `fc2` row-parallel.
  One `ttnn.all_reduce` after `out_proj` and one after `fc2`; the row-parallel biases enter the sum once (chip 0
  holds them, the others zeros — exact, no `/tp` rounding).
- **Gemma-2B VLM prefill (18 blocks, 8 q heads, 1 MQA K/V head):** per chip 2 q heads + the full K/V head
  (`wqkv_i = [wq heads 2i,2i+1 | wk | wv]`), so every chip computes the full K/V and fills its own complete KV
  cache; `o_proj` rows, `gate`/`up` columns, `down` rows; all-reduce after attention and after the MLP. The
  row-parallel partials are emitted in **bf16** (one rounding of the full sum; the single-chip path rounds its
  bf8 output once, the bf8-partials variant was 86.2 ms / PCC 0.9978 vs 87.5 ms / 0.9958 — kept bf16).
- **Expert (18 blocks on the 64-row suffix, 10 steps):** unchanged and **replicated**. Its ops are launch-bound
  (in-trace on one chip: 2-head SDPA 52 µs vs 8-head 58 µs, `rms_norm` of [64,1024] 14 µs, a TP-sharded o_proj
  11 µs vs the fused matmul+residual 22 µs) so sharding it would add 36 all-reduces per step (17 µs each) and
  un-fuse the residuals for less than it saves.
- **CCL:** `FABRIC_1D_RING` + `ttnn.Topology.Ring` (`PI05_CCL_TOPOLOGY`). Measured in-trace on this ring:
  all-reduce [64,1024] bf16 17 µs, [512,1152] 50 µs, [736,2048] 95 µs (FABRIC_1D / Linear: 24.5 / 61 / 139 µs).
- **TP-mode defaults** (applied by `FusedConfig.resolved` unless the knob is set explicitly; the 1×4 sweep of
  2026-09-17): `PI05_MLP_CHUNK=0` + `PI05_VLM_GATEUP_PC=mcast2d` (the per-chip `[736,4096]` gate/up outputs fit L1:
  87.5 → 78.5 ms, PCC 0.9958 → 0.9988), `PI05_SDPA_EXPERT_CHUNKS=64,128` (51.4 vs 58.4 µs per SDPA) and
  `PI05_EXPERT_GEGLU=1` (one `[1024, 8192]` up|gate matmul + `ttnn.geglu` instead of gate, up, multiply: −0.8 ms).
  Rejected: `PI05_SIGLIP_PC` / `PI05_VLM_ATTN_PC=mcast2d_fp32` (−2…4 ms but PCC 0.982…0.988), `PI05_KV_DTYPE=bf16`
  (82.2 ms), a matmul + `scale_mask_softmax` + matmul expert attention (62 µs vs 51 µs SDPA).
- **`ttnn.experimental.rotary_embedding_to_cache`** (C++, in this tree): rotary embedding written straight into
  the KV cache rows — the expert's K path in one launch instead of `rotary_embedding` + `fill_cache`
  (`tt/ttnn_gemma.py::rotary_embedding_to_cache` falls back to the two ops when the op is missing).

## Run

```bash
cd tt-metal && source python_env/bin/activate
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD ARCH_NAME=blackhole
export PI05_WEIGHTS_DIR=~/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba
#   (hf download lerobot/pi05_base --revision b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba)

# accuracy: PCC vs the torch reference on the 1x4 mesh (2 observations, bit-exact repeat, identical chips)
PI05_MESH=1x4 pytest models/experimental/pi0_5/tests/pcc/test_pcc_pi05_mesh.py -v -s -o timeout=3600
# speed: 50 traced requests
PI05_MESH=1x4 python models/experimental/pi0_5/tests/perf/test_perf_pi05_mesh.py --runs 50
# single chip, same code: PI05_MESH=1x1 (or the original tests/pcc/test_pcc_pi05_fused.py, tests/perf/test_perf_pi05_fused.py)

# serve (tt-dit-server ASGI app; TT_MESH_SHAPE=1x4 opens the mesh)
TT_MESH_SHAPE=1x4 PI05_NUM_IMAGES=2 PI05_TOKEN_LEN=224 python -m uvicorn --host 0.0.0.0 --port 20000 --lifespan on \
    models.experimental.pi0_5.server.app:app
python models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000
```

Knobs (all read once at model build, `common/fused_config.py`): `PI05_TP` (0 = mesh size, 1 = replicate everything),
`PI05_CCL_TOPOLOGY` (`ring` | `linear`), `PI05_MLP_CHUNK`, `PI05_VLM_GATEUP_PC`, `PI05_SDPA_EXPERT_CHUNKS`,
`PI05_KV_DTYPE` (`bf8` | `bf16`), `PI05_EXPERT_GEGLU` (fused `[up|gate]` matmul + `ttnn.geglu`), plus the single-chip
knobs of the original port (`PI05_TRACE`, `PI05_FUSED_RESIDUAL`, …). The unfused path was removed (2026-09-29):
`TT_FUSED` unset or `1` is accepted, `TT_FUSED=0` raises a `ValueError`.

Parallel single-chip jobs on this box: pass distinct `device_id`s. `TT_METAL_VISIBLE_DEVICES=<n>` renumbers the chip
to 0 in every process, and the UMD `CHIP_IN_USE_0` lock then serialises them; a fabric mesh on a subset of the ring
while other chips are busy fails its router handshake.

## Layout

```
pi0_5/
├── common/   configs, weight_loader (safetensors → categorized torch weights), fused_config (knobs), fused_host (im2col, KV plan)
├── reference/ fp32 torch reference (PCC oracle)
├── tt/       ttnn_pi0_model (fused graph + trace), ttnn_paligemma (backbone, KV caches, TP weight sharding),
│             ttnn_gemma (VLM / expert blocks), ttnn_siglip (vision tower), ttnn_suffix / ttnn_prefix, ttnn_ccl (mesh helpers)
├── server/   FastAPI app (tt-model-manager `tt-dit-server`), smoke_test
└── tests/    pcc/test_pcc_pi05_mesh.py, perf/test_perf_pi05_mesh.py (+ the single-chip fused tests)
```

## Caveats

- Fixed served geometry: 2 cameras × 224×224 (a missing camera is padded with a black image), prompt ≤224 tokens,
  batch 1, 10 denoising steps baked into the trace.
- `pi05_base` is the base checkpoint (no per-robot normalisation stats): actions are in lerobot's normalised
  QUANTILES space and need a fine-tune for a real robot. The tokenizer `google/paligemma-3b-pt-224` is gated.
- Not re-run here: the LIBERO closed-loop benchmark of the single-chip port (needs the `pi05_libero` fine-tune and
  the simulator) and the RTX 5090 comparison (no GPU on this box).

SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. · SPDX-License-Identifier: Apache-2.0 (weights under the Gemma terms)

---

# v2 (branch `changh95/pi05-v2-disagg`): batched requests and prefix / expert disaggregation

## Step 1 — B requests per trace (`sample_actions_fused` with `tokens [B, L]`, `B*N` images, `noise [B, 50, 32]`)
Per-request KV caches `[B, 1, 786, 256]` (L1 for B ≤ 2, DRAM above), SDPA over the batch dim, the row-independent
matmuls fold the batch into rows (1D-multicast / MinimalMatmul configs keyed by the row count). Per-request PCC is
identical to the single-request run (0.9976 / 0.9912 / 0.9989 / 0.9971 on observations 100..103).

| 1x4 mesh, TP=4 prefix, replicated expert | B = 1 | B = 2 | B = 4 |
|---|---:|---:|---:|
| per batch (traced, host to host) | 76.0 ms | 101.0 ms | 159.7 ms |
| per request | 76.0 ms | 50.5 ms | 39.9 ms |
| prefix + VLM / expert x10 (traced halves) | 25.4 / 49.5 | 41.4 / 57.7 | 77.9 / 77.8 |

The expert is launch-bound (B = 4 costs 1.57x B = 1), the prefix compute-bound (3.07x).

## Step 2 — disaggregation on the two boards (`tt/ttnn_disagg.py`, `PI05DisaggPipeline`)
One 1x4 parent mesh (a standalone 1x2 fabric mesh cannot initialise: auto-discovery sees the whole ring), two
sub-meshes: **A** = chips [1, 0] runs the prefix TP=2 and sends the 36 K/V tensors of a request set through mesh
sockets (`ttnn.create_socket_pair`, `send_async` / `recv_async`; A's chip c pairs with B's chip 1-c, the ring
neighbours); **B** = chips [3, 2] runs the replicated expert on two ping/pong cache sets. Both stages are traces
(the sends end A's trace, the recvs start B's), the host overlaps A(n+1) with B(n).

| pipelined, 2 socket connections (the 2 fabric links; 7.7 MB of K/V per request move in 0.49 ms) | B = 1 | B = 2 | B = 4 |
|---|---:|---:|---:|
| serial A -> B per set | 87.3 ms | 125.3 ms | 207.4 ms |
| pipelined per request | 54.0 ms | 33.7 ms | 29.5 ms |
| requests / s | **18.5** | **29.7** | **33.9** |
| latency per request (median) | 103 ms | 128 ms | 225 ms |
| PCC vs torch (TP=2 prefix) | 0.9955 | 0.9955 / 0.9873 | 0.9955 / 0.9873 / 0.9988 / 0.9949 |

Versus the monolithic batched 1x4: 13.2 / 19.8 / 25.1 requests/s (1.4x / 1.5x / 1.35x). Outputs are identical pipelined vs serial. The
remaining per-set overhead over max(prefix, expert) is the 72 socket launches and the stage ordering, not bytes: the raw
transfer is 0.49 ms.
Two variants did not pay off: receiving the next set's K/V between the expert's layers (stalls the compute stream
until A is done: 88 ms/request at B = 1) and recvs as a trace on B's second command queue (hangs on this runtime).

## Step 3 — expert megakernel (`MEGAKERNEL.md`, `tt/ttnn_fused_attn.py`, `tt/kernels/`)

Measured first (all on one p300 chip, in-trace; details and the bisection in `MEGAKERNEL.md`):

* DRAM streams at 300-420 GB/s and the wide expert matmul (`up|gate`, 8.5 MB of bf8 weights) already runs at
  312 GB/s: the expert's 3.1 GB of weights per request bound a single-chip streamed-weight expert at ~8 ms.
* Every ttnn launch costs ~5 us (tiny ops 4.8-5.7 us in a trace).
* Chaining matmul stages inside ONE `generic_op` program (`tt/kernels/mk_poc*`: weights resident in L1, the 64-row
  activation exchanged between the 32 cores over the NoC) is correct (PCC 0.997) but never faster than 18 separate
  `ttnn.linear` launches (9.0 us/stage): the exchange alone costs 4.4 us of synchronisation plus 3.4 us for a 128 KB
  multicast, and more than one concurrent multicaster is slower still. So the matmuls stay ttnn ops.

Built: **fused expert attention** (`tt/ttnn_fused_attn.py`, kernels `tt/kernels/fused_attn/`; the resolved default,
`PI05_EXPERT_ATTN=ttnn` opts out; batch 1, other batches fall back): one program on 16 cores (8 heads x 2 query
tile-rows) does RoPE(q, k), q K^T over the cache prefix plus the local suffix rows, the key mask, the row softmax,
P V and the head concat. It replaces `nlp_create_qkv_heads` + `rotary_embedding` x2 + the two cache fills + SDPA +
`nlp_concat_heads`: **40 us vs 128.8 us**, PCC 0.9996 against the ttnn ops (0.9998 vs fp32 torch, the ttnn path
0.9998). Per request that is 18 layers x 10 steps x ~88 us = 15.8 ms less expert time. The K/V prefix streams through
small L1 rings and P overwrites S, so the program needs ~440 KB of L1 per core and coexists with the L1 buffers of the
disaggregated expert chips.

Model-level, same inputs and torch reference as the v1 numbers (B = 1):

| layout | expert attention | traced median per request | PCC vs torch (obs 1 / obs 2) |
|---|---|---:|---:|
| 1x4 TP mesh (v1 layout) | ttnn ops (5 launches) | 76.1 ms | 0.9988 / 0.9986 |
| 1x4 TP mesh (v1 layout) | **fused program** | **60-61 ms** | 0.9986 / 0.9985 |
| 1x4 TP mesh (v1 layout) | **fused program + adaRMS fold** (default) | **57.5 ms** | 0.9986 / 0.9984 |
| one p300 chip | ttnn ops | 122.7 ms | 0.8995 / 0.9987 |
| one p300 chip | **fused program** | **103-104 ms** | 0.9011 / 0.9987 |
| one p300 chip | **fused program + adaRMS fold** (default) | **101.5 ms** | 0.8927 / 0.9983 |

(Per layer the fused program agrees with the ttnn ops at PCC 0.9998; the single-chip obs-1 PCC is the pre-existing
bf16 K-accumulation issue of the 16384-wide VLM matmul, unchanged.) The 2+2 disaggregated pipeline (step 2) with the fused attention on the expert board (default configuration, B = 1,
16 requests):

| expert | B | req/s | ms per request | latency (median) | PCC vs torch |
|---|---:|---:|---:|---:|---:|
| ttnn ops (step 2) | 1 | 18.6 | 53.9 | 103 ms | 0.9955 |
| fused attention | 1 | **27** | **37** | **70 ms** | 0.9947 |
| fused attention + fold (default) | 1 | 26 | 39 | 73 ms | 0.9946 |
| ttnn ops (step 2) | 2 | 29.7 | 33.7 | 128 ms | |
| fused attention + fold (default) | 2 | **32** | **31** | **120 ms** | |
| ttnn ops (step 2) | 4 | 33.9 | 29.5 | 225 ms | |
| fused attention + fold (default) | 4 | 33.6 | 29.8 | 230 ms | |

At B = 4 the pipeline is prefix-bound (the two prefix chips take ~29 ms per request), so the faster expert shows
up as latency only at B = 1-2. The pipeline is a benchmark (`tt/ttnn_disagg.py`), the served layout is the 1x4 mesh.

**Batched fused attention**: the same program on `batch x 16` cores (one launch per layer for the whole batch):
per layer 47 us at B = 2 (ttnn ops 142 us) and 122 us at B = 4 (181 us; the batch-4 caches live in DRAM, so the
K/V reads dominate). Model: B = 2 39-41 ms per request (was 50.5), B = 4 36-37 ms per request (was 39.9), PCC per
request unchanged.

**adaRMS folding** (`tt/ttnn_fused_norm.py`, `tt/kernels/row_rsqrt`, `tt/kernels/geglu_rc`; `PI05_EXPERT_NORM_FOLD`,
on by default): `rms_norm(x)*scale + shift @ W = r * (x @ diag(scale) W) + shift @ W`, so the post-attention norm of
every layer moves into per-step up|gate weights (18 x 10 folded copies, 1.5 GB of DRAM, built in 4 s at start-up)
and a bias; what remains is one small program for the per-row `r` (7.5 us) and a fused GeGLU program that applies `r`
and the bias (6 us) instead of `rms_norm` (14.7 us) + `geglu` (23 us). Per layer 7 launches instead of 8; the MLP half
42 us instead of 57 (B = 1), 70 instead of 90 (B = 4); PCC vs torch unchanged (0.9986 / 0.9984). The same fold on
the attention side (`PI05_EXPERT_NORM_FOLD_ATTN=1`) is numerically fine but not faster (the rsqrt program costs what
the norm cost), so it stays off. The folded expert needs two eager compile passes before the trace capture
(`PI05_FUSED_COMPILE_PASSES`, default 2 with the fold); small constant tiles are shared per device because every
interleaved L1 tensor takes a page on every bank.

**Serving with batching** (`server/app.py`, `PI05_BATCH_SIZES=1,2,4`, `PI05_BATCH_WINDOW_MS=4`): the model keeps one
prepared shape (inputs, KV caches, trace) per batch size and switches between them; the server gathers concurrent
requests for up to the window and runs the smallest configured batch that fits, so one robot gets the batch-1 trace
(64.5 ms end-to-end over HTTP) and 2-4 robots share a forward (2 clients: 22.7 req/s at 88 ms each; 4 clients:
26.8 req/s at 149 ms each). The HF fast tokenizer is serialised (it is not thread-safe). Published as
`changh95/pi05-base-batch-p300x2`.

**Prefix program configs (2026-09-17, after comparing with `sdawle/dvartanians/pi0.5_bh`)**: that branch reaches
64.85 ms per chunk on one Blackhole for 1 camera + 256 tokens on a >= 120-core chip. Running my model at that shape on
this 110-core p300 chip gave 74.8 ms, and a micro-benchmark of the prefix ops (`bench/prefix_op_bench.py`) showed
where the difference was: not their sharded norms (1-2 ms here) or their 2D matmul configs (my `mcast2d` configs are
faster on every VLM/SigLIP matmul at these shapes), but the matmuls my configuration still left on ttnn's automatic
program selection. At 736 rows the automatic programs are 4-8x slower: VLM qkv 261 us vs 38, o_proj 256 vs 31, SigLIP
fc2 195 vs 42. `PI05_VLM_ATTN_PC`, `PI05_SIGLIP_PC` and `PI05_VLM_GATEUP_PC` now default to `mcast2d` on every layout:

| | before | now |
|---|---:|---:|
| 1x4 mesh, B = 1 (prefix / expert) | 57.5 ms (25.4 / 33.7) | **50.5 ms** (18.4 / 33.7), PCC 0.9986 / 0.9988 |
| one chip, my shape | 102.1 ms (69.8 / 33.7) | **84 ms** (52 / 33.7) |
| one chip, their shape (1 camera, 256 tokens) | 74.8 ms | **69.0 ms** (their 64.85 on ~10 percent more cores) |
| 1x4 mesh, B = 2 / B = 4 (per request) | 39.9 / 36.5 ms | **33.0 / 29.1 ms** |
| 2+2 pipeline, B = 1 / 2 / 4 (req/s) | 25.9 / 32.1 / 33.6 | **27.9 / 46.4 / 49.0** (latency 68 / 82 / 155 ms) |

The 2D configs are built from the total row count, so for a batched prefix the batch is folded into M (the
per-request `seq_len` alone trips `num_blocks_y <= grid.y` at B >= 2). The pipeline gains most: it was prefix-bound,
and the prefix is where these matmuls live. The fp32-destination variants of the same configs were slower and,
unexpectedly, lower in PCC on the mesh; on individual observations the per-request PCC moves by a few thousandths in
either direction; over 16 random observations on the mesh (`bench/pcc_many.py`) the distribution is unchanged or
slightly better: min / mean / median / max 0.9315 / 0.9814 / 0.9896 / 0.9986 with the 2D configs vs
0.9111 / 0.9802 / 0.9846 / 0.9987 with the automatic programs.

Not built (measured to be not worth it on Blackhole): fusing the matmul chain. Remaining per-layer cost after these
is ~155-170 us, of which ~35 us is the launch floor of the 7 remaining ops and ~100 us are DRAM-bound weight streams;
the next candidate is a K-split down-projection program with the GeGLU as its prologue and the gated residual as its
epilogue (removes 2 more launches, ~20 us per layer).


## Serving profiles (`PI05_LAYOUT`, `PI05_PROFILE`; package `changh95/pi05-base-p300x2`)

Two serve profiles of one container image (`serve_profiles` in the package's `tt-model.yaml`;
`tt-model serve changh95/pi05-base-p300x2 [--profile multi-robot]`). Only one can run at a time: both need all four chips.

| | `single-robot` (default) | `multi-robot` |
|---|---|---|
| for | one robot's control loop: lowest latency per action chunk | up to 4 robots sharing the box |
| env | `PI05_LAYOUT=mesh PI05_BATCH_SIZES=1` | `PI05_LAYOUT=dp PI05_DP_GROUP=2 PI05_BATCH_SIZES=1,2 PI05_BATCH_WINDOW_MS=4` |
| chips | 1x4 mesh: prefix tensor-parallel over 4 chips, expert replicated on 4, one Metal trace | two independent 1x2 sub-meshes ("pairs"), each running the whole model (prefix TP=2, expert replicated) with its own traces and batching worker |
| routing / batching | none (requests run one after another) | a request goes to the pair with the fewest requests in flight; requests arriving within 4 ms at a busy pair share a batch of 2 |
| in process | **50.5 ms** per request (PCC 0.9986 / 0.9988) | one pair: 57.2 ms (batch 1), 78.1 ms (batch 2); both pairs concurrently: 57.9 / 82.0 ms wall (PCC 0.9955 / 0.9879, 2-chip partials) |
| HTTP, 1 client (median) | **52.7 ms** end to end | 63.1 ms |
| HTTP, 2 / 3 / 4 / 8 closed-loop clients | 19.5 / - / 19.6 / 19.6 req/s at 102 / - / 204 / 406 ms (serialised) | **31.0 / 37.5 / 47.3 / 49.2 req/s** at 64 / 81 / 84 / 158 ms |

Stage times per batch (traced, in process; the same with the SDK's fused rotary->cache op or the Python fallback,
`PI05_NO_FUSED_ROTARY=1`, which is what the published image runs):

| batch | mesh prefix (4 chips) | mesh expert x10 (4 chips) | mesh total | pair prefix (2 chips) | pair expert x10 (2 chips) | pair total |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.4 ms | 33.7 ms | 50.5 ms | 24.6 ms | 33.7 ms | 57.2 ms |
| 2 | 26.7 ms | 39.8 ms | 64.9 ms | 38.0 ms | 42.5 ms | 78.1 ms |
| 4 | 50.9 ms | 65.6 ms | 117.3 ms | 75.6 ms | 65.4 ms | - |

Why pairs: the expert does not get faster with more chips (replicated: 33.7 ms on 2 or 4 chips), so a 4-chip request
spends two thirds of its time with three chips doing duplicate work, while two pairs serve two requests in the time
one pair serves one. Per request the pairs cost 2 x 57 = 114 chip-ms, single chips would cost 84 chip-ms (one p300
chip alone: 84 ms), so for fleets well beyond 4 robots `PI05_DP_GROUP=1` (four independent chips) has the higher
ceiling; for 1-4 robots the pairs win on latency (bench: `bench/dp_run.py`; the two workers drive their sub-meshes
from two Python threads, which ttnn allows: 57.9 ms wall for two concurrent requests).

The published package builds tt-metal from the public `main` commit 975015c2f03 and ships this directory as
`extra_code`, so the fused rotary->cache op is absent there and the fallback runs (no measurable difference).

**Pairs vs four single chips** (`PI05_DP_GROUP=2` vs `1`, host HTTP, closed-loop clients, median latency; 2026-09-19,
`bench/dp_run.py DP_GROUP=1 DP_BATCHES=1,2,4`): one chip alone runs a request in 84.0 ms (batch 2: 136.0, batch 4: 243.8;
four chips concurrently 86.9 / 141.6 / 255.7 ms wall; PCC 0.9904 / 0.9956 on the two reference observations).

| clients | pairs x2, batch 1-2 | pairs x2, batch 1-4 | singles x4, batch 1-2 | singles x4, batch 1-4 |
|---:|---:|---:|---:|---:|
| 1 | **63 ms**, 15.7 req/s | | 91 ms, 10.9 req/s | |
| 2 | **65 ms**, 30.7 | | 91 ms, 21.9 | |
| 3 | **81 ms**, 36.9 | | 92 ms, 32.3 | |
| 4 | **85 ms**, 47.1 | | 94 ms, 42.3 | |
| 6 | 119 ms, 45.4 | | **95 ms**, 47.4 | |
| 8 | 160 ms, 48.5 | 154 ms, 51.0 | **146 ms**, 51.3 | 150 ms, 52.3 |
| 12 | 243 ms, 48.6 | 227 ms, 51.5 | 224 ms, 51.9 | 252 ms, 47.3 |
| 16 | 320 ms, 48.8 | 286 ms, 54.4 | 273 ms, **56.9** | 269 ms, 52.6 |

Up to 4 robots the pairs win on both axes (the 2-chip prefix, 24.6 vs ~50 ms). From 5-6 robots on, single chips have the
lower latency and from 8 on a slightly higher ceiling (about 52-57 req/s vs 49-54), because a request costs 84 chip-ms
on one chip against 2 x 57 = 114 on a pair (the replicated expert runs twice). Batch 4 per group buys little: on one
chip batch 4 is 61 ms per request against 68 for batch 2, and it lengthens every group's wait. The served
`multi-robot` profile keeps the pairs (sized for up to 4 robots); a fleet beyond that should set `PI05_DP_GROUP=1`.

**Alternative layout, not served: the 2+2 pipeline** (`PI05_LAYOUT=pipeline`, `tt/ttnn_disagg.py`): the prefix on
one board, the expert on the other, K/V over mesh sockets, one trace per stage per batch size; the server's
`_PipelineBatcher` overlaps the prefix of group n+1 with the expert of group n and splits an idle pipeline's group in
two. Measured: 66 ms alone, 27.4 / 41.9 / 45.6 req/s at 69 / 90 / 154 ms with 2 / 4 / 8 clients; pipelined per group
37.7 / 48.9 / 86.7 ms (expert-bound at 1-2, prefix-bound at 4). The pairs beat it at every client count because the
expert board's two chips do duplicate work there too, plus the K/V hop. `PI05DisaggPipeline(batches=(1, 2, 4))`
prepares every batch size at once (persistent inputs, one K/V set on each
board, one A trace and one B trace per size; `host_inputs` / `write_prefix_inputs` / `launch_prefix` /
`launch_expert` / `read_expert` are the stage API, `run_one` / `run_pipelined` the benchmarks). The server's
`_PipelineBatcher` launches group n's expert, then waits for group n+1 for at most the expert stage's duration
(`measure_stage_times` at warm-up: serial minus pipelined time per group, a safe lower bound), stages its prefix and
reads group n back, so an idle pipeline never holds a finished result. When the pipeline is idle and several requests
wait, the group is split in two: only groups in flight together overlap, so 4 closed-loop clients settle into 2 + 2
(41.9 req/s at 90 ms) instead of one group of 4 with nothing to overlap (26.4 req/s at 151 ms, the first version).
A single B cache set per batch size suffices because B's trace receives its K/V right before computing on them (the
ping / pong sets are only needed by the recv-ahead / CQ1 variants). Validation: `bench/disagg_multi_run.py` (per
request PCC identical to the single-batch pipeline, pipelined outputs bit-identical to un-pipelined ones, mixed batch
sizes through one pipeline).
