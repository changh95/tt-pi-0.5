# pi0.5 megakernel

The whole pi0.5 `sample_actions` runs as three persistent `ttnn.generic_op` programs (four with 3 or 4 cameras: one
VISION program per group of <= 2 cameras) on the 11 x 10 worker grid of a single Blackhole chip (110 cores).
`tt/ttnn_pi05_model.py` (`PI05MegakernelTTNN`) is the host. It uploads the weights once, writes each request's inputs
into fixed-address device buffers, and replays a Metal trace that holds the ops of the request's preset.

## The programs

A call runs, in one trace:

| program | builder | kernels | ops |
|---|---|---|---|
| VISION | `pe_program.PrefixEngineProgram(..., VISION_OPS, group=)` | `kernels_p2/whole_*.cpp` | prefix-engine ops 0..191: patch embedding, SigLIP x 27, post-LN, projector, on a group of <= 2 cameras (`pe_geometry.vision_groups`: 3 cameras = 2 + 1); the group's im2col and x_v rows are an address offset (tile pages interleave over the 8 DRAM banks) |
| PREFIX | `pe_program.PrefixEngineProgram(..., PREFIX_OPS)` | `kernels_p2/whole_*.cpp` | ops 192..313: language embedding, VLM prefill (writes the K / V caches) |
| EXPERT | `program.ExpertMegakernel` | `kernels/mk_*.cpp` | the N-step x 18-layer expert loop, action in / out, Euler |

Each program has three kernels, each on all 110 cores:

| kernel | RISC | NoC | role |
|---|---|---|---|
| `kernels_p2/whole_ncrisc.cpp` | RISCV_1 | NOC_0 | weight / activation reads, op descriptors for the TRISCs |
| `kernels_p2/whole_brisc.cpp` | RISCV_0 | NOC_1 | multicasts, exchanges, barriers, cache writes |
| `kernels_p2/whole_trisc.cpp` | TRISC 0-2 | | matmuls, norms, softmax, epilogues (fp32 DST accumulation) |

(the EXPERT program's `kernels/mk_{ncrisc,brisc,trisc}.cpp` take the same roles). Each `whole_*.cpp` runs the prefix
engine (`kernels_p2/pe_*.hpp`) over the op range of its runtime args. The expert loop is compiled out
(`PE_NO_EXPERT`) and the norm / attention bodies are flattened (`PE_ATT_FLAT = 2`, `pe_program.CODEGEN_DEFINES`): with
the dead expert code gone, GCC no longer specialised the LLK calls inside them (attention was 10-50 % slower); with
`flatten` they specialise their own calls and every op is as fast as or faster than before. VISION hands over to
PREFIX through the VLM residual in DRAM, PREFIX to EXPERT through the 18 bfp8 K / V caches in L1. Each program
declares only its own L1 (CB) region.

Every per-request value is tensor data at a fixed address: camera patches (im2col), token ids, the VLM key mask, the
expert key row, the four RoPE row tables and the noise. The runtime args hold only buffer addresses and shape
constants, so the programs are trace-capturable and one replay is three ops.

A preset (`presets.py`: cameras, prompt bucket, suffix bucket) fixes the prefix length, the action rows and the
attention chunking, i.e. the geometry of all three programs. A model fixes its cameras, horizon (suffix bucket: 32 rows
for H <= 32, 64 for 33..64) and step count at construction; a request runs in the smallest prompt bucket that holds
its real tokens (`prompt_bucket=` overrides it). Each preset's programs compile on its first call (`warmup()` does it
ahead). The prefix rows are tiled into rv <= 8 bands of rpb = ceil(ptv / 8) row tiles (pad row tiles are masked keys
and discarded queries); the expert's prefix keys are padded to its chunking (masked). The camera images are a
compile-time arg of the VISION and PREFIX programs (SigLIP rows 8 x images). The 2-camera presets:

| prompt bucket | prefix P | VLM row tiles (bands x rows) | S 32: key chunk x chunks (pad tiles) | S 64: key chunk x chunks (pad tiles) |
|---|---|---|---|---|
| 32 | 544 | 18 (6 x 3) | 3 x 6 (0) | 4 x 5 (1) |
| 64 | 576 | 18 (6 x 3) | 4 x 5 (1) | 4 x 5 (0) |
| 128 | 640 | 21 (7 x 3) | 4 x 6 (3) | 5 x 5 (3) |
| 224 | 736 | 24 (8 x 3) | 4 x 6 (0) | 5 x 5 (0) |

The 1-camera presets:

| prompt bucket | prefix P | VLM row tiles (bands x rows) | S 32: key chunk x chunks (pad tiles) | S 64: key chunk x chunks (pad tiles) |
|---|---|---|---|---|
| 32 | 288 | 10 (5 x 2) | 2 x 5 (0) | 3 x 4 (1) |
| 64 | 320 | 10 (5 x 2) | 2 x 6 (1) | 3 x 4 (0) |
| 128 | 384 | 12 (6 x 2) | 3 x 5 (2) | 3 x 5 (1) |
| 224 | 480 | 16 (8 x 2) | 3 x 6 (2) | 4 x 5 (3) |

**Math fidelity.** At HiFi2 a matmul keeps only 1 hidden + 6 mantissa bits of its in0 operand (truncated toward
zero), a biased error that dominated the device's deviation from fp32 on sensitive inputs. Every build therefore runs
the attention (q.K, P.V) at HiFi3 in both prefix-engine programs, the SigLIP matmuls at HiFi3 in the VISION program
(`pe_program.FIDELITY_DEFINES`) and every expert matmul at HiFi3 (`MK_FID = 3`); the VLM matmuls keep HiFi2. Cost
+1.84 ms (+3.4 %) per call at the 2-camera / 224-token / 64-row preset.

## Prefix engine (SigLIP x 2, projector, embedding, VLM prefill)

The prefix engine is a fixed sequence of 314 ops (`kernels_p2/pe_defs.hpp`; host mirror `pe_geometry.py`):

1. the patch embedding;
2. 27 SigLIP layers of 7 ops each: LN1, qkv, attention, o, LN2, fc1, fc2;
3. the post-LN, the projector and the language embedding;
4. 17 VLM layers of 7 ops each: RMS1, qkv, attention, o, RMS2, gate|up, down;
5. RMS1 + qkv of the last VLM layer, which is all the expert needs from it.

* Between ops, activations are staged in DRAM scratch, and one global barrier (hub core (10, 9)) separates
  consecutive ops.
* Matmuls run on 8 row bands x 11 columns. A feeder core (x, 8) reads each weight page once from a bank-striped
  arena and multicasts it down column x. The in0 band is read in K pieces and multicast along the band's row.
  The VLM down projection (K = 512 tiles) streams in0 in K blocks with fp32 partials.
* The norms' affine parts are folded into the consuming matmul on the host (`pe_host.fold_norms`), so the kernels
  apply only `(x - mu) * rstd` / `x * r`.
* Residual streams are fp32. VLM qkv weights are bf16; the other weights are bfp8. Matmuls run at HiFi2 with fp32
  accumulation.
* The VLM attention masks the pad keys of the prompt (the VLM key-mask row). Prefix positions are `0..P-1`, which
  equal openpi's `cumsum(valid) - 1` for a right-padded prompt.
* Work items (norm (row, column group), attention (row, head pair)) beyond one per core run in a second round. Norm
  items then use the first 108 cores (a multiple of both column-group counts, so a row's items stay on one core group),
  with the second round's statistics in their own half of the exchange ring: the statistics are unchanged. With
  4 row tiles per band (3 cameras) the RoPE pairs' accumulators fill DST, so they are parked in fp32 and the RoPE
  epilogue runs per 3-row sub-block. With 5 row tiles per band (4 cameras) a pair's accumulators exceed DST: every
  VLM matmul runs per 3-row sub-block over the pair's weight pages, which stay resident for all its sub-blocks (the
  same K accumulation order per output tile).
* The VLM attention merges its key chunks of 6 tiles. At 7 chunks (4 cameras, 224 tokens) the chunk weights fill DST,
  so the merge spills them to fp32 between its steps: the same operations in the same order, bit-identical to the
  in-DST merge where both apply (checked by forcing the spill at 4 chunks).

## Expert loop (N steps x 18 layers, time conditioning, Euler)

Core roles (logical coordinates; `geometry.py`, checked by `geometry.check_roles`):

* pair cores (x, y), x in 0..9, y in 0..3: qkv for two head-dim tiles of Q head x (x < 8), of K (x = 8) or V (x = 9),
  with the adaRMS epilogue and RoPE done locally;
* attention units: (head, key chunk, q row tile). Each computes one chunk of keys over [prefix | suffix] with the
  additive key row. The kc = 0 units merge the chunks' (max, sum, partial output);
* 32 owners (n // 4, 4 + n % 4): residual column n (fp32), o_proj column n, the column reduce of the down projection;
* 8 x 8 MLP cores: up|gate + GeGLU for two h columns, the down partial for 16 K x 4 N tiles;
* hubs H0 (10, 9) (x / x_mid rounds, adaRMS r, the per-step action in / out projections and the Euler step) and
  H1 (10, 8) (ctx round).

How the loop computes each step:

* **Time conditioning.** The adaRMS modulations depend only on the fixed timestep schedule. The host computes them
  once, then folds them into per-step weights: `rms(x) * (1 + scale) + shift @ W = r * (x @ W') + c`. So the
  qkv / up|gate inputs are the raw residual, and `r` is a per-row scalar.
* **Weights.** `arena.py` packs the per-step weights. They stream from DRAM with one bank per core
  (`geometry.plan_banks`).
* **Attention.** The action tokens attend the valid prefix keys and the H real action rows. They sit at positions
  `n_valid + [0, H)` (`common/pi05_host.attention_inputs`). The 8 head units of one (key chunk, query row) share
  the chunk (one KV head): at 64 action rows (`program.kv_mcast`) the head-0 unit reads it once and multicasts it
  along its row (`MK_KV_MCAST`, up to -1.7 ms per call); at 32 rows each unit reads its own copy, which measured
  faster there. A model keeps its K / V caches in L1 unless some program would keep < 16 KB of L1 free
  (`presets.kv_in_dram`: the 4-camera models); DRAM caches always use the multicast.
* **Row loop.** With 4 cameras, 64 action rows and >= 64 prompt tokens, one query row per unit would need more than
  the grid's 10 rows of units. A unit is then (head, key chunk) and runs both query rows; its merger takes one row's
  parts at a time and tells the column's units when the next row's slots are free.
* **Euler step.** The step `x_t += dt * v` runs in fp32 on H0. N (1..10, `num_denoising_steps`) is fixed per model:
  the host folds the adaRMS weights for `t_i = 1 - i / N`, uploads N per-step arenas and passes `ngen = 18 N` and dt
  as runtime args, so every N runs the same binaries.

`host_model.loop_decomposed` computes the kernel's decomposition in torch, and the CPU tests check it against the plain
formulas (`host_model.loop_reference`).

## Kernel-config ring

Each program must fit the kernel-config ring on its own. That ring is 136,192 B only when the device is opened with the 64 KiB
worker-L1 cut (`worker_l1_size=1_395_712`, `PI05_DEVICE_PARAMS`). The ring holds:

* the five RISC binaries;
* the runtime args;
* the CB and semaphore configuration.

`size_check.py` builds the three programs on a mock cluster (no card) and reports each footprint against a 128 KiB
gate:

```bash
TT_METAL_CACHE=$(mktemp -d) python -m models.experimental.pi0.tt.megakernel.size_check
```

| preset | vision | prefix | expert |
|---|---:|---:|---:|
| c2_l32_s32 | 72,236 B | 72,236 B | 69,188 B |
| c2_l32_s64 | 72,236 B | 72,236 B | 71,588 B |
| c2_l64_s32 | 71,996 B | 71,996 B | 69,060 B |
| c2_l64_s64 | 71,996 B | 71,996 B | 71,588 B |
| c2_l128_s32 | 71,996 B | 71,996 B | 69,172 B |
| c2_l128_s64 | 71,996 B | 71,996 B | 71,636 B |
| c2_l224_s32 | 72,156 B | 72,156 B | 69,172 B |
| c2_l224_s64 | 72,156 B | 72,156 B | 71,636 B |

The 1-camera presets:

| preset | vision | prefix | expert |
|---|---:|---:|---:|
| c1_l32_s32 | 71,676 B | 71,676 B | 69,012 B |
| c1_l32_s64 | 71,676 B | 71,676 B | 71,524 B |
| c1_l64_s32 | 71,676 B | 71,676 B | 69,124 B |
| c1_l64_s64 | 71,676 B | 71,676 B | 71,524 B |
| c1_l128_s32 | 71,900 B | 71,900 B | 69,076 B |
| c1_l128_s64 | 71,900 B | 71,900 B | 71,620 B |
| c1_l224_s32 | 71,996 B | 71,996 B | 69,188 B |
| c1_l224_s64 | 71,996 B | 71,996 B | 71,588 B |

The 3-camera presets (two VISION programs: images 0-1, image 2):

| preset | vision | prefix | expert |
|---|---:|---:|---:|
| c3_l32_s32 | 74,012 B + 73,980 B | 74,028 B | 69,204 B |
| c3_l32_s64 | 74,012 B + 73,980 B | 74,028 B | 71,652 B |
| c3_l64_s32 | 74,028 B + 73,980 B | 74,028 B | 69,204 B |
| c3_l64_s64 | 74,028 B + 73,980 B | 74,028 B | 71,652 B |
| c3_l128_s32 | 74,012 B + 73,980 B | 74,028 B | 69,204 B |
| c3_l128_s64 | 74,012 B + 73,980 B | 74,028 B | 71,652 B |
| c3_l224_s32 | 73,436 B + 73,404 B | 73,452 B | 69,220 B |
| c3_l224_s64 | 73,436 B + 73,404 B | 73,452 B | 71,668 B |

The 4-camera presets (two VISION programs: images 0-1, 2-3; K / V caches in DRAM):

| preset | vision | prefix | expert |
|---|---:|---:|---:|
| c4_l32_s32 | 81,516 B + 81,516 B | 81,516 B | 69,908 B |
| c4_l32_s64 | 81,516 B + 81,516 B | 81,516 B | 71,620 B |
| c4_l64_s32 | 81,516 B + 81,516 B | 81,532 B | 69,908 B |
| c4_l64_s64 | 81,516 B + 81,516 B | 81,532 B | 71,892 B |
| c4_l128_s32 | 81,532 B + 81,532 B | 81,532 B | 69,940 B |
| c4_l128_s64 | 81,532 B + 81,532 B | 81,532 B | 71,940 B |
| c4_l224_s32 | 82,396 B + 82,396 B | 82,396 B | 69,940 B |
| c4_l224_s64 | 82,396 B + 82,396 B | 82,396 B | 71,940 B |

## Files

| file | contents |
|---|---|
| `kernels/mk_{brisc,ncrisc,trisc}.cpp`, `mk_defs.hpp`, `mk_dm.hpp` | expert loop kernels; constants shared with the host |
| `kernels_p2/whole_*.cpp`, `pe_*.hpp` | the VISION / PREFIX programs' entry points and the prefix engine |
| `geometry.py` | expert shapes, core roles, CB table, DRAM bank plan, host invariants (parses `mk_defs.hpp`) |
| `pe_geometry.py` | prefix-engine op list, splits and L1 layouts (parses `pe_defs.hpp`) |
| `host_model.py` | expert parameters from the checkpoint (time MLP, adaRMS), reference and decomposed loops |
| `arena.py` | per-step expert weight arenas |
| `pe_host.py` | prefix-engine parameters, weight arenas, tables, host model of the decomposition |
| `presets.py` | the preset table: (cameras, prompt bucket, suffix bucket) -> program geometries |
| `program.py` | `ExpertMegakernel`: expert arenas, runtime args, the EXPERT program |
| `pe_program.py` | `PrefixTensors`, `PrefixEngineProgram`: the VISION and PREFIX programs |
| `size_check.py` | offline kernel-config ring footprint per program |
