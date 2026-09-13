# DEVICE_VALIDATION -- pi05-base-p150, branch `opt/pi05-base-p150-megakernel`

Hardware pass plan for the fused / traced graph implemented behind `TT_FUSED=1` (BRIEF §4
deliverable 4). Written 2026-09-13 from a HOST-ONLY pass: **nothing below was run on a device**;
every number marked "est." is an estimate with its arithmetic. Evaluation:
`reports/megakernel/pi05-base-p150.md` (tt-models repo). Baseline to compare against:
`reports/publish-p150/pi05-base-p150.json` -- `tt serve` of the Hub image, 2 x 224x224, 224 tokens,
10 steps, batch 1: warm-up 2 = 144.4 ms; smoke 144.2 / 148.0 ms; card example 150.4 ms inference /
152.8 total; soak x10 143.1 / 145.0 / 147.1 ms (min / median / max); PCC 0.9921 (README, 32 tokens);
LIBERO 4/5 (fine-tuned checkpoint).

## 0. What is on the branch (all opt-in, `TT_FUSED` unset = shipped behaviour bit-for-bit)

Tree: `changh95/pi05` @ `4c9fbfcceb9` (`/home/deepgadget/experiments/pi05/tt-metal`, `source.tt_metal`).

| commit | lever (evaluation §) | exactness claimed | proof on host |
|---|---|---|---|
| `Add TT_FUSED knob config ...` | `common/fused_config.py` (knobs, read once at build), `common/fused_host.py` (torch reformulations) | -- | `test_knob_*` |
| `Fused Gemma blocks ...` | 3.2 backbone-owned KV caches filled by the VLM (no per-layer-per-step prefix refill); 3.3 64-row suffix, no `slice(q)`; 3.4 fused gated residuals `dit_minimal_matmul_addcmul_fused` with bf16 stream/weights/gates; 3.5 GeGLU gate linear `activation="gelu"`; 3.8a VLM MLP fused per chunk (+ `PI05_MLP_CHUNK=0`); 3.10 cos/sin slice cache; last-VLM-layer tail skip; SDPA / dit-block knobs | 3.2, 3.3, slice cache, tail skip: exact; 3.5 rounding (same SFPU GELU, on the fp32 accumulator); 3.4 precision-affecting (improving: residual no longer rounded to bf8) | `test_kv_*`, `test_64_row_*`, `test_vlm_tail_skip_*`, `test_gate_up_split_*`, `test_gated_residual_dit_math` |
| `Fused SigLIP ...` | 3.9 host im2col persistent input (device tilize), both cameras as one batch, precomputed pos table, out_proj / fc2 biases fused, `PI05_SDPA_SIGLIP_CHUNKS` | im2col / pos table / concat order: exact; batching + bias fusion: rounding | `test_im2col_*`, `test_patch_linear_*`, `test_positional_table_*`, `test_prefix_concat_*`, `test_siglip_batched_*` |
| `Fused suffix / prefix ...` | 3.6 Euler + out-proj in one dit (`scalar=dt`, ones ternary_b, bf16 W_out); 3.10 mask-free prefix (drops the 2 in-graph host->device mask uploads), TILE language embedding | Euler fold: exact in fp64 (device: rounding, fp32 accumulate); masks / TILE embedding: exact | `test_euler_step_dit_fold_is_exact`, `test_euler_dts_*`, `test_language_embedding_*` |
| `PI0ModelTTNN: whole-graph fused path ...` | 3.1 persistent inputs, compile pass, one `begin/end_trace_capture`, `copy_host_to_device_tensor` + `execute_trace` per call, `PI05_TRACE=0` eager fallback | exact (same programs and buffers) | `test_kv_cache_plan_and_constraints` (input specs, tile constraints) |
| `Server: serve the fused traced graph ...` | device opened with `trace_region_size`, `run_inference` -> `sample_actions_fused`, capture during warm-up 1 (before READY), `/info` | contract unchanged | app import + `load_config` under `TT_FUSED=1/0` |
| `Tests: ...` (+ review fixes) | `$PKG/tests/test_fused_host.py` (22 torch-only tests; `test_bias_fusion_math` now proves the fused bias differs from linear-then-add by <= one bf16 ULP of the accumulator + one of the output, ~27% of elements differ), device harnesses `$PKG/tests/pcc/test_pcc_pi05_fused.py`, `$PKG/tests/perf/test_perf_pi05_fused.py` | -- | 22 passed |

Two corrections to the evaluation found while implementing:

1. The served model's `GemmaMLPTTNN` never had a fused `gate_up` weight: the backbone converts every
   `mlp.*` matrix to a ttnn tensor before the MLP is built, so `fused_gate_up is None` and the legacy
   MLP already runs `gate, up, gelu, multiply` (4 launches, no 8.9 MB gate_up tensor and no 4.5 MB
   slice copies). Lever 3.5 therefore saves ONE launch per MLP (the separate `gelu`), not three; the
   corrected launch counts are in §4.
2. rf-detr's validated `dit_minimal_matmul_addcmul_fused` combination on this device was bf16
   activation x bf16 weight x bf16 residual x bf16 scale (`WEIGHT_DTYPE = bfloat16`); that is the
   default here (`PI05_FUSED_RESIDUAL=bf16`, attention input typecast bf8 -> bf16, 1 launch). The
   evaluation's bf8-activation variant is the `mixed` sub-mode (saves that typecast; unexercised path).
   `bf16` changes THREE things in the expert numerics relative to the shipped path, all gated by
   step 4: (a) the residual stream is no longer rounded to bf8 before the two gated residual adds,
   (b) o_proj / down_proj weights are bf16 instead of bf8 (+0.75 GB weight reads per call), and (c)
   the expert GeGLU intermediates -- gate (`activation="gelu"`), up and gelu(gate) * up feeding
   down_proj -- are emitted in bf16 instead of bf8 (`FusedConfig.expert_mlp_act_dtype()`,
   `GemmaBlockTTNN.forward_fused_expert` -> `forward_fused_pre_down(act_dtype)`; the legacy
   `GemmaMLPTTNN.forward_pre_down` emitted bf8). (c) costs ~0.5 MB more L1 per expert layer
   (64 x 4096 x 2 B x 2 tensors) and rounds the down-proj input differently; `mixed` keeps (c) at
   bf8 (bf8 act x bf16 weight, the unexercised path), `legacy` keeps all three as shipped.
3. **(review fix)** With `PI05_FUSED_RESIDUAL` = `bf16` / `mixed` the LEGACY `sample_actions` /
   `sample_actions_traced` can NOT run on the fused-built model: the shipped `GemmaBlockTTNN.forward`
   typecasts hidden + gate to bf8 and feeds them with `self.mlp.down_proj` (now bf16) into
   `dit_minimal_matmul_addcmul_fused`, whose factory requires the residual format == the weight
   format (`minimal_matmul_program_factory.cpp:335`
   `TT_FATAL(ternary_a_data_format == in1_data_format)`) -- a hard device failure, not "slightly
   different numerics" as an earlier revision of this file said. Fix on the branch: the legacy entry
   points raise a `RuntimeError` up front (`PI0ModelTTNN._check_legacy_path_available`, decided by
   `FusedConfig.legacy_sample_actions_available` == bf8 expert weights), and both device harnesses
   run their in-process legacy comparison only under `PI05_FUSED_RESIDUAL=legacy`
   (`test_pcc_pi05_fused.py` asserts the `RuntimeError` otherwise; `test_perf_pi05_fused.py` skips
   the legacy timing with a message). The true legacy baseline for steps 3-8 is the `TT_FUSED`-unset
   run (step 0); only steps 1-2 (`PI05_FUSED_RESIDUAL=legacy`) also get the same-model legacy PCC.

## 1. Host verification already done (repeat first on the device host)

```bash
cd $ROOT/models/pi05-base-p150            # ROOT = /home/deepgadget/experiments/tt-models
TREE=/home/deepgadget/experiments/pi05/tt-metal
PKG=code/models/experimental/pi0_5         # every test / server file below lives under this package
PY="$TREE/python_env/bin/python"
export PYTHONPATH=code TT_METAL_HOME=$TREE
$PY -m pytest $PKG/tests/test_fused_host.py -q -p no:cacheprovider
# -> 22 passed in 1.90s   (2026-09-13, after the review fixes; also runs as a plain script)
for m in ttnn_gemma ttnn_siglip ttnn_paligemma ttnn_suffix ttnn_prefix ttnn_pi0_model; do
  $PY -c "import models.experimental.pi0_5.tt.$m"
done                                       # all imported with the host ttnn (no device)
$PY -m py_compile $PKG/tests/pcc/test_pcc_pi05_fused.py $PKG/tests/perf/test_perf_pi05_fused.py $PKG/server/app.py
```

## 2. Device pass -- A/B order, commands, gates

Run everything from `$ROOT/models/pi05-base-p150` with `PYTHONPATH=code` and the tree's python
(`$PY = $TREE/python_env/bin/python`; for the server add the HTTP side dir, see SERVING.md §"Run on
the HOST"). All test and server files are under `$PKG = code/models/experimental/pi0_5` (there is no
`tests/` at the repo root): `pytest` / `python` take `$PKG/tests/pcc/<file>`, `$PKG/tests/perf/<file>`,
`$PKG/server/smoke_test.py`. Weights: `PI05_WEIGHTS_DIR=<snapshot dir with model.safetensors +
config.json>` or the HF cache. Served shape everywhere: `PI05_NUM_IMAGES=2 PI05_TOKEN_LEN=224
PI05_NUM_STEPS=10`.

Gates (the model's own, unchanged): e2e PCC vs torch >= 0.93 (expect ~0.99 as today), per-step
velocity PCC >= 0.99, determinism (repeat == repeat bit-for-bit), multi-replan responsiveness >= 0.6
of torch's, LIBERO-spatial 4/5 with the fine-tuned checkpoint, served `$PKG/server/smoke_test.py`
PASS (repeat_maxdiff 0, change_maxdiff > 0). Speed: `$PY $PKG/tests/perf/test_perf_pi05_fused.py`
and the served soak (`smoke_test.py` timings / `tt serve` log) against 143 / 145 / 147 ms.

```bash
# step 0 (legacy parity, TT_FUSED unset)
$PY -m pytest $PKG/tests/pcc/test_pcc_pi05_model.py $PKG/tests/pcc/test_pcc_pi05_multireplan.py \
    $PKG/tests/pcc/test_determinism_pi05.py -v -s
$PY $PKG/tests/perf/test_perf_pi05.py
# steps 1-8 (fused; set the step's env first)
TT_FUSED=1 PI05_TRACE=0 PI05_FUSED_RESIDUAL=legacy PI05_SIGLIP_BATCHED=0 \
    $PY -m pytest $PKG/tests/pcc/test_pcc_pi05_fused.py -v -s
TT_FUSED=1 $PY $PKG/tests/perf/test_perf_pi05_fused.py --runs 10        # add --skip-legacy to save time
# step 9 (served): after the server is READY
$PY $PKG/server/smoke_test.py --url http://127.0.0.1:20000
```

| step | env | what it isolates | expected | gate |
|---|---:|---|---|---|
| 0 | `TT_FUSED` unset | this branch's legacy path == shipped | PCC / perf identical to the baseline (`pytest $PKG/tests/pcc/test_pcc_pi05_model.py $PKG/tests/pcc/test_pcc_pi05_multireplan.py $PKG/tests/pcc/test_determinism_pi05.py`; `$PY $PKG/tests/perf/test_perf_pi05.py`) | bit-identical actions, 143-150 ms |
| 1 | `TT_FUSED=1 PI05_TRACE=0 PI05_FUSED_RESIDUAL=legacy PI05_SIGLIP_BATCHED=0` | the EXACT set, eager: KV hoist, 64-row suffix, im2col + pos table, mask removal, TILE embedding, tail skip, cos/sin cache; plus the two rounding-level fusions (fused gelu, fused biases) | `pytest $PKG/tests/pcc/test_pcc_pi05_fused.py -v -s`: PCC(fused, torch) ~ legacy's; PCC(fused, legacy ttnn on the same model) >= 0.99 (differences only from the fused gelu / bias rounding) | PCC >= 0.93, ratio >= 0.6 |
| 2 | step 1 + `PI05_TRACE=1` | the trace | output bit-identical to step 1; first call = compile + capture (`trace_region_size` 160 MB fits?) | `torch.equal`; `$PY $PKG/tests/perf/test_perf_pi05_fused.py` (legacy timing in-process: still `PI05_FUSED_RESIDUAL=legacy`) |
| 3 | step 2 + `PI05_SIGLIP_BATCHED=1` (default), keep `PI05_FUSED_RESIDUAL=legacy` | cameras batched (M=512 matmul programs) | rounding-level | PCC gates |
| 4 | defaults (`PI05_FUSED_RESIDUAL=bf16`) | fused gated residuals: bf16 stream / o_proj / down_proj (+0.75 GB weight reads) AND bf16 GeGLU intermediates (gate, up, gelu*up into down_proj; §0 correction 2c) | PCC expected UP (no bf8 residual rounding; down-proj input rounded once in bf16 instead of bf8); +1.5..3 ms weight reads est. From here on the harness has NO in-process legacy comparison (legacy `sample_actions` raises on bf16 weights) -- compare against step 0 / step 3 | per-step + e2e PCC, multi-replan, LIBERO 4/5 |
| 5 | `PI05_FUSED_RESIDUAL=mixed` | bf8 activation x bf16 weight into the fused op (drops 180 typecasts) | may fail validate / differ; keep only if PCC holds | PCC gates |
| 6 | `PI05_MLP_CHUNK=0` | unchunked VLM MLP (-3.9 GB est. repeated weight reads; intermediates in DRAM) | L1/DRAM fit and program selection unknown | PCC gates + speed |
| 7 | `PI05_SDPA_VLM_CHUNKS=128,256`, `PI05_SDPA_EXPERT_CHUNKS=64,256`, `PI05_SDPA_SIGLIP_CHUNKS=64,256` (one at a time) | SDPA program configs on the full grid | precision-affecting (online-softmax rescale) | per-step PCC + LIBERO |
| 8 | `PI05_DIT_BLOCKS=4,4,4,2,2` | explicit fused-matmul blocks (rf-detr's L1-safe config) if the default blocks clash with the L1 working set | -- | no OOM, PCC unchanged |
| 9 | serve: `TT_FUSED=1 tt serve ...` (or the host uvicorn recipe) | warm-up captures before READY; `/info.hardware` says traced | `$PY $PKG/server/smoke_test.py --url ...` PASS; soak x10 | contract + timing |

Per-step PCC: `$PKG/tests/pcc/test_pcc_pi05_per_step.py` drives the legacy API (so it needs
`TT_FUSED` unset or `PI05_FUSED_RESIDUAL=legacy`); for the fused graph the same check is
`sample_actions_fused` with `PI05_NUM_STEPS=1` vs the torch reference's first velocity
(x_1 = x_0 + dt * v), or add a debug flag that returns the per-step x_t list from
`_fused_device_graph` in eager mode.

## 3. Op constraints checked in the sources, NOT verified on hardware

- `dit_minimal_matmul_addcmul_fused` (`minimal_matmul_program_factory.cpp:334-335`): `ternary_a`
  format == weight format -> bf16 hidden with bf16 o_proj / down_proj / W_out; `ternary_b` any dtype
  (bf16 -> `mul_tiles_bcast` "workaround" path, factory comment) -- the bf16 gate broadcast
  `[1, 1024]` over M = 64 rows and the `[1, 32]` ones vector with N = 32 (1 tile, default N block 8 >
  N tiles) were not run. Both addcmul inputs are placed in the same buffer type (residual + gates in
  DRAM for the expert, x_t + ones in L1 for the Euler step); this tree gives each its own
  TensorAccessor (factory lines 108/111), the rf-detr break was on the newer gbp-tt tree.
- `mixed` sub-mode: bf8 activation x bf16 weight in the same op -- validate allows (no act == weight
  check), kernel path unexercised on Blackhole by this port.
- `rotary_embedding_to_cache(k, cos, sin, cache, 736)` with a 64-row k: `736 % 32 == 0`,
  `736 + 64 <= 800`, bf8 == bf8 (`rotary_embedding.cpp:107-123`); `fill_cache(cache, k_rope[736 rows],
  0, update_idx=0)` from the VLM (`update_cache_device_operation.cpp:72-102`). Cache logical rows 786,
  padded 800: SDPA masks rows >= 786 as it did for the legacy concat buffer (`sdpa_device_operation.cpp`
  allows padding on the seq dim only) -- relied upon, not re-measured.
- `ttnn.embedding(tokens, table, layout=TILE_LAYOUT)` fused tilize (`embedding.cpp:45-54`:
  `224 % 32 == 0`, `2048 % 32 == 0`) with a ROW_MAJOR bf16 weight table of 257152 rows.
- `ttnn.tilize([2, 256, 608] bf16 ROW_MAJOR, memory_config=L1, use_multicore=True)`, `ttnn.to_device`
  / `copy_host_to_device_tensor` of a ROW_MAJOR uint32 `[1, 224]` and an L1 TILE bf16 `[1, 64, 32]`.
- Trace: `trace_region_size=160 MB` (author's loop-only trace was ~71 MB; whole graph est. ~95 MB);
  deallocations inside the captured graph; the SigLIP pos table / cos-sin slices / KV caches are
  allocated on the compile pass and live across replays (~8 MB L1 for the caches as today).
- SigLIP B = 2 through `nlp_create_qkv_heads` / SDPA / `nlp_concat_heads` (all documented to carry
  the batch dim); `ttnn.reshape([2, 256, 2048] -> [1, 512, 2048])` as a tile-aligned view.
- `ttnn.linear(..., activation="gelu")` on the Gemma gate matmuls with a `compute_kernel_config`
  (the port already does this for SigLIP fc1; `unary_op_utils.cpp:838` maps "gelu" to the
  non-approximate GELU, the same as `ttnn.gelu`'s default).
- Unchunked VLM MLP (`PI05_MLP_CHUNK=0`): `[736, 2048] x [2048, 16384]` bf8 output in DRAM -- program
  selection and L1 CB fit unknown (the author chunked for L1).
- SDPA chunk knobs: `k_chunk=256` against K padded 736 / 800 (non-multiple) relies on the kernel's tail
  handling; accuracy moved with chunk size on rf-detr.
- Legacy `sample_actions` / `sample_actions_traced` on a `TT_FUSED=1` model: runnable ONLY with
  `PI05_FUSED_RESIDUAL=legacy` (bf8 o_proj / down_proj as shipped; it then sees a bf16 `W_out` --
  `ttnn.linear` accepts that -- and no fused gate_up copy, so numerics differ slightly from the
  shipped legacy). With `bf16` / `mixed` it raises a `RuntimeError` before touching the device
  (§0 correction 3); the earlier claim that it "still works" was wrong. Use `TT_FUSED` unset for the
  true legacy A/B (step 0).

## 4. Launch counts and expected gain (est., corrected for the fused-gate_up finding)

Counted from the code for the served shape (reshape views and no-op slices not counted):

| stage | shipped | fused defaults | with `mixed` / `MLP_CHUNK=0` |
|---|---:|---:|---:|
| SigLIP (2 cameras) | 2 x 361 = 722 | tilize 1 + patch 1 + pos add 1 + 27 x 11 + post-LN 1 + projector 1 = 302 | 302 |
| prefix assembly | ~12 (+2 host writes in-graph) | 3 (embedding, mul, concat) | 3 |
| VLM (18 layers) | 18 x 34 + 1 = 613 | 17 x 29 + last layer 6 = 499 | 17 x 16 + 6 = 278 (unchunked) |
| denoise x10 | 10 x (18 x 23 + 5) = 4190 | 10 x (18 x 15 + 3) = 2730 | 10 x (18 x 14 + 3) = 2550 (`mixed`) |
| total | ~5537 | ~3534 (-36%) | ~3133 (-43%) |

Expert layer, fused (15): adaRMS, qkv, heads, rot_q, rot_k->cache, fill_v, SDPA, concat, typecast,
dit(o + gated residual), adaRMS, gate+gelu, up, mul, dit(down + gated residual). Per step: embed,
final adaRMS, Euler dit.

Time (est., per-op anchors from the evaluation §1.3: ~5-10 us device cost and ~3-8 us extra host
dispatch per tiny op untraced): trace -12..-32 ms; ~2000 fewer launches -10..-20 ms; removed copies
(72 MB prefix-KV refills, mask uploads) -0.5..-1 ms; bf16 o_proj / down_proj / W_out +0.75 GB weight
reads +1.5..+3 ms. **Expected 145 ms -> ~95-120 ms (-17..-35%) with the defaults; ~85-105 ms if
`PI05_MLP_CHUNK=0` (-3.9 GB est. re-streamed weights) and the SDPA configs pass their gates.** Not
the rf-detr 3.3x: ~45-50 ms of VLM compute / weight streaming stays. The §1C unified kernel for the
expert layer (scoped in the evaluation §4) is not part of this pass.

## 5. Not verified at all (host pass limitations)

No device run of any kind: no PCC, no timing, no L1 / trace-region fit, no confirmation that the
fused ops accept the exact shapes / dtypes / buffer types listed in §3, no check that trace replay is
deterministic, no LIBERO. The server's fused branch was import-checked and its config plumbing
exercised (`load_config` under `TT_FUSED=1/0`), not booted. The two device harnesses
(`$PKG/tests/pcc/test_pcc_pi05_fused.py`, `$PKG/tests/perf/test_perf_pi05_fused.py`) were only
compile-checked; the `RuntimeError` guard on the legacy entry points is exercised on the host only
through `FusedConfig.legacy_sample_actions_available` (`test_knob_legacy_sample_actions_availability`),
its raise inside `PI0ModelTTNN` needs a built model.

## Results (device, 2026-09-13)

Hardware pass on the p150a (tt-metal `changh95/pi05` @ `4c9fbfcceb9`; host python_env of the tree for
the A/B steps, the shipped image `tt-model/pi05-base-p150:0dd4e9efec2d` + pytest = `pi05-base-dev:latest`
for the served A/B and the final gate). Evidence: `logs/megakernel-validate/pi05-base/` in the tt-models
repo (every log, probe and the exact commands: `env.sh`, `s0_legacy.sh`, `run_fused.sh`, `run_dev.sh`,
`serve_dev.sh`, `serve_ab.sh`, `gate_dev.sh`, `probe_*.py`), one row per experiment in
`reports/megakernel/VALIDATION.md`. Served shape everywhere: 2 x 224x224, 224 tokens, 10 steps, batch 1.
Run-to-run noise of the traced number on this host: +-0.5 ms (129.2 .. 130.2 ms across six runs of the
same configuration).

### 0. Legacy regression (`TT_FUSED` unset) -- pass, unchanged

The model's own tests through a stand-in `device` / `device_params` fixture plugin
(`pi05_device_plugin.py`: the tests live outside the tree, so pytest does not load the tree conftest, and
importing it as a plugin would shadow this package's `models`): `test_pcc_pi05_model` PCC **0.992823**
(README 0.9921); `test_pcc_pi05_multireplan` PCC(A) 0.9870 / PCC(B) 0.9888, response ratio 1.142;
`test_determinism_pi05` bit_equal=True -- 3 passed. `tests/perf/test_perf_pi05.py` (32-token port
config): 132.9 / 133.1 / 135.1 ms (avg 133.7; a second process 136.5 / 136.4 / 135.5) vs README 132.7.
Legacy `sample_actions` at the served shape (224 tokens), in-process leg of `test_perf_pi05_fused.py`,
4 processes: medians 143.5 / 144.0 / 143.2 / 143.2 ms (min 142.1, max 144.6) = the published served
143.1 / 145.0 / 147.1.

### 1. What the hardware rejected, and the fixes

* **`dit_minimal_matmul_addcmul_fused` with the op-default blocks clashes with L1.** First seen in the perf
  harness (legacy run first, then the fused compile pass): `TT_THROW program.cpp:1476 Statically allocated
  circular buffers in program 293 clash with L1 buffers on core range [0-0 - 10-9]. L1 buffer allocated at
  1285440 and static circular buffer region ends at 1323520`. The default 8x8x8 blocks on the 110-core grid
  reserve ~1.26 MB of CBs per core; in the fused-only process the lowest persistent L1 buffer sat at
  ~1299 KB of the 1403.5 KB bank (`ttnn.get_memory_view`: 104.5 KB allocated) -- a 6.5 KB margin. Isolated
  probe of the three fused shapes x 12 configs (`probe_dit_blocks.py`, output bit-identical for all):
  Euler [64,1024]x[1024,32] 27.3 -> **10.0 us** (M1 K8 N1 sub 1x1, 11x2 grid); o_proj [64,2048]x[2048,1024]
  40.2 -> **21.6 us**, down_proj [64,4096]x[4096,1024] 58.5 -> **39.4 us** (M1 K8 N4 sub 1x4, 11x2). New
  defaults `PI05_DIT_BLOCKS=1,8,4,1,4,0,2`, `PI05_EULER_DIT_BLOCKS=1,8,1,1,1,0,2` (7-tuple knob with an
  optional core grid; `op` = the op default). Effect in the graph: 137.2 -> 130.2 ms, PCC unchanged; the
  clash sequence re-run (`s8b`: legacy first, 183 KB/bank allocated) captures fine -> legacy 143.2 vs fused
  132.8 ms in one process. Commit `7c210fe`.
* A failed trace capture is now closed before the exception propagates (`_fused_prepare`), and the perf
  harness prints the per-bank DRAM / L1 / trace allocator view after the first fused call.
* Everything else in §3 of the plan ran first time: `rotary_embedding_to_cache` at update_idx 736 into the
  800-row cache, `fill_cache` at 0 and 736, TILE `embedding`, `tilize` of the ROW_MAJOR im2col input,
  `copy_host_to_device_tensor` into the persistent inputs, bf16 `ternary_b` broadcast, batched SigLIP
  through `nlp_create_qkv_heads` / SDPA / `nlp_concat_heads`, `linear(activation="gelu")`, whole-graph
  trace capture (27-32 MB of the 160 MB region; DRAM 4.27 GB allocated of 30.6 GB).

### 2. Fused A/B (host python; `test_pcc_pi05_fused.py` = PCC vs torch on two random observations A / B,
`test_perf_pi05_fused.py --runs 10` = fused traced min / med / max)

| step | env | PCC A / B | fused traced ms | verdict |
|---|---|---|---|---|
| 1 | `PI05_TRACE=0 PI05_FUSED_RESIDUAL=legacy PI05_SIGLIP_BATCHED=0` (exact set, eager) | 0.9964 / 0.9973; vs legacy ttnn 0.9990; response ratio 1.197 | -- | pass |
| 2 | + trace | 0.9964 / 0.9973, eager == traced, replay == replay | -- | pass |
| 3 | + batched SigLIP | 0.9964 / 0.9973 | -- | keep |
| 4 | defaults (bf16 fused residuals) | **0.9984 / 0.9987**; per-step velocity PCC worst 0.999022; ratio 1.231 | 136.8 / 137.2 / 138.6 (op-default dit blocks) | keep |
| 5 | `PI05_FUSED_RESIDUAL=mixed` | 0.9983 / 0.9989; robustness 8 min 0.9059 / mean 0.9824 | 135.6 / 136.0 / 136.6 | pass, knob (see §4) |
| 6 | `PI05_MLP_CHUNK=0` / 384 / 512 (auto programs) | 0.9988 / 0.9989; 0.9988 / 0.9989; 0.9904 / 0.9987 | 329.4; 317.5; 241.6 | drop |
| 7a/b/c | SDPA chunks VLM 128,256 / expert 64,256 / SigLIP 64,256 | A 0.9931 / 0.9985 / **0.9403**, B 0.9986 / 0.9987 / 0.9989 | 129.3 / 129.9 / 130.0 (all incl. tuned dit blocks = 130.2 alone: noise) | knobs only |
| 8 | tuned dit blocks (new defaults) | 0.9984 / 0.9987 (bit-identical to 4) | **129.7 / 130.2 / 130.5** | keep |
| 10b | + `PI05_EXPERT_MM=mcast1d` (1D-multicast program for the expert qkv / up) | 0.9988 / 0.9988; robust16 mean 0.9863, >= legacy 15/16, seed 106 -0.0106 | **125.7 / 125.8 / 126.5** | see fp32 variant (s14b) |
| 11c | unchunked VLM MLP with explicit 2D configs | 0.9982 / 0.9988; robustness 8 below legacy on 105 (-0.020) / 106 (-0.014) | 113.6 / 113.9 / 114.8 | **drop (accuracy)** |
| 12 | + `PI05_VLM_ATTN_PC=mcast2d` (2D-multicast programs for the VLM qkv / o_proj) | 0.9950 / 0.9990; per-step worst 0.999027; robustness 8: seed 106 **-0.032** vs legacy | 116.1 / 116.2 / 123.9 | drop as is (bf16 partial sums) |
| 13a | + `PI05_SIGLIP_PC=mcast2d` (SigLIP qkv / wo / fc1 / fc2) | 0.9917 / 0.9989; per-step worst 0.999041; robustness 8: seed 106 **-0.036** | **104.2 / 104.4 / 105.3** | drop as is (bf16 partial sums) |
| 14 | the same three with fp32 destination accumulation (`mcast1d_fp32`, `mcast2d_fp32`) | A **0.9233** / B 0.9982 (test FAILED); robust16 mean 0.9852, min 0.9002; per-step 0.998844 | 104.3 / 104.4 / 104.8 | drop (2D programs) |
| 14b | `PI05_EXPERT_MM=mcast1d_fp32` alone | **0.9977 / 0.9987**; per-step 0.999012; robust16 mean 0.9840 (legacy 0.9786), >= legacy 15/16, worst -0.0081 | **125.7 / 125.9 / 126.6** | **keep -> default** |

### 3. Where the time goes (segment profile by difference of partial traces, `probe_segments.py`)

Tuned dit blocks, chunked MLP (130 ms): served call 130.2 = host inputs 0.68 (im2col + 3 from_torch) +
copies 0.05 + **execute_trace 128.9** + readback 0.03. Trace = **SigLIP 22.6** (27 blocks x 0.85 ms:
attention 0.30, MLP 0.51) + language embed / concat 0.03 + **VLM 49.9** (18 x 2.77: qkv + heads 0.42,
RoPE + cache writes 0.06, SDPA 0.10, concat + o_proj + add 0.42, rms + chunked MLP 2.06) + **10 expert
steps 56.3** (5.63 ms per step, 18 layers x 15 launches: adaRMS 0.31, qkv 0.84, RoPE + cache 0.52, SDPA
1.08, concat + typecast 0.51, o_proj dit 0.38, adaRMS 0.29, GeGLU 1.02, down dit 0.70; ~21 us per launch =
the in-trace launch floor). The plan's "-17..-35 %" assumed a launch-bound loop; the loop is 44 % of the
trace and already at the floor, SigLIP + VLM (56 %) are program-selection / bandwidth bound. Isolated
probes showed `ttnn.linear`'s auto program far from optimal on this device: VLM qkv 0.387 -> 0.035 ms and
o_proj 0.405 -> 0.029 ms with explicit 2D-multicast configs, SigLIP fc2 0.247 -> 0.102, expert qkv 23.5 ->
10.0 us with a 1D-multicast config; and inside the graph the auto program of the unchunked
[736,2048]x[2048,16384] MLP matmuls selects a kernel 25x slower than in isolation (10.7 vs 0.4 ms), which is
why `PI05_MLP_CHUNK=0` collapsed (§2 step 6) -- explicit configs fix the speed (11c) but change the K
accumulation enough to fail the accuracy rule.

### 4. Accuracy: the e2e PCC gate, the robustness set and the acceptance rule

* Per-step velocity PCC (torch x_t fed to the fused expert at every step, `probe_perstep_fused.py`):
  worst 0.999022 (defaults), 0.999004-0.999027 for every knob tried (7a/b/c, 12, 13a), 0.998978 for
  the unchunked MLP -- far above the 0.99 gate and insensitive to the levers. The Euler fold
  `dit(x_t + dt*(hW+b)*1)` differs from the host `x_t + dt*v` by 1-2 bf16 ULP of x (max abs 7e-3..1.25e-2).
* The e2e PCC of the 10-step chunk is the sensitive metric. A robustness set of 8 random observations
  (uniform / gaussian / smooth low-frequency / black cameras x 2, random 224-token prompts, random
  noise; seeds 100-107) gave, for the fused defaults, 0.9929 / 0.9979 / 0.9978 / 0.9986 / 0.9928 /
  **0.9003** / 0.9761 / 0.9951 -- one observation below the plan's 0.93 gate. The SHIPPED legacy path
  on the same observations (`probe_legacy_robust.py`): 0.9884 / 0.9943 / 0.9966 / 0.9966 / 0.9811 /
  **0.8946** / 0.9797 / 0.9870. The low observation (seed 105: gaussian cameras; reference chunk std
  0.048) is inherent to the port's bf8 numerics, and the fused path is more accurate than legacy on 7 of
  the 8 observations (mean 0.9814 vs 0.9773). Both paths are bit-deterministic (repeat == repeat 8/8).
* Acceptance rule used for every precision-affecting lever (documented in `VALIDATION.md`): on the same
  observation set, mean PCC >= the legacy path's AND no observation more than 0.01 below legacy.
  Rejected by it despite passing the plan's gates: SigLIP SDPA 64/256 (one observation 0.9403), the
  unchunked VLM MLP (seed 105 -0.020, 106 -0.014 vs legacy; -16 ms), the explicit-program linears with
  bf16 partial sums (s12: seed 106 -0.032, s13a: -0.036; -10 / -26 ms). `mixed` residual passes
  (worst -0.0024, mean +0.0051) but was not made default (-1.2 ms measured only against the op-default
  dit blocks). `mcast1d` with bf16 partial sums misses the rule by 0.0006 on one of 16 observations
  (seed 106, -0.0106) and stays a knob; its fp32-accumulate variant `mcast1d_fp32` passes (worst -0.0081,
  mean +0.0054 vs legacy on 16 observations) and is the new default. On 16 observations the fused
  defaults score 0.9853 mean vs legacy 0.9786 (>= legacy on 15 / 16; legacy itself has 2 observations
  below 0.93: seeds 105 and 115).
* LIBERO (4/5 tasks) could not be re-measured: the simulator stack (`libero`, `robosuite`, MuJoCo/EGL) is
  not installed on this host (the fine-tuned checkpoint is cached). The card keeps the LIBERO row as a
  legacy-path result.

### 5. Served A/B (shipped image `tt-model/pi05-base-p150:0dd4e9efec2d` + pytest = `pi05-base-dev:latest`,
working-tree code bind-mounted over `/opt/tt-metal/models/experimental/pi0_5`, the `tt-model serve --print`
flags, `serve_ab.sh ab 100`)

| | legacy (`TT_FUSED` unset at cb0ad1b) | fused (`TT_FUSED=1`, defaults) |
|---|---|---|
| boot | model built 33.2 s, warm-up 721.0 / 143.3 ms, READY 42 s | model built 32.2 s, warm-up 1 **8435 ms** (fused-kernel JIT into the image cache + trace capture), warm-up 2 126.0 ms, READY 50 s |
| `/info.hardware` | untraced sample_actions | fused whole-graph sample_actions_fused, one Metal trace per request |
| 100 warm requests, `timing_ms.inference` median / min / max | **171.7 / 142.1 / 188.1 ms** (climbs 142 -> 188 over the run: first-20 median 143.4 = the published 145.0, last-20 183.9), total 173.6, client wall 175.5 | **125.9 / 125.6 / 126.5 ms** (flat), total **127.1**, client wall 128.6 |
| output | repeat identical | repeat identical; PCC vs the legacy served actions (card example) 0.999288, max abs 0.066 |
| `smoke_test.py` | PASS (repeat_maxdiff 0, change_maxdiff 0.0391) | PASS (repeat_maxdiff 0, change_maxdiff 0.0352) |
| malformed image / num_steps=5 / 4 images | 400 / 400 / 400 | 400 / 400 / 400 |
| 1 image + seed, tokens-only | 200 / 200 | 200 / 200 |
| `docker stop -t 120` | "Closing device", exit in 2 s, `docker ps` empty, `tt-smi -s` OK | same |

The legacy drift (monotonic 142 -> 188 ms over 100 back-to-back requests, also seen in the smoke test run
right after: 188 / 185 ms) is host-side: the traced path in the same session is flat to 1 ms, so it is not
the device. It is a property of the untraced dispatch path under sustained load, not measured by the
10-request soak of the publish stage.

### 6. Final gate in the shipped image (`gate_dev.sh final`, NO env: the flipped default)

`gate_dev.sh final` (RUNS=10, PROBE=1, E2E_SEEDS=8; `TT_FUSED` NOT set -> the flipped default; python 3.12 /
torch 2.11 of the image): `test_pcc_pi05_fused.py` PASSED -- PCC A **0.9977** / B **0.9987**, response
ratio 1.310, replay == replay, legacy entry point refuses as designed; `test_perf_pi05_fused.py` fused
traced **125.8 / 126.1 / 126.6 ms** (first call 508 ms with the kernels cached by the served run), L1 per
bank 104.5 KB allocated / 1299 KB free, trace **26.6 MB** of the 160 MB region; 8-observation robustness
0.9919 / 0.9978 / 0.9979 / 0.9985 / 0.9922 / 0.8865 / 0.9798 / 0.9940 (mean 0.9798 vs legacy 0.9773 on
the same observations, worst -0.0081 at seed 105, seed 106 +0.0001), repeat == repeat 8/8. Identical to
the host-python numbers: the device does the arithmetic. GATE_EXIT=0.

### 7. Kept / dropped / unverified

Kept (the defaults of `FusedConfig`): whole-graph trace with persistent inputs; backbone-owned expert KV
cache filled by the VLM; 64-row suffix; bf16 fused gated residuals (`PI05_FUSED_RESIDUAL=bf16`) and
fused Euler step with the measured `MinimalMatmulConfig` blocks (`PI05_DIT_BLOCKS=1,8,4,1,4,0,2`,
`PI05_EULER_DIT_BLOCKS=1,8,1,1,1,0,2`); GeGLU with the fused GELU; batched SigLIP from a host im2col with
the pos table and fused biases; TILE language embedding; mask removal; cos/sin cache; VLM tail skip;
expert qkv / up through the 1D-multicast program with fp32 accumulation (`PI05_EXPERT_MM=mcast1d_fp32`).
`TT_FUSED=0` restores the previously shipped path bit-for-bit.

Dropped (knobs kept, defaults unchanged): unchunked VLM MLP (`PI05_MLP_CHUNK=0`, -16 ms with explicit
programs but two observations 0.014-0.020 below legacy); explicit 2D-multicast programs for the VLM
attention and SigLIP linears (`PI05_VLM_ATTN_PC`, `PI05_SIGLIP_PC`: -10 / -26 ms, one observation 0.03
below legacy; the fp32-accumulate variant fails the harness gate); SDPA chunk configs (noise-level speed,
single observations moved); `mixed` residual (passes the rule, -1.2 ms measured only against the
op-default dit blocks -- not re-measured, left as a knob); bf16-partial-sum `mcast1d` (0.0006 over the
rule threshold on one observation).

Unverified / open: LIBERO closed loop (simulator not installed here; the card row stays a legacy-path
result); the fused path's kernels are not in the image's kernel cache, so the first boot of a fresh
`~/.cache/tt-model/pi05-base-p150/cache` JIT-compiles them (8.4 s warm-up 1 here with the legacy kernels
already cached; a fully cold cache was not timed); the shipped image `0dd4e9efec2d` still carries the
pre-pass code -- `tt-model package` must be rebuilt before pushing; the numerically-sound unchunked VLM
MLP and the 2D-multicast attention / SigLIP programs are the next levers (together ~-40 ms measured, i.e.
~85 ms per chunk) once a program with the auto kernel's K accumulation (or an fp32 intermediate that
this device executes correctly) is available.
