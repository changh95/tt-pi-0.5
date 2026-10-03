# pi05-base-p150 — Blackhole p150a vs RTX 5090 (same host, same weights, same input)

- GPU pass: 2026-09-14.
- The p150a side has updates of 2026-10-01 and 2026-10-03 (the next sections).
- Original note: this file gives facts only.
  - The 2026-09-14 pass measured every GPU number below.
  - Every p150a number is a copy from the validation / publish reports and logs, with its source line.
  - The pass did NOT touch the p150a.

## Update 2026-10-03: the multi-config megakernel (current image)

- The p150a now serves the multi-config megakernel (`PI05MegakernelTTNN`).
  - It runs vision | prefix | expert as three persistent `ttnn.generic_op` programs for each call.
  - With 3-4 cameras, it runs four programs for each call.
- tt-metal: `main` @ `f856a38a361`.
- Image: `tt-model/pi05-base-p150:4dd06e9d6fd3`.
- The test used 100 warm requests at the served shape of the sections below:
  - 2 × 224² (two cameras), the card's 142-token prompt.
  - H = 50: the action chunk has 50 actions.
  - 10 flow-matching steps (N).
- Results of the 100 warm requests:
  - `timing_ms.inference`: median **56.40 ms** (p90 56.60).
  - `timing_ms.total`: 57.63 ms.
  - Client wall: 59.16 ms.
  - The 2026-10-01 image gave 55.84 / 57.13 ms.
- This test did not measure the GPU rows below again.

Closed-loop comparison of the same day:

- Benchmark: LIBERO-spatial with `lerobot/pi05_libero`.
- Each row has 100 paired episodes; openpi's client sends the requests.
- The GPU runs openpi's `PI0Pytorch` on the RTX 5090 of this host.
- The p150a latency is the mean of the server-side policy call.
  - This time includes the host input preparation.
- This file does not report the GPU latency of these runs.
  - That measurement used a different method (openpi model time only, on a shared, loaded host).
  - Thus the GPU latency is not comparable.

| cameras | N | p150a policy call (server side, incl. host inputs) | p150a success | RTX 5090 success |
|---:|---:|---:|---:|---:|
| 2 | 10 | 52.7 ms | 99 / 100 | 100 / 100 |
| 2 | 5 | 45.1 ms | 100 / 100 | 99 / 100 |
| 2 | 1 | 39.1 ms | 100 / 100 | 99 / 100 |
| 1 | 10 | 40.2 ms | 1 / 100 | 2 / 100 |
| 1 | 5 | 32.8 ms | 1 / 100 | 2 / 100 |
| 1 | 1 | 27.0 ms | 0 / 100 | 0 / 100 |

- With 1 camera, both backends fail, because the request does not include the wrist view.
- Thus the 1-camera rows are a record, not an accuracy signal.

## Update 2026-10-01: new p150a measurements (the 2026-10-01 image, the whole-model megakernel)

- On 2026-10-01, the p150a served the whole-model megakernel on tt-metal `main` @ `668c2907575`.
- For each request, ONE persistent `ttnn.generic_op` on 110 cores runs all of these parts:
  - SigLIP on the two cameras.
  - The projector and the language embedding.
  - The VLM prefill, which writes the K / V caches.
  - The full action-expert loop of 10 steps × 18 layers.
  - The action in/out projections and the Euler updates.
- This op is the only device op of the Metal trace replay.

Measurement of 2026-10-01:

- This test measured these numbers again on 2026-10-01 with this repo's image `tt-model/pi05-base-p150:6fb244df57ff` on the p150a.
- The test sent 100 warm requests of the card's request after 5 warm-ups.
- The shape was the served shape: 2 × 224² + 224 tokens, H = 50, 10 steps.
- `timing_ms.inference`: median **55.84 ms** (p10 55.62, p90 56.13).
- `timing_ms.total`: **57.13 ms** (p90 57.44).
- preprocess 1.27 ms; client wall 58.52 ms.
- The in-process trace replay alone (`execute_trace`, host wall, median of 30) is 54.1 ms.
  - Source: port repo `docs/megakernel/integrate_p2/results/A_whole_base_r1.json`.

Previous images:

- The previous image (`fe0d2e3d68a7`, 2026-09-30):
  - It ran the SigLIP / VLM prefix as traced stock TT-NN ops and the expert loop as one generic_op.
  - It measured 70.81 ms inference and 72.16 ms total, in the same benchmark cycle as the current figures.
  - Its replay took 69.6 ms (`A_expert_base_r1.json`, the same session as the current replay).
- The 2026-09-29 image (`672900e23919`):
  - It used stock TT-NN ops plus 3 custom programs.
  - It measured 84.05 ms inference, 85.30 ms total and an 82.8 ms replay.

GPU numbers:

- **This test did not measure the GPU numbers again.**
- They are the 2026-09-14 runs below, of the torch reference at that time, before the mask / RoPE fix.
- The fix changes the attention mask and the RoPE positions but not the tensor shapes.
- This test did not measure the GPU cost of the fixed reference.
- Ratio = p150a ms / GPU ms (> 1 means the GPU is faster).

| row | p150a ms | GPU setting | GPU ms (2026-09-14) | ratio p150a/GPU |
|---|---:|---|---:|---:|
| device forward (p150a `timing_ms.inference` vs GPU incl_h2d) | 55.84 | fp32 strict | 144.066 | **0.39** |
|  |  | tf32 | 108.693 | **0.51** |
|  |  | bf16 autocast (fp32 weights) | 121.476 | **0.46** |
|  |  | fp16 autocast (fp32 weights) | 123.674 | **0.45** |
|  |  | bf16 weights resident (eager) | 99.869 | **0.56** |
|  |  | bf16 autocast + `torch.compile` default | 88.419 | **0.63** |
|  |  | bf16 autocast + `torch.compile` reduce-overhead | 98.388 | **0.57** |
|  |  | bf16 weights resident + `torch.compile` reduce-overhead | 59.97 | **0.93** |
|  |  | bf16 weights resident + whole-request `torch.compile` default | 46.62 | **1.20** |
|  |  | bf16 weights resident + whole-request `torch.compile` reduce-overhead | 48.609 | **1.15** |
| forward only (p150a trace replay vs GPU excl_h2d) | 54.1 | fp32 strict | 143.77 | 0.38 |
|  |  | tf32 | 108.54 | 0.50 |
|  |  | bf16 autocast | 121.19 | 0.45 |
|  |  | fp16 autocast | 126.91 | 0.43 |
|  |  | bf16 weights resident (eager) | 99.78 | 0.54 |
|  |  | bf16 weights resident + whole-request `torch.compile` default | 46.39 | 1.17 |
| served e2e (p150a `timing_ms.total` vs GPU served-like) | 57.13 | fp32 strict | 156.514 | **0.37** |
|  |  | tf32 | 110.221 | **0.52** |
|  |  | bf16 autocast (fp32 weights) | 122.868 | **0.46** |
|  |  | bf16 weights resident (eager) | 101.484 | **0.56** |

Interpretation:

- The p150a (55.84 ms) is faster than every eager GPU row.
- The fastest eager row is bf16 weights resident (99.869 ms, ratio 0.56).
- The p150a is also faster than the stage-wise compiled rows (best 59.97 ms, ratio 0.93).
- With bf16 weights and a compiled whole-request graph, the GPU gets to 46.6 ms, 1.20x faster than the p150a.
- This test did not measure the p150a power, so this file makes no efficiency comparison.

Historical part:

- The rest of this file is the 2026-09-14 pass as recorded.
- That pass compared the GPU with the previous p150a image (125.84 ms, tt-metal fork `changh95/pi05` @ `4c9fbfcceb9`, before the fix).
- Its references to `DEVICE_VALIDATION.md` now mean `docs/history/DEVICE_VALIDATION_2026-09-13.md` in the port repo [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5).

## What the pass ran

| | |
|---|---|
| Model | pi-0.5 (`lerobot/pi05_base`), 3.617 B parameters. Vision tower: SigLIP-So400m/14 (27 layers, 1152 wide, 256 tokens per 224x224 camera). VLM: PaliGemma / Gemma-2B (18 layers, 2048 wide, MQA 8/1 heads, 16384 MLP). Action expert: the Gemma-300M flow-matching action expert (18 layers, 1024 wide, adaRMS). The GPU runs the port's own torch reference `models/pi05-base-p150/code/models/experimental/pi0_5/reference/torch_pi0_model.py::PI0Model` (`pi05=True`). This is the config that the lifespan of `server/app.py` builds for the TT model. `tests/pcc/test_pcc_pi05_fused.py` used this network as the gate for the p150a. It is plain Python over raw weight tensors (no `nn.Module`). The script moves it to `cuda`: it moves every tensor of the categorized state dict |
| Weights | `lerobot/pi05_base` @ `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`). Files: `model.safetensors` (14.47 GB, 812 fp32 tensors) + `config.json`. Source: the HF cache `~/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`, with `HF_HUB_OFFLINE=1`. Tokenizer: `google/paligemma-3b-pt-224` (gated, cached, logged-in token) |
| Input | Images: `media/sample_base.png` + `media/sample_wrist.png` (224x224 RGB). They are byte-identical to `smoke_test.synthetic_image('base'/'wrist')`. Thus they are exactly the payload of the Hub warm run `logs/publish-megakernel/pi05-base/warm_client.py`. `server/app.py::decode_image` makes RGB, does a bilinear squash-resize to 224x224, /255 and (x-0.5)/0.5. The result is two `[1,3,224,224]` fp32 tensors. Prompt: `"pick up the cube"` + zero state -> `build_prompt` -> `Task: pick up the cube, State: 128 128 1… 128;\nAction: `. The PaliGemma tokenizer right-pads it to 224 tokens (143 real). Noise: fixed seeded noise (`torch.manual_seed(42); randn(1,50,32)`), the same noise that `PI0ModelTTNN.__init__` draws in the server. 10 Euler flow-matching steps; batch 1. H2D payload: 1.21 MB |
| Output | `[1,50,32]` normalized actions (host `actions[0].tolist()` in the served-like loop) |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600.00 W, 32607 MiB; idle 28.6 W |
| venv | A dedicated GPU venv (`.venv-gpu/pi05`). Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, transformers 4.53.0 (= the tt-model.yaml pin). Other packages: numpy 2.5.3, safetensors 0.8.0, huggingface_hub 0.36.2, sentencepiece 0.2.2, protobuf 7.36.1, triton 3.6.0. This pass added pillow 12.3.0, fastapi 0.141.1 and pydantic 2.13.5. With them, `server/app.py` (the served preprocess code) imports; no ttnn |
| Scripts | `logs/gpu-vs-p150/pi05-base/bench_pi05_gpu.py` (eager legs, uses `logs/gpu-vs-p150/bench_common.py`). `bench_pi05_compile.py` (torch.compile leg). `make_report.py` (makes this file from the JSONs). Logs: `full_run.log`, `compile_run.log`; smoke runs `smoke.log` / `smoke2.log`. Raw JSON: `result.json`, `compile_result.json` (merged into `reports/gpu-vs-p150/pi05-base.json`). CPU reference: `cpu_fp32_reference.pt` |
| Commands | `HF_HUB_OFFLINE=1 .venv-gpu/pi05/bin/python bench_pi05_gpu.py --skip-cpu --iters 50 --warmup 10 --served-iters 50 --stage-iters 20`, then `bench_pi05_compile.py --iters 50 --warmup 10`. The first `smoke.log` run of the same script without `--skip-cpu` made the CPU reference |
| Loop | For each precision: 10 warm-ups + 50 timed iterations, with `torch.cuda.synchronize()` before and after each. Wall-clock time (perf_counter) is the primary number; the script also records the CUDA-event time. `nvidia-smi -lms 200` samples the power during the timed loop |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `reports/megakernel/PUBLISH_SUMMARY.md:18` (Hub `tt serve`, 40 warm requests: inference **125.84** ms, total **127.0**). `logs/publish-megakernel/pi05-base/warm-hub.json` adds preprocess 1.17 ms and client wall 128.44. `models/pi05-base-p150/DEVICE_VALIDATION.md:262`: harness 125.7 / 125.9 / 126.6. `:266-271`: segment profile of the 130.2 ms configuration. Segments: host inputs 0.68 + copies 0.05 + execute_trace 128.9 + readback 0.03. Trace = SigLIP 22.6 + VLM 49.9 + 10 expert steps 56.3. `logs/megakernel-validate/pi05-base/serve/verify_fused_probe.log:7`: 100 warm, 125.9 / 125.6 / 126.3, total 127.1 |

The GPU run has two differences from the literal reference. Both are necessary for CUDA, and both are in `bench_pi05_gpu.py`:

- `precompute_freqs_cis` is cached and returns the RoPE tables on the model's device.
  - The reference calculates the tables again on the CPU at every expert call, and CPU tables cannot multiply CUDA tensors.
  - The p150a port also keeps a cos/sin cache.
- The noise sampler returns the fixed seeded tensor (as `test_pcc_pi05_fused.py` does).
- All other code is the reference's own code path.
  - This includes its per-call `weight.to(dtype)` casts and Python-level loops.

Time definitions (the same as the p150a `timing_ms` keys):

- **incl_h2d** = `.to('cuda')` of the two images, tokens, mask, state and noise + `sample_actions` + `actions.cpu()`.
  - The inputs are pageable host tensors, the same as what `predict()` produces.
  - `sample_actions` = embed_prefix (SigLIP x2 + projector + language embed), VLM prefill of 736 tokens with KV cache, and 10 expert steps.
  - Compare with p150a `timing_ms.inference` = 3 host->device copies + `execute_trace` of the whole graph + readback (**125.84 ms**, PUBLISH_SUMMARY.md:18).
- **excl_h2d** = the same forward with resident inputs; the actions stay on the device.
  - Compare with the p150a `execute_trace` alone: 128.9 ms in the 130.2 ms configuration (DEVICE_VALIDATION.md:267).
  - With the same split, the shipped 125.8 ms configuration has ~124.5 ms of trace.
- **served-like** = base64 decode + PNG decode + resize/normalize (`decode_image` x2) + `build_prompt` + `tokenize_prompt` + incl_h2d forward + `actions.tolist()`.
  - Compare with p150a `timing_ms.total` (**127.0 ms** = preprocess 1.17 + inference 125.84; PUBLISH_SUMMARY.md:18, warm-hub.json).
  - The p150a `preprocess` also contains its base64/PNG decode, because `predict()` starts its clock before `decode_image`.
- **stages** = embed_prefix / VLM prefill / 10 denoise steps, each measured separately with a sync after it (inputs resident).
  - Compare with the p150a segment profile: SigLIP 22.6 / VLM 49.9 / expert 56.3 ms (DEVICE_VALIDATION.md:267-270).
  - That profile is of the 130.2 ms configuration.
  - The shipped default is 4.4 ms faster in the expert, thus ~51.9 ms.

## Correctness check (GPU vs CPU fp32 reference)

- CPU fp32 reference: `PI0Model` on the host, 16 threads.
  - One `sample_actions` takes 13.9 s.
- The GPU side uses fp32 strict (no TF32) on the same inputs.
- The metric is the PCC over the `[1,50,32]` action chunk.
  - The p150a e2e gate uses the same quantity.

| metric | value |
|---|---:|
| PCC actions (1600 values) | **1.000000** |
| max abs diff | 5.66e-07 |
| actions[0][:6] GPU / CPU | [-0.02314, -0.02479, -0.04036, 0.06249, -0.05698, 0.03722] / [-0.02314, -0.02479, -0.04036, 0.06249, -0.05698, 0.03722] |
| first GPU call (fp32 strict, incl. CUDA init / cuBLAS handles) | 372.7 ms |

- PCC > 0.999 holds; the GPU runs the correct model.
- Cross-check against the p150a itself:
  - The same GPU fp32 model runs on the `serve_probe.py` payload.
  - The payload has the same two images and the prompt `"pick up the cube"`.
  - It has the state `[0.1, -0.2, 0.3, 0, 0, 0, 0.5, -0.5]` and the model's default seed-42 noise.
  - The comparison uses the actions that the p150a served in the verifier session (`logs/megakernel-validate/pi05-base/serve/verify_fused_actions.json`, fused bf16/bf8 trace).
  - Result: **PCC 0.999530**, max abs diff 2.643e-02.
- For scale: the port's harness reports torch-vs-fused PCC 0.9977 / 0.9987 on two random observations (DEVICE_VALIDATION.md:262).

Accuracy of each GPU precision compared with the CPU fp32 reference (same input):

| GPU precision | PCC actions | max abs diff |
|---|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **1.000000** | 5.66e-07 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **1.000000** | 6.18e-04 |
| bf16 autocast (fp32 weights, +TF32 remainder) | **0.999931** | 1.53e-02 |
| fp16 autocast (fp32 weights, +TF32 remainder) | **0.999999** | 1.21e-03 |
| bf16 weights resident (eager, no autocast) | **0.999873** | 2.05e-02 |
| bf16 autocast + `torch.compile` default | **0.999958** | 1.17e-02 |
| bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) | **0.999958** | 1.17e-02 |
| bf16 weights resident + `torch.compile` reduce-overhead (CUDA graphs) | **0.999896** | 1.92e-02 |
| bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps) | **0.999982** | 7.26e-03 |
| bf16 weights resident + whole-request `torch.compile` reduce-overhead (one CUDA graph per request) | **0.999982** | 7.26e-03 |
| p150a (bf16 activations, bf8/bf16 weights, fused trace; DEVICE_VALIDATION.md:262, VS:38) | 0.9977 / 0.9987 (harness A/B, random observations); 16-observation mean 0.9853, min 0.8865 | — |

- fp16 autocast is numerically correct on this input: the Gemma-2B activations have no overflow, and the PCC is above 0.999.
  - Thus the pass timed it.
- Every bf16 GPU row is well above the p150a's own e2e PCC.
  - The p150a e2e PCC is 0.9977 / 0.9987 on the harness inputs.
  - It is 0.99953 on the served probe payload, compared with this fp32 reference.

## GPU latency (batch 1, 2 x 224x224 + 224 tokens + 10 steps; median / min / p90 of 50 iterations, wall-clock ms)

- Eager PyTorch, with fp32 weights on the device (as the brief specifies).
- The autocast rows cast the fp32 weights again at every call.
  - Under `inference_mode`, the autocast weight cache is off.

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power mean W (excl / incl loop) | GPU util % | peak mem alloc / reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **144.07 / 143.32 / 147.63** | **143.77 / 143.26 / 146.86** | 143.76 | 142.91 | 465.9 / 463.2 | 92.1 | 14052 / 14410 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **108.69 / 108.06 / 110.28** | **108.54 / 108.03 / 110.95** | 108.52 | 109.35 | 333.9 / 332.9 | 74.8 | 14052 / 14440 |
| bf16 autocast (fp32 weights, +TF32 remainder) | **121.48 / 121.11 / 123.70** | **121.19 / 120.61 / 122.53** | 121.17 | 205.59 | 291.5 / 291.2 | 74.3 | 13986 / 14504 |
| fp16 autocast (fp32 weights, +TF32 remainder) | **123.67 / 121.49 / 130.50** | **126.91 / 122.73 / 150.60** | 126.88 | 229.08 | 270.0 / 280.8 | 68.1 | 13985 / 14504 |

Weights resident in bf16:

- The script casts every tensor once with `.to(bfloat16)`: 6.76 GiB on the device.
- The script casts the images to bf16 on the device, inside the timed region.
- SigLIP / VLM / expert run in bf16 with the reference's fp32 softmax.
- The script calculates the time embedding in fp32 and then casts it.
- The script casts the velocity to fp32, so the Euler update `x_t + dt*v` stays fp32.
  - The p150a keeps x_t in bf16.
- This configuration is the closest match to the p150a, which holds bf16 / bf8 weights on the device.

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power W (excl / incl) | GPU util % | peak mem MiB | PCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 weights resident (eager, no autocast) | **99.87 / 99.18 / 101.32** | **99.78 / 99.31 / 101.08** | 99.77 | 100.83 | 280.3 / 279.9 | 64.0 | 7047 | 0.999873 |

Stage split:

- Inputs resident, a sync after each stage, median of 20.
- The p150a row is the segment profile from DEVICE_VALIDATION.md:267-270 (130.2 ms configuration).

| precision | embed_prefix (SigLIP x2 + projector + lang embed) | VLM prefill (18 layers, 736 tokens) | 10 expert steps | per step | sum |
|---|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | 23.3 | 57.8 | 62.7 | 6.27 | 143.8 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 11.4 | 34.8 | 65.0 | 6.50 | 111.2 |
| bf16 autocast (fp32 weights, +TF32 remainder) | 12.3 | 28.9 | 80.4 | 8.04 | 121.6 |
| fp16 autocast (fp32 weights, +TF32 remainder) | 15.5 | 27.0 | 107.4 | 10.74 | 149.8 |
| bf16 weights resident (eager, no autocast) | 8.0 | 20.7 | 70.8 | 7.08 | 99.5 |
| bf16 autocast + `torch.compile` default | 12.4 | 25.5 | 35.2 | 3.52 | 73.1 |
| bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) | 12.4 | 35.0 | 50.9 | 5.09 | 98.3 |
| bf16 weights resident + `torch.compile` reduce-overhead (CUDA graphs) | 8.1 | 23.9 | 28.1 | 2.81 | 60.1 |
| bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps) | (one graph) | (one graph) | (one graph) | — | 46.4 (excl_h2d) |
| bf16 weights resident + whole-request `torch.compile` reduce-overhead (one CUDA graph per request) | (one graph) | (one graph) | (one graph) | — | 48.5 (excl_h2d) |
| p150a fused trace (bf16; SigLIP incl. host im2col + concat 0.03) | 22.6 | 49.9 | 56.3 (shipped default ~51.9 after `mcast1d_fp32`, -4.4 ms) | 5.63 (~5.2) | 128.9 (execute_trace) |

Launch overhead:

- The eager reference is launch-bound in the expert loop.
  - 18 layers x ~30 small kernels x 10 steps ≈ 5–6 k launches per chunk at 50 tokens.
  - Thus the 10 steps cost 65–85 ms at all precisions.
  - bf16 autocast is *slower* than fp32 there, because every step casts the 300 M expert weights again.
  - The prefill and SigLIP have real work in each launch, so their time changes with precision as expected.
- The p150a whole-graph Metal trace removes exactly this range.
- On the GPU, `torch.compile` with CUDA graphs solves it.

`torch.compile` configuration:

- Backend: inductor, `dynamic=False`.
- The stage-wise variants compile the three stage functions `embed_image` / `forward_vlm` / `_denoise_forward` separately.
  - The Euler loop and the 10 calls stay in Python.
- The reduce-overhead (CUDA graphs) variants need two host-side additions:
  - The `embed_image` output is `.clone()`d outside the graph. The graph replays once for each camera, and the second replay overwrites the output buffer of the first camera.
  - Without the clone, the first attempt failed with 'accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run' (`compile_run.log`).
  - The script calls `torch.compiler.cudagraph_mark_step_begin()` once for each request.
- The `whole-request` variants compile `forward_inference` as ONE function.
  - dynamo unrolls the 10-step Euler loop.
  - SigLIP x2 + VLM prefill + 10 expert steps are in a single graph.
  - Under reduce-overhead, each request is one CUDA graph replay.
  - This is the closest equivalent of the p150a's single Metal trace.
  - A stage split is not possible for these variants.
- The pass checked the accuracy again after compile.

| variant | compile s (first call / ready after 3 more calls) | incl_h2d median / min / p90 | excl_h2d median / min / p90 | power W (excl loop) | GPU util % | peak mem MiB | PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 autocast + `torch.compile` default | 31.7 / 31.9 | **88.42 / 87.87 / 89.85** | **87.94 / 87.51 / 89.58** | 336.5 | 76.9 | 13959 | 0.999958 |
| bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) | 4.4 / 4.9 | **98.39 / 98.24 / 102.18** | **98.19 / 98.11 / 99.19** | 383.6 | 98.0 | 13829 | 0.999958 |
| bf16 weights resident + `torch.compile` reduce-overhead (CUDA graphs) | 3.3 / 3.6 | **59.97 / 59.86 / 60.46** | **59.83 / 59.74 / 60.52** | 427.1 | 95.7 | 6937 | 0.999896 |
| bf16 autocast + whole-request `torch.compile` reduce-overhead (one CUDA graph per request) | — / — | not timed | | | | | failed: `OutOfMemoryError('CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 31.31 GiB of which 11.19 MiB is free. Including non-PyTorch mem` |
| bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps) | 126.5 / 126.7 | **46.62 / 46.43 / 47.15** | **46.39 / 46.31 / 47.73** | 407.0 | 89.2 | 7077 | 0.999982 |
| bf16 weights resident + whole-request `torch.compile` reduce-overhead (one CUDA graph per request) | 120.0 / 120.2 | **48.61 / 48.53 / 52.46** | **48.51 / 48.43 / 49.48** | 418.9 | 97.2 | 6915 | 0.999982 |

Other facts:

- Model load: 14.47 GB fp32 safetensors -> host took 1.1 s in the first run (`smoke.log`).
  - After that run, the file is page-cached: 0.01 s here.
- host -> cuda + model build: **1.6 s**.
  - The first run took 6.24 s, because the pages were cold (`smoke.log`).
- 13.48 GiB of fp32 weights on the device.
- First fp32 call: 372.7 ms.
- Idle GPU power: 28.6 W.
- Only the fp32 strict loop brings the GPU near its 600 W limit (466 W, 92 % util).
- Every other precision is bandwidth/launch-bound at batch 1 (64-75 % util).
- The fp32 served-like inference median (152.8 ms) is above the standalone incl_h2d median (144.1 ms), with a wide p90.
  - Cause: the fp32 loop runs at the power limit, and the host pre/post work between forwards adds jitter.
  - This file gives the value as measured.

Served-like loop (same host work as `server/app.py::predict`, 50 iterations, medians ms; p90 of total in brackets):

| GPU precision | preprocess (base64 + PNG decode + resize/normalize x2 + build_prompt + tokenize 224) | inference (incl_h2d) | postprocess (`tolist`) | **total** | power W |
|---|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | 1.97 | 152.82 | 0.06 | **156.51** (186.10) | 408.0 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 1.43 | 108.73 | 0.03 | **110.22** (113.55) | 335.4 |
| bf16 autocast (fp32 weights, +TF32 remainder) | 1.50 | 121.27 | 0.03 | **122.87** (124.28) | 290.9 |
| bf16 weights resident (eager, no autocast) | 1.46 | 100.03 | 0.04 | **101.48** (102.59) | 277.5 |
| p150a Hub run (PUBLISH_SUMMARY.md:18; warm-hub.json) | 1.17 | 125.84 | (inside inference: readback) | **127.0** (client wall 128.44) | not measured |

- The host stages are the same code on the two sides (`decode_image` x2 + `build_prompt` + `tokenize_prompt`).
- The GPU-side preprocess runs in the PIL/transformers of this venv.
- Its time is within a millisecond of the p150a server's 1.17 ms.

## Comparison with the p150a (same definitions)

- Ratio = p150a ms / GPU ms (> 1 means the GPU is faster).
- p150a precision:
  - bf16 activations, bf8 weights in places (VS:38 'inherent bf8 numerics').
  - Expert projections: `mcast1d_fp32`.
  - One Metal trace of the whole graph for each request (DEVICE_VALIDATION.md §2 step 14b, §5).

| row | p150a (definition) | GPU precision | GPU ms | ratio p150a/GPU |
|---|---:|---|---:|---:|
| device forward (p150a `timing_ms.inference` incl. uploads + readback vs GPU incl_h2d) | 125.84 (PUBLISH_SUMMARY.md:18, Hub tt serve; 125.9 in the 100-request verifier session) | fp32 strict (`allow_tf32=False`, `'highest'`) | 144.066 | **0.87** |
|  |  | tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 108.693 | **1.16** |
|  |  | bf16 autocast (fp32 weights, +TF32 remainder) | 121.476 | **1.04** |
|  |  | fp16 autocast (fp32 weights, +TF32 remainder) | 123.674 | **1.02** |
|  |  | bf16 weights resident (eager, no autocast) | 99.869 | **1.26** |
| | | bf16 autocast + `torch.compile` default | 88.419 | **1.42** |
| | | bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) | 98.388 | **1.28** |
| | | bf16 weights resident + `torch.compile` reduce-overhead (CUDA graphs) | 59.970 | **2.10** |
| | | bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps) | 46.620 | **2.70** |
| | | bf16 weights resident + whole-request `torch.compile` reduce-overhead (one CUDA graph per request) | 48.609 | **2.59** |
| GPU forward only (excl_h2d) vs p150a `execute_trace` 128.9 (DEVICE_VALIDATION.md:267, 130.2 ms configuration) | 128.9 | fp32 strict / tf32 / bf16 autocast / fp16 autocast / bf16 resident | 143.77 / 108.54 / 121.19 / 126.91 / 99.78 | 0.90 / 1.19 / 1.06 / 1.02 / 1.29 |
| | | bf16 autocast + `torch.compile` default (excl_h2d) | 87.94 | 1.47 |
| | | bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) (excl_h2d) | 98.19 | 1.31 |
| | | bf16 weights resident + `torch.compile` reduce-overhead (CUDA graphs) (excl_h2d) | 59.83 | 2.15 |
| | | bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps) (excl_h2d) | 46.39 | 2.78 |
| | | bf16 weights resident + whole-request `torch.compile` reduce-overhead (one CUDA graph per request) (excl_h2d) | 48.51 | 2.66 |
| served e2e (p150a `timing_ms.total` vs GPU served-like total) | 127.0 (PUBLISH_SUMMARY.md:18, Hub tt serve, 40 warm; 127.1 in the 100-request verifier session) | fp32 strict (`allow_tf32=False`, `'highest'`) | 156.514 | **0.81** |
|  |  | tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 110.221 | **1.15** |
|  |  | bf16 autocast (fp32 weights, +TF32 remainder) | 122.868 | **1.03** |
|  |  | bf16 weights resident (eager, no autocast) | 101.484 | **1.25** |

Interpretation:

- In strict fp32, the eager RTX 5090 takes 144.1 ms for each action chunk (H = 50).
- The p150a's fused bf16 trace takes 125.84 ms (ratio 0.87).
- The best eager GPU row is bf16 weights resident (eager, no autocast) at 99.9 ms (1.26x).
- The best compiled row is bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps).
  - With it, the GPU gets to 46.6 ms with transfers, 2.70x the p150a.
- The difference between eager and compiled is launch overhead in the expert loop, not arithmetic.
- The p150a number already contains the equivalent fix (one Metal trace for each request).
  - Thus the compiled rows compare the two *deployed* paths directly.
  - The eager rows compare the p150a's fused deployment with an unoptimized GPU run of the reference code.
- This test did not measure the p150a power in this pass, so this file makes no power or cost statement.

## Not measured / limitations

- p150a power: no pass measured it (BRIEF).
- GPU power is the mean of `nvidia-smi` samples during each timed loop.
- The p150a per-stage split:
  - It is from the 130.2 ms configuration (DEVICE_VALIDATION.md:266-271).
  - The shipped default (125.8 ms) is different only in the expert projections (-4.4 ms).
  - This pass did not measure a p150a number again.
- The input description in `p150_numbers.json`:
  - It describes the pi05 input as '3 camera images ... 32 tokens'.
  - The served / Hub configuration that gave 125.84 ms is `PI05_NUM_IMAGES=2`, `PI05_TOKEN_LEN=224` (tt-model.yaml `serve.env`).
  - DEVICE_VALIDATION.md:205 gives 'Served shape everywhere: 2 x 224x224, 224 tokens, 10 steps'.
  - This pass ran that configuration.
- Precision difference:
  - The GPU runs the reference's fp32 softmax / adaRMS / Euler math.
  - The p150a runs bf16 activations with bf8 weights in places.
  - The bf16 GPU rows are the closest precision match.
  - None of them reproduce the bf8 weight quantization.
- Autocast rows cast the fp32 weights again at every call (no autocast cache under `inference_mode`).
  - Thus bf16/fp16 autocast are not faster than tf32 in the expert loop.
  - The resident-bf16 rows do not have this cost.
- torch.compile:
  - The stage-wise variants compile three functions of a hand-written class-based model.
  - The 10-step loop and the two `embed_image` calls stay in Python.
  - The whole-request variants compile `forward_inference` once, with the loop unrolled.
  - The compile times of the re-runs use a warm inductor cache (`compile_run2.log`, `compile_run3.log`).
  - The cold compile of the stage functions took 31.7 s (`compile_run.log`).
- The stage split of the stage-wise CUDA-graph variants:
  - It is valid only with a `cudagraph_mark_step_begin` for each iteration (`compile_run3.log`).
  - The split in `compile_run2.log` did not have this call, and it recorded the graphs again.
  - Thus its 600 ms denoise numbers are not valid (replaced).
- The whole-request compile of the *autocast* model (fp32 weights resident):
  - The compilation failed because the GPU did not have sufficient memory (`compile_run3.log`: 29.6 GiB allocated by PyTorch of 31.3).
  - The unrolled single graph holds the bf16 copies of the 13.5 GiB fp32 weights plus its intermediates.
  - This file reports it as failed, not timed.
  - The bf16-resident whole-graph variants (6.8 GiB of weights) compile in 120-127 s and are the best GPU rows.
- Host preprocess:
  - On the GPU side, it runs in `.venv-gpu/pi05` (pillow 12.3, transformers 4.53).
  - The p150a preprocess runs inside its container (same host CPU).

## Evidence

- `logs/gpu-vs-p150/pi05-base/full_run.log`, `compile_run.log`, `smoke.log` (first run incl. the CPU reference), `smoke2.log`
- `logs/gpu-vs-p150/pi05-base/result.json`, `compile_result.json`, `cpu_fp32_reference.pt`; merged `reports/gpu-vs-p150/pi05-base.json`
- p150a: `reports/megakernel/PUBLISH_SUMMARY.md:18`, `logs/publish-megakernel/pi05-base/warm-hub.json`, `models/pi05-base-p150/DEVICE_VALIDATION.md:198-271`, `logs/megakernel-validate/pi05-base/serve/verify_fused_probe.log:7`, `verify_fused_actions.json`
