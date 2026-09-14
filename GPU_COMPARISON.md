# pi05-base-p150 — Blackhole p150a vs RTX 5090 (same host, same weights, same input)

Date 2026-09-14. Facts only; every GPU number below was measured in this pass, every p150a number is copied (with its source line) from the validation / publish reports and logs. The p150a was NOT touched.

## What was run

| | |
|---|---|
| Model | pi-0.5 (`lerobot/pi05_base`): SigLIP-So400m/14 vision tower (27 layers, 1152 wide, 256 tokens per 224x224 camera) + PaliGemma / Gemma-2B VLM (18 layers, 2048 wide, MQA 8/1 heads, 16384 MLP) + Gemma-300M flow-matching action expert (18 layers, 1024 wide, adaRMS), 3.617 B parameters. The port's own torch reference `models/pi05-base-p150/code/models/experimental/pi0_5/reference/torch_pi0_model.py::PI0Model` (`pi05=True`, the config `server/app.py`'s lifespan builds for the TT model) — the network `tests/pcc/test_pcc_pi05_fused.py` gated the p150a against. It is plain Python over raw weight tensors (no `nn.Module`), moved to `cuda` by moving every tensor of the categorized state dict |
| Weights | `lerobot/pi05_base` @ `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` (tt-model.yaml `weights.revision` = `serve.env.TT_WEIGHTS_REVISION`): `model.safetensors` (14.47 GB, 812 fp32 tensors) + `config.json` from the HF cache `/home/deepgadget/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`, `HF_HUB_OFFLINE=1`. Tokenizer `google/paligemma-3b-pt-224` (gated, cached, logged-in token) |
| Input | `media/sample_base.png` + `media/sample_wrist.png` (224x224 RGB; byte-identical to `smoke_test.synthetic_image('base'/'wrist')`, i.e. exactly the payload of the Hub warm run `logs/publish-megakernel/pi05-base/warm_client.py`) -> `server/app.py::decode_image`: RGB, bilinear squash-resize to 224x224, /255, (x-0.5)/0.5 -> two `[1,3,224,224]` fp32 tensors; prompt `"pick up the cube"` + zero state -> `build_prompt` -> `Task: pick up the cube, State: 128 128 1… 128;\nAction: ` -> PaliGemma tokenizer right-padded to 224 tokens (143 real); fixed seeded noise (`torch.manual_seed(42); randn(1,50,32)` = what `PI0ModelTTNN.__init__` draws in the server); 10 Euler denoise steps; batch 1. H2D payload 1.21 MB |
| Output | `[1,50,32]` normalised actions (host `actions[0].tolist()` in the served-like loop) |
| GPU | NVIDIA GeForce RTX 5090 (sm_120), driver 580.126.18, power limit 600.00 W, 32607 MiB; idle 28.6 W |
| venv | `/home/deepgadget/experiments/tt-models/.venv-gpu/pi05` — Python 3.12.13, torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, transformers 4.53.0 (= the tt-model.yaml pin), numpy 2.5.3, safetensors 0.8.0, huggingface_hub 0.36.2, sentencepiece 0.2.2, protobuf 7.36.1, triton 3.6.0; added in this pass: pillow 12.3.0, fastapi 0.141.1, pydantic 2.13.5 (so `server/app.py` — the served preprocessing — imports; no ttnn) |
| Scripts | `logs/gpu-vs-p150/pi05-base/bench_pi05_gpu.py` (eager legs, uses `logs/gpu-vs-p150/bench_common.py`), `bench_pi05_compile.py` (torch.compile leg), `make_report.py` (this file from the JSONs); logs `full_run.log`, `compile_run.log`, smoke runs `smoke.log` / `smoke2.log`; raw JSON `result.json`, `compile_result.json` (merged into `reports/gpu-vs-p150/pi05-base.json`); CPU reference `cpu_fp32_reference.pt` |
| Commands | `HF_HUB_OFFLINE=1 .venv-gpu/pi05/bin/python bench_pi05_gpu.py --skip-cpu --iters 50 --warmup 10 --served-iters 50 --stage-iters 20` (the CPU reference was produced by the first `smoke.log` run of the same script without `--skip-cpu`), then `bench_pi05_compile.py --iters 50 --warmup 10` |
| Loop | per precision: 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before/after each; wall-clock (perf_counter) is the primary number, CUDA-event time recorded alongside; power sampled by `nvidia-smi -lms 200` during the timed loop |
| p150a source | `reports/gpu-vs-p150/p150_numbers.json` -> `reports/megakernel/PUBLISH_SUMMARY.md:18` (Hub `tt serve`, 40 warm requests: inference **125.84** ms, total **127.0**; `logs/publish-megakernel/pi05-base/warm-hub.json` adds preprocess 1.17 ms, client wall 128.44), `models/pi05-base-p150/DEVICE_VALIDATION.md:262` (harness 125.7 / 125.9 / 126.6), `:266-271` (segment profile of the 130.2 ms configuration: host inputs 0.68 + copies 0.05 + execute_trace 128.9 + readback 0.03; trace = SigLIP 22.6 + VLM 49.9 + 10 expert steps 56.3), `logs/megakernel-validate/pi05-base/serve/verify_fused_probe.log:7` (100 warm: 125.9 / 125.6 / 126.3, total 127.1) |

Two deviations from the literal reference, needed to run it on CUDA (both in `bench_pi05_gpu.py`): `precompute_freqs_cis` is cached and returns the RoPE tables on the model's device (the reference recomputes them on the CPU on every expert call, which cannot multiply CUDA tensors; the p150a port also keeps a cos/sin cache), and the noise sampler returns the fixed seeded tensor (as `test_pcc_pi05_fused.py` does). Everything else is the reference's own code path, including its per-call `weight.to(dtype)` casts and Python-level loops.

Timing definitions (matching the p150a `timing_ms` keys):

- **incl_h2d** = `.to('cuda')` of the two images, tokens, mask, state and noise (pageable host tensors, what `predict()` produces) + `sample_actions` (embed_prefix: SigLIP x2 + projector + language embed; VLM prefill of 736 tokens with KV cache; 10 expert steps) + `actions.cpu()`. Compare with p150a `timing_ms.inference` = 3 host->device copies + `execute_trace` of the whole graph + readback (**125.84 ms**, PUBLISH_SUMMARY.md:18).
- **excl_h2d** = the same forward with inputs resident and the actions left on the device. Compare with the p150a `execute_trace` alone (128.9 ms in the 130.2 ms configuration, DEVICE_VALIDATION.md:267; the shipped 125.8 ms configuration has ~124.5 ms of trace by the same split).
- **served-like** = base64 decode + PNG decode + resize/normalise (`decode_image` x2) + `build_prompt` + `tokenize_prompt` + incl_h2d forward + `actions.tolist()`. Compare with p150a `timing_ms.total` (**127.0 ms** = preprocess 1.17 + inference 125.84; PUBLISH_SUMMARY.md:18, warm-hub.json). The p150a's base64/PNG decode is inside its `preprocess` too (`predict()` starts its clock before `decode_image`).
- **stages** = embed_prefix / VLM prefill / 10 denoise steps measured separately with a sync after each (inputs resident). Compare with the p150a segment profile SigLIP 22.6 / VLM 49.9 / expert 56.3 ms (DEVICE_VALIDATION.md:267-270; that profile is of the 130.2 ms configuration — the shipped default is 4.4 ms faster in the expert, i.e. ~51.9 ms).

## Correctness check (GPU vs CPU fp32 reference)

CPU fp32 reference (`PI0Model` on the host, 16 threads): one `sample_actions` = 13.9 s. GPU fp32 strict (no TF32) on the same inputs, PCC over the `[1,50,32]` action chunk (the quantity the p150a's e2e gate uses):

| metric | value |
|---|---:|
| PCC actions (1600 values) | **1.000000** |
| max abs diff | 5.66e-07 |
| actions[0][:6] GPU / CPU | [-0.02314, -0.02479, -0.04036, 0.06249, -0.05698, 0.03722] / [-0.02314, -0.02479, -0.04036, 0.06249, -0.05698, 0.03722] |
| first GPU call (fp32 strict, incl. CUDA init / cuBLAS handles) | 372.7 ms |

PCC > 0.999 holds; the GPU runs the right model. Cross-check against the p150a itself: the same GPU fp32 model on the `serve_probe.py` payload (same two images, prompt `"pick up the cube"`, state `[0.1, -0.2, 0.3, 0, 0, 0, 0.5, -0.5]`, the model's default seed-42 noise) vs the actions the p150a served in the verifier session (`logs/megakernel-validate/pi05-base/serve/verify_fused_actions.json`, fused bf16/bf8 trace): **PCC 0.999530**, max abs diff 2.643e-02. For scale, the port's harness reports torch-vs-fused PCC 0.9977 / 0.9987 on two random observations (DEVICE_VALIDATION.md:262).

Per-precision accuracy vs the CPU fp32 reference (same input):

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

fp16 autocast is numerically fine on this input (no overflow in the Gemma-2B activations; PCC above 0.999), so it is timed. Every bf16 GPU row is well above the p150a's own e2e PCC (0.9977 / 0.9987 on the harness inputs, 0.99953 on the served probe payload vs this fp32 reference).

## GPU latency (batch 1, 2 x 224x224 + 224 tokens + 10 steps; median / min / p90 of 50 iterations, wall-clock ms)

Eager PyTorch, fp32 weights on the device (as the brief prescribes), autocast rows re-cast the fp32 weights on every call (the autocast weight cache is off under `inference_mode`):

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power mean W (excl / incl loop) | GPU util % | peak mem alloc / reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | **144.07 / 143.32 / 147.63** | **143.77 / 143.26 / 146.86** | 143.76 | 142.91 | 465.9 / 463.2 | 92.1 | 14052 / 14410 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | **108.69 / 108.06 / 110.28** | **108.54 / 108.03 / 110.95** | 108.52 | 109.35 | 333.9 / 332.9 | 74.8 | 14052 / 14440 |
| bf16 autocast (fp32 weights, +TF32 remainder) | **121.48 / 121.11 / 123.70** | **121.19 / 120.61 / 122.53** | 121.17 | 205.59 | 291.5 / 291.2 | 74.3 | 13986 / 14504 |
| fp16 autocast (fp32 weights, +TF32 remainder) | **123.67 / 121.49 / 130.50** | **126.91 / 122.73 / 150.60** | 126.88 | 229.08 | 270.0 / 280.8 | 68.1 | 13985 / 14504 |

Weights resident in bf16 (every tensor `.to(bfloat16)` once, 6.76 GiB on the device; images cast to bf16 on the device inside the timed region; SigLIP / VLM / expert in bf16 with the reference's fp32 softmax; time embedding computed in fp32 then cast; velocity cast to fp32 so the Euler update `x_t + dt*v` stays fp32 — the p150a keeps x_t in bf16). This is the closest match to the p150a, which holds bf16 / bf8 weights on the device:

| precision | incl_h2d median / min / p90 | excl_h2d median / min / p90 | CUDA-event excl | first call ms | power W (excl / incl) | GPU util % | peak mem MiB | PCC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 weights resident (eager, no autocast) | **99.87 / 99.18 / 101.32** | **99.78 / 99.31 / 101.08** | 99.77 | 100.83 | 280.3 / 279.9 | 64.0 | 7047 | 0.999873 |

Stage split (inputs resident, a sync after each stage, median of 20; p150a segment profile from DEVICE_VALIDATION.md:267-270, 130.2 ms configuration):

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

The eager reference is launch-bound in the expert loop: 18 layers x ~30 small kernels x 10 steps ≈ 5–6 k launches per chunk at 50 tokens, so the 10 steps cost 65–85 ms regardless of precision (bf16 autocast is *slower* than fp32 there because every step re-casts the 300 M expert weights); the prefill and SigLIP, which have real work per launch, scale with precision as expected. This is exactly the regime the p150a's whole-graph Metal trace removes, and what `torch.compile` with CUDA graphs addresses on the GPU:

`torch.compile` (inductor, `dynamic=False`, the three stage functions `embed_image` / `forward_vlm` / `_denoise_forward` compiled separately; the Euler loop and the 10 calls stay in Python). The reduce-overhead (CUDA graphs) variants need two host-side additions: the `embed_image` output is `.clone()`d outside the graph (it is replayed once per camera and the second replay overwrites the first camera's output buffer -- the first attempt failed with 'accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run', `compile_run.log`) and `torch.compiler.cudagraph_mark_step_begin()` is called once per request. The `whole-request` variants compile `forward_inference` as ONE function (dynamo unrolls the 10-step Euler loop; SigLIP x2 + VLM prefill + 10 expert steps in a single graph, one CUDA graph replay per request under reduce-overhead) -- the closest analogue of the p150a's single Metal trace; no stage split is possible for them. Accuracy re-checked after compile:

| variant | compile s (first call / ready after 3 more calls) | incl_h2d median / min / p90 | excl_h2d median / min / p90 | power W (excl loop) | GPU util % | peak mem MiB | PCC |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 autocast + `torch.compile` default | 31.7 / 31.9 | **88.42 / 87.87 / 89.85** | **87.94 / 87.51 / 89.58** | 336.5 | 76.9 | 13959 | 0.999958 |
| bf16 autocast + `torch.compile` reduce-overhead (CUDA graphs) | 4.4 / 4.9 | **98.39 / 98.24 / 102.18** | **98.19 / 98.11 / 99.19** | 383.6 | 98.0 | 13829 | 0.999958 |
| bf16 weights resident + `torch.compile` reduce-overhead (CUDA graphs) | 3.3 / 3.6 | **59.97 / 59.86 / 60.46** | **59.83 / 59.74 / 60.52** | 427.1 | 95.7 | 6937 | 0.999896 |
| bf16 autocast + whole-request `torch.compile` reduce-overhead (one CUDA graph per request) | — / — | not timed | | | | | failed: `OutOfMemoryError('CUDA out of memory. Tried to allocate 16.00 MiB. GPU 0 has a total capacity of 31.31 GiB of which 11.19 MiB is free. Including non-PyTorch mem` |
| bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps) | 126.5 / 126.7 | **46.62 / 46.43 / 47.15** | **46.39 / 46.31 / 47.73** | 407.0 | 89.2 | 7077 | 0.999982 |
| bf16 weights resident + whole-request `torch.compile` reduce-overhead (one CUDA graph per request) | 120.0 / 120.2 | **48.61 / 48.53 / 52.46** | **48.51 / 48.43 / 49.48** | 418.9 | 97.2 | 6915 | 0.999982 |

Other facts: model load (14.47 GB fp32 safetensors -> host 1.1 s in the first run (`smoke.log`; the file is page-cached afterwards, 0.01 s here); host -> cuda + model build **1.6 s** (6.24 s in the first run when the pages were cold, `smoke.log`); 13.48 GiB of fp32 weights on the device); first fp32 call 372.7 ms; idle GPU power 28.6 W. Only the fp32 strict loop pulls the GPU near its 600 W limit (466 W, 92 % util); every other precision is bandwidth/launch-bound at batch 1 (64-75 % util). The fp32 served-like inference median (152.8 ms) is above the standalone incl_h2d median (144.1 ms) with a wide p90 -- the fp32 loop runs at the power limit and the host pre/post between forwards adds jitter; reported as measured.

Served-like loop (same host work as `server/app.py::predict`, 50 iterations, medians ms; p90 of total in brackets):

| GPU precision | preprocess (base64 + PNG decode + resize/normalise x2 + build_prompt + tokenize 224) | inference (incl_h2d) | postprocess (`tolist`) | **total** | power W |
|---|---:|---:|---:|---:|---:|
| fp32 strict (`allow_tf32=False`, `'highest'`) | 1.97 | 152.82 | 0.06 | **156.51** (186.10) | 408.0 |
| tf32 (`allow_tf32=True`, `'high'`; PyTorch default is `'highest'`) | 1.43 | 108.73 | 0.03 | **110.22** (113.55) | 335.4 |
| bf16 autocast (fp32 weights, +TF32 remainder) | 1.50 | 121.27 | 0.03 | **122.87** (124.28) | 290.9 |
| bf16 weights resident (eager, no autocast) | 1.46 | 100.03 | 0.04 | **101.48** (102.59) | 277.5 |
| p150a Hub run (PUBLISH_SUMMARY.md:18; warm-hub.json) | 1.17 | 125.84 | (inside inference: readback) | **127.0** (client wall 128.44) | not measured |

The host stages are the same code on both sides (`decode_image` x2 + `build_prompt` + `tokenize_prompt`); the GPU-side preprocess is measured in this venv's PIL/transformers and is within a millisecond of the p150a server's 1.17 ms.

## Comparison with the p150a (matching definitions)

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster). p150a precision: bf16 activations, bf8 weights in places (VS:38 'inherent bf8 numerics'), expert projections `mcast1d_fp32`, one Metal trace of the whole graph per request (DEVICE_VALIDATION.md §2 step 14b, §5).

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

Reading: in strict fp32 the eager RTX 5090 takes 144.1 ms per 50-step chunk against the p150a's 125.84 ms fused bf16 trace (ratio 0.87); the best eager GPU row is bf16 weights resident (eager, no autocast) at 99.9 ms (1.26x). With `torch.compile` (bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps)) the GPU reaches 46.6 ms incl. transfers, 2.70x the p150a. The gap between eager and compiled is launch overhead in the expert loop, not arithmetic: the p150a number already includes the equivalent fix (one Metal trace per request), so the compiled rows compare the two *deployed* paths like for like, while the eager rows compare the p150a's fused deployment with an un-optimised GPU run of the reference code. p150a power was not measured in this pass, so no power or cost statement is made.

## Not measured / caveats

- p150a power: not measured in any pass (BRIEF). GPU power is the mean of `nvidia-smi` samples during each timed loop.
- p150a per-stage split is from the 130.2 ms configuration (DEVICE_VALIDATION.md:266-271); the shipped default (125.8 ms) differs only in the expert projections (-4.4 ms). No p150a number was re-measured.
- `p150_numbers.json` describes the pi05 input as '3 camera images ... 32 tokens'; the served / Hub configuration that produced 125.84 ms is `PI05_NUM_IMAGES=2`, `PI05_TOKEN_LEN=224` (tt-model.yaml `serve.env`, DEVICE_VALIDATION.md:205 'Served shape everywhere: 2 x 224x224, 224 tokens, 10 steps'), which is what was run here.
- The GPU runs the reference's fp32 softmax / adaRMS / Euler math; the p150a runs bf16 activations with bf8 weights in places. The bf16 GPU rows are the closest precision match; none of them replicate the bf8 weight quantisation.
- Autocast rows re-cast the fp32 weights every call (no autocast cache under `inference_mode`), which is why bf16/fp16 autocast are not faster than tf32 in the expert loop; the resident-bf16 rows avoid that.
- torch.compile: the stage-wise variants compile three functions of a hand-written class-based model (the 10-step loop and the two `embed_image` calls remain Python); the whole-request variants compile `forward_inference` once with the loop unrolled. Compile times are with a warm inductor cache for the re-runs (`compile_run2.log`, `compile_run3.log`); the cold compile of the stage functions took 31.7 s (`compile_run.log`). The stage split of the stage-wise CUDA-graph variants is valid only with a `cudagraph_mark_step_begin` per iteration (`compile_run3.log`; the split in `compile_run2.log` lacked it and re-recorded graphs, hence its 600 ms denoise numbers -- superseded).
- The whole-request compile of the *autocast* model (fp32 weights resident) ran out of GPU memory during compilation (`compile_run3.log`: 29.6 GiB allocated by PyTorch of 31.3): the unrolled single graph holds the bf16 copies of the 13.5 GiB fp32 weights plus its intermediates. It is reported as failed, not timed; the bf16-resident whole-graph variants (6.8 GiB of weights) compile in 120-127 s and are the best GPU rows.
- Host preprocess on the GPU side runs in `.venv-gpu/pi05` (pillow 12.3, transformers 4.53); the p150a's runs inside its container (same host CPU).

## Evidence

- `logs/gpu-vs-p150/pi05-base/full_run.log`, `compile_run.log`, `smoke.log` (first run incl. the CPU reference), `smoke2.log`
- `logs/gpu-vs-p150/pi05-base/result.json`, `compile_result.json`, `cpu_fp32_reference.pt`; merged `reports/gpu-vs-p150/pi05-base.json`
- p150a: `reports/megakernel/PUBLISH_SUMMARY.md:18`, `logs/publish-megakernel/pi05-base/warm-hub.json`, `models/pi05-base-p150/DEVICE_VALIDATION.md:198-271`, `logs/megakernel-validate/pi05-base/serve/verify_fused_probe.log:7`, `verify_fused_actions.json`

