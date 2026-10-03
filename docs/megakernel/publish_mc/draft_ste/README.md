---
tags:
- blackhole
- p150
- tt-dit-server
- tt-model-cache
- tt-model-container
- tenstorrent
- ttnn
- tt-metal
- tt-nn
- vla
- robot-control
- flow-matching
- pi0
- tt-model-catalog
license: gemma
base_model:
- lerobot/pi05_base
license_link: https://ai.google.dev/gemma/terms
pipeline_tag: robotics
---

# pi05-base-p150

This package runs the pi-0.5 vision-language-action policy of Physical Intelligence on one Tenstorrent Blackhole p150a.

- The model is `lerobot/pi05_base`: a SigLIP vision encoder, a Gemma-2B VLM and a Gemma-300M flow-matching action expert.
- One image supports these configurations:
  - 1 to 4 cameras.
  - An action chunk of 1 to 64 actions (the action chunk has H actions).
  - 1 to 10 flow-matching steps (N).
- The server builds one model for one configuration when it starts. Three environment variables select the configuration (see "How to change the configuration").
- A prompt bucket is a fixed prompt length that the device programs use: 32, 64, 128 or 224 tokens. Each request uses the smallest prompt bucket that holds its prompt.
- A device program is one persistent `ttnn.generic_op` megakernel on 110 Tensix cores. Each request uses three device programs (four with 3 or 4 cameras):
  - VISION: SigLIP and the projector (one device program for each group of 2 cameras or fewer).
  - PREFIX: the language embedding and the Gemma-2B prefill, which writes the K / V caches.
  - EXPERT: the action expert (N steps × 18 layers) with its adaRMS time conditioning, the action input and output projections and the Euler steps.
- One Metal trace for each prompt bucket holds these device programs. Each request is one trace replay of that trace.
- The default serve profile has 2 cameras, an action chunk of 50 actions and 10 steps.
- With this profile and a prompt of about 143 tokens, the inference time is 56.4 ms (`timing_ms.inference`).

Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) ·
Paper: [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) ·
Upstream code: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) ·
Port: [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5)

- This package runs on **p150** (mesh `P150`, one p150a).
- The package has one serve profile, `p150`. Three environment variables change its configuration (see "How to change the configuration").

- The package uses [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) 0.1.0 (manifest schema 5.1).

## Quickstart

```bash
tt-model pull  changh95/pi05-base-p150 --with-weights
tt-model serve changh95/pi05-base-p150
```

- `tt-model pull` downloads the weights [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) at `b211f3d44c36` into your HF cache. The image does not contain the weights.
- The server uses port 20000. If that port is busy, the server uses the next free port.
- The server is ready when the log shows `Application startup complete`.

### Run with tt-cli

1. Start the server, send one request and stop the server:

```bash
tt serve changh95/pi05-base-p150
printf '{"images":["%s","%s"],"prompt":"pick up the cube","state":[0.1,-0.2,0.3,0,0,0,0.5,-0.5]}' \
  "$(base64 -w0 media/sample_base.png)" "$(base64 -w0 media/sample_wrist.png)" > req.json
curl -s localhost:20000/predict -H 'Content-Type: application/json' -d @req.json
tt model stop changh95/pi05-base-p150
```

`POST /predict` accepts these fields:

- `images`: one base64 PNG or JPEG image for each camera of the server (default 2), in the order `[base/exterior, wrist, ...]`.
  - The server refuses masked cameras and a wrong number of images.
  - If you have fewer cameras, start a server for that number of cameras.
- `prompt` (the task text) or `tokens` (PaliGemma token ids, 224 real tokens or fewer).
- `state` (optional): the proprioceptive state, 32 or fewer floats, normalized to the range [-1, 1]. The default is zeros.
- `seed` (optional): the seed of the initial flow-matching noise. The default noise is fixed, thus the output is deterministic.
- `prompt_bucket` (optional): 32, 64, 128 or 224. The request then uses that prompt bucket, not the smallest prompt bucket that holds the prompt.

Other endpoints:

- `GET /health` shows the server status.
- `GET /info` (path `/info`) shows the configuration, the kernel digest and the number of device programs for each call.

### How to change the configuration

- The package has one serve profile, `p150`. Its default configuration has 2 cameras, an action chunk of 50 actions and 10 flow-matching steps.
- The server builds one model for one configuration when it starts. To change the configuration, start the server again with other values.
- The prompt bucket is the only item that can change for each request.

| Item | Environment variable | Python field (`PI0ModelConfig`) | Range | Default | Fixed when |
|---|---|---|---|---|---|
| Number of cameras | `PI05_NUM_IMAGES` | `num_cameras` | 1 to 4 | 2 | The server starts |
| Action chunk length (H) | `PI05_ACTION_HORIZON` | `action_horizon` | 1 to 64 | 50 | The server starts |
| Flow-matching (denoising) steps (N) | `PI05_NUM_STEPS` | `num_denoising_steps` | 1 to 10 | 10 | The server starts |
| Prompt bucket | none (request field `prompt_bucket`) | `sample_actions(..., prompt_bucket=)` | 32, 64, 128, 224 tokens | The smallest bucket that holds the prompt | Each request |
| Batch size | none | none | 1 only | 1 | Always |
| Image size | none | none | 224 × 224 only | 224 × 224 | Always |
| Models on one device | none | none | 1 live model | 1 | Always |

Where in the code (paths in `code/`):

| Item | The server reads it | The model checks it (refusal) | The model field |
|---|---|---|---|
| Number of cameras | `models/experimental/pi0_5/server/app.py:251` | `models/experimental/pi0/tt/ttnn_pi05_model.py:139`; the server check `models/experimental/pi0_5/server/mc_backend.py:48` | `models/experimental/pi0/common/configs.py:147` |
| Action chunk length (H) | `models/experimental/pi0_5/server/app.py:264` | `models/experimental/pi0/tt/megakernel/geometry.py:391` (from `ttnn_pi05_model.py:138`) | `models/experimental/pi0/common/configs.py:132` |
| Flow-matching steps (N) | `models/experimental/pi0_5/server/app.py:252` | `models/experimental/pi0/tt/megakernel/geometry.py:391` (from `ttnn_pi05_model.py:138`) | `models/experimental/pi0/common/configs.py:141` |
| All three at server start | `models/experimental/pi0_5/server/app.py:274` | `models/experimental/pi0_5/server/mc_backend.py:34` | `models/experimental/pi0_5/server/mc_backend.py:72` |
| Prompt bucket | `models/experimental/pi0_5/server/app.py:1050` (request field), `app.py:1277` | `models/experimental/pi0/tt/ttnn_pi05_model.py:305` | the compiled buckets: `models/experimental/pi0/tt/megakernel/presets.py:69` |
| Image count of a request | `models/experimental/pi0_5/server/app.py:1221` | `models/experimental/pi0/tt/ttnn_pi05_model.py:344` (masked cameras: line 339) | none |
| Batch size, image size | none | `models/experimental/pi0/tt/ttnn_pi05_model.py:347`, `:363` | none |
| One live model on each device | none | `models/experimental/pi0/tt/ttnn_pi05_model.py:133` | none |

To change the configuration of the server:

1. Pull the package: `tt-model pull changh95/pi05-base-p150 --with-weights`.
2. Write the launch command of the serve profile to a variable. `tt-model serve` has no flag for the environment.
3. Change the three `PI05_*` values in the command. The server reads them only at start.
4. Start the container. Then examine `GET /info`: it shows the new configuration.
5. Run the smoke test or send requests. Send exactly `PI05_NUM_IMAGES` images in each request.

```bash
CMD=$(tt-model serve changh95/pi05-base-p150 --print | grep '^docker run')
CMD=$(echo "$CMD" | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=3/' \
                        -e 's/PI05_ACTION_HORIZON=50/PI05_ACTION_HORIZON=10/' \
                        -e 's/PI05_NUM_STEPS=10/PI05_NUM_STEPS=5/' -e 's/^docker run /docker run --detach /')
eval "$CMD"
curl -s localhost:20000/info | python3 -c "import json,sys; print(json.load(sys.stdin)['megakernel']['program'])"
docker rm -f tt-model-pi05-base-p150-p150      # stop the server
```

To use another configuration in Python:

1. Open the device with `PI05_DEVICE_PARAMS` (the 64 KiB worker-L1 cut is necessary).
2. Make a `PI0ModelConfig` with `num_cameras`, `action_horizon` and `num_denoising_steps`.
3. Build the model in a `with` block. Only one model can use the device at a time.
4. Call `sample_actions` with exactly `num_cameras` images. The model selects the prompt bucket.

```python
import ttnn
from models.experimental.pi0.common.configs import PI0ModelConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.tt.ttnn_pi05_model import PI05_DEVICE_PARAMS, PI05MegakernelTTNN

device = ttnn.open_device(device_id=0, **PI05_DEVICE_PARAMS)  # the 64 KiB worker-L1 cut is required
cfg = PI0ModelConfig(action_horizon=10, num_denoising_steps=5, num_cameras=3, pi05=True)
with PI05MegakernelTTNN(cfg, PI0WeightLoader("lerobot/pi05_base"), device) as model:  # one live model per device
    actions = model.sample_actions(images, None, lang_tokens, lang_masks=lang_masks, noise=noise)  # [1, 10, 32]
```

Cameras:

- Send the images in this order: base (exterior) camera, then the wrist camera, then the other cameras.
- Send only real cameras. The model refuses masked camera slots.
- If you have fewer cameras, start a server (or build a model) for that number of cameras.

Values out of range:

- If a value is out of range, the server does not start. The error message gives the name and the value.
- For example, `PI05_NUM_IMAGES=5` gives: `pi0.5 megakernel refused: cameras = 5 (compiled: 1, 2, 3, 4) [PI05_NUM_IMAGES=5, PI05_ACTION_HORIZON=50, PI05_NUM_STEPS=10]`.
- The server also refuses a mesh, a `PI05_BATCH_SIZES` value other than 1 and a `PI05_TOKEN_LEN` value larger than 224.

### Response

```json
{"actions": [[-0.0640, -0.1553, 0.2969, 0.0991, -0.0430, 0.0879, 0.4102, -0.4961, ...], ...],
 "action_horizon": 50, "action_dim": 32, "normalized": true, "denoising_steps": 10,
 "num_tokens": 142, "token_len": 224, "prompt_bucket": 224, "prompt_truncated": false,
 "images_used": 2, "images_padded": 0, "image_size": [224, 224], "seed": null,
 "timing_ms": {"preprocess": 1.54, "inference": 56.55, "total": 58.09}}
```

- `actions` is the action chunk: H rows and 32 columns.
- The values are in the **normalized** QUANTILES action space of lerobot.
- The columns after your action dimension are zero.

To get the actions for your robot:

1. Denormalize the actions with `(a+1)*(q99-q01)/2+q01` and the statistics of your dataset.
2. Use only the columns of your action dimension (for example, the first 7 columns for LIBERO).

Input processing:

- The server resizes each image to 224×224 and does not keep the aspect ratio.
- Then it normalizes each image as openpi does: `x * float32(1/255) * 2 - 1`. The result is bit-exact to the PyTorch input of openpi.
- The server puts `state` into the prompt as 256 bins: `Task: <prompt>, State: b0 … b31;\nAction: `.
- The attention masks the pad tokens (openpi semantics). Thus the pad tokens have no effect on the output.

### Demo

| Base camera (`media/sample_base.png`, synthetic) | Wrist camera (`media/sample_wrist.png`, synthetic) |
|:---:|:---:|
| ![](media/sample_base.png) | ![](media/sample_wrist.png) |

- The end of this card shows a LIBERO closed-loop demo of the same code with the LIBERO fine-tuned weights.

### What runs where

| Part of one request | Where it runs | Device programs for each call |
|---|---|---:|
| Image decode, resize and normalization. The im2col of the patches. The prompt and its tokens. The attention-mask rows and the RoPE rows. The initial noise. The copies from the host to the device. The readback. | The host. The host only prepares the inputs and the outputs. | 0 |
| SigLIP (patch embedding + 27 layers + post-LN) and the projector for the cameras. | VISION: one persistent `ttnn.generic_op` on 110 cores for each group of 2 cameras or fewer. | 1 (2 with 3-4 cameras) |
| The language embedding. The Gemma-2B prefill (18 layers), which writes 18 bfp8 K / V caches. | PREFIX: one persistent `ttnn.generic_op`. | 1 |
| The time MLP and the adaRMS conditioning. The action input projection. N Euler steps × 18 Gemma-300M expert layers. The action output projection. | EXPERT: one persistent `ttnn.generic_op`. | 1 |

- One Metal trace for each prompt bucket holds the three (or four) device programs.
- A request writes its inputs into fixed device buffers. Then it replays that trace.
- A preset is one combination of a camera count, a prompt bucket and an action-row bucket.
- The 32 presets are all combinations of 4 camera counts, 4 prompt buckets and 2 action-row buckets (32 and 64 rows).
- If H is 32 or less, the request uses the 32-row bucket.
- The model includes N in the expert weights when it builds the model.
- For 1 to 3 cameras, the K / V caches stay in L1. For 4 cameras, the K / V caches are in DRAM.
- The image build checks the 32 presets and each C++ header that the kernels include.
- The image build also checks the kernel digest that the device gates used (`1429d5bea05c31ad`).

### Accuracy

Test conditions:

- One p150a, tt-metal `main` @ `f856a38a361`, batch size 1 and HiFi3 for each expert matmul.
- The full test matrix has 320 sets: 32 presets × N 1..10.
- A2: each set compares the device output with the fp32 torch reference on 6 padded prompts.
- A4: the expert against an fp32 expert loop with the K / V caches of the device.
- A3 and A6: the K / V caches.
- The trace replays, which must be bit-identical.
- An independent cross-check.

| Check | Result |
|---|---|
| The device output against the openpi GPU policy. The test used `lerobot/pi05_libero`, 8 real LIBERO observations, 2 cameras, H = 10 and N = 10. The PCC is over the 7 action dims. This image did the measurement. | Mean **0.999981**, min **0.999958**. The host inputs of the image are bit-identical to the inputs of openpi (images, tokens, mask, noise). |
| A2: the full call against the fp32 reference on 6 prompts. The gate is PCC min ≥ 0.95 and mean ≥ 0.98. | **314/320** sets pass (see the limitation below). |
| A4: the expert against an fp32 expert loop with the K / V caches of the device. The gate is ≥ 0.999 for N ≥ 2. | **286/288** sets pass. For N = 1, the card gives the values, but there is no gate. |
| The K / V caches, the replay identity and the cross-check. | 320/320, 320/320, 320/320. |
| Negative controls: inputs with a known error must fail the gates. | They fail as necessary on 320/320 (A2) and 320/320 (A4). |

### Benchmarks

Device time of one trace replay in ms for each preset (the mean of 2 builds; each build gives the median of 60 trace replays):

| Cameras | Action chunk (H) | N | Prompt bucket 32 | 64 | 128 | 224 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | H ≤ 32 | 10 | 38.93 | 39.56 | 39.87 | 41.13 |
| 1 | H 33-64 | 10 | 41.26 | 41.34 | 42.30 | 43.75 |
| 2 | H ≤ 32 | 10 | 51.05 | 50.23 | 51.89 | 52.49 |
| 2 | H 33-64 | 10 | 53.68 | 53.72 | 54.85 | 55.57 |
| 3 | H ≤ 32 | 10 | 67.93 | 68.12 | 68.45 | 69.87 |
| 3 | H 33-64 | 10 | 72.60 | 72.76 | 73.15 | 74.91 |
| 4 | H ≤ 32 | 10 | 84.06 | 84.17 | 85.73 | 86.77 |
| 4 | H 33-64 | 10 | 86.03 | 91.75 | 93.63 | 94.65 |
| 2 | H ≤ 32 | 1 | 37.40 | 37.42 | 38.49 | 40.11 |
| 2 | H 33-64 | 1 | 37.70 | 37.90 | 39.02 | 40.35 |
| 2 | H ≤ 32 | 5 | 43.43 | 43.17 | 44.34 | 45.52 |
| 2 | H 33-64 | 5 | 44.74 | 44.86 | 45.78 | 46.98 |

- Each value is the device time of one trace replay of all device programs of one request (in-process, release code, batch size 1).
- Served over HTTP with the default configuration (2 cameras, H = 50, N = 10), the median `timing_ms.inference` was **56.40 ms** for 100 warm requests.
- [`PERF_PRESETS.md`](PERF_PRESETS.md) gives the standard errors, the † marks and the build log of these measurements.

### LIBERO closed loop (TT vs GPU)

Test conditions:

- The tests used `lerobot/pi05_libero` with this code and the `pi05_libero` conventions of openpi.
- The conventions are H = 10, the openpi normalization statistics, the openpi client and the LIBERO loop of openpi.
- The test set is libero_spatial: 10 tasks × initial states 0-9.
- Each episode on the p150a has a pair: the same episode with the PyTorch policy of openpi on an RTX 5090.
- Both runs used the same client, initial states and noise seeds for each call.
- The p150a latency is the time of one policy call on the server, with host input preparation, device time and readback.
- This card does not give the GPU latency, because its measurement was different (openpi model time only, on a shared host with load).

| Cameras | N | p150a successes | RTX 5090 successes | Discordant pairs (only TT / only GPU) | Mean time of one p150a policy call on the server, with the host inputs |
|---:|---:|---:|---:|---:|---:|
| 2 | 10 | **99 / 100** | 100 / 100 | 0 / 1 | 52.7 ms |
| 2 | 5 | **100 / 100** | 99 / 100 | 1 / 0 | 45.1 ms |
| 2 | 1 | **100 / 100** | 99 / 100 | 1 / 0 | 39.1 ms |
| 1 | 10 | **1 / 100** | 2 / 100 | 0 / 1 | 40.2 ms |
| 1 | 5 | **1 / 100** | 2 / 100 | 0 / 1 | 32.8 ms |
| 1 | 1 | **0 / 100** | 0 / 100 | 0 / 0 | 27.0 ms |

- For each row, the exact McNemar test of the discordant pairs gives p = 1.
- With 1 camera, the policy does not get the wrist image.
- Then both backends fail (0 to 2 successes of 100), because pi05_libero needs the wrist view.
- These rows are a record. They do not show accuracy.

### Limitations

- **One input-sensitive trajectory (A2).**
  - All 6 failed A2 sets have the same input.
  - The input has 2 cameras, a full 224-token prompt, 64 action rows (H = 64) and matrix seed 5, at N = 5 to 10.
  - The bf16 GPU policy of openpi also fails on this input for N = 6 and more.
  - But the result of this device is lower at each N.
  - For the other five prompts of this cell, the PCC is 0.9983 or more at N = 5 to 10.
  - At each N from 1 to 10, it is 0.9978 or more.

| steps | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| openpi GPU bf16 vs fp32 | 0.966 | 0.936 | 0.933 | 0.901 | 0.828 | 0.895 |
| this megakernel vs fp32 | 0.928 | 0.870 | 0.845 | 0.803 | 0.813 | 0.763 |

  - If you round the K / V caches of the reference to bf16 or bfp8, the reference stays at a PCC of 0.99994 or more.
  - A higher fidelity for the VLM matmuls does not help.
  - This trajectory increases the reduced-precision error of the prefix.
  - The closed-loop LIBERO test is the end-to-end check.
- **A4 with one action row.**
  - Two A4 sets fail at N = 2 or more. Both sets have H = 1. The gate is 0.999.
  - The first set has 1 camera, the 64-token bucket and N = 5 (0.998746).
  - The second set has 2 cameras, the 128-token bucket and N = 3 (0.997803).
  - At N = 1, 21 of 96 (preset, H) values are below 0.999.
  - The lowest value is 0.993544, with 1 camera, the 32-token bucket and H = 1.
  - HiFi4 for each expert matmul removes both failures. But HiFi4 is slower for some configurations, thus this release does not use it.
  - Expert fidelity for each preset is a future task.
- **Supported inputs.**
  - The server supports only batch size 1 and images of 224 × 224.
  - It supports 1 to 4 real cameras, with no masked cameras.
  - The prompt must have 224 real tokens or fewer.
  - H must be 64 or less, and N must be 10 or less.
  - The camera count, H and N stay the same until the server stops.
- **One model for each device.** Only one model can use a device at a time. If a second model starts, the server refuses it until the first model closes.
- **Worker grid.** The worker grid is 11 × 10 (110 cores) with Tensix dispatch. A later release will use the 12 × 10 grid with Ethernet dispatch.
- **Gated tokenizer.**
  - Accept the Gemma terms of `google/paligemma-3b-pt-224`. Then use `hf auth login` before `tt serve`.
  - The prompt tokenizer of this model is **gated** under the Gemma terms.
  - If you cannot get access, send `tokens` instead of `prompt`.
- **Base checkpoint.**
  - The outputs are normalized actions of the **base** checkpoint.
  - This checkpoint has no task-specific fine-tune.
  - The rows with real inputs (openpi golden, LIBERO) use `lerobot/pi05_libero` with the same code.
- **API.** The API is not OpenAI-compatible. `GET /v1/models` is only a stub.
- **Previous paths.**
  - `code/` also has the single-configuration paths of the previous releases (`PI05_MEGAKERNEL=whole` / `expert` / `off`, the mesh layouts) as comparators.
  - The tests for these paths used tt-metal `668c2907575`, not the tt-metal tree of this image.

### License

- Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base), [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
  - This repository does not include the weights. `tt-model` downloads them into your HF cache.
  - The tokenizer [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) is gated under the same terms.
- Port and server code (`code/`): Apache-2.0 headers, with distribution under the same Gemma terms.
  - The code is from [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ [`5edf139`](https://github.com/changh95/tt-pi-0.5/commit/5edf139d458e9acabd31eea641cd364079962677).

## Provenance

- The next table shows the sources of the image.
- `code/` in this repository is byte-identical to the model code in the image.

| component | built from |
| --- | --- |
| tt-metal | [`f856a38a361939888f92d88f9e69b2f8a83fb713`](https://github.com/tenstorrent/tt-metal/commit/f856a38a361939888f92d88f9e69b2f8a83fb713) |
| `code/` digest | `184c58b636fb7621` (sha256, first 16 hex digits) |
| built | 2026-10-03T06:47:04+00:00 by tt-model 0.1.0 |

## LIBERO closed-loop demo

<video controls width="480" src="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4" poster="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial_poster.png"></video>

- The video shows the multi-config megakernel of this image in a LIBERO-Spatial closed-loop test in MuJoCo on one p150a.
- A screen capture recorded the test in real time ([video](https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4)).
- [`demo/README.md`](demo/README.md) gives the procedure for the demo and shows the four clips.
- [`demo/manifest.json`](demo/manifest.json) is the machine-readable record of the clips (probes, step counts, latency, backend stamps, eval score).

| libero_spatial (10 tasks x initial states 0-9) | Device | Successes |
|---|---|---:|
| `lerobot/pi05_libero` on `PI05MegakernelTTNN`, 2 cameras, N = 10. | p150a | **99 / 100** (0 errors). The t9/i4 episode stopped at the step limit. |
| openpi `PI0Pytorch` with the same weights, client and noise seeds. | RTX 5090 | 100 / 100 |

- The mean policy latency on the p150a is **52.7 ms** for each call (server side, p90 53.0, 2,182 calls).
- The code is tt-metal-pr `changh95/pi05-megakernel-mc` @ `7a622a3a570`. This is the model code of `code/`, identical to `fae9cd03fa4`.
- The tt-metal version is `f856a38`, and the kernel digest is `1429d5bea05c31ad`.

**Caution:** Do not use this demo to estimate the results of `lerobot/pi05_base`. The demo uses different weights and a different configuration:

- The weights are the LIBERO fine-tuned `lerobot/pi05_libero` @ `a217bfd3b146` with the `pi05_libero` normalization statistics of openpi.
- The configuration is the LIBERO configuration of openpi.
  - The action chunk has H = 10 actions. The robot executes 5 actions of each chunk.
  - The prompts have 32 tokens.
- The image serves `lerobot/pi05_base` with the same model code.
- The LIBERO websocket server is `code/models/experimental/pi0_5/server/serve_pi05_libero.py`.
- The client and the normalization statistics are not in this repository.
