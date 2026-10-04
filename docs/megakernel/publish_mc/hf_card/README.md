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

This package runs the [Pi-0.5](https://huggingface.co/lerobot/pi05_base) VLA model of Physical Intelligence on one Tenstorrent Blackhole p150a.

- Supports the following configurations:
  - num_cameras: 1/2/3/4
  - Prompt bucket: 32/64/128/224
  - Action chunk: 1~64 (in 32/64 bucket)
  - 1 to 10 flow-matching steps (N).
  - batch: 1
- Built on megakernel principle using `ttnn.generic_op` - not in a sense that the entire model is a single operation, but rather, main operations are fused into a single operation.
  - VISION: SigLIP and the projector (one device program for each group of 2 cameras)
  - PREFIX: the Language embedding and the Gemma-2B prefill, which writes the KV caches
  - EXPERT: the action expert (N steps x 18 layers) with its adaRMS time conditioning, the action input and output projections and the Euler steps.
- The default profile with 2 cameras, action chunk of 50 actions and 10 steps, with 143 token-long prompt, the inference time is 56.4 ms.

Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) ·
Paper: [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) ·
Upstream code: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) ·
Port: [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5)

- This package runs on **p150** (mesh `P150`, one p150a).
- The package uses [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) 0.1.0 (manifest schema 5.1).

## Demo

<video controls width="480" src="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4" poster="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial_poster.png"></video>

- The video shows the multi-config megakernel of this image in a LIBERO-Spatial closed-loop test in MuJoCo on one Tenstorrent Blackhole p150a.
- Achieves 99/100 success.
- The mean policy latency on the p150a is **52.7 ms** for each call (server side, p90 53.0, 2,182 calls).
- CAUTION: The weights used for this demo are the LIBERO fine-tuned `lerobot/pi05_libero`, not the `lerobot/pi05_base` that this repository points to. To run this demo, you need to swap the weights.

## Quickstart

```bash
tt-model pull  changh95/pi05-base-p150 --with-weights
tt-model serve changh95/pi05-base-p150
```

- `tt-model pull` downloads the weights [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) into your HF cache. The image does not contain the weights.
- The server uses port 20000. If that port is busy, the server uses the next free port.


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

- The `p150` profile
  - Assumes batch=1
  - Default configuration has 2 cameras, an action chunk of 50 actions and 10 flow-matching steps
  - The server builds the model on boot. To change the configuration, start the server again with other values.
  - The prompt bucket is the only item that can change for each request.
  - There are total 32 combinations of configuration (camera: 1/2/3/4, prompt bucket: 32/64/128/224, action-row bucket: 32/64)

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

### Implementation implications

- For 1 to 3 cameras, the KV cache lives in L1. For 4 cameras, the KV cache is in DRAM.
- A Metal trace for each program (VISION/PREFIX/EXPERT) fixes the model pipeline in cold state. Warm runs are simply replays of the traced replay. This means we are assuming you are feeding 1 robot's data into the server, instead of multiple robots with different configurations.


### Accuracy

| Check | Result |
|---|---|
| The device output against the openpi GPU policy. The test used `lerobot/pi05_libero`, 8 real LIBERO observations, 2 cameras, H = 10 and N = 10. The PCC is over the 7 action dims. This image did the measurement. | Mean **0.999981**, min **0.999958**. The host inputs of the image are bit-identical to the inputs of openpi (images, tokens, mask, noise). |
| A2: the full call against the fp32 reference on 6 prompts. The gate is PCC min ≥ 0.95 and mean ≥ 0.98. | **314/320** sets pass (see the limitation below). |
| A4: the expert against an fp32 expert loop with the K / V caches of the device. The gate is ≥ 0.999 for N ≥ 2. | **286/288** sets pass. For N = 1, the card gives the values, but there is no gate. |
| The K / V caches, the replay identity and the cross-check. | 320/320, 320/320, 320/320. |
| Negative controls: inputs with a known error must fail the gates. | They fail as necessary on 320/320 (A2) and 320/320 (A4). |

### Benchmarks

Device time of one trace replay in ms for each preset (the mean of 2 builds; each build gives the median of 60 trace replays):

| Cameras | Action chunk (H) | N | inference time (ms) for prompt bucket 32 | ...for 64 | ...for 128 | ...for 224 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | H ≤ 32 | 10 | 38.93 | 39.56 | 39.87 | 41.13 |
| 1 | H 33-64 | 10 | 41.26 | 41.34 | 42.30 | 43.75 |
| 2 | H ≤ 32 | 10 | 51.05 | 50.23 | 51.89 | 52.49 |
| 2 | H 33-64 | 10 | 53.68 | 53.72 | 54.85 | 55.57 |
| 3 | H ≤ 32 | 10 | 67.93 | 68.12 | 68.45 | 69.87 |
| 3 | H 33-64 | 10 | 72.60 | 72.76 | 73.15 | 74.91 |
| 4 | H ≤ 32 | 10 | 84.06 | 84.17 | 85.73 | 86.77 |
| 4 | H 33-64 | 10 | 86.03 | 91.75 | 93.63 | 94.65 |

- Served over HTTP with the default configuration (2 cameras, H = 50, N = 10), the median inference time was **56.40 ms** for 100 warm requests.

| Cameras | Action chunk (H) | N | inference time (ms) for prompt bucket 32 | ...for 64 | ...for 128 | ...for 224 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | H ≤ 32 | 1 | 37.40 | 37.42 | 38.49 | 40.11 |
| 2 | H 33-64 | 1 | 37.70 | 37.90 | 39.02 | 40.35 |
| 2 | H ≤ 32 | 5 | 43.43 | 43.17 | 44.34 | 45.52 |
| 2 | H 33-64 | 5 | 44.74 | 44.86 | 45.78 | 46.98 |


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


### Limitations

- **Supported inputs.**
  - The server supports only batch size 1 and images of 224 × 224, as it assumes single robot case.
  - It does not support >4 cameras.
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
