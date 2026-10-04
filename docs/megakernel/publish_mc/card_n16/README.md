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
  - 1 to 16 flow-matching steps (N).
  - batch: 1
  - Total 32 presets are available.
- Built on megakernel principle using `ttnn.generic_op` - not in a sense that the entire model is a single operation, but rather, main operations are fused into a single operation.
  - VISION: SigLIP and the projector (one device program for each group of 2 cameras)
  - PREFIX: the Language embedding and the Gemma-2B prefill, which writes the KV caches
  - EXPERT: the action expert (N steps x 18 layers) with its adaRMS time conditioning, the action input and output projections and the Euler steps.
- The default configuration with 2 cameras, action chunk of 50 actions and 10 steps, with 142 token-long prompt, the inference time is 55.2 ms (non-scalable profile) and 56.5 ms (scalable profile).

Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) ·
Paper: [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) ·
Upstream code: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) ·
Port: [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5)

- This package runs on **p150** (mesh `P150`, one p150a), with two serve profiles:
  - `non-scalable` (default): Ethernet cores do the dispatch, so the vision and prefix programs get a 12 x 10 worker grid. The chip cannot join a multi-chip fabric in this mode.
  - `scalable`: Tensix cores do the dispatch (11 x 10 worker grid). The Ethernet cores stay free for a multi-chip fabric.
- The package uses [tt-model-manager](https://github.com/tenstorrent/tt-model-manager) 0.1.0 (manifest schema 5.1).

## Demo

<video controls width="480" src="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4" poster="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial_poster.png"></video>

- The video shows the multi-config megakernel of this image (`non-scalable` profile) in a LIBERO-Spatial closed-loop test in MuJoCo on one Tenstorrent Blackhole p150a.
- Achieves 99/100 success (the libero_spatial run of the table below, `non-scalable`, N = 10).
- The median policy latency on the p150a is **49.3 ms** for each call (server side, p10 49.1, p90 49.8, 85 calls of the recording).
- CAUTION: The weights used for this demo are the LIBERO fine-tuned `lerobot/pi05_libero`, not the `lerobot/pi05_base` that this repository points to. To run this demo, you need to swap the weights.

## Quickstart

```bash
tt-model pull  changh95/pi05-base-p150 --with-weights
tt-model serve changh95/pi05-base-p150
```

- `tt-model pull` downloads the weights [`lerobot/pi05_base`](https://huggingface.co/lerobot/pi05_base) into your HF cache. The image does not contain the weights.
- The server uses port 20000. If that port is busy, the server uses the next free port.
- The default serve profile is `non-scalable`. To use the other one: `tt-model serve changh95/pi05-base-p150 --profile scalable`.


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

- The `non-scalable` (default) and `scalable` profiles
  - Assumes batch=1
  - Default configuration has 2 cameras, an action chunk of 50 actions and 10 flow-matching steps
  - The server builds the model on boot. To change the configuration, start the server again with other values.
  - The prompt bucket is the only item that can change for each request.
  - There are total 32 combinations of configuration (camera: 1/2/3/4, prompt bucket: 32/64/128/224, action-row bucket: 32/64)

| Item | Environment variable | Python field (`PI0ModelConfig`) | Range | Default | Fixed when |
|---|---|---|---|---|---|
| Number of cameras | `PI05_NUM_IMAGES` | `num_cameras` | 1 to 4 | 2 | The server starts |
| Action chunk length (H) | `PI05_ACTION_HORIZON` | `action_horizon` | 1 to 64 | 50 | The server starts |
| Flow-matching (denoising) steps (N) | `PI05_NUM_STEPS` | `num_denoising_steps` | 1 to 16 | 10 | The server starts |
| Device profile (dispatch) | `PI05_DISPATCH` (set by the serve profile) | none (read at import; `open_pi05_device` uses it) | `eth` (non-scalable), `tensix` (scalable) | `eth` | The server starts |
| Prompt bucket | none (request field `prompt_bucket`) | `sample_actions(..., prompt_bucket=)` | 32, 64, 128, 224 tokens | The smallest bucket that holds the prompt | Each request |
| Batch size | none | none | 1 only | 1 | Always |
| Image size | none | none | 224 × 224 only | 224 × 224 | Always |
| Models on one device | none | none | 1 live model | 1 | Always |

Where in the code (paths in `code/`):

| Item | The server reads it | The model checks it (refusal) | The model field |
|---|---|---|---|
| Number of cameras | `models/experimental/pi0_5/server/app.py:251` | `models/experimental/pi0/tt/ttnn_pi05_model.py:171`; the server check `models/experimental/pi0_5/server/mc_backend.py:51` | `models/experimental/pi0/common/configs.py:147` |
| Action chunk length (H) | `models/experimental/pi0_5/server/app.py:264` | `models/experimental/pi0/tt/megakernel/geometry.py:410` (from `ttnn_pi05_model.py:170`) | `models/experimental/pi0/common/configs.py:132` |
| Flow-matching steps (N) | `models/experimental/pi0_5/server/app.py:252` | `models/experimental/pi0/tt/megakernel/geometry.py:410` (from `ttnn_pi05_model.py:170`) | `models/experimental/pi0/common/configs.py:141` |
| All three at server start | `models/experimental/pi0_5/server/app.py:274` | `models/experimental/pi0_5/server/mc_backend.py:37` | `models/experimental/pi0_5/server/mc_backend.py:77` |
| Device profile | `models/experimental/pi0/tt/megakernel/profile.py:18` | `models/experimental/pi0/tt/ttnn_pi05_model.py:301` (`device_refusal`, the worker grid) | `models/experimental/pi0/tt/ttnn_pi05_model.py:81` (`open_pi05_device`) |
| Prompt bucket | `models/experimental/pi0_5/server/app.py:1050` (request field), `app.py:1277` | `models/experimental/pi0/tt/ttnn_pi05_model.py:341` | the compiled buckets: `models/experimental/pi0/tt/megakernel/presets.py:71` |
| Image count of a request | `models/experimental/pi0_5/server/app.py:1221` | `models/experimental/pi0/tt/ttnn_pi05_model.py:380` (masked cameras: line 375) | none |
| Batch size, image size | none | `models/experimental/pi0/tt/ttnn_pi05_model.py:383`, `:399` | none |
| One live model on each device | none | `models/experimental/pi0/tt/ttnn_pi05_model.py:165` | none |

To change the configuration of the server:

1. Pull the package: `tt-model pull changh95/pi05-base-p150 --with-weights`.
2. Write the launch command of the serve profile to a variable (add `--profile scalable` for the other profile). `tt-model serve` has no flag for the environment.
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
docker rm -f tt-model-pi05-base-p150-non-scalable      # stop the server
```

To use another configuration in Python:

1. Open the device with `open_pi05_device(0)`: it uses the dispatch cores of `PI05_DISPATCH` (default `eth`) and the 64 KiB worker-L1 cut.
2. Make a `PI0ModelConfig` with `num_cameras`, `action_horizon` and `num_denoising_steps`.
3. Build the model in a `with` block. Only one model can use the device at a time.
4. Call `sample_actions` with exactly `num_cameras` images. The model selects the prompt bucket.

```python
import ttnn
from models.experimental.pi0.common.configs import PI0ModelConfig
from models.experimental.pi0.common.weight_loader import PI0WeightLoader
from models.experimental.pi0.tt.ttnn_pi05_model import PI05MegakernelTTNN, open_pi05_device

device = open_pi05_device(0)  # PI05_DISPATCH=eth (default) or tensix; the 64 KiB worker-L1 cut
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
 "timing_ms": {"preprocess": 1.27, "inference": 55.21, "total": 56.53}}
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
- Both profiles give bit-identical outputs (checked on this image: c1-c4, N = 16 and the 8 openpi records); they differ only in speed and in what the chip can do next to the model (fabric or not).
- With an action chunk of up to 32 actions and a prompt bucket of up to 128 tokens, the expert streams its weights through two-layer rings when the L1 allows it (automatic; outputs bit-identical).
- With 4 cameras and 33-64 actions, the expert attention uses wider key chunks instead of a row loop (automatic).
- A Metal trace for each program (VISION/PREFIX/EXPERT) fixes the model pipeline in cold state. Warm runs are simply replays of the traced replay. This means we are assuming you are feeding 1 robot's data into the server, instead of multiple robots with different configurations.


### Accuracy

| Check | Result |
|---|---|
| The device output against the openpi GPU policy. The test used `lerobot/pi05_libero`, 8 real LIBERO observations, 2 cameras, H = 10 and N = 10. The PCC is over the 7 action dims. This image did the measurement, on both profiles. | Mean **0.999981**, min **0.999958** on each profile. The host inputs of the image are bit-identical to the inputs of openpi (images, tokens, mask, noise). |
| The matrix: 32 presets (cameras × prompt bucket × action-row bucket), N = 1 to 16, 3 action chunk lengths and 6 prompts for each set, on both profiles (1,024 sets). | The `scalable` outputs are bit-identical to the `non-scalable` outputs on 9,216/9,216 calls. So the rows below hold for both profiles. |
| A2: the full call against the fp32 reference on 6 prompts. The gate is PCC min ≥ 0.95 and mean ≥ 0.98. | **1,524/1,536** sets pass. The 12 failures are one input (2 cameras, 224-token prompt, H = 64, prompt 5) at N = 5 to 16. The GPU bf16 policy also fails this input at N = 6 to 16. |
| A4: the expert against an fp32 expert loop with the K / V caches of the device. The gate is ≥ 0.999 for N ≥ 2. | **1,437/1,440** sets pass for N ≥ 2 (min 0.9978). For N = 1, the card gives the values, but there is no gate (1,512/1,536 for all N). |
| The K / V caches, the replay identity and the cross-check. | 96/96 prefixes (min PCC 0.9926), 1,536/1,536, 576/576. |
| Negative controls: inputs with a known error must fail the gates. | They fail as necessary on 128/128 jobs (A2 and A4) and 96/96 prefixes (K / V). |

### Benchmarks

Device time of one trace replay in ms for each preset (the mean of 2 builds; each build gives the median of 60 trace replays):

#### `non-scalable` (default)

| Cameras | Action chunk (H) | N | inference time (ms) for prompt bucket 32 | ...for 64 | ...for 128 | ...for 224 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | H ≤ 32 | 10 | 36.80 | 36.80 | 37.45 | 39.89 |
| 1 | H 33-64 | 10 | 40.45 | 40.52 | 41.60 | 42.92 |
| 2 | H ≤ 32 | 10 | 47.16 | 47.66 | 48.20 | 51.09 |
| 2 | H 33-64 | 10 | 52.38 | 52.42 | 53.66 | 54.92 |
| 3 | H ≤ 32 | 10 | 63.13 | 63.29 | 63.67 | 69.42 |
| 3 | H 33-64 | 10 | 68.40 | 68.53 | 68.88 | 73.88 |
| 4 | H ≤ 32 | 10 | 77.63 | 77.94 | 80.75 | 84.69 |
| 4 | H 33-64 | 10 | 83.63 | 84.23 | 86.68 | 87.95 |

| Cameras | Action chunk (H) | N | inference time (ms) for prompt bucket 32 | ...for 64 | ...for 128 | ...for 224 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | H ≤ 32 | 1 | 36.09 | 36.24 | 37.66 | 39.41 |
| 2 | H 33-64 | 1 | 36.45 | 36.63 | 38.06 | 39.81 |
| 2 | H ≤ 32 | 5 | 40.92 | 41.19 | 42.31 | 44.55 |
| 2 | H 33-64 | 5 | 43.53 | 43.60 | 44.93 | 46.65 |
| 2 | H ≤ 32 | 16 | 54.65 | 55.38 | 55.67 | 59.41 |
| 2 | H 33-64 | 16 | 63.05 | 63.15 | 64.56 | 65.25 |

#### `scalable`

| Cameras | Action chunk (H) | N | inference time (ms) for prompt bucket 32 | ...for 64 | ...for 128 | ...for 224 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | H ≤ 32 | 10 | 37.58 | 37.61 | 38.07 | 41.12 |
| 1 | H 33-64 | 10 | 41.24 | 41.37 | 42.28 | 43.77 |
| 2 | H ≤ 32 | 10 | 48.93 | 48.72 | 49.34 | 52.71 |
| 2 | H 33-64 | 10 | 53.64 | 53.77 | 54.82 | 55.93 |
| 3 | H ≤ 32 | 10 | 67.65 | 67.86 | 68.22 | 69.97 |
| 3 | H 33-64 | 10 | 72.63 | 72.77 | 73.15 | 75.06 |
| 4 | H ≤ 32 | 10 | 80.00 | 80.19 | 82.71 | 86.93 |
| 4 | H 33-64 | 10 | 86.05 | 86.64 | 88.59 | 89.99 |

| Cameras | Action chunk (H) | N | inference time (ms) for prompt bucket 32 | ...for 64 | ...for 128 | ...for 224 |
|---:|---|---:|---:|---:|---:|---:|
| 2 | H ≤ 32 | 1 | 37.21 | 37.32 | 38.60 | 40.40 |
| 2 | H 33-64 | 1 | 37.73 | 37.79 | 39.08 | 40.70 |
| 2 | H ≤ 32 | 5 | 42.39 | 42.35 | 43.15 | 45.81 |
| 2 | H 33-64 | 5 | 44.80 | 44.85 | 45.85 | 47.44 |
| 2 | H ≤ 32 | 16 | 56.67 | 56.32 | 56.93 | 61.70 |
| 2 | H 33-64 | 16 | 64.37 | 64.45 | 65.72 | 66.38 |

- The host had no other workload during these measurements. The only load was the benchmark itself, while it built each model.
- Served over HTTP with the default configuration (2 cameras, H = 50, N = 10), the median inference time was **55.21 ms** (non-scalable) and **56.50 ms** (scalable) for 100 warm requests. The only other load on the host was the model server itself (1-min load average 3.3 or less).


### LIBERO closed loop (TT vs GPU)

Test conditions:

- The tests used `lerobot/pi05_libero` with this code and the `pi05_libero` conventions of openpi.
- The conventions are H = 10, the openpi normalization statistics, the openpi client and the LIBERO loop of openpi.
- The test set is libero_spatial: 10 tasks × initial states 0-9.
- Each episode on the p150a has a pair: the same episode with the PyTorch policy of openpi on an RTX 5090.
- Both runs used the same client, initial states and noise seeds for each call.
- Both profiles ran the same 100 episodes.
- This table does not give latency, because the host had other load during these runs. For latency, see the Benchmarks section.

| Profile | Cameras | N | p150a successes | RTX 5090 successes | Discordant pairs (only TT / only GPU) |
|---|---:|---:|---:|---:|---:|
| `non-scalable` | 2 | 10 | **99 / 100** | 100 / 100 | 0 / 1 |
| `non-scalable` | 2 | 5 | **100 / 100** | 99 / 100 | 1 / 0 |
| `non-scalable` | 2 | 1 | **100 / 100** | 99 / 100 | 1 / 0 |
| `non-scalable` | 2 | 16 | **100 / 100** | 100 / 100 | 0 / 0 |
| `scalable` | 2 | 10 | **99 / 100** | 100 / 100 | 0 / 1 |
| `scalable` | 2 | 5 | **100 / 100** | 99 / 100 | 1 / 0 |
| `scalable` | 2 | 1 | **100 / 100** | 99 / 100 | 1 / 0 |
| `scalable` | 2 | 16 | **100 / 100** | 100 / 100 | 0 / 0 |

- The paired difference is not significant in any row (exact McNemar p = 1).


### Limitations

- **Accuracy.**
  - A2 fails on one input: 2 cameras, a 224-token prompt, H = 64 and prompt (seed) 5, at N = 5 to 16. The other 5 prompts of that preset stay at 0.998 or more.
  - The GPU bf16 openpi policy also fails this input at N = 6 to 16, but the device is lower at every N (A2 min PCC against fp32):

    | N | p150a | GPU bf16 |
    |---|---|---|
    | 5 | 0.928 | 0.966 |
    | 6 | 0.870 | 0.936 |
    | 7 | 0.845 | 0.933 |
    | 8 | 0.803 | 0.901 |
    | 9 | 0.813 | 0.828 |
    | 10 | 0.763 | 0.895 |
    | 11 | 0.774 | 0.890 |
    | 12 | 0.772 | 0.836 |
    | 13 | 0.730 | 0.878 |
    | 14 | 0.724 | 0.858 |
    | 15 | 0.770 | 0.798 |
    | 16 | 0.729 | 0.807 |

  - A4 at N ≥ 2: 3 of 1,440 sets are below 0.999: 1 camera / 64-token prompt / N = 5 / H = 1 (0.9988), 2 cameras / 128 / N = 3 / H = 1 (0.9978), and 2 cameras / 128 / N = 16 / H = 50, prompt 5 (0.9989, the same as before the 64-row change).
  - At N = 1, A4 is reported, not gated: 21 of 96 sets are below 0.999.
- **Supported inputs.**
  - The server supports only batch size 1 and images of 224 × 224, as it assumes single robot case.
  - It does not support >4 cameras.
  - The prompt must have 224 real tokens or fewer.
  - H must be 64 or less, and N must be 16 or less.
  - The camera count, H and N stay the same until the server stops.
- **One model for each device.** Only one model can use a device at a time. If a second model starts, the server refuses it until the first model closes.
- **Profiles.** `non-scalable` (the default) uses Ethernet dispatch and a 12 × 10 grid for vision and prefix. It needs the tt-metal runtime of this image (PR #57142 patches). The chip cannot join a multi-chip fabric in this mode. `scalable` uses Tensix dispatch (11 × 10) and runs on any runtime.
- **Gated tokenizer.**
  - Accept the Gemma terms of `google/paligemma-3b-pt-224`. Then use `hf auth login` before `tt serve`.
  - The prompt tokenizer of this model is **gated** under the Gemma terms.
  - If you cannot get access, send `tokens` instead of `prompt`.
- **Base checkpoint.**
  - The outputs are normalized actions of the **base** checkpoint.
  - This checkpoint has no task-specific fine-tune.
  - The rows with real inputs (openpi golden, LIBERO) use `lerobot/pi05_libero` with the same code.
- **API.** The API is not OpenAI-compatible. `GET /v1/models` is only a stub.

### TODO

- **Blocked matmul for 64 action rows (S64)**
  - With an action chunk of 33-64 actions, the expert processes two 32-row tiles of action rows.
  - Today each expert matmul (qkv, o_proj, up/gate, down) is called once for each row tile. So each weight tile is unpacked twice.
  - Plan: call each matmul once for both row tiles (`rt_dim=2`). Each weight tile is then unpacked once. The output is expected to stay bit-identical.
  - Measured ceiling: removing the second row tile's matmuls saves 10.5 us per layer (qkv 2.3, o_proj 2.4, MLP 5.7) at 2 cameras / 224-token bucket / H = 50. That is at most about 0.19 ms per denoising step, or 1.9 ms per call at N = 10. The real gain will be smaller.
  - Open risk: a blocked call uses twice the DST tiles. Some matmuls may need a new DST split.
- **Expert fidelity per preset**
  - HiFi4 on the expert fixes the two A4 cells at H = 1, but its speed changes with the preset (-2.5 to +1.6 ms). Choose the fidelity per preset class.

### License

- Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base), [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
  - This repository does not include the weights. `tt-model` downloads them into your HF cache.
  - The tokenizer [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) is gated under the same terms.
- Port and server code (`code/`): Apache-2.0 headers, with distribution under the same Gemma terms.
  - The code is from [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ [`33528a8`](https://github.com/changh95/tt-pi-0.5/commit/33528a82f4dc524a00336b6552041b76dac9448e).

## Provenance

- The next table shows the sources of the image.
- `code/` in this repository is byte-identical to the model code in the image.

| component | built from |
| --- | --- |
| tt-metal | `c718b5df9b9589f8920e2f960af56145e4bf91dc` = [`f856a38a361`](https://github.com/tenstorrent/tt-metal/commit/f856a38a361939888f92d88f9e69b2f8a83fb713) + `a69a83df5ad` (PR #57142, squashed: Ethernet dispatch on harvested Blackhole ETH grids) + `c718b5df9b9` (the fetch-queue command-size check kept in Release builds); describe `v0.80.0-dev20261001-19-gc718b5df9b` |
| `code/` digest | `0a542bdf5a9d3af8` (sha256, first 16 hex digits) |
| built | 2026-10-04T12:36:25+00:00 by tt-model 0.1.0 |
