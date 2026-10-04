# Serve pi-0.5 (`lerobot/pi05_base`) on Blackhole with tt-model-manager

This repo is a **tt-model container package**. It contains these parts:

- `tt-model.yaml`.
- The port in `code/`.
- The built OCI image in `image/`.

The commands and the server:

- `tt-model package --container` builds the image.
- The image contains tt-metal (built from source), the port and the HTTP stack.
- `tt-model serve` runs the image on the card.
- The server is `code/models/experimental/pi0_5/server/app.py` (FastAPI, `kind: tt-dit-server`).
- The default backend of the server is the multi-config pi0.5 megakernel `PI05MegakernelTTNN` (`code/models/experimental/pi0/tt/ttnn_pi05_model.py`).

## Terms

- Camera count: the number of real camera images in each request (1 to 4).
- Action chunk: the actions that one request returns. The action chunk has H actions.
- Flow-matching steps (N): the number of Euler steps of the action expert (1 to 10).
- Configuration: one set of a camera count, H and N. The server builds one model for one configuration.
- Serve profile: a named configuration of the server (`--profile`). This package has one serve profile, `p150`.
- Prompt bucket: a fixed prompt length that the device programs use: 32, 64, 128 or 224 tokens.
- Action-row bucket (S): a fixed number of action rows that the device programs use: 32 or 64.
- Preset: one combination of a camera count, a prompt bucket and an action-row bucket. There are 32 presets.
- Device program: one persistent `ttnn.generic_op` megakernel on the 11 x 10 worker grid. A request uses 3 device programs (4 with 3 or 4 cameras).
- Trace replay: one replay of the Metal trace that holds the device programs of one request.
- Request: one `POST /predict` call.
- Inference time: `timing_ms.inference`.

## Layout

```
tt-model.yaml                               authoring manifest (schema 5.1): build, serve env, 1 serve profile (p150), verify, card
tt_kernel_manifest.json, image/             the wire manifest and the OCI image `tt-model pull` loads
SERVING.md                                  this file
PERF_PRESETS.md                             per-preset device replay latency (32 presets at N = 10, c2 at N = 1 / 5) + build log
media/sample_base.png, media/sample_wrist.png   synthetic 224x224 placeholders for the curl example / smoke test
demo/                                       LIBERO closed-loop demo (pi05_libero weights; see demo/README.md)
code/models/__init__.py, code/models/experimental/__init__.py   empty: make `models` a regular package
code/models/common/lightweightmodule.py     the schema's mandatory tt-metal entry (nothing imports it)
code/models/experimental/pi0/               the model: the multi-config megakernel (from the tenstorrent/tt-metal PR branch
                                            changh95/pi05-megakernel-mc @ fae9cd03fa4, unchanged)
    common/      configs, weight loader, pi05_host.py (im2col, masks, RoPE rows, K / V cache plan)
    tt/          ttnn_pi05_model.py (PI05MegakernelTTNN) + tt/megakernel/: presets.py (the 32 presets), the vision /
                 prefix programs (pe_program.py, kernels_p2/) and the expert program (program.py, kernels/mk_*)
    reference/   torch reference (the PCC oracle, openpi semantics)
    tests/       host proofs, device PCC / perf tests (need a tt-metal tree + pytest; not used by the server)
code/models/experimental/pi0_5/             the server and the earlier single-config paths
    server/      app.py (ASGI app; mc_backend.py = the default backend), smoke_test.py, serve_pi05_libero.py
                 (openpi websocket server for the LIBERO client; not the image's entry)
    common/, tt/, reference/, tests/        the single-config paths of the 09-29 / 09-30 / 10-01 releases (comparators)
```

- `code/models/` is byte-identical to `models/` of [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ `5edf139d458e9acabd31eea641cd364079962677`.
- In the image, `code/models/` is at `/opt/tt-metal/models/...`, with `PYTHONPATH=/opt/tt-metal`.
- tt-model copies the tt-metal tree without its own `models/`. Thus this `pi0` package is the only `pi0` package in the image.
- The server finds the kernel sources relative to the Python files.
- The JIT compiler compiles the kernel sources at the first boot of each configuration.

## tt-metal tree

- `source.tt_metal` is a clean clone of tenstorrent/tt-metal `main` @ `f856a38a361939888f92d88f9e69b2f8a83fb713` (v0.80.0-dev20261001-17).
- The wire manifest shows `dirty: false` for this tree.
- The validation of the multi-config megakernel used this tree.
- The previous releases pinned `main` @ `668c2907575`.
- These items did not change from the previous releases:
  - The base image (`ghcr.io/tenstorrent/tt-metal/tt-metalium/ubuntu-22.04-dev-amd64`).
  - `torch==2.11.0` (+cpu).
  - `transformers==5.12.1`.
- `verify:` checks these items:
  - The 32 presets.
  - The 14 kernel sources and their digest (`1429d5bea05c31ad`).
  - Each `#include` of the C++ sources of the megakernel, against `tt_metal/hw/inc` in the image.
  - The default configuration of the server.
  - The checks of the earlier single-config paths.

## Run on the HOST for validation (no Docker)

Use this procedure to validate the port on the host without Docker:

1. Set `T` to a built tt-metal tree @ `f856a38a361`. Set `R` to this repo.
2. Export `PYTHONPATH`, `TT_METAL_HOME`, `ARCH_NAME` and the model variables.
3. Install fastapi and uvicorn into a side directory. Add that directory to `PYTHONPATH`.
4. Start the server with `uvicorn`.
5. Run `smoke_test.py` against the server.

```bash
T=/path/to/tt-metal                      # a built tree @ f856a38a361
R=/path/to/this/repo
export PYTHONPATH=$R/code:$T/ttnn TT_METAL_HOME=$T ARCH_NAME=blackhole
export HF_MODEL=lerobot/pi05_base TT_WEIGHTS_REVISION=b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba
export TT_MESH_SHAPE=1x1 TT_DEVICE_ID=0 PI05_NUM_IMAGES=2 PI05_ACTION_HORIZON=50 PI05_NUM_STEPS=10 PI05_TOKEN_LEN=224
# fastapi / uvicorn are not in the tree venv: install them into a side directory and add it to PYTHONPATH
$T/python_env/bin/python -m uvicorn --host 0.0.0.0 --port 20000 --lifespan on models.experimental.pi0_5.server.app:app
python $R/code/models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:20000
```

- The tree venv does not contain fastapi or uvicorn. Thus step 3 is necessary.
- The server writes these boot log phrases in this order. `tt-model serve` uses them for its checklist:
  - `Loading weights`
  - `Loading tokenizer`
  - `Opening device`
  - `Loading pipeline`
  - `Warming up: compiling and capturing the 32 / 64 / 128 / 224-token prompt buckets`
  - `Warmup 1/2`
  - `Warmup complete (... -- N device ops per call, one trace per prompt bucket)`
  - `Application startup complete` (from uvicorn)
- The server compiles and captures each prompt bucket before READY. Thus READY means that the server is warm.
- If the startup fails, the server raises an exception and uvicorn exits with a non-zero code.
- The server does not use the CPU as a fallback.

Offline overrides:

- `PI05_WEIGHTS_DIR=<dir with model.safetensors + config.json>`: a local directory for the weights.
- `PI05_TOKENIZER_DIR=<local tokenizer dir>`: a local directory for the tokenizer.
- `PI05_TOKENIZER_REQUIRED=0`: the server starts without a tokenizer. Then only `tokens` requests work.

## Package and serve (from the repo directory)

1. Set `ROOT` to your tt-model-manager checkout. The checkout contains `bin/docker-env.sh` and `.venv`.
2. Run `source $ROOT/bin/docker-env.sh` to use rootless Docker. This script sets `PATH` and `DOCKER_HOST`.
3. Build the package with `tt-model package --container`.
4. Start the server with `tt-model serve`.
5. Run `smoke_test.py` against the port that `tt-model serve` printed. Then stop the server with `tt-model stop`.

```bash
ROOT=/path/to/tt-models                              # your tt-model-manager checkout (bin/docker-env.sh, .venv)
source $ROOT/bin/docker-env.sh                       # rootless Docker (PATH + DOCKER_HOST)
$ROOT/.venv/bin/tt-model package --container tt-model.yaml --out $ROOT/build
$ROOT/.venv/bin/tt-model serve $ROOT/build/pi05-base-p150/tt_kernel_manifest.json
python code/models/experimental/pi0_5/server/smoke_test.py --url http://127.0.0.1:<port serve printed>
$ROOT/.venv/bin/tt-model stop  $ROOT/build/pi05-base-p150/tt_kernel_manifest.json
```

With tt-cli:

- To start the server, use `tt serve changh95/pi05-base-p150`.
- To stop the server, use `tt model stop changh95/pi05-base-p150`.
- To see the serve profile, use `tt-model profiles changh95/pi05-base-p150`.

Caches and boot times:

- The host cache is `~/.cache/tt-model/pi05-base-p150/{cache,weights,tensors}`. The JIT kernels stay in this cache from one boot to the next.
- The weights are in `~/.cache/huggingface/hub/models--lerobot--pi05_base`.
- Measured boot times of this image on the p150a (2026-10-03, wall time from `tt-model serve` to READY):
  - Default configuration, cold (empty JIT cache): 66.2 s. The four prompt buckets compile and capture in 11.8 s.
  - Default configuration, warm: 41.2 s. The four prompt buckets compile and capture in 0.8 s.
  - Other configurations (1, 3 or 4 cameras; H = 10; N = 5): 35-57 s. Each configuration compiles its own device programs at its first boot.

## Request and response contract

- `GET /health` returns `{"status": "ok" | "starting", "model": "pi05-base-p150", "device": ...}`.
  - The status code is always 200.
  - The status is `ok` only after the warm-up.
- `GET /info` returns these fields:
  - model, task, hardware.
  - `megakernel`: {backend `mc`, program {kernel_digest, cameras, action_horizon, suffix_rows, num_steps, prompt_buckets, kv_dram, device_ops_per_call}}.
  - `weights`, `tokenizer`.
  - `source`: {repo, commit, path}.
  - `inputs`: num_images, prompt_buckets, missing_images: refused.
  - `outputs`: actions [H, 32].
  - `limits`, `warmup_latency_ms`, `license`.
- `GET /v1/models` returns an OpenAI-shaped stub. The stub prevents a 404 on the tt-model ready card. This endpoint is not a chat API.

`POST /predict` accepts JSON with these fields:

| field | type | description |
|---|---|---|
| `images` | `[str]`, exactly `PI05_NUM_IMAGES` | Base64 PNG/JPEG images in the order `[base/exterior camera, wrist camera, ...]`. The server squash-resizes each image to 224x224 (bilinear) and does not resample a 224x224 image. Then it applies `x * float32(1/255) * 2 - 1` (the PyTorch input of openpi, bit-exact). If the request has fewer or more images, the server returns 400. The server does not serve masked or padded cameras. |
| `prompt` | `str` | The task instruction. The server builds the pi0.5 prompt of lerobot: `Task: <prompt>, State: b0 ... b31;\nAction: ` |
| `state` | `[float]` <= 32, optional | The proprioceptive state, ALREADY normalized to [-1, 1]. The server zero-pads it and discretizes it into 256 bins for the prompt. The default is zeros. |
| `tokens` | `[int]` <= `PI05_TOKEN_LEN` (224), optional | Pre-tokenized PaliGemma ids. These ids bypass the gated tokenizer. The server right-pads them with `<pad>` = 0. |
| `num_steps` | `int`, optional | Must be equal to `PI05_NUM_STEPS`. The server refuses other values (400). |
| `seed` | `int`, optional | The seed of the initial flow-matching noise. The default is the fixed seeded noise of the model (deterministic policy). |
| `prompt_bucket` | `int`, optional | 32 / 64 / 128 / 224: the prompt bucket for the request. It must hold the prompt. The default is the smallest prompt bucket that holds the prompt. |

- The server masks the pad tokens out of the attention (openpi semantics).
- Thus the output does not change with the padding or with the extra pad keys of the bucket.
- The response (200) contains these fields:
  - `actions`: the action chunk, H x 32 values in the normalized QUANTILES space, zero-padded to 32 dimensions.
  - `action_horizon`, `action_dim`, `normalized: true`, `denoising_steps`.
  - `prompt`, `num_tokens`, `token_len`, `prompt_bucket`, `prompt_truncated`.
  - `images_used`, `images_padded` (0), `original_sizes`, `image_size`, `seed`.
  - `timing_ms` {preprocess, inference, total}. The inference time is `timing_ms.inference`.
- Errors:
  - 400: the input is bad. The response gives the reason.
  - 503: the server did not complete its startup.
  - 500: the device call failed. The response contains the exception text.
- Each request has batch 1.
- One lock serializes the requests.

## Configuration

Set these environment variables before the server starts. The server reads them only at start.

| env | range (default) | fixed |
|---|---|---|
| `PI05_NUM_IMAGES` | 1..4 (2) | at start |
| `PI05_ACTION_HORIZON` | 1..64 (50). The action-row bucket is 32 rows for H <= 32, and 64 rows for a larger H. | at start |
| `PI05_NUM_STEPS` | 1..10 (10). The adaRMS folds depend on the schedule. | at start |
| `PI05_TOKEN_LEN` | <= 224 (224): the buffer of the tokenizer. The server selects the prompt bucket for each request. | at start |
| `PI05_MEGAKERNEL` | `mc` (default on one chip) / `whole` / `expert` / `off` | at start |

- At start, the server refuses these settings with the message of the model:
  - An out-of-range value.
  - A mesh (`TT_MESH_SHAPE` other than 1x1).
  - `PI05_BATCH_SIZES` other than 1.
  - A different layout.
- `whole` / `expert` / `off` select the single-config paths of the earlier releases.
- These single-config paths support only 2 cameras, H = 50 and N = 10.
- The validation of these single-config paths used tt-metal `668c2907575`, not this tree.

## How to change the configuration

- The package has one serve profile, `p150`. Its default configuration is in `tt-model.yaml` (`serve.env`): 2 cameras, an action chunk of 50 actions and 10 flow-matching steps.
- `tt-model serve` has no flag for the environment, and it does not pass the host environment to the container.
- This procedure is the tested method (2026-10-03, image `tt-model/pi05-base-p150:4dd06e9d6fd3`):

1. Write the launch command of the serve profile to a variable with `tt-model serve ... --print`.
2. Change the three `PI05_*` values in the command.
3. Start the container.
4. Examine `GET /info`. It shows the new configuration.

```bash
CMD=$(tt-model serve changh95/pi05-base-p150 --print | grep '^docker run')
CMD=$(echo "$CMD" | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=3/' \
                        -e 's/PI05_ACTION_HORIZON=50/PI05_ACTION_HORIZON=10/' \
                        -e 's/PI05_NUM_STEPS=10/PI05_NUM_STEPS=5/' -e 's/^docker run /docker run --detach /')
eval "$CMD"
curl -s localhost:20000/info | python3 -c "import json,sys; print(json.load(sys.stdin)['megakernel']['program'])"
docker rm -f tt-model-pi05-base-p150-p150      # stop the server
```

Result of the test:

- `/info` showed cameras 3, `action_horizon` 10, `num_steps` 5, `suffix_rows` 32 and 4 device programs for each call.
- `smoke_test.py` passed: actions (10, 32), repeat difference 0.0000, inference 65.93 / 64.71 ms.
- With `PI05_NUM_IMAGES=5`, the container stopped at start (exit code 3).
  - The message was `pi0.5 megakernel refused: cameras = 5 (compiled: 1, 2, 3, 4) [PI05_NUM_IMAGES=5, PI05_ACTION_HORIZON=50, PI05_NUM_STEPS=10]`.

Served latency of six configurations:

- The first row (the default configuration) is from image `4dd06e9d6fd3`. The other rows are from image `36f651704bf1`.

- The request is the request of the model card. Image `4dd06e9d6fd3` has the same packages and libraries as `36f651704bf1` (only the serve profile changed).
- With 3 or 4 cameras, the request repeated the two images of the card.
- The values are the medians of warm requests.


| Cameras | H | N | Device programs for each call | Prompt bucket | Inference ms | Total ms | Requests |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 50 | 10 | 3 | 224 | 56.40 | 57.63 | 100 |
| 1 | 50 | 10 | 3 | 224 | 44.33 | 45.08 | 30 |
| 3 | 50 | 10 | 4 | 224 | 76.16 | 77.72 | 30 |
| 4 | 50 | 10 | 4 | 224 | 96.28 | 98.37 | 30 |
| 2 | 10 | 10 | 3 | 224 | 53.43 | 54.58 | 30 |
| 2 | 10 | 5 | 3 | 224 | 45.86 | 47.00 | 30 |

## Caveats

- **Gated tokenizer.**
  - `google/paligemma-3b-pt-224` needs an HF token for an account that accepted the Gemma terms.
  - `tt-model serve` mounts `~/.cache/huggingface`, which contains the token file.
  - If the token is not available, the boot fails with the default `PI05_TOKENIZER_REQUIRED=1`.
- **State.** The proprioceptive state goes into the model only through the prompt (pi0.5 semantics).
- **Megakernel limits.**
  - Single p150a (11 x 10 worker grid, Tensix dispatch).
  - Batch 1.
  - 224 x 224 images.
  - The device must open with the 64 KiB worker-L1 cut (`PI05_DEVICE_PARAMS`). The server does this.
  - One live model for each device.
  - With 4 cameras, the K / V caches stay in DRAM.
- **Accuracy limits** (see the card):
  - One input-sensitive trajectory of the 320-set matrix (2 cameras, a 224-token prompt, H = 64, N 5-10).
  - The expert-oracle residual at one action row.
  - `pi05_base` is a base checkpoint. Fine-tune it before you use its actions on an arbitrary robot.
- **Host RAM.**
  - The server materializes the fp32 checkpoint (14.5 GB) on the host while it builds the model.
  - The server releases the checkpoint after the conversion (`PI05_FREE_HOST_WEIGHTS=1`).
  - Make sure that the host has about 20 GB of RAM for the peak.
