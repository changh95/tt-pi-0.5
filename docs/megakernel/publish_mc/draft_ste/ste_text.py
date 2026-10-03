# STE (ASD-STE100) prose of the card. Imported by card_ste.py, which supplies the measured values in V.
import re


def perf_md(rt, f):
    body = rt[rt.index("## 10 denoising steps"):rt.index("## Build log")].strip()
    fn = [l for l in body.split("\n") if l.startswith("Footnote:")]; dg = [l for l in body.split("\n") if l.startswith("† ")]
    assert len(fn) == 1 and len(dg) == 1
    m = re.search(r"host 1-min load (\d+)-(\d+); the (\d+) presets also timed on a quiet host agree within ([0-9.]+) ms \((.*) ms here vs the quiet ABAB holds of (\S+) (\S+) / (\S+)\)\.$", fn[0])
    assert m, fn[0]
    lo, hi, npre, within, deltas, day, t1, t2 = m.groups()
    md = re.search(r"Build-to-build se is <= ([0-9.]+) ms at 10 steps; at 1 / 5 steps it reaches ([0-9.]+) ms\.$", dg[0]); assert md, dg[0]
    se10, se15 = md.groups()
    words = {"4": "Four"}
    fn_ste = (f"Notes:\n\n"
              f"- The host had a 1-minute load average of {lo} to {hi} during these measurements.\n"
              f"- {words.get(npre, npre)} presets also had measurements on a quiet host ({day} {t1} and {t2}).\n"
              f"- The differences are {within} ms or less: {deltas} ms.")
    dg_ste = (f"- † A different CPU process was active during one of the two builds. [`PERF_PRESETS.md`](PERF_PRESETS.md) gives the time windows.\n"
              f"- The standard error between the builds is {se10} ms or less at 10 steps. At 1 or 5 steps, it is {se15} ms or less.")
    body = body.replace(fn[0], fn_ste).replace(dg[0], dg_ste)
    body = body.replace("## 10 denoising steps (all 32 presets)", "**10 flow-matching steps (all 32 presets)**").replace(
        "## 2 cameras at 1 and 5 denoising steps", "**2 cameras at 1 and 5 flow-matching steps**")
    intro = rt.split("\n\n")[1].strip()
    mi = re.match(r"Blackhole p150a, trace replay \(blocking\), ms: median of (\d+) replays per build, mean of (\d+) builds \(± se of the build means\)\. 1 batch, H = (\d+) for (\d+) action rows and H = (\d+) for (\d+) \(the device programs depend on the bucket, not H\)\.$", intro)
    assert mi, intro
    nrep, nb, h32, s32, h64, s64 = mi.groups()
    return ("- The next tables show the device time of one trace replay for each preset.\n"
            "- The data is from in-process measurements on the release code.\n"
            "- L is the prompt bucket in tokens. S is the action-row bucket.\n"
            f"- Each value is in ms on the Blackhole p150a. It is the mean of {nb} builds.\n"
            f"- Each build gives the median of {nrep} trace replays. Each trace replay blocks the host until it is complete.\n"
            "- The ± value is the standard error of the build means.\n"
            "- The batch size is 1.\n"
            f"- For {s32} action rows, H is {h32}. For {s64} action rows, H is {h64}. The device programs change with the action-row bucket, not with H.\n"
            "- [`PERF_PRESETS.md`](PERF_PRESETS.md) has the full source table and the build log for the † mark.\n\n" + body)


def description(V, f):
    return f"""This package runs the pi-0.5 vision-language-action policy of Physical Intelligence on one Tenstorrent Blackhole p150a.

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
- With this profile and a prompt of about 143 tokens, the inference time is {f(V['inf'],1)} ms (`timing_ms.inference`).

Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) ·
Paper: [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) ·
Upstream code: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) ·
Port: [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5)
"""


def quickstart(V, f):
    fr, g, pc, T, a4, n1 = V["fr"], V["g"], V["pc"], V["T"], V["a4"], V["n1"]
    return f"""### Run with tt-cli

1. Start the server, send one request and stop the server:

```bash
tt serve changh95/pi05-base-p150
printf '{{"images":["%s","%s"],"prompt":"pick up the cube","state":[0.1,-0.2,0.3,0,0,0,0.5,-0.5]}}' \\
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
CMD=$(echo "$CMD" | sed -e 's/PI05_NUM_IMAGES=2/PI05_NUM_IMAGES=3/' \\
                        -e 's/PI05_ACTION_HORIZON=50/PI05_ACTION_HORIZON=10/' \\
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
{{"actions": [[{', '.join(f'{x:.4f}' for x in fr['actions_head'][0])}, ...], ...],
 "action_horizon": {fr['action_horizon']}, "action_dim": {fr['action_dim']}, "normalized": true, "denoising_steps": {fr['denoising_steps']},
 "num_tokens": {fr['num_tokens']}, "token_len": {fr['token_len']}, "prompt_bucket": {fr['prompt_bucket']}, "prompt_truncated": false,
 "images_used": {fr['images_used']}, "images_padded": 0, "image_size": [224, 224], "seed": null,
 "timing_ms": {{"preprocess": {fr['timing_ms']['preprocess']}, "inference": {fr['timing_ms']['inference']}, "total": {fr['timing_ms']['total']}}}}}
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
- The server puts `state` into the prompt as 256 bins: `Task: <prompt>, State: b0 … b31;\\nAction: `.
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
- The image build also checks the kernel digest that the device gates used (`{V['digest']}`).

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
| The device output against the openpi GPU policy. The test used `lerobot/pi05_libero`, 8 real LIBERO observations, 2 cameras, H = 10 and N = 10. The PCC is over the 7 action dims. This image did the measurement. | Mean **{f(g['pcc7_mean'],6)}**, min **{f(g['pcc7_min'],6)}**. The host inputs of the image are bit-identical to the inputs of openpi (images, tokens, mask, noise). |
| A2: the full call against the fp32 reference on 6 prompts. The gate is PCC min ≥ 0.95 and mean ≥ 0.98. | **{pc['A2']}** sets pass (see the limitation below). |
| A4: the expert against an fp32 expert loop with the K / V caches of the device. The gate is ≥ 0.999 for N ≥ 2. | **{pc['A4']}** sets pass. For N = 1, the card gives the values, but there is no gate. |
| The K / V caches, the replay identity and the cross-check. | {pc['A3']}, {pc['replay']}, {pc['crosscheck']}. |
| Negative controls: inputs with a known error must fail the gates. | They fail as necessary on {T['control_failed_as_required']['A2']} (A2) and {T['control_failed_as_required']['A4']} (A4). |

### Benchmarks

Device time of one trace replay in ms for each preset (the mean of 2 builds; each build gives the median of 60 trace replays):

{V['bench_md']}

- Each value is the device time of one trace replay of all device programs of one request (in-process, release code, batch size 1).
- Served over HTTP with the default configuration (2 cameras, H = 50, N = 10), the median `timing_ms.inference` was **{f(V['inf'])} ms** for 100 warm requests.
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
{V['libero_rows']}

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
  - For the other five prompts of this cell, the PCC is {f(V['others'],4)} or more at N = 5 to 10.
  - At each N from 1 to 10, it is {f(V['others_all'],4)} or more.

{V['a2_tab']}

  - If you round the K / V caches of the reference to bf16 or bfp8, the reference stays at a PCC of 0.99994 or more.
  - A higher fidelity for the VLM matmuls does not help.
  - This trajectory increases the reduced-precision error of the prefix.
  - The closed-loop LIBERO test is the end-to-end check.
- **A4 with one action row.**
  - Two A4 sets fail at N = 2 or more. Both sets have H = 1. The gate is 0.999.
  - The first set has 1 camera, the 64-token bucket and N = 5 ({f(a4[0]['min'],6)}).
  - The second set has 2 cameras, the 128-token bucket and N = 3 ({f(a4[1]['min'],6)}).
  - At N = 1, {n1['below_0.999']} of {n1['n_values']} (preset, H) values are below 0.999.
  - The lowest value is {f(n1['worst']['min'],6)}, with 1 camera, the 32-token bucket and H = 1.
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
  - The code is from [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ [`{V['src'][:7]}`](https://github.com/changh95/tt-pi-0.5/commit/{V['src']}).
"""


def bench_md(rt):
    """ONE table from RELEASE_TABLE.md: rows cameras x action-row bucket x N, columns the 4 prompt buckets (median ms;
    the ± se and † marks stay in PERF_PRESETS.md)."""
    def table(after):
        sec = rt[rt.index(after):]
        lines = [l for l in sec.split("\n") if l.startswith("|")]
        hdr = [c.strip() for c in lines[0].strip("|").split("|")]
        rows = []
        for l in lines[2:]:
            c = [x.strip() for x in l.strip("|").split("|")]
            if not c[0].isdigit():
                break
            rows.append((int(c[0]), {hdr[i]: c[i].split("±")[0].strip() for i in range(1, len(c))}))
        return hdr, rows
    hdr10, n10 = table("## 10 denoising steps")
    hdr15, n15 = table("## 2 cameras at 1 and 5 denoising steps")
    assert hdr10[1:] == hdr15[1:] == ["L32 S32", "L64 S32", "L128 S32", "L224 S32", "L32 S64", "L64 S64", "L128 S64", "L224 S64"]
    out = ["| Cameras | Action chunk (H) | N | Prompt bucket 32 | 64 | 128 | 224 |", "|---:|---|---:|---:|---:|---:|---:|"]
    def row(cams, n, vals):
        for S, hl in (("S32", "H ≤ 32"), ("S64", "H 33-64")):
            out.append(f"| {cams} | {hl} | {n} | " + " | ".join(vals[f"L{L} {S}"] for L in (32, 64, 128, 224)) + " |")
    for cams, vals in n10:
        row(cams, 10, vals)
    for n, vals in n15:
        row(2, n, vals)
    assert len(out) == 2 + 12
    return "\n".join(out)
