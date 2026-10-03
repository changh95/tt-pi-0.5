"""Write the card: block (description + quickstart) of the mc staging tt-model.yaml from measured files only.
usage: card_mc.py --bench BENCH_JSON --image TAG --check IMG_CHECK_JSON [--perf PERF_JSON] [--served SERVED_JSON]
Every number below is read from a file named in SRC (asserted where the file and a fixed value must agree)."""
import argparse, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)  # image bench, default profile (c2 H50 N10), 100 warm requests
ap.add_argument("--image", required=True)
ap.add_argument("--check", required=True)  # img_check.json of the image
ap.add_argument("--profiles", required=True)  # dir with bench-<profile>-*.json of the validate run (30 requests each)
ap.add_argument("--source-commit", required=True)
ap.add_argument("--yaml", default="/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc/tt-model.yaml")
a = ap.parse_args()

MK = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel"
W = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/wp6"
LM = "/home/deepgadget/experiments/gr00t/libero_eval/pi05/tt_mc_matrix"
SRC = {}

bench = json.load(open(a.bench)); SRC["bench"] = a.bench
assert bench["n"] == 100 and bench["identical_actions_all"]
st = bench["stats"]
inf, p90, tot, wall = st["inference"]["median"], st["inference"]["p90"], st["total"]["median"], st["client_wall"]["median"]
prev = json.load(open(f"{MK}/publish_p2/results/bench-c2-mkp2-20261001-131455.json"))["stats"]
assert round(prev["inference"]["median"], 2) == 55.84
prev_inf, prev_tot = prev["inference"]["median"], prev["total"]["median"]
fr = bench["first_response"]

chk = json.load(open(a.check)); SRC["check"] = a.check
g = chk["golden"]; assert g["all_inputs_equal"] and g["n"] == 8
bit = chk["bitid"]
assert chk["refuse_all_ok"]

T = json.load(open(f"{W}/table.json"))
assert T["sets_evaluated"] == T["sets_expected"] == 320 and not T["missing_control"]
pc = T["pass_counts"]
assert pc["A2"] == "314/320" and pc["A4"] == "286/288"
a2 = sorted([c for c in T["failing_cells"] if c["gate"] == "A2"], key=lambda c: c["N"])
a4 = [c for c in T["failing_cells"] if c["gate"] == "A4"]
n1 = T["n1_a4"]
GI = json.load(open(f"{W}/gpu_inc_c2_L224_H64.json"))
gpu_bf16 = {int(N): [s["pcc"] for s in v["seeds"] if s["seed"] == 5][0] for N, v in GI["arms"]["bf16"]["per_N"].items()}
mk_s5 = {c["N"]: [s["pcc"] for s in c["seeds"] if s["seed"] == 5][0] for c in a2}
others = min(s["pcc"] for c in a2 for s in c["seeds"] if s["seed"] != 5)  # over the 6 failing (N = 5-10) sets

P = json.load(open(f"{LM}/paired_vs_gpu.json"))
LS = {k: json.load(open(f"{LM}/{k}_summary.json")) for k in P}
for k in P:
    assert LS[k]["episodes"] == 100 and not LS[k]["errors"] and LS[k]["success"] == P[k]["tt_success"]

f = lambda x, d=2: f"{x:.{d}f}"
import glob
PROF = [("c2-h50-n10", None, 2, 50, 10), ("c1-h50-n10", "c1-h50-n10", 1, 50, 10), ("c3-h50-n10", "c3-h50-n10", 3, 50, 10),
        ("c4-h50-n10", "c4-h50-n10", 4, 50, 10), ("c2-h10-n10", "c2-h10-n10", 2, 10, 10), ("c2-h10-n5", "c2-h10-n5", 2, 10, 5)]
prof_rows = []
for label, pname, cams, H, N in PROF:
    if pname is None:
        b = bench
    else:
        fs = glob.glob(f"{a.profiles}/bench-{pname}-*.json"); assert len(fs) == 1, fs
        b = json.load(open(fs[0])); SRC[pname] = fs[0]
        assert b["identical_actions_all"] and b["n"] == 30
    fr_ = b["first_response"]
    assert (fr_["images_used"], fr_["action_horizon"], fr_["denoising_steps"]) == (cams, H, N), (label, fr_)
    tag = "c2-default-c2" if pname is None else pname
    fi = glob.glob(f"{a.profiles}/info-{tag}-*.json"); assert len(fi) == 1, fi
    info = json.load(open(fi[0])); SRC[f"info {tag}"] = fi[0]
    ops = info["megakernel"]["program"]["device_ops_per_call"]
    assert info["megakernel"]["backend"] == "mc" and ops == (4 if cams >= 3 else 3), (tag, ops)
    prof_rows.append(f"| `{label}`{' (default)' if pname is None else ''} | {cams} | {H} | {N} | {ops} | {fr_['prompt_bucket']} | **{f(b['stats']['inference']['median'])}** | "
                     f"{f(b['stats']['total']['median'])} | {b['n']} |")
prof_md = "\n".join(prof_rows)

# ---------------------------------------------------------------- tables
libero_rows = "\n".join(
    f"| {k[1]} | {k.split('_n')[1]} | **{P[k]['tt_success']} / 100** | {P[k]['gpu_success']} / 100 | "
    f"{len(P[k]['tt_only'])} / {len(P[k]['gpu_only'])} | {f(LS[k]['policy_latency']['mean_ms'], 1)} ms | {f(P[k]['gpu_latency_mean_ms'], 1)} ms |"
    for k in ["c2_n10", "c2_n5", "c2_n1", "c1_n10", "c1_n5", "c1_n1"])

bit_rows = "\n".join(
    f"| {r['cameras']} | {r['preset'][1]} ({r['n_tokens']} real tokens) | {r['preset'][2]} (H = {r['H']}) | {r['N']} | "
    f"{r['device_ops_per_call']} | {'DRAM' if r['kv_dram'] else 'L1'} |"
    for r in (bit[c] for c in ["c1", "c2", "c3", "c4"]))

RT = "/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/wp6/perf/out/RELEASE_TABLE.md"
rt = open(RT).read(); SRC["perf"] = RT
body = rt[rt.index("## 10 denoising steps"):rt.index("## Build log")].strip()
assert "Footnote:" in body and "†" in body
body = body.replace("## 10 denoising steps (all 32 presets)", "**10 denoising steps (all 32 presets)**").replace(
    "## 2 cameras at 1 and 5 denoising steps", "**2 cameras at 1 and 5 denoising steps**")
intro = rt.split("\n\n")[1].strip()  # the method line under the title
perf_md = ("Device replay latency per preset: the trace replay of a call's programs, measured in-process on the release code "
           "(L = prompt bucket in tokens, S = action-row bucket; the full source table with the build log the † refers to is "
           "[`PERF_PRESETS.md`](PERF_PRESETS.md)). " + intro + "\n\n" + body)

a2_tab = ("| steps | " + " | ".join(str(N) for N in sorted(mk_s5)) + " |\n|---|" + "---|" * len(mk_s5) + "\n"
          "| openpi GPU bf16 vs fp32 | " + " | ".join(f(gpu_bf16[N], 3) for N in sorted(mk_s5)) + " |\n"
          "| this megakernel vs fp32 | " + " | ".join(f(mk_s5[N], 3) for N in sorted(mk_s5)) + " |")

desc = f"""Physical Intelligence's pi-0.5 vision-language-action policy (lerobot/pi05_base: SigLIP + Gemma-2B VLM + Gemma-300M flow-matching action expert) running entirely on one Tenstorrent Blackhole p150a. One image serves 1-4 cameras, action chunks of 1-64 steps and 1-10 flow-matching steps (a model per configuration, chosen at server start by serve profile or environment); each request runs in the smallest of four prompt buckets (32 / 64 / 128 / 224 tokens) that holds its prompt. Every request runs the whole model as three persistent custom-kernel programs (`ttnn.generic_op` megakernels on 110 Tensix cores; four with 3-4 cameras): VISION (SigLIP + projector), PREFIX (language embedding + Gemma-2B prefill writing the K / V caches) and EXPERT (the N-step x 18-layer action expert with its adaRMS time conditioning, the action in / out projections and the Euler steps), replayed from one Metal trace per prompt bucket. The default profile (2 cameras, 50 actions, 10 steps, a ~143-token prompt) answers in {f(inf,1)} ms (`timing_ms.inference`).

Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) ·
Paper: [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) ·
Upstream code: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) ·
Port: [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5)
"""

quick = f"""### Run with tt-cli

```bash
tt serve changh95/pi05-base-p150
printf '{{"images":["%s","%s"],"prompt":"pick up the cube","state":[0.1,-0.2,0.3,0,0,0,0.5,-0.5]}}' \\
  "$(base64 -w0 media/sample_base.png)" "$(base64 -w0 media/sample_wrist.png)" > req.json
curl -s localhost:20000/predict -H 'Content-Type: application/json' -d @req.json
tt model stop changh95/pi05-base-p150
```

- `POST /predict` takes:
  - `images`: exactly as many base64 PNG/JPEG as the server's cameras (default 2), ordered `[base/exterior, wrist, ...]`. Masked or missing cameras are refused: serve a model with that many cameras instead;
  - `prompt` (task text) or `tokens` (pre-tokenised PaliGemma ids, ≤224 real tokens);
  - optional `state`: ≤32 floats already normalised to [-1, 1] (default zeros);
  - optional `seed`: the initial flow-matching noise (the default is fixed, so the output is deterministic);
  - optional `prompt_bucket` (32 / 64 / 128 / 224): run in that bucket instead of the smallest one that holds the prompt.
- `GET /health`, `GET /info` (`/info` names the configuration, the kernel digest and the device ops per call).

### Choosing cameras, horizon and steps

A model is built for one (cameras, horizon H, steps N) at server start; the prompt bucket is picked per request. Serve profiles (`tt-model profiles changh95/pi05-base-p150`):

| Profile | Cameras | H | N |
|---|---:|---:|---:|
| `c2-h50-n10` (default) | 2 | 50 | 10 |
| `c1-h50-n10` | 1 | 50 | 10 |
| `c3-h50-n10` | 3 | 50 | 10 |
| `c4-h50-n10` | 4 | 50 | 10 |
| `c2-h10-n10` (LIBERO shape) | 2 | 10 | 10 |
| `c2-h10-n5` | 2 | 10 | 5 |

```bash
tt-model serve changh95/pi05-base-p150 --profile c1-h50-n10   # one camera
tt-model serve changh95/pi05-base-p150 --profile c2-h10-n5    # 10 actions, 5 flow-matching steps
```

Any other combination in range is three environment variables of the container: `PI05_NUM_IMAGES` (1..4), `PI05_ACTION_HORIZON` (1..64), `PI05_NUM_STEPS` (1..10). An out-of-range value refuses at start with a message that names it; so do a mesh, `PI05_BATCH_SIZES` other than 1, and `PI05_TOKEN_LEN` > 224. In Python (the code in `code/`):

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
{{"actions": [[{', '.join(f'{x:.4f}' for x in fr['actions_head'][0])}, ...], ...],
 "action_horizon": {fr['action_horizon']}, "action_dim": {fr['action_dim']}, "normalized": true, "denoising_steps": {fr['denoising_steps']},
 "num_tokens": {fr['num_tokens']}, "token_len": {fr['token_len']}, "prompt_bucket": {fr['prompt_bucket']}, "prompt_truncated": false,
 "images_used": {fr['images_used']}, "images_padded": 0, "image_size": [224, 224], "seed": null,
 "timing_ms": {{"preprocess": {fr['timing_ms']['preprocess']}, "inference": {fr['timing_ms']['inference']}, "total": {fr['timing_ms']['total']}}}}}
```

- `actions` is H × 32 in lerobot's **normalised** QUANTILES action space, zero-padded to 32 dims. Denormalise with `(a+1)*(q99-q01)/2+q01` using your own dataset's stats, then slice to your action dim (e.g. the first 7 for LIBERO).
- Images are squash-resized to 224×224 and normalised as openpi does (`x * float32(1/255) * 2 - 1`, bit-exact to openpi's PyTorch inputs). `state` is discretised into 256 bins inside the prompt (`Task: <prompt>, State: b0 … b31;\\nAction: `); the pad tokens are masked out of attention (openpi semantics), so the output does not depend on the padding.

### Demo

| Base camera (`media/sample_base.png`, synthetic) | Wrist camera (`media/sample_wrist.png`, synthetic) |
|:---:|:---:|
| ![](media/sample_base.png) | ![](media/sample_wrist.png) |

The end of this card has a LIBERO closed-loop demo of the same code, with the LIBERO fine-tune swapped in.

### What runs where

| Part of one request | Where it runs | Device ops per call |
|---|---|---:|
| image decode / resize / normalisation, im2col of the patches, prompt building and tokenisation, the attention-mask and RoPE rows, the initial noise, the host-to-device copies, the readback | host (input / output formatting only) | 0 |
| SigLIP (patch embedding + 27 layers + post-LN) and the projector on the cameras | VISION: one persistent `ttnn.generic_op` on 110 cores per group of ≤ 2 cameras | 1 (2 with 3-4 cameras) |
| language embedding, Gemma-2B prefill (18 layers) writing 18 bfp8 K / V caches | PREFIX: one persistent `ttnn.generic_op` | 1 |
| time MLP / adaRMS conditioning, action in-projection, N Euler steps × 18 Gemma-300M expert layers, action out-projection | EXPERT: one persistent `ttnn.generic_op` | 1 |

The three (four) programs are captured in one Metal trace per prompt bucket; a request writes its inputs into fixed device buffers and replays that trace. The 32 compiled presets are cameras {{1, 2, 3, 4}} × prompt buckets {{32, 64, 128, 224}} × suffix buckets {{32, 64}} action rows (H ≤ 32 runs in the 32-row bucket); N is folded into the expert's weights at construction. The K / V caches stay in L1 for 1-3 cameras and go to DRAM for 4. The image's build checks the 32 presets, every C++ header the kernels include, and the kernel digest the device gates ran on (`{chk['kernel_digest']}`).

### Accuracy

Validated on one p150a with tt-metal `main` @ `f856a38a361`, batch 1, every expert matmul at HiFi3. The full matrix: 32 presets × N 1..10 = 320 (preset, steps) sets, each against the fp32 torch reference on 6 padded prompts (A2), plus an fp32 expert loop fed the device's own K / V caches (A4), the K / V caches (A3, A6), bit-identical replays and an independent cross-check.

| Check | Result |
|---|---|
| vs the openpi GPU policy (`lerobot/pi05_libero`, 8 real LIBERO observations, 2 cameras, H = 10, N = 10, PCC over the 7 action dims), measured in this image | mean **{f(g['pcc7_mean'],6)}**, min **{f(g['pcc7_min'],6)}**; the image's host inputs are bit-identical to openpi's (images, tokens, mask, noise) |
| whole call vs fp32 reference, gate PCC min ≥ 0.95 and mean ≥ 0.98 over 6 prompts (A2) | **{pc['A2']}** sets pass (see the limitation below) |
| expert vs an fp32 expert loop on the device's K / V, gate ≥ 0.999 (A4, N ≥ 2) | **{pc['A4']}** sets pass; N = 1 recorded, not gated |
| K / V caches, replay identity, cross-check | {pc['A3']}, {pc['replay']}, {pc['crosscheck']} |
| negative controls (perturbed inputs must fail the gates) | fail as required on {T['control_failed_as_required']['A2']} (A2) / {T['control_failed_as_required']['A4']} (A4) |

### Speed

Served by this image over HTTP, default profile (2 × 224² images, the card's prompt = {fr['num_tokens']} tokens -> the {fr['prompt_bucket']}-token bucket, H = 50, N = 10), median of 100 warm requests: **{f(inf)} ms** `timing_ms.inference` (p90 {f(p90)}), {f(tot)} ms `timing_ms.total`, {f(wall)} ms client wall. The previous release (one fused op for this shape only, image `6fb244df57ff`, same benchmark) was {f(prev_inf)} / {f(prev_tot)} ms.

Every serve profile, served by this image (the card's request; 3-4 cameras repeat the two card images; `timing_ms` medians, warm requests):

| Profile | Cameras | H | N | Device programs per call | Prompt bucket | inference ms | total ms | requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{prof_md}

{perf_md}

### LIBERO closed loop (TT vs GPU)

`lerobot/pi05_libero` through this code (openpi's `pi05_libero` conventions: H = 10, openpi norm stats, openpi's client and LIBERO loop), libero_spatial 10 tasks × init states 0-9, paired episode by episode with openpi's PyTorch policy on an RTX 5090 (same client, init states and per-call noise seeds). Latency = server-side policy call (host inputs + device + readback), mean.

| Cameras | N | p150a | RTX 5090 | discordant pairs (TT-only / GPU-only) | p150a latency | RTX 5090 latency |
|---:|---:|---:|---:|---:|---:|---:|
{libero_rows}

Every discordant split has exact McNemar p = 1. With 1 camera (the wrist image dropped) both backends collapse (0-2 / 100): pi05_libero needs the wrist view, so those rows are a record, not an accuracy signal.

### Limitations

- **One input-sensitive trajectory (A2).** All 6 failing A2 sets are one input: 2 cameras, a full 224-token prompt, 64 action rows (H = 64), matrix seed 5, at N = 5-10. openpi's GPU policy in bf16 also fails there from N = 6, but this megakernel is lower at every N; the other five prompts of that preset stay ≥ {f(others,4)} at those step counts:

{a2_tab}

  Rounding the reference's own K / V to bf16 / bfp8 leaves it ≥ 0.99994 and a higher VLM matmul fidelity does not help: the trajectory amplifies the prefix's reduced-precision error. Closed-loop LIBERO is the end-to-end check.
- **A4 at one action row.** The two A4 misses at N ≥ 2 have H = 1: 1 camera / 64-token bucket / N = 5 ({f(a4[0]['min'],6)}) and 2 cameras / 128-token bucket / N = 3 ({f(a4[1]['min'],6)}), against 0.999. At N = 1, {n1['below_0.999']} of {n1['n_values']} (preset, H) values are below 0.999, the lowest {f(n1['worst']['min'],6)} (1 camera, 32-token bucket, H = 1). HiFi4 on every expert matmul fixes both cells but is slower on some shapes, so it is not used; per-preset expert fidelity is a follow-up.
- Batch 1, 224 × 224 images, 1-4 real cameras (no masked slots), ≤ 224 real prompt tokens, H ≤ 64, N ≤ 10; N, H and the camera count are fixed per server.
- One live model per device (a second one is refused until the first is closed).
- The worker grid is 11 × 10 (110 cores) with Tensix dispatch; the 12 × 10 Ethernet-dispatch grid is a later release.
- The prompt tokenizer `google/paligemma-3b-pt-224` is **gated** (Gemma terms): accept the terms and `hf auth login` before `tt serve`, or send `tokens`.
- Outputs are normalised actions of the **base** checkpoint, not fine-tuned for any task; the real-input rows (openpi golden, LIBERO) use `lerobot/pi05_libero` through the same code.
- Not an OpenAI-compatible API (`GET /v1/models` is a stub).
- The single-config paths of the previous releases (`PI05_MEGAKERNEL=whole` / `expert` / `off`, the mesh layouts) are still in `code/` as comparators; they were validated on tt-metal `668c2907575`, not on this image's tree.

### Licensing

- Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base), [Gemma Terms of Use](https://ai.google.dev/gemma/terms). They are not redistributed here but fetched into your HF cache. The tokenizer [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) is gated under the same terms.
- Port and serving code (`code/`): Apache-2.0 headers, distributed under the same Gemma terms, from [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ [`{a.source_commit[:7]}`](https://github.com/changh95/tt-pi-0.5/commit/{a.source_commit}).
"""

def block(text):
    return "\n".join(("    " + l) if l.strip() else "" for l in text.rstrip("\n").split("\n"))

y = open(a.yaml).read()
i = y.index("\ncard:\n")
y = y[:i] + "\ncard:\n  description: |\n" + block(desc) + "\n  quickstart: |\n" + block(quick) + "\n"
open(a.yaml, "w").write(y)
import yaml
d = yaml.safe_load(open(a.yaml))
assert d["card"]["quickstart"].startswith("### Run with tt-cli")
print("card written; sources:", json.dumps(SRC))
print("bitid table (for the report):\n" + bit_rows)
