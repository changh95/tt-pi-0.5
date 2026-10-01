"""Write the card: block of tt-model.yaml (description + quickstart) from the measured files (bench JSON arg 1, image tag arg 2)."""
import json, sys, yaml
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"
MK = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel"
R = f"{MK}/integrate_p2/results"
bench = json.load(open(sys.argv[1])); img = sys.argv[2]
st = bench["stats"]; fr = bench["first_response"]
assert bench["identical_actions_all"] and bench["n"] == 100
inf, tot, wall = st["inference"]["median"], st["total"]["median"], st["client_wall"]["median"]
p90 = st["inference"]["p90"]
t = fr["timing_ms"]
# previous image (phase 1, fe0d2e3d68a7), SAME benchmark cycle (c2) as the current figures
prev = json.load(open(f"{MK}/publish_p1/results/bench-c2-mkp1-20260930-172411.json"))["stats"]
prev_inf, prev_tot = prev["inference"]["median"], prev["total"]["median"]
assert (round(prev_inf, 2), round(prev_tot, 2)) == (70.81, 72.16), (prev_inf, prev_tot)
G = json.load(open(f"{R}/G_gates_r1.json")); sg = G["seed_gate_base"]["summary"]
assert sg["n"] == 32 and sg["pass_vs_off"] == 32 and sg["pass_vs_expert"] == 31
kv_b, kv_l = G["p23_base"], G["p23_libero"]
A = {a: json.load(open(f"{R}/A_{a}_base_r1.json")) for a in ("whole", "expert", "off")}
rep = {a: A[a]["replay_ms"]["median"] for a in A}; assert all(len(A[a]["replay_ms"]["all"]) == 30 for a in A)
AL = {a: json.load(open(f"{R}/A_{a}_libero_r1.json")) for a in ("whole", "expert")}
SS = json.load(open(f"{R}/S_struct.json"))
wb = SS["whole_base"]; assert wb["n_sessions"] == 21 and wb["ops_per_session"] == [1] and wb["core_counts"] == ["110"]
dev_whole = wb["device_ms"]["median"]; dev_exp = SS["expert_base"]["session_device_ms_median"]; dev_off = SS["off_base"]["session_device_ms_median"]
n_exp, n_off = SS["expert_base"]["n_sessions"], SS["off_base"]["n_sessions"]
assert SS["expert_base"]["ops_per_session"] == [892] and SS["off_base"]["ops_per_session"] == [2551]
pad = json.load(open(f"{MK}/p2/gates/results/g1_pcc_whole_base.json"))
assert pad["pad_ids_invisible_bit_identical"] and pad["stamp"]["megakernel_program"]["kernel_digest"] == "4aa02cdf21ed0c94"
M = json.load(open(f"{R}/M_size_base.json"))
L = json.load(open("/home/deepgadget/experiments/gr00t/libero_eval/pi05/megakernel-p2/tt_spatial_summary.json"))
ol = json.load(open("/home/deepgadget/experiments/gr00t/libero_eval/pi05/megakernel-p2/openloop_pcc_mkp2.json"))["summary"]
assert L["successes"] == 99 and L["paired_vs_gpu_init0_4"]["tt_success_init0_4"] == 49
lib_ms = L["server_infer_ms"]["median"]
GW = {a: json.load(open(f"{R}/A_{a}_libero_r1.json")) for a in ("whole", "expert", "off")}
gm = {a: (GW[a]["pcc7_mean"], GW[a]["pcc7_min"]) for a in GW}
assert abs(gm["whole"][0] - ol["mean_pcc_7"]) < 1e-6 and abs(gm["whole"][1] - ol["min_pcc_7"]) < 1e-6
fp = M["footprint"]; assert fp["pass"] and fp["ring"] == 136192
s7 = [r for r in G["seed_gate_base"]["rows"] if r["tag"] == 707]; assert len(s7) == 1; s7 = s7[0]
assert s7["margin_vs_expert"] < 0 < s7["margin_vs_off"] and [r["tag"] for r in G["seed_gate_base"]["rows"] if r["margin_vs_expert"] < 0] == [707]
desc = """Physical Intelligence's pi-0.5 vision-language-action policy (lerobot/pi05_base: SigLIP + Gemma-2B VLM + Gemma-300M flow-matching action expert) running entirely on one Tenstorrent Blackhole p150a: two camera images + a task prompt in, a 50-step chunk of normalised actions out, in %.1f ms per request (`timing_ms.inference`). The whole model runs as ONE fused op per request: a single persistent custom-kernel program (`ttnn.generic_op` megakernel on 110 Tensix cores) that executes SigLIP on both cameras, the projector, the language embedding, the VLM prefill writing the K / V caches, the whole 10-step flow-matching loop of the action expert (18 layers per step), the action in / out projections and the Euler updates. It is captured in a Metal trace whose replay holds exactly that one device op; the host only formats inputs and outputs.

Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) ·
Paper: [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) ·
Upstream code: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) ·
Port: [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5)
""" % inf
acts = ", ".join("%.4f" % a for a in fr["actions_head"][0])
q = f"""### Run with tt-cli

```bash
tt serve changh95/pi05-base-p150
printf '{{"images":["%s","%s"],"prompt":"pick up the cube","state":[0.1,-0.2,0.3,0,0,0,0.5,-0.5]}}' \\
  "$(base64 -w0 media/sample_base.png)" "$(base64 -w0 media/sample_wrist.png)" > req.json
curl -s localhost:20000/predict -H 'Content-Type: application/json' -d @req.json
tt model stop changh95/pi05-base-p150
```

- `POST /predict` takes:
  - `images`: 1-2 base64 PNG/JPEG, ordered `[base/exterior, wrist]`;
  - `prompt` (task text) or `tokens` (pre-tokenised PaliGemma ids, ≤224);
  - optional `state`: ≤32 floats already normalised to [-1, 1] (default zeros);
  - optional `seed`: the initial flow-matching noise (the default is fixed, so the output is deterministic).
- `GET /health`, `GET /info` (`/info` names the megakernel backend and its kernel digest).

### Response

```json
{{"actions": [[{acts}, ...], ...],
 "action_horizon": 50, "action_dim": 32, "normalized": true, "denoising_steps": 10,
 "prompt": "Task: pick up the cube, State: ...;\\nAction: ", "num_tokens": {fr["num_tokens"]}, "token_len": 224, "prompt_truncated": false,
 "images_used": 2, "images_padded": 0, "original_sizes": [[224, 224], [224, 224]], "image_size": [224, 224], "seed": null,
 "timing_ms": {{"preprocess": {t["preprocess"]}, "inference": {t["inference"]}, "total": {t["total"]}}}}}
```

- `actions` is 50 × 32 in lerobot's **normalised** QUANTILES action space, zero-padded to 32 dims. Denormalise with `(a+1)*(q99-q01)/2+q01` using your own dataset's stats, then slice to your action dim (e.g. the first 7 for LIBERO).
- Images are squash-resized to 224×224. `state` is discretised into 256 bins inside the prompt (`Task: <prompt>, State: b0 … b31;\\nAction: `), and the prompt is right-padded to 224 tokens. The pad tokens are masked out of attention (openpi semantics), so the output does not depend on the padding.

### Demo

| Base camera (`media/sample_base.png`, synthetic) | Wrist camera (`media/sample_wrist.png`, synthetic) |
|:---:|:---:|
| ![](media/sample_base.png) | ![](media/sample_wrist.png) |

The end of this card has a LIBERO closed-loop demo of the same code path, with the LIBERO fine-tune swapped in.

### What runs where

| Part of one request | Where it runs | Device ops per trace replay |
|---|---|---:|
| image resize / normalisation, im2col of the patches, prompt building and tokenisation, the attention-mask and RoPE rows, the initial noise, the host-to-device copies, the readback | host (input / output formatting only) | 0 |
| SigLIP on both cameras (patch embedding + 27 layers + post-LN), projector, language embedding, Gemma-2B VLM prefill (18 layers, writing 18 bf8 K / V caches), action in-projection, 10 Euler steps × 18 Gemma-300M expert layers, action out-projection | **ONE persistent `ttnn.generic_op`** on 110 cores (`code/.../tt/megakernel/pe_program.py`, kernels `tt/megakernel/kernels_p2/whole_{{brisc,ncrisc,trisc}}.cpp`, which include the expert-loop kernels `tt/megakernel/kernels/mk_*`) | **1** |

The image's build checks that every C++ header the kernels include is in the image and that the kernel digest is the one the device gates ran on (`4aa02cdf21ed0c94`). Kernel-config ring footprint of the program: {fp["total_load"]:,} B of the {fp["ring"]:,} B ring (offline mock-cluster compile). The two earlier paths stay in the code only as comparators: `PI05_MEGAKERNEL=expert` (the previous image: SigLIP / VLM prefix as traced stock TT-NN ops, the expert loop as one generic_op; 892 ops per replay) and `PI05_MEGAKERNEL=off` (stock TT-NN ops plus 3 custom generic_op programs; 2,551 ops per replay).

### Accuracy and speed

Measured 2026-10-01 on one p150a (AICLK 1350 MHz), tt-metal `668c2907575`, batch 1, 10 flow-matching steps. "Previous" is the 2026-09-30 image (`fe0d2e3d68a7`), the phase-1 path that is still in the code as `PI05_MEGAKERNEL=expert`. The two served rows (and the 100 repeated requests) come from this image's own benchmark. The replay, device-time, accuracy and determinism rows were measured in-process on the same code, with the paths alternated in the same session ([`docs/megakernel/integrate_p2/results/`](https://github.com/changh95/tt-pi-0.5/tree/main/docs/megakernel/integrate_p2/results); the padding check in `docs/megakernel/p2/gates/results/`). The LIBERO row is the closed-loop run at the end of this card; the GPU rows are in [`GPU_COMPARISON.md`](GPU_COMPARISON.md).

| Metric | Value |
|---|---:|
| Inference, served over HTTP by this image (2×224×224, 224 tokens, H = 50; median of 100 warm requests) | **{inf} ms** device (`timing_ms.inference`, p90 {p90}) · {tot} ms end-to-end (`timing_ms.total`) · {wall} ms client wall per 50-step chunk |
| Previous image (`fe0d2e3d68a7`), same request, same benchmark cycle | {prev_inf:.2f} ms device · {prev_tot:.2f} ms end-to-end |
| In-process trace replay (host wall, median of 30) | {rep["whole"]:.1f} ms (previous path {rep["expert"]:.1f} ms; `off` {rep["off"]:.1f} ms) |
| Device time per replay (device profiler) | **{dev_whole:.2f} ms as one op** (median of 21 profiled replay sessions); previous path {dev_exp:.2f} ms over 892 ops, `off` {dev_off:.2f} ms over 2,551 ops (sum of the op durations, median of {n_exp} / {n_off} profiled sessions) |
| PCC vs the openpi GPU golden (8 real LIBERO observations, same code with `lerobot/pi05_libero`, prompt right-padded to 32 tokens, H = 10, 7 action dims) | mean **{gm["whole"][0]:.6f}**, min **{gm["whole"][1]:.6f}** (previous path {gm["expert"][0]:.6f} / {gm["expert"][1]:.6f}; `off` {gm["off"][0]:.6f} / {gm["off"][1]:.6f}) |
| Whole call vs the fp32 torch reference of the whole model on the same inputs (served shape, `pi05_base`, 32 random inputs with 1-224 real tokens) | mean **{sg["mean"]["whole"]:.5f}**, min {sg["min"]["whole"]:.5f}; closer than `off` on **32 / 32** (`off` mean {sg["mean"]["off"]:.5f}, min {sg["min"]["off"]:.5f}) and than the previous path on 31 / 32 (previous mean {sg["mean"]["expert"]:.5f}, min {sg["min"]["expert"]:.5f}) |
| Per-layer VLM K / V (18 layers × K, V × 8 inputs) vs the fp32 reference's own cache | closer than the TT-NN caches on **288 / 288** at both shapes (min PCC {kv_b["whole_min_pcc"]:.5f} served / {kv_l["whole_min_pcc"]:.5f} LIBERO vs TT-NN {kv_b["ttnn_min_pcc"]:.5f} / {kv_l["ttnn_min_pcc"]:.5f}) |
| Padding is exact | random ids in the padded slots under the same mask give a bit-identical output |
| Determinism | ten trace replays are bit-identical. Alternating prompts and a shape switch reproduce fresh-model outputs bit-for-bit (20 / 20). 20 consecutive processes ran with no hang and gave one output digest. 100 repeated served requests returned identical actions |
| LIBERO-spatial closed loop (`lerobot/pi05_libero`, 10 tasks × 10 init states, same code) | **99 / 100**. Paired init states 0-4: 49 / 50 vs openpi on an RTX 5090 50 / 50. {lib_ms} ms per policy call (previous path 99 / 100 at 66.4 ms) |
| Same forward on an RTX 5090 (same host, the port's torch reference, eager PyTorch, batch 1, incl. H2D/D2H; measured 2026-09-14) | fp32 strict 144.1 ms; bf16 / fp16 autocast 121.5 / 123.7 ms; bf16 weights resident 99.9 ms. The p150a is {99.869/inf:.2f}-{144.066/inf:.2f}× faster than every eager row. The best whole-request `torch.compile` run is 46.6 ms (the GPU is {inf/46.62:.2f}× faster) |

### Caveats

- Fixed geometry:
  - 1-2 images squashed to 224×224; a missing wrist camera is padded with a black image, which is still attended;
  - prompt ≤224 tokens, right-padded, pads masked;
  - `state` ≤32 floats;
  - batch 1;
  - `num_steps` is baked in at 10, and a per-request value is rejected.
- The megakernel is a single-p150a, batch-1 program built for 2 cameras. It needs the 64 KiB worker-L1 cut, which the server applies when it opens the device. It refuses by name at startup on a mesh, with batch sizes > 1, with bf16 K/V caches, with step counts ≠ 10, or with `PI05_NUM_IMAGES` ≠ 2. On a mesh an unset `PI05_MEGAKERNEL` falls back to `off`.
- Accuracy against fp32 on random inputs is limited by the bf16 / bfp8 arithmetic of the vision / language prefix, so the release gate is "per input at least as close to fp32 as the `off` path", not a fixed floor. One of the 32 seeds (707) is closer under the previous path ({s7["expert"]:.5f} vs {s7["whole"]:.5f}; both closer than `off`, {s7["off"]:.5f}).
- The prompt tokenizer `google/paligemma-3b-pt-224` is **gated** (Gemma terms). Accept the terms on huggingface.co and run `hf auth login` on the host before `tt serve`, or send pre-tokenised `tokens`. The weights themselves are ungated but under the same Gemma terms.
- Outputs are normalised actions of the **base** checkpoint, which is not fine-tuned for any task. `pi05_base` ships no per-feature stats, so its outputs are not usable on an arbitrary robot without fine-tuning. The real-input rows (openpi golden, LIBERO) use the fine-tuned `lerobot/pi05_libero` through the same code, not `pi05_base`.
- Not an OpenAI-compatible API. `GET /v1/models` is a stub so the tt-model ready card does not 404.
- Validated on tt-metal `main` @ `668c2907575` (v0.79.0-dev20260914), single p150a only. The mesh / tensor-parallel path, `PI05_EXPERT_ATTN=ttnn` and batch > 1 exist only on the `PI05_MEGAKERNEL=off` path, and none of them was re-validated after the mask / RoPE fix.
- GPU comparison: the RTX 5090 rows are the 2026-09-14 runs of the torch reference before the mask / RoPE fix. The shapes are the same; the rows were not re-measured. They use fp32 weights + autocast unless stated, no TensorRT, medians of 50 iterations after warm-up, H2D/D2H included. p150a power was not measured, so no efficiency comparison is made. Full table: [`GPU_COMPARISON.md`](GPU_COMPARISON.md).

### Licensing

- Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base), [Gemma Terms of Use](https://ai.google.dev/gemma/terms). They are not redistributed here but fetched into your HF cache. The tokenizer [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) is gated under the same terms.
- Port and serving code (`code/`): Apache-2.0 headers, distributed under the same Gemma terms, from [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ [`821e8c5`](https://github.com/changh95/tt-pi-0.5/commit/821e8c528dfffa0d1d6e73ad6abf181a749b39d9) (`main`, merge of PR #3).
"""
s0 = open(f"{S}/tt-model.yaml").read()
head = s0.split("\ncard:\n", 1)[0]
class D(yaml.SafeDumper): pass
D.add_representer(str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style="|" if "\n" in v else None))
card = yaml.dump({"card": {"description": desc, "quickstart": q}}, Dumper=D, sort_keys=False, allow_unicode=True, width=10**6)
if "--write" in sys.argv:
    open(f"{S}/tt-model.yaml", "w").write(head + "\n" + card)
    d = yaml.safe_load(open(f"{S}/tt-model.yaml"))
    assert d["card"]["description"] == desc and d["card"]["quickstart"] == q
    print("card written", len(desc), len(q))
else:
    print(desc); print(q[q.index("### What runs"):q.index("### Caveats")])
