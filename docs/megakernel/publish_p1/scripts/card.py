"""Write the card: block of tt-model.yaml (description + quickstart) from the measured files (bench JSON arg 1)."""
import json, sys, yaml
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"
R = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/integrate_p1/results"
bench = json.load(open(sys.argv[1]))
st = bench["stats"]; fr = bench["first_response"]
assert bench["identical_actions_all"] and bench["n"] == 100
inf, tot, wall = st["inference"]["median"], st["total"]["median"], st["client_wall"]["median"]
p90 = st["inference"]["p90"]
t = fr["timing_ms"]
O = json.load(open(f"{R}/O_ip1.json")); rows = O["rows"]; osum = O["summary"]
rng = lambda k: (min(x[k] for x in rows), max(x[k] for x in rows), sum(x[k] for x in rows) / len(rows))
dmin, dmax, dmean = rng("default_vs_fullref"); fmin, fmax, fmean = rng("off_vs_fullref"); omin, omax, _ = rng("oracle_vs_fullref")
assert osum["n"] == 22 and osum["all_pass"]
A = json.load(open(f"{R}/A_default_base.json")); replay = A["replay_ms"]["median"]
L = json.load(open("/home/deepgadget/experiments/gr00t/libero_eval/pi05/megakernel-p1/tt_spatial_summary.json"))
ol = json.load(open("/home/deepgadget/experiments/gr00t/libero_eval/pi05/megakernel-p1/openloop_pcc_mkp1.json"))["summary"]
assert L["successes"] == 99 and L["paired_vs_gpu_init0_4"]["tt_success_init0_4"] == 49
lib_ms = L["server_infer_ms"]["median"]
desc = """Physical Intelligence's pi-0.5 vision-language-action policy (lerobot/pi05_base: SigLIP + Gemma-2B VLM + Gemma-300M flow-matching action expert) running entirely on one Tenstorrent Blackhole p150a: two camera images + a task prompt in, a 50-step chunk of normalised actions out, in %.1f ms on the device. Each request is one Metal trace. The SigLIP tower and the VLM prefill run as traced TT-NN ops. The whole 10-step flow-matching loop of the action expert (18 layers per step, plus the action in / out projections and the Euler updates) runs as ONE persistent custom-kernel program: a `ttnn.generic_op` megakernel on 110 Tensix cores.

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

### Accuracy and speed

Measured 2026-09-30 on one p150a (AICLK 1350 MHz), tt-metal `668c2907575`, batch 1, 10 flow-matching steps. "Previous" is the 2026-09-29 image, whose expert loop ran as stock / custom TT-NN ops in the same trace. That path is still in the code as `PI05_MEGAKERNEL=off`.

| Metric | Value |
|---|---:|
| Inference, served over HTTP by this image (2×224×224, 224 tokens, H = 50; median of 100 warm requests) | **{inf} ms** device (`timing_ms.inference`, p90 {p90}) · {tot} ms end-to-end (`timing_ms.total`) · {wall} ms client wall per 50-step chunk |
| Previous image (`672900e23919`), same request | 84.02 ms device · 85.35 ms end-to-end |
| In-process trace replay (host wall, median of 60) | {replay:.1f} ms (previous path 82.8 ms) |
| Device time of the expert loop (profiler, median of 21 replays) | **17.49 ms as one op** (previous path 30.31 ms over 1,660 ops); the prefix is 891 traced ops |
| PCC vs the openpi GPU golden (8 real LIBERO observations, same code with `lerobot/pi05_libero`, prompt right-padded to 32 tokens, H = 10, 7 action dims) | mean **{ol["mean_pcc_7"]:.6f}**, min **{ol["min_pcc_7"]:.6f}** (previous path 0.999839 / 0.999712) |
| Expert vs an fp32 expert-loop oracle fed the device's own prefix K/V and noise (served shape, `pi05_base`, 22 random inputs with 1-224 real tokens) | mean **{osum["mean_mk_vs_oracle"]:.5f}**, min {osum["min_mk_vs_oracle"]:.5f}. Closer than the previous path on **22 / 22** (previous mean {osum["mean_off_vs_oracle"]:.5f}) |
| Whole call vs the fixed fp32 torch reference, same 22 inputs | {dmin:.3f}-{dmax:.4f} (mean {dmean:.3f}; previous path {fmin:.3f}-{fmax:.4f}, mean {fmean:.3f}). The spread comes from the bf16 / bf8 prefix: an fp32 expert fed the device's own K/V scores {omin:.3f}-{omax:.4f} against the same reference |
| Padding is exact | random ids in the padded slots under the same mask give a bit-identical output |
| Determinism | ten trace replays are bit-identical. Alternating prompts and a shape switch reproduce fresh-model outputs bit-for-bit. 20 consecutive processes ran with no hang and gave one output digest. 100 repeated served requests returned identical actions |
| LIBERO-spatial closed loop (`lerobot/pi05_libero`, 10 tasks × 10 init states, same code) | **99 / 100**. Paired init states 0-4: 49 / 50 vs openpi on an RTX 5090 50 / 50. {lib_ms} ms per policy call (previous path 98 / 100 at 77.6 ms) |
| Same forward on an RTX 5090 (same host, the port's torch reference, eager PyTorch, batch 1, incl. H2D/D2H; measured 2026-09-14) | fp32 strict 144.1 ms; bf16 / fp16 autocast 121.5 / 123.7 ms; bf16 weights resident 99.9 ms. The p150a is {99.869/inf:.2f}-{144.066/inf:.2f}× faster than every eager row. The best whole-request `torch.compile` run is 46.6 ms (the GPU is {inf/46.62:.2f}× faster) |

### Caveats

- Fixed geometry:
  - 1-2 images squashed to 224×224; a missing wrist camera is padded with a black image, which is still attended;
  - prompt ≤224 tokens, right-padded, pads masked;
  - `state` ≤32 floats;
  - batch 1;
  - `num_steps` is baked in at 10, and a per-request value is rejected.
- The megakernel is a single-p150a, batch-1 program. It needs the 64 KiB worker-L1 cut, which the server applies when it opens the device. It refuses by name on a mesh, with batch sizes > 1, with bf16 K/V caches, or with step counts ≠ 10. On a mesh an unset `PI05_MEGAKERNEL` falls back to the previous path.
- The prompt tokenizer `google/paligemma-3b-pt-224` is **gated** (Gemma terms). Accept the terms on huggingface.co and run `hf auth login` on the host before `tt serve`, or send pre-tokenised `tokens`. The weights themselves are ungated but under the same Gemma terms.
- Outputs are normalised actions of the **base** checkpoint, which is not fine-tuned for any task. `pi05_base` ships no per-feature stats, so its outputs are not usable on an arbitrary robot without fine-tuning. The real-input rows (openpi golden, LIBERO) use the fine-tuned `lerobot/pi05_libero` through the same code, not `pi05_base`.
- Not an OpenAI-compatible API. `GET /v1/models` is a stub so the tt-model ready card does not 404.
- Validated on tt-metal `main` @ `668c2907575` (v0.79.0-dev20260914), single p150a only. The mesh / tensor-parallel path, `PI05_EXPERT_ATTN=ttnn` and batch > 1 exist only on the `PI05_MEGAKERNEL=off` path, and none of them was re-validated after the mask / RoPE fix.
- GPU comparison: the RTX 5090 rows are the 2026-09-14 runs of the torch reference before the mask / RoPE fix. The shapes are the same; the rows were not re-measured. They use fp32 weights + autocast unless stated, no TensorRT, medians of 50 iterations after warm-up, H2D/D2H included. p150a power was not measured, so no efficiency comparison is made. Full table: [`GPU_COMPARISON.md`](GPU_COMPARISON.md).

### Licensing

- Weights: [lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base), [Gemma Terms of Use](https://ai.google.dev/gemma/terms). They are not redistributed here but fetched into your HF cache. The tokenizer [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) is gated under the same terms.
- Port and serving code (`code/`): Apache-2.0 headers, distributed under the same Gemma terms, from [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ [`f7f173b`](https://github.com/changh95/tt-pi-0.5/commit/f7f173bd61241f09b286e56a9eb4038f76ee0782) (`main`, merge of PR #2).
"""
s = open(f"{S}/tt-model.yaml").read()
head = s.split("\ncard:\n", 1)[0]
class D(yaml.SafeDumper): pass
D.add_representer(str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style="|" if "\n" in v else None))
card = yaml.dump({"card": {"description": desc, "quickstart": q}}, Dumper=D, sort_keys=False, allow_unicode=True, width=10**6)
open(f"{S}/tt-model.yaml", "w").write(head + "\n" + card)
d = yaml.safe_load(open(f"{S}/tt-model.yaml"))
assert d["card"]["description"] == desc and d["card"]["quickstart"] == q
print("card ok", len(desc), len(q))
