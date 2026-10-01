"""HF GPU_COMPARISON.md: replace the p150a update section (HF cf08fb95) with the 2026-10-01 measurements of this image (bench JSON arg 1, image tag arg 2)."""
import json, sys
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"
BAK = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pub2/hf_head/GPU_COMPARISON.md"
MK = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel"
R = f"{MK}/integrate_p2/results"
b = json.load(open(sys.argv[1])); st = b["stats"]; img = sys.argv[2]
assert b["n"] == 100 and b["identical_actions_all"]
inf, tot = st["inference"]["median"], st["total"]["median"]
A = json.load(open(f"{R}/A_whole_base_r1.json")); assert len(A["replay_ms"]["all"]) == 30
replay = round(A["replay_ms"]["median"], 1)
prev_replay = round(json.load(open(f"{R}/A_expert_base_r1.json"))["replay_ms"]["median"], 1)
prev = json.load(open(f"{MK}/publish_p1/results/bench-c2-mkp1-20260930-172411.json"))["stats"]
old = open(BAK).read()
i = old.index("## Update 2026-09-30"); j = old.index("The rest of this file is the 2026-09-14 pass")
gpu_dev = [("fp32 strict", 144.066), ("tf32", 108.693), ("bf16 autocast (fp32 weights)", 121.476), ("fp16 autocast (fp32 weights)", 123.674),
           ("bf16 weights resident (eager)", 99.869), ("bf16 autocast + `torch.compile` default", 88.419), ("bf16 autocast + `torch.compile` reduce-overhead", 98.388),
           ("bf16 weights resident + `torch.compile` reduce-overhead", 59.97), ("bf16 weights resident + whole-request `torch.compile` default", 46.62),
           ("bf16 weights resident + whole-request `torch.compile` reduce-overhead", 48.609)]
gpu_fwd = [("fp32 strict", 143.77), ("tf32", 108.54), ("bf16 autocast", 121.19), ("fp16 autocast", 126.91), ("bf16 weights resident (eager)", 99.78),
           ("bf16 weights resident + whole-request `torch.compile` default", 46.39)]
gpu_srv = [("fp32 strict", 156.514), ("tf32", 110.221), ("bf16 autocast (fp32 weights)", 122.868), ("bf16 weights resident (eager)", 101.484)]
for _, v in gpu_dev + gpu_fwd + gpu_srv: assert f"| {v} |" in old[i:j], v   # copied, not re-measured
def block(label, p, rows, bold):
    out = []
    for k, (name, g) in enumerate(rows):
        r = f"{p / g:.2f}"; r = f"**{r}**" if bold else r
        out.append(f"| {label if k == 0 else ''} | {p if k == 0 else ''} | {name} | {g} | {r} |")
    return out
tab = ["| row | p150a ms | GPU setting | GPU ms (2026-09-14) | ratio p150a/GPU |", "|---|---:|---|---:|---:|"]
tab += block("device forward (p150a `timing_ms.inference` vs GPU incl_h2d)", inf, gpu_dev, True)
tab += block("forward only (p150a trace replay vs GPU excl_h2d)", replay, gpu_fwd, False)
tab += block("served e2e (p150a `timing_ms.total` vs GPU served-like)", tot, gpu_srv, True)
eager_best = min(g for n, g in gpu_dev if "compile" not in n)
stage_best = min(g for n, g in gpu_dev if "compile" in n and "whole-request" not in n)
sec = f"""## Update 2026-10-01: the p150a side re-measured (current image, the whole-model megakernel)

The p150a now serves the whole-model megakernel on tt-metal `main` @ `668c2907575`. Every request runs SigLIP on both cameras, the projector, the language embedding, the VLM prefill (writing the K / V caches), the whole 10-step × 18-layer action-expert loop, the action in/out projections and the Euler updates as ONE persistent `ttnn.generic_op` on 110 cores. It is the only device op of the Metal trace replay.

Its numbers were re-measured on 2026-10-01 by serving this repo's image `{img}` on the p150a: 100 warm requests of the card's request after 5 warm-ups, at the served shape (2 × 224² + 224 tokens, H = 50, 10 steps).
- `timing_ms.inference`: median **{inf} ms** (p10 {st["inference"]["p10"]}, p90 {st["inference"]["p90"]}).
- `timing_ms.total`: **{tot} ms** (p90 {st["total"]["p90"]}).
- preprocess {st["preprocess"]["median"]} ms; client wall {st["client_wall"]["median"]} ms.
- The in-process trace replay alone (`execute_trace`, host wall, median of 30; port repo `docs/megakernel/integrate_p2/results/A_whole_base_r1.json`) is {replay} ms.

The previous image (`fe0d2e3d68a7`, 2026-09-30) ran the SigLIP / VLM prefix as traced stock TT-NN ops and the expert loop as one generic_op. It measured {prev["inference"]["median"]:.2f} ms inference and {prev["total"]["median"]:.2f} ms total (same benchmark cycle as the current figures) and a {prev_replay} ms replay (`A_expert_base_r1.json`, same session as the current replay). The 2026-09-29 image (`672900e23919`, stock TT-NN ops plus 3 custom programs) measured 84.05 ms inference, 85.30 ms total and an 82.8 ms replay.

**The GPU numbers are not re-measured.** They are the 2026-09-14 runs below, of the torch reference as it was then, before the mask / RoPE fix. The fix changes the attention mask and the RoPE positions but not the tensor shapes. The GPU cost of the fixed reference was not measured.

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster).

""" + "\n".join(tab) + f"""

Reading: the p150a ({inf} ms) is faster than every eager GPU row. The fastest eager row is bf16 weights resident ({eager_best} ms, ratio {inf / eager_best:.2f}). It is also faster than the stage-wise compiled rows (best {stage_best} ms, ratio {inf / stage_best:.2f}). With bf16 weights and a compiled whole-request graph the GPU reaches 46.6 ms, {inf / 46.62:.2f}x faster than the p150a. p150a power was not measured, so no efficiency comparison is made.

"""
new = old[:i] + sec + old[j:]
a = "GPU pass: 2026-09-14; p150a side updated 2026-09-30 (next section)."
assert new.count(a) == 1
new = new.replace(a, "GPU pass: 2026-09-14; p150a side updated 2026-10-01 (next section).", 1)
open(f"{S}/GPU_COMPARISON.md", "w").write(new)
print(sec)
