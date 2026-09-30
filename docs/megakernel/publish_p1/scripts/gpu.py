"""GPU_COMPARISON.md: replace the p150a update section with the 2026-09-30 measurements (bench JSON arg 1)."""
import json, sys
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"
BAK = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pub1/bak/GPU_COMPARISON.md"
R = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/integrate_p1/results"
b = json.load(open(sys.argv[1])); st = b["stats"]; img = sys.argv[2]
inf, tot = st["inference"]["median"], st["total"]["median"]
replay = round(json.load(open(f"{R}/A_default_base.json"))["replay_ms"]["median"], 1)
old = open(BAK).read()
i = old.index("## Update 2026-09-29"); j = old.index("The rest of this file is the 2026-09-14 pass")
gpu_dev = [("fp32 strict", 144.066), ("tf32", 108.693), ("bf16 autocast (fp32 weights)", 121.476), ("fp16 autocast (fp32 weights)", 123.674),
           ("bf16 weights resident (eager)", 99.869), ("bf16 autocast + `torch.compile` default", 88.419), ("bf16 autocast + `torch.compile` reduce-overhead", 98.388),
           ("bf16 weights resident + `torch.compile` reduce-overhead", 59.97), ("bf16 weights resident + whole-request `torch.compile` default", 46.62),
           ("bf16 weights resident + whole-request `torch.compile` reduce-overhead", 48.609)]
gpu_fwd = [("fp32 strict", 143.77), ("tf32", 108.54), ("bf16 autocast", 121.19), ("fp16 autocast", 126.91), ("bf16 weights resident (eager)", 99.78),
           ("bf16 weights resident + whole-request `torch.compile` default", 46.39)]
gpu_srv = [("fp32 strict", 156.514), ("tf32", 110.221), ("bf16 autocast (fp32 weights)", 122.868), ("bf16 weights resident (eager)", 101.484)]
# every GPU value must already be in the 2026-09-29 table (copied, not re-measured)
for _, v in gpu_dev + gpu_fwd + gpu_srv: assert f"| {v} |" in old[i:j], v
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
sec = f"""## Update 2026-09-30: the p150a side re-measured (current image, phase-1 expert megakernel)

The p150a now serves the phase-1 expert megakernel on tt-metal `main` @ `668c2907575`. The SigLIP + VLM prefix is still traced TT-NN ops. The whole 10-step × 18-layer action-expert loop, with the action in/out projections and the Euler updates, is ONE persistent `ttnn.generic_op` on 110 cores inside the same Metal trace.

Its numbers were re-measured on 2026-09-30 by serving this repo's image `{img}` on the p150a: 100 warm requests of the card's request after 5 warm-ups, at the served shape (2 × 224² + 224 tokens, H = 50, 10 steps).
- `timing_ms.inference`: median **{inf} ms** (p10 {st["inference"]["p10"]}, p90 {st["inference"]["p90"]}).
- `timing_ms.total`: **{tot} ms** (p90 {st["total"]["p90"]}).
- preprocess {st["preprocess"]["median"]} ms; client wall {st["client_wall"]["median"]} ms.
- The in-process trace replay alone (`execute_trace`, host wall, median of 60; port repo `docs/megakernel/integrate_p1/results/A_default_base.json`) is {replay} ms.

The previous image (`672900e23919`, 2026-09-29) ran the expert loop as stock / custom TT-NN ops. It measured 84.02 ms inference, 85.35 ms total and an 82.8 ms replay.

**The GPU numbers are not re-measured.** They are the 2026-09-14 runs below, of the torch reference as it was then, before the mask / RoPE fix. The fix changes the attention mask and the RoPE positions but not the tensor shapes. The GPU cost of the fixed reference was not measured.

Ratio = p150a ms / GPU ms (> 1 means the GPU is faster).

""" + "\n".join(tab) + f"""

Reading: the p150a ({inf} ms) is faster than every eager GPU row. The fastest eager row is bf16 weights resident ({eager_best} ms, ratio {inf / eager_best:.2f}). With bf16 weights and a compiled whole-request graph the GPU reaches 46.6 ms, {inf / 46.62:.2f}x faster than the p150a. p150a power was not measured, so no efficiency comparison is made.

"""
new = old[:i] + sec + old[j:]
new = new.replace("GPU pass: 2026-09-14; p150a side updated 2026-09-29 (next section).", "GPU pass: 2026-09-14; p150a side updated 2026-09-30 (next section).", 1)
assert "updated 2026-09-30" in new
open(f"{S}/GPU_COMPARISON.md", "w").write(new)
print(sec)
