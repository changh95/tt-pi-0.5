# GPU_COMPARISON.md: ASD-STE100 draft, change notes (2026-10-03)

This is a draft only. Nothing was published, pushed or committed. The source `models/pi05-base-p150-mc/GPU_COMPARISON.md` is unchanged (md5 `1c4b061f6ad7e7c63531c9bc0ca80204` before and after).

- Draft: `draft_ste/GPU_COMPARISON.md`.
- Rules: `draft_ste/STE_RULES.md`.

## Summary of the method

- Every section and its order stay. Four headings have new STE words:
  - "Update 2026-10-01: the p150a side re-measured (...)" became "Update 2026-10-01: new p150a measurements (...)".
  - "What was run" became "What the pass ran".
  - "Comparison with the p150a (matching definitions)" became "Comparison with the p150a (same definitions)".
  - "Not measured / caveats" became "Not measured / limitations".
- Prose paragraphs became bullet lists: one fact for each bullet, at most two levels (2-space indent).
- Long paragraphs without a section heading start with a short label ("Interpretation:", "Previous images:", "Other facts:"). The labels keep the topics apart.
- The data tables are byte-identical to the source, with one exception: the served-like table header "resize/normalise" became "resize/normalize" (US spelling).
- The "What was run" table stays a two-column table. Its 9 cells are rewritten as short STE sentences; every number, `code` span, path and hash stays.
- The historical sections (2026-09-14 pass, 2026-10-01 update) use the simple past for the measurements.
- "Reading:" (an -ing noun) became "Interpretation:". "Timing definitions" became "Time definitions". "un-optimised", "normalised", "quantisation" became US spelling.
- Passive forms became active where possible ("We did not measure ...", "The script casts ...", "This file reports ..."). States stay passive ("is cached", "is page-cached").

## Before / after pairs (exact text)

### Title and intro

- Before: "Original note: Facts only; every GPU number below was measured in this pass, every p150a number is copied (with its source line) from the validation / publish reports and logs."
  - After: "Original note: this file gives facts only."
  - After: "The 2026-09-14 pass measured every GPU number below."
  - After: "Every p150a number is a copy from the validation / publish reports and logs, with its source line."
- Before: "The p150a was NOT touched."
  - After: "The pass did NOT touch the p150a."

### Update 2026-10-03

- Before: "The p150a now serves the multi-config megakernel (`PI05MegakernelTTNN`: vision | prefix | expert as three persistent `ttnn.generic_op` programs per call, four with 3-4 cameras) on tt-metal `main` @ `f856a38a361`, image `tt-model/pi05-base-p150:36f651704bf1`."
  - After: "The p150a now serves the multi-config megakernel (`PI05MegakernelTTNN`)."
  - After: "It runs vision | prefix | expert as three persistent `ttnn.generic_op` programs for each call."
  - After: "With 3-4 cameras, it runs four programs for each call."
- Before: "The GPU latency of these runs is not reported: it was measured differently (openpi model time only, on a shared, loaded host), so it is not comparable:"
  - After: "This file does not report the GPU latency of these runs."
  - After: "That measurement used a different method (openpi model time only, on a shared, loaded host)."
  - After: "Thus the GPU latency is not comparable."
- Before: "With 1 camera both backends collapse (the wrist view is dropped), so those rows are a record, not an accuracy signal."
  - After: "With 1 camera, both backends fail, because the request does not include the wrist view."
  - After: "Thus the 1-camera rows are a record, not an accuracy signal."

### Update 2026-10-01

- Before: "Every request runs SigLIP on both cameras, the projector, the language embedding, the VLM prefill (writing the K / V caches), the whole 10-step × 18-layer action-expert loop, the action in/out projections and the Euler updates as ONE persistent `ttnn.generic_op` on 110 cores."
  - After: "For each request, ONE persistent `ttnn.generic_op` on 110 cores runs all of these parts:"
  - After: "The VLM prefill, which writes the K / V caches."
  - After: "The full action-expert loop of 10 steps × 18 layers."
- Before: "**The GPU numbers are not re-measured.**"
  - After: "**We did not measure the GPU numbers again.**"
- Before: "With bf16 weights and a compiled whole-request graph the GPU reaches 46.6 ms, 1.20x faster than the p150a. p150a power was not measured, so no efficiency comparison is made."
  - After: "With bf16 weights and a compiled whole-request graph, the GPU gets to 46.6 ms, 1.20x faster than the p150a."
  - After: "We did not measure the p150a power, so this file makes no efficiency comparison."

### What was run

- Before (table cell): "a dedicated GPU venv (`.venv-gpu/pi05`) — Python 3.12.13, torch 2.11.0+cu128, ... triton 3.6.0; added in this pass: pillow 12.3.0, fastapi 0.141.1, pydantic 2.13.5 (so `server/app.py` — the served preprocessing — imports; no ttnn)"
  - After: "This pass added pillow 12.3.0, fastapi 0.141.1 and pydantic 2.13.5."
  - After: "With them, `server/app.py` (the served preprocess code) imports; no ttnn"
- Before (table cell): "per precision: 10 warm-ups + 50 timed iterations, `torch.cuda.synchronize()` before/after each; wall-clock (perf_counter) is the primary number, CUDA-event time recorded alongside; power sampled by `nvidia-smi -lms 200` during the timed loop"
  - After: "For each precision: 10 warm-ups + 50 timed iterations, with `torch.cuda.synchronize()` before and after each."
  - After: "`nvidia-smi -lms 200` samples the power during the timed loop"
- Before: "Two deviations from the literal reference, needed to run it on CUDA (both in `bench_pi05_gpu.py`): `precompute_freqs_cis` is cached and returns the RoPE tables on the model's device (the reference recomputes them on the CPU on every expert call, which cannot multiply CUDA tensors; ...)"
  - After: "The GPU run has two deviations from the literal reference."
  - After: "The reference calculates the tables again on the CPU at every expert call, and CPU tables cannot multiply CUDA tensors."
- Before: "Compare with the p150a segment profile SigLIP 22.6 / VLM 49.9 / expert 56.3 ms (DEVICE_VALIDATION.md:267-270; that profile is of the 130.2 ms configuration — the shipped default is 4.4 ms faster in the expert, i.e. ~51.9 ms)."
  - After: "Compare with the p150a segment profile: SigLIP 22.6 / VLM 49.9 / expert 56.3 ms (DEVICE_VALIDATION.md:267-270)."
  - After: "The shipped default is 4.4 ms faster in the expert, thus ~51.9 ms."

### Correctness check

- Before: "PCC > 0.999 holds; the GPU runs the right model."
  - After: "PCC > 0.999 holds; the GPU runs the correct model."
- Before: "fp16 autocast is numerically fine on this input (no overflow in the Gemma-2B activations; PCC above 0.999), so it is timed."
  - After: "fp16 autocast is numerically correct on this input: the Gemma-2B activations have no overflow, and the PCC is above 0.999."
  - After: "Thus the pass timed it."
- Before: "Per-precision accuracy vs the CPU fp32 reference (same input):"
  - After: "Accuracy of each GPU precision compared with the CPU fp32 reference (same input):"

### GPU latency

- Before: "The eager reference is launch-bound in the expert loop: 18 layers x ~30 small kernels x 10 steps ≈ 5–6 k launches per chunk at 50 tokens, so the 10 steps cost 65–85 ms regardless of precision (...)"
  - After: "The eager reference is launch-bound in the expert loop."
  - After: "Thus the 10 steps cost 65–85 ms at all precisions."
  - After: "bf16 autocast is *slower* than fp32 there, because every step casts the 300 M expert weights again."
- Before: "... the `embed_image` output is `.clone()`d outside the graph (it is replayed once per camera and the second replay overwrites the first camera's output buffer -- the first attempt failed with '...', `compile_run.log`) ..."
  - After: "The graph replays once for each camera, and the second replay overwrites the output buffer of the first camera."
- Before: "Only the fp32 strict loop pulls the GPU near its 600 W limit (466 W, 92 % util); every other precision is bandwidth/launch-bound at batch 1 (64-75 % util)."
  - After: "Only the fp32 strict loop brings the GPU near its 600 W limit (466 W, 92 % util)."
  - After: "Every other precision is bandwidth/launch-bound at batch 1 (64-75 % util)."

### Comparison with the p150a

- Before: "The gap between eager and compiled is launch overhead in the expert loop, not arithmetic: the p150a number already includes the equivalent fix (one Metal trace per request), so the compiled rows compare the two *deployed* paths like for like, while the eager rows compare the p150a's fused deployment with an un-optimised GPU run of the reference code."
  - After: "The difference between eager and compiled is launch overhead in the expert loop, not arithmetic."
  - After: "Thus the compiled rows compare the two *deployed* paths directly."
  - After: "The eager rows compare the p150a's fused deployment with an unoptimized GPU run of the reference code."
- Before: "With `torch.compile` (bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps)) the GPU reaches 46.6 ms incl. transfers, 2.70x the p150a."
  - After: "The best compiled row is bf16 weights resident + whole-request `torch.compile` default (one graph: prefix + VLM + 10 unrolled steps)."
  - After: "With it, the GPU gets to 46.6 ms with transfers, 2.70x the p150a."

### Not measured / caveats

- Before: "The whole-request compile of the *autocast* model (fp32 weights resident) ran out of GPU memory during compilation (`compile_run3.log`: 29.6 GiB allocated by PyTorch of 31.3): ..."
  - After: "The compilation failed because the GPU did not have sufficient memory (`compile_run3.log`: 29.6 GiB allocated by PyTorch of 31.3)."
- Before: "The bf16 GPU rows are the closest precision match; none of them replicate the bf8 weight quantisation."
  - After: "The bf16 GPU rows are the closest precision match."
  - After: "None of them reproduce the bf8 weight quantization."
- Before: "... (`compile_run3.log`; the split in `compile_run2.log` lacked it and re-recorded graphs, hence its 600 ms denoise numbers -- superseded)."
  - After: "The split in `compile_run2.log` did not have this call, and it recorded the graphs again."
  - After: "Thus its 600 ms denoise numbers are not valid (superseded)."

### Evidence

- No change: the section is a list of paths.

## Check results (final)

`ste_check.py GPU_COMPARISON.md --all` (summary lines):

```
sentences 213 (instructions 0, descriptive 213); over limit 0; paragraphs > 6 sentences 0; max length 22; mean 11.3
table prose cells: sentences 125, over 25 0, max 20
British spellings: []
list items 208; list items nested deeper than 2 levels 0; prose paragraphs (not lists) 16
```

For comparison, the source gives: sentences 89, over limit 33, max length 74, mean 24.3; table cells over 25: 6; British spellings `normalise`, `normalised`, `optimised`; list items 19; prose paragraphs 27.

`integrity_check.py <source> GPU_COMPARISON.md`:

```
OK   front matter identical 
OK   code blocks identical (0 vs 0)
OK   no number lost []
OK   no new number []
OK   no `code` span lost []
OK   URLs identical []
OK   hex ids identical []
RESULT PASS
```

Extra check: 110 of the 120 source table rows occur byte-identical in the draft. The 10 changed rows are the 9 "What was run" cells and the served-like table header (`normalise` to `normalize`).

## Prose paragraphs that remain (16), and why

| # | paragraph | why it stays prose |
|---|---|---|
| 1 | "Closed-loop comparison of the same day:" | label: it starts a second topic in the 2026-10-03 section |
| 2 | "Measurement of 2026-10-01:" | label of the measurement bullets |
| 3 | "Previous images:" | label of the bullets about the 2026-09-30 and 2026-09-29 images |
| 4 | "GPU numbers:" | label of the bullets about the GPU numbers that were not measured again |
| 5 | "Interpretation:" (2026-10-01 section) | label of the result bullets (was "Reading:") |
| 6 | "Historical part:" | label: it marks where the 2026-09-14 record starts |
| 7 | "The GPU run has two deviations from the literal reference. Both are necessary for CUDA, and both are in `bench_pi05_gpu.py`:" | lead-in: it gives the count, the reason and the file of the two deviations in the list |
| 8 | "Time definitions (the same as the p150a `timing_ms` keys):" | lead-in: it names the keys that the four definitions match |
| 9 | "Accuracy of each GPU precision compared with the CPU fp32 reference (same input):" | caption of the table below it |
| 10 | "Weights resident in bf16:" | label of the bullets that describe this configuration |
| 11 | "Stage split:" | label of the bullets above the stage table |
| 12 | "Launch overhead:" | label of the launch-overhead bullets |
| 13 | "`torch.compile` configuration:" | label of the compile bullets |
| 14 | "Other facts:" | label (from the source text) |
| 15 | "Served-like loop (same host work as `server/app.py::predict`, 50 iterations, medians ms; p90 of total in brackets):" | caption of the table below it; it gives the column units |
| 16 | "Interpretation:" (comparison section) | label of the result bullets (was "Reading:") |

## Sentences over the limits

- None. The longest prose sentence has 22 words; the longest table-cell sentence has 20 words.

## Notes and questions for the reviewer

1. (Resolved by pi05-mc-ship after this report: the draft now says "for each action chunk (H = 50)"; ste_check and integrity_check still pass.) "50-step chunk" (Interpretation of the comparison section) stayed as in the source. The chunk has H = 50 actions and N = 10 flow-matching steps, so "50-step" is not accurate. Do you want "for each action chunk (H = 50)"?
2. The glossary term "flow-matching steps (N)" occurs in the 2026-10-03 update and in the Input cell ("10 Euler flow-matching steps"). Other places keep the source words "10 steps", "10 expert steps" and "10 denoise steps", because they name table columns and stage names. Is that acceptable?
3. Words that may not be approved STE words: "deviations", "analogue", "regime", "jitter", "addresses", "superseded". "superseded" stays in parentheses as the status word of the source. Please confirm or give replacements.
4. "We" occurs in descriptive sentences ("We did not measure the GPU numbers again."). This makes the sentences active. If the house style does not allow "we", the alternative is "This pass did not measure ...".
5. Some kept noun clusters have more than 3 words, because they are row labels or configuration names in the tables and in the text: "bf16 weights resident + whole-request `torch.compile` default", "p150a segment profile", "server-side policy call". Configuration names stay as in the tables so that the text and the tables match.
6. The quoted error text 'accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run' and the quoted 'Served shape everywhere: ...' stay verbatim, because they are quotes.
7. The `code` spans are all present; no span was merged.
- 2026-10-03 15:21:28 lead answers applied: deviations/analogue/regime/jitter/addresses/superseded -> difference/equivalent/range/variation/solves/replaced; 'We' -> 'This test' (6 sentences). Checks re-run: PASS.
- 2026-10-03 15:30:35 user correction: 'variation' reverted to 'jitter' (line 309); jitter is a kept VLA term.
