# SERVING.md, PERF_PRESETS.md, demo/README.md: ASD-STE100 draft, change notes (2026-10-03)

This is a draft only. Nothing was published, pushed or committed. The sources under `models/pi05-base-p150-mc/` are unchanged (md5 before and after: `d72eabfe…` SERVING.md, `0693d5b6…` PERF_PRESETS.md, `a674cd84…` demo/README.md).

| draft | source |
|---|---|
| `draft_ste/SERVING.md` | `models/pi05-base-p150-mc/SERVING.md` |
| `draft_ste/PERF_PRESETS.md` | `models/pi05-base-p150-mc/PERF_PRESETS.md` (prose only) |
| `draft_ste/demo/README.md` | `models/pi05-base-p150-mc/demo/README.md` |

Rules: `STE_RULES.md` of this folder. The code blocks of the drafts were copied from the sources by a script, so they are byte-identical.

## 1. SERVING.md

### General changes

- New section "Terms" after the introduction. It defines each glossary term at its first use: camera count, action chunk (H actions), flow-matching steps (N), configuration, serve profile, prompt bucket, action-row bucket (S), preset, device program, trace replay, request, inference time.
- The device program definition says "on the 11 x 10 worker grid", not "on 110 Tensix cores". The number 110 does not occur in the source, and the integrity rule forbids a new number.
- Headings: "Serving pi-0.5 …" became "Serve pi-0.5 …" (no -ing form). "Request / response contract" became "Request and response contract". "Package / serve (from the repo dir)" became "Package and serve (from the repo directory)".
- Table header "meaning" became "description" (no -ing form).
- British spellings fixed: normalised → normalized, discretised → discretizes, pre-tokenised → Pre-tokenized, serialised → serializes, materialised → materializes.
- Numbered lists: a numbered list has at most 5 items. The number 6 does not occur in the SERVING.md prose, so a step "6." would be a new number. The boot log phrases are a bullet list for the same reason; the lead-in says that they occur in this order.

### Introduction

- Before: "`tt-model package --container` builds the image (tt-metal built from source + the port + the HTTP stack), `tt-model serve` runs it against the card."
  - After: "`tt-model package --container` builds the image."
  - After: "The image contains tt-metal (built from source), the port and the HTTP stack."
  - After: "`tt-model serve` runs the image on the card."
- Before: "The server is `code/models/experimental/pi0_5/server/app.py` (FastAPI, `kind: tt-dit-server`); its default backend is the multi-config pi0.5 megakernel `PI05MegakernelTTNN` (`code/models/experimental/pi0/tt/ttnn_pi05_model.py`)."
  - After: "The server is `code/models/experimental/pi0_5/server/app.py` (FastAPI, `kind: tt-dit-server`)."
  - After: "The default backend of the server is the multi-config pi0.5 megakernel `PI05MegakernelTTNN` (`code/models/experimental/pi0/tt/ttnn_pi05_model.py`)."

### Layout

- Before: "tt-model copies the tt-metal tree without its own `models/`, so this `pi0` package is the only one in the image; the kernel sources are located relative to the Python files and JIT-compiled at the first boot of each configuration."
  - After: "tt-model copies the tt-metal tree without its own `models/`."
  - After: "Thus this `pi0` package is the only `pi0` package in the image."
  - After: "The server finds the kernel sources relative to the Python files."
  - After: "The JIT compiler compiles the kernel sources at the first boot of each configuration."
- Before: "In the image it lands at `/opt/tt-metal/models/...` with `PYTHONPATH=/opt/tt-metal`."
  - After: "In the image, `code/models/` is at `/opt/tt-metal/models/...`, with `PYTHONPATH=/opt/tt-metal`."

### tt-metal tree

- Before: "`source.tt_metal` is a clean clone of tenstorrent/tt-metal `main` @ `f856a38a361939888f92d88f9e69b2f8a83fb713` (v0.80.0-dev20261001-17; `dirty: false` in the wire manifest), the tree the multi-config megakernel was validated on."
  - After: "`source.tt_metal` is a clean clone of tenstorrent/tt-metal `main` @ `f856a38a361939888f92d88f9e69b2f8a83fb713` (v0.80.0-dev20261001-17)."
  - After: "The wire manifest shows `dirty: false` for this tree."
  - After: "The validation of the multi-config megakernel used this tree."
- Before: "`verify:` checks the 32 presets, the 14 kernel sources and their digest (`1429d5bea05c31ad`), every `#include` of the megakernel's C++ sources against `tt_metal/hw/inc` in the image, the server's default configuration, and the earlier single-config paths' own checks."
  - After: "Each `#include` of the C++ sources of the megakernel, against `tt_metal/hw/inc` in the image."
  - After: "The checks of the earlier single-config paths."

### Run on the HOST for validation (no Docker)

- New numbered list before the code block (one step for each part of the code block).
  - After: "Install fastapi and uvicorn into a side directory."
  - After: "Add that directory to `PYTHONPATH`."
- Before: "Every prompt bucket is compiled and captured before READY, so READY means warm."
  - After: "The server compiles and captures each prompt bucket before READY."
  - After: "Thus READY means that the server is warm."
- Before: "Startup failures raise (uvicorn exits non-zero); nothing falls back to CPU."
  - After: "If the startup fails, the server raises an exception and uvicorn exits with a non-zero code."
  - After: "The server does not use the CPU as a fallback."

### Package and serve

- New numbered list (5 steps) before the code block.
  - After: "Run `source $ROOT/bin/docker-env.sh` to use rootless Docker."
  - After: "To select a serve profile, add `[--profile c1-h50-n10]`."
- Before: "Measured boot of this image on the p150a (2026-10-03, `tt-model serve` wall to READY): default profile 66.2 s cold (empty JIT cache; the four prompt buckets compile + capture in 11.8 s) and 41.2 s warm (0.8 s); the other profiles 35-57 s (their own programs compiled at first boot)."
  - After: "Default serve profile, cold (empty JIT cache): 66.2 s."
  - After: "The four prompt buckets compile and capture in 11.8 s."
  - After: "Default serve profile, warm: 41.2 s."
  - After: "These serve profiles compile their own programs at the first boot."
- Before: "(JIT kernels persist across boots)"
  - After: "The JIT kernels stay in this cache from one boot to the next."

### Request and response contract

- Before: "`GET /v1/models` -> OpenAI-shaped stub so the tt-model ready card does not 404; this is not a chat API."
  - After: "`GET /v1/models` returns an OpenAI-shaped stub."
  - After: "The stub prevents a 404 on the tt-model ready card."
  - After: "This endpoint is not a chat API."
- Before (table cell): "proprio state ALREADY normalised to [-1, 1]; zero-padded; discretised into 256 bins for the prompt; default zeros"
  - After: "The proprioceptive state, ALREADY normalized to [-1, 1]."
  - After: "The server zero-pads it and discretizes it into 256 bins for the prompt."
- Before: "Errors: 400 (bad input, with the reason), 503 while starting, 500 with the exception text if the device call fails. Batch 1 per request; requests are serialised on one lock."
  - After: "503: the server did not complete its startup."
  - After: "500: the device call failed."
  - After: "One lock serializes the requests."

### Configuration

- Before (table cell): "1..64 (50); suffix bucket 32 rows for H <= 32, else 64"
  - After: "1..64 (50). The action-row bucket is 32 rows for H <= 32, and 64 rows for a larger H."
- Before: "An out-of-range value, a mesh (`TT_MESH_SHAPE` other than 1x1), `PI05_BATCH_SIZES` other than 1 or another layout is refused at start with the model's own message."
  - After: "At start, the server refuses these settings with the message of the model:"
  - After: "A mesh (`TT_MESH_SHAPE` other than 1x1)."
- Before: "they were validated on tt-metal `668c2907575`, not on this tree."
  - After: "The validation of these single-config paths used tt-metal `668c2907575`, not this tree."

### Caveats

- Before: "`google/paligemma-3b-pt-224` needs an HF token with the Gemma terms accepted;"
  - After: "`google/paligemma-3b-pt-224` needs an HF token for an account that accepted the Gemma terms."
- Before: "`pi05_base` is a base checkpoint, so actions on an arbitrary robot need fine-tuning."
  - After: "`pi05_base` is a base checkpoint."
  - After: "Fine-tune it before you use its actions on an arbitrary robot."
- Before: "The fp32 checkpoint (14.5 GB) is materialised on the host while the model is built and released after conversion (`PI05_FREE_HOST_WEIGHTS=1`); budget ~20 GB peak."
  - After: "The server materializes the fp32 checkpoint (14.5 GB) on the host while it builds the model."
  - After: "The server releases the checkpoint after the conversion (`PI05_FREE_HOST_WEIGHTS=1`)."
  - After: "Make sure that the host has about 20 GB of RAM for the peak."

### Check results (SERVING.md)

```
sentences 162 (instructions 18, descriptive 144); over limit 0; paragraphs > 6 sentences 0; max length 18; mean 7.9
table prose cells: sentences 27, over 25 0, max 17
British spellings: []
list items 128; list items nested deeper than 2 levels 0; prose paragraphs (not lists) 8
```

```
OK   front matter identical 
OK   code blocks identical (3 vs 3)
OK   no number lost []
OK   no new number []
OK   no `code` span lost []
OK   URLs identical []
OK   hex ids identical []
RESULT PASS
```

### Remaining prose paragraphs (SERVING.md): 8

| paragraph | why it stays prose |
|---|---|
| "This repo is a tt-model container package. It contains these parts:" | The lead-in names the package type (new information) and introduces the list of parts. |
| "The commands and the server:" | A label that separates the second list from the list of parts. |
| "Use this procedure to validate the port on the host without Docker:" | The lead-in of the numbered procedure. |
| "Offline overrides:" | A label for the list of override variables. |
| "With tt-cli:" | A label for the tt-cli command list. |
| "Caches and boot times:" | A label for the cache and boot-time list. |
| "`POST /predict` accepts JSON with these fields:" | The lead-in of the request table. |
| "Set these environment variables before the server starts. The server reads them only at start." | The lead-in of the configuration table; it gives the instruction for the table. |

## 2. PERF_PRESETS.md

### Scope

- Rewritten: the title, the method line under the title (now a bullet list) and the two table section headings.
- Verbatim by instruction: both measurement tables, the "Footnote:" line, the "†" line and the whole "## Build log" section (its heading, its table and the "Foreign windows" paragraph).
- `diff` of the source against the draft shows four hunks only: line 1 (title), line 3 (method line → lines 3-13), the heading "10 … steps" and the heading "2 cameras …". All other lines (both tables, Footnote, †, Build log heading, Build log table, Foreign windows) are byte-identical.

### Title and method line

- Before: "# pi0.5 megakernel: per-preset replay latency (release build fae9cd03fa4)"
  - After: "# pi0.5 megakernel: trace replay latency for each preset (release build fae9cd03fa4)"
- Before: "Blackhole p150a, trace replay (blocking), ms: median of 60 replays per build, mean of 2 builds (± se of the build means)."
  - After: "Each value is the time of one trace replay in ms."
  - After: "The host waits for the end of each trace replay."
  - After: "Each value is the mean of 2 builds."
  - After: "Each build gives the median of 60 trace replays."
- Before: "1 batch, H = 10 for 32 action rows and H = 50 for 64 (the device programs depend on the bucket, not H)."
  - After: "H = 10 for the action-row bucket of 32 rows."
  - After: "H = 50 for the action-row bucket of 64 rows."
  - After: "The device programs depend on the action-row bucket, not on H."
- New (definitions; no new number): "In the column names, L is the prompt bucket in tokens and S is the action-row bucket in rows."

### Section headings

- Before: "## 10 denoising steps (all 32 presets)"
  - After: "## 10 flow-matching steps (all 32 presets)"
- Before: "## 2 cameras at 1 and 5 denoising steps"
  - After: "## 2 cameras with 1 and 5 flow-matching steps"

### Check results (PERF_PRESETS.md)

Whole file:

```
sentences 20 (instructions 0, descriptive 20); over limit 2; paragraphs > 6 sentences 0; max length 59; mean 15.9
  OVER 59/25 descriptive: Footnote: trace replay, median of 60 per build, mean of 2 builds; measured at host 1-min load 2-9; the 4 presets also timed on a quiet host agree within 0.09 ms (c2 L64 S32 +0.027, c3 L224 S32 +0.090, c4 L224 S32 +0.057, c4 L224 S64 +0.081 ms here vs the quiet ABAB holds of 2026-10-03 09:38 / 10:00).
  OVER 36/25 descriptive: Foreign windows: pid 2847144 seen at 11:13:09 (1 s old; how long it ran is unknown, so only the build spanning 11:13 is marked); pi05-mc-ship pytest 11:19:10-11:20:15 and pytest + git clone 11:21-11:22 (its own report).
table prose cells: sentences 10, over 25 0, max 17
British spellings: []
list items 11; list items nested deeper than 2 levels 0; prose paragraphs (not lists) 3
```

- The 2 sentences over the limit and the 3 prose paragraphs are the Footnote line, the † line and the Foreign windows paragraph. These parts are verbatim by instruction. They are exempt.
- The 10 table prose cells are cells of the Build log table (verbatim). None is over 25.

Rewritten prose only (the same file without the verbatim lines and tables):

```
sentences 14 (instructions 0, descriptive 14); over limit 0; paragraphs > 6 sentences 0; max length 19; mean 11.4
table prose cells: sentences 0, over 25 0
British spellings: []
list items 11; list items nested deeper than 2 levels 0; prose paragraphs (not lists) 0
```

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

### Remaining prose paragraphs (PERF_PRESETS.md): 3, all verbatim

| paragraph | why it stays prose |
|---|---|
| "Footnote: trace replay, median of 60 per build, …" | Verbatim by instruction. |
| "† one of the two builds ran while a foreign CPU process was active …" | Verbatim by instruction. |
| "Foreign windows: pid 2847144 seen at 11:13:09 …" | Verbatim by instruction (part of the Build log section). |

## 3. demo/README.md

### General changes

- The intro paragraph, the paragraph under the Clips table and the final paragraph became bullet lists.
- The "What runs" bullets now have one fact for each sub-bullet. They define device program, prompt bucket, trace replay and action chunk (H) at their first use.
- Heading "How it was made" (passive) became "The demo pipeline".
- British spellings fixed: Normalisation → normalization, normalises → normalizes.
- The noun "recording" (-ing noun) became "demo capture". "captioned clip" became "clip with captions".
- All 12 "*not published*" markers stay. The relative links (`../code/models/experimental/pi0/tt/ttnn_pi05_model.py`, `../code/models/experimental/pi0`, `../code/models/experimental/pi0_5/server/serve_pi05_libero.py`, `manifest.json` ×2) are identical in target and count.
- The task strings in the Clips table ("pick up the black bowl …") are LIBERO task names. They stay as they are, with the phrasal verb "pick up".
- The sub-bullets under the numbered steps 2 and 3 use a 3-space indent, as in the source. A 2-space indent does not nest a bullet under "2. " in CommonMark. The checker counts these items as level 2.

### Intro

- Before: "Recorded 2026-10-03 11:35 KST on the host that built this repository's image."
  - After: "Date and time of the demo capture: 2026-10-03 11:35 KST."
  - After: "Host of the demo capture: the host that built the image of this repository."
- Before: "Files marked *not published* below stayed on that host; everything else in this folder is on the Hub."
  - After: "Files marked *not published* below stayed on that host."
  - After: "All other files in this folder are on the Hub."

### Files and Clips

- Before (table cell): "every clip with its ffmpeg probe and bytes, the recorded and eval step counts, latency, backend stamps, code commit, eval score and the paired-vs-GPU result"
  - After: "Each clip with its ffmpeg probe and its size in bytes."
  - After: "Also the result of the paired comparison against the GPU."
- Before: "Each clip is a fresh episode served live by the p150a, not a replay."
  - After: "Each clip is a new episode that the p150a served live."
  - After: "It is not a replay."
- Before: "These are the same tasks, init state and order as the previous HF demo (megakernel-p2, 2026-10-01)."
  - After: "The tasks, the init state and the order are the same as in the previous HF demo (megakernel-p2, 2026-10-01)."

### What runs

- Before: "They are replayed from one Metal trace per prompt bucket. Every LIBERO prompt runs in the 32-token bucket."
  - After: "One Metal trace for each prompt bucket holds the device programs."
  - After: "Every LIBERO prompt runs in the 32-token prompt bucket."
- Before: "**Weights:** `lerobot/pi05_libero` @ `a217bfd3b14673cf2ce597e69997ab21866438dd`, loaded through `PI0WeightLoader`. Normalisation uses openpi's `pi05_libero` norm stats, sha256 `b3a44bb2...bd84`."
  - After: "`PI0WeightLoader` loads the weights."
  - After: "The normalization uses the `pi05_libero` norm stats of openpi, sha256 `b3a44bb2...bd84`."
- Before: "This matches the matrix's server copy digit for digit."
  - After: "These values are the same as the values of the server copy of the matrix, to the last digit."
- Before: "It covers host input building, the trace replay and the readback."
  - After: "The latency includes the host input build, the trace replay and the readback."

### The demo pipeline

- Before: "The pipeline stops unless every input is `torch.equal` and PCC7 is at least 0.999."
  - After: "The pipeline stops if an input is not `torch.equal` or if PCC7 is less than 0.999."
- Before: "It burns in the captions, builds the combined clip with its title and end cards, and grabs the poster frame at 8 s."
  - After: "It writes the captions into the video frames."
  - After: "It builds the combined clip with its title and end cards."
  - After: "It takes the poster frame at 8 s."
- Before: "That window is captured with ffmpeg x11grab on DISPLAY=:1 at 30 fps, in real time."
  - After: "ffmpeg x11grab captures that window on DISPLAY=:1 at 30 fps, in real time."
- Before: "The poster, the title card (1 s), t3 running (5.5 s), t7 succeeding (14 s), the end card (27.5 s) and a 5-frame strip across the combined clip were checked by eye: no other window covered the viewer in any of them."
  - After: "A person checked these frames by eye:"
  - After: "t3 during its run (5.5 s)."
  - After: "Result: no other window covered the viewer in these frames."

### Check results (demo/README.md)

```
sentences 76 (instructions 0, descriptive 76); over limit 0; paragraphs > 6 sentences 0; max length 24; mean 10.3
table prose cells: sentences 16, over 25 0, max 19
British spellings: []
list items 58; list items nested deeper than 2 levels 0; prose paragraphs (not lists) 3
```

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

### Remaining prose paragraphs (demo/README.md): 3

| paragraph | why it stays prose |
|---|---|
| "`tools/run_demo.sh 3:0,7:0,0:0,5:0` runs these four steps (*not published*, like the other scripts in this section):" | The lead-in of the numbered steps. It names the command and carries a *not published* marker. |
| "A person checked these frames by eye:" | The lead-in of the list of checked frames. |
| "Result: no other window covered the viewer in these frames." | The result of the check after the list. A bullet in the frame list would read as one more frame. |

## Sentences over the limits

None in the rewritten prose of the three files. The only sentences over the limit are the 2 verbatim sentences of PERF_PRESETS.md (Footnote 59 words, Foreign windows 36 words).

## Questions for the reviewer

1. demo/README.md: the caption text was a quote of 24 words inside a 27-word sentence. The draft now shows it as two `code` spans ("first part" and "second part (after the `—` character)"). Is a code span correct for caption text, or do you prefer one quote and a sentence over the limit?
2. demo/README.md: should the LIBERO task strings in the Clips table stay as they are ("pick up …")? They are data, so the draft keeps them.
3. SERVING.md: the device program definition omits "110 Tensix cores" because 110 is a new number for this file. Is "the 11 x 10 worker grid" correct, or should 110 be added to the source first?
4. SERVING.md: the boot log phrases are a bullet list (a numbered list would need the new number 6). Is the lead-in "in this order" enough?
5. PERF_PRESETS.md: the "Build log" heading stays verbatim because the instruction keeps the whole section. Should it also get the STE heading treatment (it is already STE-compatible)?
6. Sub-bullets under numbered steps use 3 spaces (CommonMark needs it). Is that acceptable against the "2-space indent" rule?
