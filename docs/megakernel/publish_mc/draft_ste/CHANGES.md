# pi05-base-p150 model card: ASD-STE100 draft, change notes (2026-10-03)

This is a draft only. Nothing in this folder has been published, pushed or committed. The published card (HF `ab447b79`, staging `models/pi05-base-p150-mc/README.md`) is unchanged.

## Files

| file | what |
|---|---|
| `README.md` | the rendered draft card (HF front matter of the published card, unchanged) |
| `tt-model.yaml` | the build-time `tt-model.yaml` with the draft `card:` block (draft only; not the staging file) |
| `ste_text.py` | all the STE prose of the card (`description`, `quickstart`, the per-preset table text) |
| `card_ste.py` | the card generator (a copy of `scripts/card_mc.py`): the same measured files, asserts and numbers; it calls `ste_text.py` |
| `demo_ste.py` → `demo_section_ste.md` | the STE demo section of the card (from the demo `manifest.json` and the LIBERO matrix files) |
| `readme_ste.py` | renders `README.md`, and rewrites the sentences that `render_model_card` generates (Runs on / Packaged / Quickstart bullets / Serve profiles / Provenance) |
| `ste_check.py` → `ste_check.txt` | the word-count check of the draft (every sentence, its length, its type and its limit) |
| `ste_check_published.txt` | the same check on the published card, for comparison |

Re-render: `python3 card_ste.py --bench … --image … --check … --profiles … --source-commit …` (the same arguments as `scripts/card_mc.py`), then `python3 demo_ste.py`, then `readme_ste.py` with the tt-model venv, then `python3 ste_check.py README.md --all`.

## Word-count result (after the bullet pass, 2026-10-03)

| | published card | STE draft |
|---|---:|---:|
| sentences checked (prose outside code, tables, HTML and front matter) | 77 | 209 |
| instructions (limit 20 words) | 1 | 16 |
| descriptive sentences (limit 25 words) | 76 | 193 |
| sentences over the limit | 17 | **0** |
| longest sentence (words) | 70 | 25 (descriptive); 16 (instruction) |
| mean sentence length (words) | 21.6 | 11.1 |
| paragraphs with more than 6 sentences | 0 | **0** |
| list items | 22 | 160 |
| list items nested deeper than 2 levels | 0 | **0** |
| prose paragraphs (not lists) | 24 | 18 (all lead-ins or labels; see below) |

Draft length distribution: 99 sentences have 10 words or fewer, 77 have 11-15, 22 have 16-20, and 11 have 21-25. The 21-25 group contains descriptive sentences only. `ste_check.txt` lists every sentence with its length, type and limit, and every remaining prose paragraph.

Every sentence is under its limit, so the list of sentences that do not obey the limit is empty.

Integrity checks (draft vs published): the front matter is identical. The 5 code blocks are identical. The 71 table rows are identical. Each number in the draft prose also occurs in the published card. The draft drops repeated numbers only: the footnote no longer repeats "median of 60" (the intro above it has it), and the bucket list "32 / 64 / 128 / 224" now occurs once in the description.

## Bullet pass (user rule: use bullet points as much as possible)

- Each prose paragraph became a list where the content allows it: one fact or one instruction for each bullet, one STE sentence for each bullet where possible.
- Numbered lists for sequences: "To use a serve profile" (2 steps), "To use a different configuration in the range" (4 steps: set the three variables, then start the server), "To get the actions for your robot" (2 steps), and the tt-cli quickstart step.
- Plain bullets for facts: the description, the request fields, the endpoints, the input processing, "What runs where", the accuracy test conditions, the speed results, the LIBERO conditions and results, every limitation, the license, the demo section.
- Limitations: each limitation is one top-level bullet with a bold label; its facts are second-level bullets (two levels at most).
- Captions and notes under tables became bullets (profile table, LIBERO table, per-preset notes and † note, demo table).
- The sentences that `render_model_card` generates are bullets too (Runs on, Packaged, Quickstart, Serve profiles, Provenance).

### Prose paragraphs that remain (18), and why

| # | paragraph | why it stays prose |
|---|---|---|
| 1 | "This package runs the pi-0.5 vision-language-action policy of Physical Intelligence on one Tenstorrent Blackhole p150a." | the one-sentence summary under the title; the HF card list and the search preview show this first line |
| 2 | "Weights: … · Paper: … · Upstream code: … · Port: …" | a line of links, not sentences (unchanged from the published card) |
| 3 | "`POST /predict` accepts these fields:" | lead-in: it names the endpoint that the list describes |
| 4 | "Other endpoints:" | label of the second endpoint list |
| 5 | "To use a serve profile:" | lead-in of a numbered procedure (the goal of the steps) |
| 6 | "To use a different configuration in the range:" | lead-in of a numbered procedure |
| 7 | "If the server does not start:" | condition label of the troubleshooting bullets |
| 8 | "To use the model in Python, use the code in `code/`:" | instruction that introduces a code block (a code block cannot be a list item without a change to the code block) |
| 9 | "To get the actions for your robot:" | lead-in of a numbered procedure |
| 10 | "Input processing:" | label of a list of facts |
| 11 | "Test conditions:" (Accuracy) | label of a list of facts |
| 12 | "Default serve profile, served by this image over HTTP:" | label: it gives the profile and the method of the speed list |
| 13 | "Each serve profile of this image:" | caption above a table (tables stay tables) |
| 14 | "**10 flow-matching steps (all 32 presets)**" | table title (from the per-preset table) |
| 15 | "**2 cameras at 1 and 5 flow-matching steps**" | table title |
| 16 | "Notes:" | label of the per-preset notes |
| 17 | "Test conditions:" (LIBERO) | label of a list of facts |
| 18 | "**Caution:** Do not use this demo to estimate the results of `lerobot/pi05_base`. The demo uses different weights and a different configuration:" | an STE caution: the instruction first, then the reason; the details follow as bullets |

Each remaining paragraph is one sentence or a label, except #18 (2 sentences).

## Interim defaults (the lead's instruction while the 7 open questions are with the user)

- Table cells: kept as they are (the 71 table rows are identical to the published card).
- Per-preset footnote: STE text in the card (numbers from `RELEASE_TABLE.md` by regex); the original text stays verbatim in `PERF_PRESETS.md`.
- Spelling: British, as in the published card ("normalise", "denormalise").

## Glossary (one term for one thing; each term is defined at its first use in the card)

| term | meaning in this card |
|---|---|
| camera count / cameras | the number of real camera images in each request (1 to 4) |
| action horizon (H) | the number of actions in one output chunk (1 to 64) |
| flow-matching steps (N) | the number of Euler steps of the action expert (1 to 10); replaces "denoising steps" in prose |
| configuration | one set of camera count, H and N; the server builds one model for one configuration |
| serve profile | a named configuration of the server (`--profile`) |
| prompt bucket | a fixed prompt length that the device programs use: 32, 64, 128 or 224 tokens |
| action-row bucket (S) | a fixed number of action rows that the device programs use: 32 or 64 |
| preset | one combination of a camera count, a prompt bucket and an action-row bucket (32 presets) |
| device program | one persistent `ttnn.generic_op` megakernel on 110 Tensix cores; a request uses 3 (4 with 3 or 4 cameras) |
| trace replay | one replay of the Metal trace that holds the device programs of one request |
| request / call | one `POST /predict` (HTTP) or one `sample_actions` call (Python) |
| inference time | `timing_ms.inference` |

Technical names are used as-is: product names, API fields, environment variables, file names, model names (SigLIP, Gemma-2B, adaRMS, K / V caches, prefill, im2col, bf16 / bfp8 / fp32, HiFi3 / HiFi4, PCC, McNemar, p90).

## Before / after, by section

In an "After" quote, sentences that are on separate bullets in the draft are separated by " / " or written one after the other; every quoted sentence occurs in the draft (checked).


**Description**

- Before (68 words): "Every request runs the whole model as three persistent custom-kernel programs (`ttnn.generic_op` megakernels on 110 Tensix cores; four with 3-4 cameras): VISION (…), PREFIX (…) and EXPERT (…), replayed from one Metal trace per prompt bucket."
  After (bullet): "A device program is one persistent `ttnn.generic_op` megakernel on 110 Tensix cores. Each request uses three device programs (four with 3 or 4 cameras):" + three second-level bullets VISION / PREFIX / EXPERT.
- Before (49 words): "One image serves 1-4 cameras, action chunks of 1-64 steps and 1-10 flow-matching steps (a model per configuration, chosen at server start by serve profile or environment); each request runs in the smallest of four prompt buckets (32 / 64 / 128 / 224 tokens) that holds its prompt."
  After (bullets): "One image supports these configurations:" + "1 to 4 cameras." / "An action horizon (H) of 1 to 64 actions." / "1 to 10 flow-matching steps (N)."; then "The server builds one model for one configuration when it starts. A serve profile or three environment variables select the configuration."
- Before: "… running entirely on one Tenstorrent Blackhole p150a." (-ing modifier) After: "This package runs the pi-0.5 vision-language-action policy of Physical Intelligence on one Tenstorrent Blackhole p150a."

**Quickstart / request fields**

- Before: "Masked or missing cameras are refused: serve a model with that many cameras instead;" (passive, two topics)
  After: "The server refuses masked cameras and a wrong number of images. If you have fewer cameras, start a server for that number of cameras."
- Before: "optional `seed`: the initial flow-matching noise (the default is fixed, so the output is deterministic);"
  After: "`seed` (optional): the seed of the initial flow-matching noise. The default noise is fixed, thus the output is deterministic."
- Before (generated): "Serves on port 20000 (or the next free port); ready when the log says `Application startup complete`."
  After: "The server uses port 20000. If that port is busy, the server uses the next free port." / "The server is ready when the log shows `Application startup complete`."

**Select the cameras, the action horizon and the flow-matching steps** (heading was "Choosing cameras, horizon and steps": -ing noun)

- Before: "A model is built for one (cameras, horizon H, steps N) at server start; the prompt bucket is picked per request." (passive, phrasal "picked")
  After: "The server builds one model for one set of cameras, action horizon (H) and flow-matching steps (N) when it starts. The prompt bucket can change for each request."
- Before: "Any other combination in range is three environment variables of the container: `PI05_NUM_IMAGES` (1..4), `PI05_ACTION_HORIZON` (1..64), `PI05_NUM_STEPS` (1..10)."
  After (numbered procedure): "To use a different configuration in the range:" 1. "Set `PI05_NUM_IMAGES` on the container to the number of cameras (1 to 4)." 2. "Set `PI05_ACTION_HORIZON` to the number of actions (1 to 64)." 3. "Set `PI05_NUM_STEPS` to the number of flow-matching steps (1 to 10)." 4. "Start the server."
- Before: "An out-of-range value refuses at start with a message that names it; so do a mesh, …"
  After: "If a value is out of range, the server does not start. The error message gives the name of the value. The server also refuses a mesh, a `PI05_BATCH_SIZES` value other than 1 and a `PI05_TOKEN_LEN` value larger than 224."

**Response**

- Before: "Denormalise with `(a+1)*(q99-q01)/2+q01` using your own dataset's stats, then slice to your action dim (e.g. the first 7 for LIBERO)."
  After (numbered procedure): "To get the actions for your robot:" 1. "Denormalise the actions with `(a+1)*(q99-q01)/2+q01` and the statistics of your dataset." 2. "Use only the columns of your action dimension (for example, the first 7 columns for LIBERO)."
- Before: "Images are squash-resized to 224×224 and normalised as openpi does (…, bit-exact to openpi's PyTorch inputs)."
  After: "The server resizes each image to 224×224 and does not keep the aspect ratio. Then it normalises each image as openpi does: `x * float32(1/255) * 2 - 1`. The result is bit-exact to the PyTorch input of openpi."

**What runs where**

- Before: "The 32 compiled presets are cameras {1, 2, 3, 4} × prompt buckets {32, 64, 128, 224} × suffix buckets {32, 64} action rows (H ≤ 32 runs in the 32-row bucket); N is folded into the expert's weights at construction."
  After: "A preset is one combination of a camera count, a prompt bucket and an action-row bucket. The 32 presets are all combinations of 4 camera counts, 4 prompt buckets and 2 action-row buckets (32 and 64 rows). If H is 32 or less, the request uses the 32-row bucket. The model includes N in the expert weights when it builds the model."

**Accuracy**

- Before: "The full matrix: 32 presets × N 1..10 = 320 (preset, steps) sets, each against the fp32 torch reference on 6 padded prompts (A2), plus an fp32 expert loop fed the device's own K / V caches (A4), the K / V caches (A3, A6), bit-identical replays and an independent cross-check."
  After (bullets under "Test conditions:"): "The full test matrix has 320 sets: 32 presets × N 1..10." / "A2: each set compares the device output with the fp32 torch reference on 6 padded prompts." / "A4: the expert against an fp32 expert loop with the K / V caches of the device." / "A3 and A6: the K / V caches." / …

**Speed**

- Before: "Served by this image over HTTP, default profile (2 × 224² images, the card's prompt = 142 tokens -> the 224-token bucket, H = 50, N = 10), median of 100 warm requests: …"
  After (bullets under "Default serve profile, served by this image over HTTP:"): "The request had 2 images of 224 × 224 and the prompt of this card." / "The prompt had 142 tokens, thus it used the 224-token bucket. H was 50 and N was 10." / "For 100 warm requests, the median `timing_ms.inference` was 56.44 ms (p90 56.65)." / …
- Before (per-preset intro): "Device replay latency per preset: the trace replay of a call's programs, measured in-process on the release code (L = prompt bucket in tokens, S = action-row bucket; …)."
  After: "The next tables show the device time of one trace replay for each preset. The data is from in-process measurements on the release code. L is the prompt bucket in tokens. S is the action-row bucket."
- Before (footnote): "† one of the two builds ran while a foreign CPU process was active (windows below). Build-to-build se is <= 0.042 ms at 10 steps; at 1 / 5 steps it reaches 0.19 ms."
  After: "† A different CPU process was active during one of the two builds. `PERF_PRESETS.md` gives the time windows. The standard error between the builds is 0.042 ms or less at 10 steps. At 1 or 5 steps, it is 0.19 ms or less."

**LIBERO closed loop**

- Before: "With 1 camera (the wrist image dropped) both backends collapse (0-2 / 100): pi05_libero needs the wrist view, so those rows are a record, not an accuracy signal."
  After (bullets under the table): "With 1 camera, the policy does not get the wrist image." / "Then both backends fail (0 to 2 successes of 100), because pi05_libero needs the wrist view." / "These rows are a record. They do not show accuracy."
- Before: "The GPU latency is not reported: it was measured differently (…), so it is not comparable."
  After: "This card does not give the GPU latency, because its measurement was different (openpi model time only, on a shared host with load)."

**Limitations**

- Before: "Rounding the reference's own K / V to bf16 / bfp8 leaves it ≥ 0.99994 and a higher VLM matmul fidelity does not help: the trajectory amplifies the prefix's reduced-precision error."
  After: "If you round the K / V caches of the reference to bf16 or bfp8, the reference stays at a PCC of 0.99994 or more. A higher fidelity for the VLM matmuls does not help. This trajectory increases the reduced-precision error of the prefix."
- Before (warning, reason first): "The prompt tokenizer `google/paligemma-3b-pt-224` is **gated** (Gemma terms): accept the terms and `hf auth login` before `tt serve`, or send `tokens`."
  After (instruction first; a limitation bullet "**Gated tokenizer.**" with second-level bullets): "Accept the Gemma terms of `google/paligemma-3b-pt-224`. Then use `hf auth login` before `tt serve`." / "The prompt tokenizer of this model is **gated** under the Gemma terms." / "If you cannot get access, send `tokens` instead of `prompt`."
- Before: "One live model per device (a second one is refused until the first is closed)."
  After: "Only one model can use a device at a time. If a second model starts, the server refuses it until the first model closes."

**Demo section, Caution** (instruction first)

- Before: "**Caveat: swapped weights.** The demo uses the LIBERO fine-tune …"
  After: "**Caution:** Do not use this demo to estimate the results of `lerobot/pi05_base`. The demo uses different weights and a different configuration:" + the details in a second paragraph.

## Sentences not brought under the limits

None: all 209 checked sentences obey the limits (instructions ≤ 20 words, descriptive ≤ 25), and every paragraph has 6 sentences or fewer.

## Open questions for the review

1. **Prose in table cells.** The brief says not to change table data cells, so 71 table rows are unchanged. Some cells contain long non-STE prose: the "What runs where" rows, the "Check" column of the Accuracy table, the LIBERO column header, and the serve-profile "configuration" column (from the `description:` of each profile in `tt-model.yaml`). Rewrite these cells too, or keep them?
2. **The per-preset footnote.** You earlier asked to keep the `RELEASE_TABLE.md` footnote and daggers verbatim. The draft rewrites the footnote and the † note in STE: every number comes from the file (regex-checked against the source text), and `PERF_PRESETS.md` keeps the original text verbatim. Is that acceptable, or must the card keep the original footnote?
3. **STE dictionary.** The draft applies the rules in your brief (sentence length, active voice, simple tenses, no -ing forms as nouns or modifiers, no phrasal verbs, idioms or contractions, articles, noun clusters, one term per meaning). I did not have the ASD-STE100 dictionary, so I could not check each word against its approved list. Words to check: "serve" / "served" (also a CLI command), "supports", "executes", "replays", "readback", "megakernel", "deterministic", "bit-exact", "comparators".
4. **-ing words that remain**: "flow-matching", "language embedding", "adaRMS time conditioning" are model terms, and "during" is a preposition. I kept these as technical names. "Fine-tune" is used as a noun ("no task-specific fine-tune").
5. **State, not passive**: "The default noise is fixed", "The outputs are normalised actions", "the tokenizer … is gated" describe a state (adjective use), so I kept them. Change them if the review counts them as passive.
6. **Not in scope of this draft**: `demo/README.md` (pi05-libero-gpu-base's text), `SERVING.md`, `GPU_COMPARISON.md`, `PERF_PRESETS.md`, comments inside code blocks, and the YAML front matter. Tell me if any of these must also be in STE.
7. **Spelling.** The draft keeps the British spellings of the published card ("normalise", "denormalise"). ASD-STE100 uses US spelling ("normalize"). Change to US spelling?

## Round 3 (2026-10-03): the user's answers applied

The user answered the 7 open questions. The draft now applies:

1. **VLA domain terms are kept technical names** (see the glossary addition below). "Action chunk" is the main term; "the action chunk has H actions" is the only definition of H. The card no longer uses "action horizon" except in identifiers (`PI05_ACTION_HORIZON`, `action_horizon`).
2. **Table cells get the STE rewrite**, with every number and identifier kept: "What runs where" (all cells and the header), the Accuracy "Check" and "Result" columns, the serve-profile "Configuration" column (STE text in `readme_ste.py`; the staging `tt-model.yaml` keeps its own text), the LIBERO header, the speed-table header and the demo table.
3. **US spelling everywhere** ("normalize", "normalized", "denormalize", "tokens"; this also matches the API field `normalized`).
4. **Scope**: the card plus `SERVING.md`, `GPU_COMPARISON.md`, `PERF_PRESETS.md` (prose only) and the staged `demo/README.md`; see the file index at the end.

### Card counts after round 3

| | published card | STE draft |
|---|---:|---:|
| prose sentences | 77 | 213 |
| over the limit | 17 | **0** |
| longest sentence | 70 | 25 |
| mean length | 21.6 | 11.1 |
| table-cell sentences (prose cells) | 32, 2 over 25 (longest 28) | 61, 0 over 25 (longest 17) |
| British spellings (distinct words) | 6 | **0** |
| list items / nested deeper than 2 levels | 22 / 0 | 163 / **0** |
| prose paragraphs | 24 | 18 (the same lead-ins and labels as in the round-2 table) |

`integrity_check.py` (published card vs draft): front matter and the 5 code blocks identical; no number lost, no new number; no `code` span lost; URLs and hex ids identical: **PASS**. The table rows are no longer identical, by the user's answer 1.

Representative before / after pairs of round 3:

- Before: "An action horizon (H) of 1 to 64 actions." After: "An action chunk of 1 to 64 actions (the action chunk has H actions)."
- Before (table cell): "image decode / resize / normalisation, im2col of the patches, prompt building and tokenisation, the attention-mask and RoPE rows, the initial noise, the host-to-device copies, the readback" After: "Image decode, resize and normalization. The im2col of the patches. The prompt and its tokens. The attention-mask rows and the RoPE rows. The initial noise. The copies from the host to the device. The readback."
- Before (table header): "p150a policy call (server side, incl. host inputs; mean)" After: "Mean time of one p150a policy call on the server, with the host inputs"
- Before (profile description): "4 cameras, 50-action chunk, 10 steps (two vision programs per call; K/V caches in DRAM)" After: "4 cameras. An action chunk of 50 actions. 10 flow-matching steps. Two vision device programs for each call. The K / V caches are in DRAM."

### Glossary addition: kept technical names (VLA domain terms, used as they are)

jitter, prefix, expert, action expert, decode, action chunk, flow matching / flow-matching, denoising step, proprioceptive state, prefill, embedding (patch / language embedding), adaRMS (time) conditioning, pad tokens / padding, K / V caches, im2col, readback, megakernel, Euler step, bf16 / bfp8 / fp32, HiFi2 / HiFi3 / HiFi4, PCC, McNemar, p10 / p90, SigLIP, Gemma, openpi, lerobot, LIBERO.

## File index (all drafts; the published / staged files are unchanged)

| draft | source (unchanged) | change notes | sentence listing |
|---|---|---|---|
| `README.md` (the card, generated) | `models/pi05-base-p150-mc/README.md` | this file | `ste_check.txt` |
| `SERVING.md` | `models/pi05-base-p150-mc/SERVING.md` | `CHANGES_SERVING_PERF_DEMO.md` §1 | `ste_check_SERVING.txt` |
| `GPU_COMPARISON.md` | `models/pi05-base-p150-mc/GPU_COMPARISON.md` | `CHANGES_GPU_COMPARISON.md` | `ste_check_GPU_COMPARISON.txt` |
| `PERF_PRESETS.md` (prose only) | `models/pi05-base-p150-mc/PERF_PRESETS.md` | `CHANGES_SERVING_PERF_DEMO.md` §2 | `ste_check_PERF_PRESETS.txt` |
| `demo/README.md` | `models/pi05-base-p150-mc/demo/README.md` | `CHANGES_SERVING_PERF_DEMO.md` §3 | `ste_check_demo_README.txt` |

Rules: `STE_RULES.md`. Tools: `ste_check.py` (word counts, prose cells, British spellings, nesting, prose paragraphs) and `integrity_check.py` (numbers, code blocks, `code` spans, URLs, hex ids against the source).

## Round 4 (2026-10-03): one serve profile `p150`, configuration through the environment

- `tt-model.precard.yaml` / `tt-model.yaml` (draft): one serve profile `p150` (default configuration 2 cameras, H 50, N 10 from `serve.env`); the 5 other profiles are removed. `build/tt_kernel_manifest.json`: the draft wire manifest with the same single profile (the image `36f651704bf1` is unchanged).
- Mechanism (tested on the device, `build/env_test.sh`, `build/env_test.log`): `tt-model serve` has no `--env` flag and does not pass the host environment (`build/print_default.txt` and `build/print_hostenv.txt` are identical with `PI05_NUM_IMAGES=3` set on the host). The tested method: `tt-model serve … --print` gives the `docker run` command; change the three `PI05_*` values in it and run it. Result: `/info` cameras 3, `action_horizon` 10, `num_steps` 5, 4 device programs; smoke test PASS; `PI05_NUM_IMAGES=5` stops the container at start (exit code 3) with the model's message.
- Card: new section "How to change the configuration" (configuration table, "Where in the code" table, numbered steps for the server and for Python, cameras, out-of-range values). Every file:line was verified by content (`line_refs_check.txt`: 23 / 23 OK, against the staged `code/`).
- Card: "Benchmarks" = ONE table (cameras × action-row bucket × N rows; prompt bucket 32 / 64 / 128 / 224 columns; values = the means of `RELEASE_TABLE.md`, without the ± se and the † marks) + 3 bullets. The 6-profile served-latency table, the previous-release comparison and the per-preset ± tables are removed from the card. The served latency of the six configurations moved to `SERVING.md` ("How to change the configuration").
- `integrity_check.py` now reports expected differences for the card and `SERVING.md`: removed numbers (the profile table, the ± values, the previous-release line), new numbers (line numbers of "Where in the code", the tested results), and a new code block (the tested commands).

- User correction (2026-10-03): "jitter" is a kept VLA term. GPU_COMPARISON.md:309 is back to "adds jitter" (the lead's replacement "variation" is reverted); "jitter" is in the glossary of kept technical names (here and in STE_RULES.md).
