# STE draft rules for the pi05-base-p150 documents (ASD-STE100 Issue 9, as briefed by the user)

Draft only. Do not publish, push or commit anything. Do not edit any file outside `draft_ste/`. The published / staged
files under `/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc/` must stay unchanged.

## Sentences and paragraphs

- Instructions (procedural sentences): 20 words or fewer. Descriptive sentences: 25 words or fewer.
- Paragraphs: 6 sentences or fewer; one topic in each paragraph.
- Instructions use the imperative ("Set `PI05_NUM_IMAGES` to 3."). One instruction in each sentence, except actions that occur at the same time.
- Active voice in procedures. Avoid the passive in descriptions where possible. ("is fixed", "is gated", "are normalized" as a state are acceptable.)
- Simple tenses only: present, simple past, simple future. No -ing form as a noun or as a modifier ("the default profile runs", not "running the default profile gives"). Exception: kept technical names (below).
- Approved words with one meaning: "use" (not utilize/leverage), "start" (not spin up), "show" (not illustrate), "about" (not approximately). No phrasal verbs (set up, pick up, go to), no idioms, no slang, no contractions.
- Use articles ("the", "a"). No noun clusters of more than 3 words: break them up with "of", "for", etc.
- Conditions before the instruction ("If the prompt has more than 224 tokens, ..."). Warnings and cautions: the instruction first, then the reason.
- Use the same term for the same thing everywhere (glossary below); define each term at its first use in each document.

## Bullets (user rule: use bullet points as much as possible)

- Turn prose paragraphs into bullet lists wherever the content allows. One fact or one instruction for each bullet; one STE sentence for each bullet where possible.
- Numbered lists for sequential steps; plain bullets for facts, limitations and features.
- Keep a short lead-in sentence before a list only if it adds information. Do not write paragraphs that only repeat the list.
- Nest at most two levels (a second-level bullet is indented by 2 spaces; never indent a bullet by 4 or more spaces).
- Tables stay tables. A caption or a note under a table becomes bullets.

## What must not change

- Numbers (every number value must still occur), code blocks (byte-identical), commands, file names, paths, identifiers, API fields, environment variables, URLs, commit / image hashes, model names, YAML front matter.
- Table cells: rewrite their prose in STE (each cell sentence 25 words or fewer), but keep every number and identifier in the cell.
- Spelling: US English everywhere ("normalize", "normalized", "tokenize", "behavior", "license" as a noun too).

## Glossary (one term for one thing)

| term | meaning |
|---|---|
| action chunk | the actions that one request returns; the main term. "The action chunk has H actions" defines the action horizon (H); do not use "action horizon" as a separate term otherwise |
| flow-matching steps (N) | the number of Euler steps of the action expert (1 to 10). "denoising step" is a kept technical name and may occur in identifiers / quotes |
| camera count / cameras | the number of real camera images in each request (1 to 4) |
| configuration | one set of camera count, H and N; the server builds one model for one configuration |
| serve profile | a named configuration of the server (`--profile`) |
| prompt bucket | a fixed prompt length that the device programs use: 32, 64, 128 or 224 tokens |
| action-row bucket (S) | a fixed number of action rows that the device programs use: 32 or 64 |
| preset | one combination of a camera count, a prompt bucket and an action-row bucket (32 presets) |
| device program | one persistent `ttnn.generic_op` megakernel on 110 Tensix cores; a request uses 3 (4 with 3 or 4 cameras) |
| trace replay | one replay of the Metal trace that holds the device programs of one request |
| request / call | one `POST /predict` (HTTP) or one `sample_actions` call (Python) |
| inference time | `timing_ms.inference` |

Kept technical names (VLA domain terms, used as they are): jitter, prefix, expert, action expert, decode, action chunk, flow matching / flow-matching, denoising step, proprioceptive state, prefill, embedding, adaRMS (time) conditioning, padding / pad tokens, K / V caches, im2col, readback, megakernel, bf16 / bfp8 / fp32, HiFi2 / HiFi3 / HiFi4, PCC, McNemar, p10 / p90, SigLIP, Gemma, openpi, lerobot, LIBERO, autocast, torch.compile, TensorRT. Product names, API fields and environment variables are allowed as-is.

## Check

Run `python3 /home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc/draft_ste/ste_check.py <draft file> --all`.
It must show: 0 sentences over the limit, 0 table-cell sentences over 25, no British spellings, 0 list items nested deeper than 2 levels, 0 paragraphs over 6 sentences. Report the remaining prose paragraphs (not lists) and why each one stays prose.
Also check the integrity against the published file: every number of the published file occurs in the draft and the draft has no new number; code blocks identical.
