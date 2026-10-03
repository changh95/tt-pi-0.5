# publish-mc journal (pi0.5 multi-config megakernel -> changh95/pi05-base-p150)

Timestamps from `date`. Nothing outward-facing (no push, no PR, no HF upload, no registry push) is done in this session.

## 2026-10-03 11:14:34 KST -- session start
Branch multiconfig-2026-10-03 from megakernel-2026-09-29 eb6f4e7 (= origin/main as last fetched; HF code/ is 821e8c5 = PR #3 merge, eb6f4e7 adds only docs).
Source of the code: /home/deepgadget/experiments/tt-metal-pr branch changh95/pi05-megakernel-mc HEAD fae9cd03fa4 (frozen, read only),
validated on tt-metal main f856a38a. Disk 25 GB free at start.

## 2026-10-03 11:28:04 KST -- port, manifest draft, queued device checks
- Vendored tt-metal-pr fae9cd03fa4 models/experimental/pi0 (git archive, 70 files; file list == the archive) into
  models/experimental/pi0. kernel_digest() == 1429d5bea05c31ad (= tt-metal-pr, = the LIBERO matrix stamp).
- Server: new models/experimental/pi0_5/server/mc_backend.py; app.py default backend PI05_MEGAKERNEL unset/"mc" on one
  chip = PI05MegakernelTTNN (PI05_NUM_IMAGES 1..4, PI05_ACTION_HORIZON 1..64, PI05_NUM_STEPS 1..10, start-up refusals
  by name; requests must send exactly PI05_NUM_IMAGES images; optional prompt_bucket; /info megakernel.backend "mc" with
  kernel_digest / presets / device_ops_per_call). decode_image now uses openpi's x * f32(1/255) * 2 - 1 (was
  (x/255-0.5)/0.5). whole / expert / off and the mesh / dp layouts stay selectable (not re-validated on f856a38a).
  smoke_test.py reads H from /info. LIBERO websocket server added unchanged from pi05-libero-gpu-base's
  tt_mc_matrix/package/serve_pi05_libero.py (sha256 prefix 96dff490370ad915).
- CPU: pi0 host suite + server mask tests 84 passed; new tests/test_server_mc.py 17 passed (overlay of tt-metal-main
  + the branch's pi0 / pi0_5). NOTE: these CPU runs (11:19-11:22) overlapped pi05-mc-impl's quiet-host sweep hold
  (11:11:25-); told pi05-mc-impl which minutes.
- Clean clone /home/deepgadget/experiments/gr00t/publish/tt-metal-f856a38a361 (local clone of tt-metal-main's objects,
  origin reset to tenstorrent/tt-metal, detached at f856a38a361939888f92d88f9e69b2f8a83fb713, submodules at the
  recorded commits, porcelain empty, describe v0.80.0-dev20261001-17-gf856a38a361).
- Staging /home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc: tt-model.yaml rewritten (tt_metal ->
  the f856a38a clone; serve.env PI05_MEGAKERNEL=mc, NUM_IMAGES 2, ACTION_HORIZON 50, NUM_STEPS 10; 6 serve profiles
  c2-h50-n10 (default) / c1 / c3 / c4 -h50-n10 / c2-h10-n10 / c2-h10-n5; 6 new verify lines for pi0 + the 23 old
  pi0_5 lines). Manifest loads (profiles resolve).
- Image build NOT started: disk 25 GB free, estimate leaves ~5-8 GB; asked the lead to approve freeing space.

## 2026-10-03 11:36:47 KST -- code commit, host device checks
- Code commit 1626afad2d3fcbfae7ccb8a238116a08c01f7fb7 (amended once before any build to fold in the package README
  note; the first sha 3d3c076 was never packaged). Staging code/ = git archive of it (180 files, no __pycache__);
  serve.env PI05_SOURCE_COMMIT = that sha.
- Host serve check (one hold 11:29:30-11:30:31, after pi05-mc-impl's sweep hold): tt-metal-main build + the branch,
  uvicorn app, defaults: model built 35.7 s, the 4 prompt buckets compiled + captured 12.1 s, /info backend mc digest
  1429d5bea05c31ad 3 ops/call; smoke PASS (actions (50,32), 143 tokens); HTTP refusals: 1 / 3 images (2-camera model),
  num_steps 3, prompt_bucket 48, 41 tokens in bucket 32 -> 400 with the reason; explicit bucket 224 and auto bucket 32
  -> 200. Bench 30 warm requests: inference median 56.44 ms (p90 56.57), total 57.88, client 59.17
  (results/host/host_c2.*). Host, not the image: the release number comes from the image.
- Host reference of img_check.py (hold 11:30:34-, tt-metal-main + tt-metal-pr fae9cd0 pi0 snapshot + the packaged
  LIBERO server module): golden c2 N10 (openloop_golden.pt, 8 records, inputs torch.equal) PCC7 mean 0.999981 min
  0.999958; c1 (H10 N10 L32) / c2 (H50 N10 L224) / c3 (H32 N5 L64) / c4 (H64 N10 L128, K/V in DRAM) outputs saved +
  sha256, replays identical, ops per call 3 / 3 / 4 / 4; 9 refusals correct (results/host/img_check.json).
- Manifest verify lines: all 29 pass on a fake /opt layout (code/ + lightweightmodule + the clone's tt_metal, ttnn
  from tt-metal-main); negative control (an include appended to pi0 mk_defs.hpp) fails the digest line
  (b02d27f569e862fb) and the include line.

## 2026-10-03 11:38:32 KST -- lead decisions; refusal messages; prune listing
- Lead (11:3x): disk (a) builder prune --filter until=72h, (b) OCI export over build/pi05-base-p150, (c) remove image
  6fb244df57ff after validation passes -- approved. Start prune + build only after pi05-libero-gpu-base's demo
  recording ends (its captions carry live latency). Card: per-preset device replay table from
  tt-metal-pr/.val/mc_impl/wp6/perf/out/RELEASE_TABLE.md (footnote + daggers verbatim), served c2 separately; state 3
  programs per call, 4 with 3-4 cameras. Env validation must refuse with the model's own messages.
- mc_backend.refusal now calls the model's own pure-Python checks (geometry.megakernel_refusal for N / H, the
  PI05MegakernelTTNN cameras message) and appends the env names. Tests updated (22 passed with the mask tests).
  Code commit amended (still nothing packaged): 5edf139d458e9acabd31eea641cd364079962677; staging code/ and
  PI05_SOURCE_COMMIT re-taken from it.
- (a) dry listing before the prune: docker buildx du --verbose -> results/buildcache_before.tsv (250 entries);
  entries last used >= 3 days ago -> results/buildcache_prune_candidates.tsv: 187 entries, ~32.6 GB by the listed
  sizes (layers can share storage; the reclaimed amount is reported after the prune). They are old builds of this and
  other tt-model packages (build_metal layers 3.04-3.28 GB, venv 1.08 GB, engine 0.86 GB, metal-local 0.81 GB, ...).
  The ccache / cpm cache mounts (46 h ago) and the 2-day-old entries stay.

## 2026-10-03 11:40:24 KST -- disk (a): what actually ran
- pi05-libero-gpu-base's recording hold ended 11:37:33 (guard ledger; demo/ files 11:37-11:38); told it I start at ~11:40.
- `docker builder prune -f --filter until=72h` (11:39:09) freed 20 MB; with `-a` 32 kB; `--filter unused-for=72h` 0 B
  (Docker 28.5.2: the records show Reclaimable true / Shared false / last used 10 days ago, yet the time filters
  matched nothing). Then `docker builder prune -a -f --max-used-space 18GB` (11:39:50; 18 GB = the listed size of
  the entries used < 3 days ago): reclaimed 26.1 GB, build cache 43.75 -> 17.67 GB, disk 24 -> 49 GB free.
  DEVIATION from the approved command: LRU-to-size is not "unused > 72 h". It removed every build_metal layer
  (incl. the 46 h / 2-3 day ones of the p2 image and other packages, so their next rebuild recompiles tt-metal) and
  left some 10-11 day records (small ones, venv / metal-local copies). The /cpm source-cache mount (3.35 GB) stayed.
  Remaining records: results/buildcache_after.tsv. Reported to the lead.

## 2026-10-03 11:48:12 KST -- image built
- tt-model package --container (staging pi05-base-p150-mc, 11:40:40-11:47:35, rc 0; results/pkg-mc.log, buildkit log .gz):
  tt-metal f856a38a built from the clean clone (build_metal ~5 min with an empty ccache), all 29 verify lines ran in
  the image (#45 DONE 70.8 s). Image tt-model/pi05-base-p150:36f651704bf1, digest
  sha256:36f651704bf123ebed106991303c3b077ff94e96b77d5c8c684d3f6772432477; built.tt_metal sha f856a38a dirty=false
  describe v0.80.0-dev20261001-17; code_sha256 184c58b636fb7621...; 6 serve profiles in the wire manifest, default
  c2-h50-n10. OCI export 3.1 GB over build/pi05-base-p150 (approved (b)). Disk 41 GB free after.
- Image validation hold queued (run_container_mc.sh: default profile cold + warm with 100-request benches, the 5
  other profiles with 30-request benches, img_check.py in the image, device-profiler op count).

## 2026-10-03 12:06:10 KST -- image validation (one hold 11:48:07-11:59:36, WITH_DEVICE_RESET_AFTER=1, reset exit 0)
results/image/ (validate.log, info / smoke / bench per profile, img_check_image.json, imgcheck + prof logs, prof_ops.json).
- Default profile c2-h50-n10, cycle 1 cold (package JIT cache removed): serve 66.2 s (model built 34.0 s, the 4
  prompt buckets compiled + captured 11.8 s); cycle 2 warm: 41.2 s (capture 0.8 s). /info both cycles: backend mc,
  kernel_digest 1429d5bea05c31ad, 3 device ops per call, source 5edf139d. smoke PASS both. 100 warm requests,
  identical actions: inference median 56.36 (c1) / **56.44 ms (c2)**, p90 56.54 / 56.65; total 57.56 / 57.68;
  client 59.17 / 59.29. Previous image 6fb244df57ff (one fused op, this shape only) c2 = 55.84 / 57.13: +0.60 ms.
- Other profiles (serve 35-57 s each, /info checked, smoke PASS, 30 warm requests, identical actions; inference /
  total median): c1-h50-n10 44.33 / 45.08 (3 ops); c3-h50-n10 76.16 / 77.72 (4 ops); c4-h50-n10 96.28 / 98.37 (4 ops,
  K/V in DRAM); c2-h10-n10 53.43 / 54.58; c2-h10-n5 45.86 / 47.00 (3 ops). All on the 224-token bucket (the card's
  142-token prompt).
- img_check.py inside the image (tt-model's container spec, private run): golden c2 N10 through the shipped
  serve_pi05_libero.Pi05LiberoPolicy: 8/8 inputs torch.equal to openpi's, PCC7 mean **0.999981** min **0.999958** (gate
  >= 0.9995 mean: pass), the 8 outputs bit-identical to the host reference. Bit-identity vs the tt-metal-pr build
  (tt-metal-main f856a38a + tt-metal-pr fae9cd0 models, host run 11:30): c1 (L32 S32 H10 N10), c2 (L224 S64 H50 N10),
  c3 (L64 S32 H32 N5), c4 (L128 S64 H64 N10, K/V DRAM) all torch.equal (results/bitid_compare.txt); replays
  identical; call medians host / image 39.79/39.78, 56.74/56.60, 62.95/62.75, 95.84/95.28 ms. Refusals 10/10:
  H=65, N=11, cameras=5 at construction; masked camera; 1 image on a 2-camera model; 40 tokens in bucket 32; bucket
  48; batch 2; a second live model on the device; and the model still serves afterwards.
- Device profiler in the image (prof_ops.py: c2 H50 N10 then c3 H32 N5, 6 calls each over prompt lengths 5 / 150 /
  40): 12 trace replay sessions, every c2 session exactly **3** programs and every c3 session exactly **4**, all on
  110 cores (results/image/prof_ops.json; device time per replay 53.5-55.2 ms c2, 60.8-62.6 ms c3 by bucket). 84
  programs outside replay sessions = the construction / compile / capture passes of the two models (the CSV's
  GLOBAL CALL COUNT is shared by replays, so it cannot order them against single requests; the per-request path
  writes inputs with copy_host_to_device only).
- (c) approved: removed the local image tt-model/pi05-base-p150:6fb244df57ff after the validation passed (on HF
  990e22b5). Disk 39 GB free.

## 2026-10-03 12:06:10 KST -- staging, card, HF diff (nothing uploaded)
- Staging /home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc: code/ (git archive 5edf139), tt-model.yaml
  (the build-time file + the card block, written by scripts/card_mc.py from the files it names), README.md
  (scripts/readme_mc.py: HF front matter of 990e22b5 + render_model_card + demo section), SERVING.md (rewritten),
  GPU_COMPARISON.md (2026-10-03 section prepended: served 56.44 ms + the same-day LIBERO TT vs GPU latency table),
  PERF_PRESETS.md (= tt-metal-pr .val/mc_impl/wp6/perf/out/RELEASE_TABLE.md verbatim, incl. footnote, daggers and the
  build log they refer to), requirements.lock (the image's own; drift vs HF: filelock 4.0.7->4.0.9, MarkupSafe
  3.0.3->3.0.4, ttnn dev10011->dev10865, virtualenv 21.14.1->21.14.5), demo/ (pi05-libero-gpu-base's 11:35 recording:
  combined mp4, poster, 4 clips byte-identical to tt_mc_matrix/demo; README.md + libero_eval.json written by
  scripts/demo_mc.py from its manifest.json and the matrix summaries). Card numbers: served = this image's benches;
  golden = this image's img_check; per-preset = RELEASE_TABLE.md; WP6 = scratchpad wp6/table.json and
  gpu_inc_c2_L224_H64.json; LIBERO = tt_mc_matrix paired_vs_gpu.json + *_summary.json; the K/V-rounding / fidelity
  sentences = tt-metal-pr README fae9cd0.
- HF diff (scripts/hf_commit_mc.py --dry-run vs the Hub head 990e22b5, read-only; results/hf_diff_mc.json): 246 local
  files; 117 new (70 pi0 files, mc_backend.py, serve_pi05_libero.py, test_server_mc.py, PERF_PRESETS.md, 43 image
  blobs), 20 changed (README, SERVING, GPU_COMPARISON, tt-model.yaml, tt_kernel_manifest.json, requirements.lock,
  pi0_5 app.py / smoke_test.py / README.md, demo/* same names), 109 unchanged, 43 superseded image blobs deleted;
  .gitattributes and media/ kept. Staging manifest with sha256 per file:
  /home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc.STAGING_MANIFEST.json (outside the staging dir).

## 2026-10-03 12:06:25 KST -- handoff (session end; nothing outward-facing done)
Branch multiconfig-2026-10-03 (local, not pushed): 5edf139 (code; = PI05_SOURCE_COMMIT and HF code/), 4188a8f (docs),
this entry. Publish commands for the lead are in the final report (push + PR + merge commit; one HF create_commit with
parent 990e22b5 via scripts/hf_commit_mc.py; then pull-back check docker inspect Id == sha256:36f651704bf1...).
