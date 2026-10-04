"""N16 companion docs (user decision 10-05: update them): SERVING.md, PERF_PRESETS.md, GPU_COMPARISON.md from their HF
head versions (061677cf, *.hf_head_061677cf.md), same style as the card (STE, short bullets, VLA terms, US spelling).
Every number is read from its file and asserted; every edit must match its target text exactly once.
usage: companions_n16.py OUT_DIR"""
import json, os, re, sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
PUB = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc"
SP = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc"
MAN = json.load(open("/home/deepgadget/experiments/tt-models/build/pi05-base-p150/tt_kernel_manifest.json"))["container"]
B, IMG = MAN["built"], MAN["image"]
TAG = IMG["tag"].split(":")[1]; assert TAG == "fe6ecd0b4e86" and IMG["digest"].startswith("sha256:fe6ecd0b4e86")
TM = B["tt_metal"]; assert TM["sha"].startswith("c718b5df9b9") and not TM["dirty"]
SRC_COMMIT = [l for l in open("/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc/tt-model.yaml") if "PI05_SOURCE_COMMIT:" in l][0].split('"')[1]
assert SRC_COMMIT.startswith("33528a8")
DIGEST = "26f0c46b7f1721c1"


class Doc:
    def __init__(self, name):
        self.name, self.s = name, open(f"{HERE}/{name}.hf_head_061677cf.md").read()

    def rep(self, a, b):
        n = self.s.count(a)
        assert n == 1, (self.name, n, a[:90])
        self.s = self.s.replace(a, b)

    def write(self):
        assert "PENDING[" not in self.s
        open(f"{OUT}/{self.name}.md", "w").write(self.s)
        print(f"{self.name}.md: {len(self.s.splitlines())} lines")


# ---------------------------------------------------------------- inputs
BENCH = {p: json.load(open(f"{PUB}/results/n16/bench/bench-{p}.json")) for p in ("non-scalable", "scalable")}
for p, b in BENCH.items():
    assert b["n"] == 100 and b["identical_actions_all"] and b["info_profile"]["name"] == p
VAL = open(f"{PUB}/results/n16/val/val16.log").read()
assert VAL.count("PASS pi05") == 6 and "done RC=0" in VAL


def boot(p):  # container start -> READY, and the prompt-bucket compile + capture, from the container log timestamps
    ts = {}
    for l in open(f"{PUB}/results/n16/bench/container-{p}.log"):
        for k in ("Started server process", "Warming up", "Warmup complete", "Application startup complete"):
            if k in l and k not in ts:
                ts[k] = datetime.fromisoformat(l.split()[0][:26])
    return (ts["Application startup complete"] - ts["Started server process"]).total_seconds(), \
           (ts["Warmup complete"] - ts["Warming up"]).total_seconds()


BOOT = {p: boot(p) for p in BENCH}
RT = open("/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/wpv/perf/out/RELEASE_TABLE.md").read()
assert "Commit dd9431fa10f" in RT and "c718b5df9b9" in RT
W = json.load(open("/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/wpv/summary.json"))
E = W["eth"]; assert W["profiles_identical"] and not E["new"]
a2f = sorted((int(c[0].split("_N")[1]), c[3], c[4]) for c in E["known"] + E["traj"] if c[1] == "A2")
a4f = [c for c in E["known"] + E["traj"] if c[1] == "A4"]; a4n = [c for c in a4f if not c[0].endswith("_N1")]
assert [n for n, _, _ in a2f] == list(range(5, 17)) and len(a4f) == 24 and len(a4n) == 3
gpu = {}
for f in ("gpu_inc_c2_L224_H64.json", "gpu_inc_c2_L224_H64_n16.json"):
    for n, d in json.load(open(f"/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/wp6/{f}"))["arms"]["bf16"]["per_N"].items():
        gpu[int(n)] = [x["pcc"] for x in d["seeds"] if x["seed"] == 5][0]
LD = "/home/deepgadget/experiments/gr00t/libero_eval/pi05"
PV = json.load(open(f"{LD}/tt_dispatch_matrix/paired_vs_gpu.json"))

# ---------------------------------------------------------------- PERF_PRESETS.md: the release sweep, verbatim
sec = {}
for title in ("non-scalable", "scalable"):
    sec[title] = "## " + title + " (" + RT.split(f"\n## {title} (")[1].split("\n## ")[0].rstrip("\n")
diff = "## non-scalable minus scalable" + RT.split("\n## non-scalable minus scalable")[1].split("\n## ")[0].rstrip("\n")
foot = RT[RT.index("\n* = the 1-min load"):].strip("\n")
pp = ("# pi0.5 megakernel: trace replay latency for each preset (release build dd9431fa10f, both device profiles)\n\n"
      "- Hardware: one Blackhole p150a. Runtime: tt-metal `c718b5df9b9` (the runtime of image `%s`).\n"
      "- Each value is the time of one trace replay in ms. A trace replay is one replay of the Metal trace that holds the device programs of one request.\n"
      "- The host waits for the end of each trace replay.\n"
      "- Each value is the mean of 2 builds. Each build gives the median of 60 trace replays.\n"
      "- The +- value is the standard error (se) of the build means. p10 / p90 are of the 60 replays (median over the builds).\n"
      "- The batch size is 1.\n"
      "- H is the number of actions in the action chunk (the actions that one request returns).\n"
      "- H = 10 for the action-row bucket of 32 rows. H = 50 for the action-row bucket of 64 rows.\n"
      "- The device programs depend on the action-row bucket, not on H.\n"
      "- In the column names, L is the prompt bucket in tokens and S is the action-row bucket in rows.\n"
      "- A preset is one combination of a camera count, a prompt bucket and an action-row bucket.\n"
      "- The device profile selects the dispatch cores: `non-scalable` (default) uses Ethernet dispatch and a 12 x 10 grid for vision and prefix; `scalable` uses Tensix dispatch and an 11 x 10 grid.\n"
      "- The sections below are copied without change from the release sweep (`RELEASE_TABLE.md` of the WP-V perf run, generated %s).\n\n"
      % (TAG, re.search(r"Generated (\S+ \S+ KST)", RT).group(1))
      + sec["non-scalable"] + "\n\n" + sec["scalable"] + "\n\n" + diff + "\n\n" + foot + "\n")
assert pp.count("| cameras |") == 7 and pp.count("| steps |") == 2 and pp.count("| # | rep |") == 2
os.makedirs(OUT, exist_ok=True)
open(f"{OUT}/PERF_PRESETS.md", "w").write(pp); print(f"PERF_PRESETS.md: {len(pp.splitlines())} lines")

# ---------------------------------------------------------------- SERVING.md
S = Doc("SERVING")
S.rep("- Flow-matching steps (N): the number of Euler steps of the action expert (1 to 10).",
      "- Flow-matching steps (N): the number of Euler steps of the action expert (1 to 16).")
S.rep("- Serve profile: a named configuration of the server (`--profile`). This package has one serve profile, `p150`.",
      "- Serve profile: a named configuration of the server (`--profile`). This package has two serve profiles, `non-scalable` (the default) and `scalable`. They differ only in the device profile.\n"
      "- Device profile (`PI05_DISPATCH`): the cores that do the dispatch.\n"
      "  - `eth` (`non-scalable`): Ethernet cores do the dispatch. The vision and prefix programs get a 12 x 10 worker grid. The chip cannot join a multi-chip fabric.\n"
      "  - `tensix` (`scalable`): Tensix cores do the dispatch. The worker grid is 11 x 10. The Ethernet cores stay free for a multi-chip fabric.")
S.rep("- Device program: one persistent `ttnn.generic_op` megakernel on the 11 x 10 worker grid. A request uses 3 device programs (4 with 3 or 4 cameras).",
      "- Device program: one persistent `ttnn.generic_op` megakernel on the worker grid of the device profile. A request uses 3 device programs (4 with 3 or 4 cameras).")
S.rep("tt-model.yaml                               authoring manifest (schema 5.1): build, serve env, 1 serve profile (p150), verify, card",
      "tt-model.yaml                               authoring manifest (schema 5.1): build, serve env, 2 serve profiles (non-scalable, scalable), verify, card")
S.rep("PERF_PRESETS.md                             per-preset device replay latency (32 presets at N = 10, c2 at N = 1 / 5) + build log",
      "PERF_PRESETS.md                             per-preset device replay latency for both profiles (32 presets at N = 10, c2 at N = 1 / 5 / 16) + build logs\nGPU_COMPARISON.md                           p150a vs RTX 5090: LIBERO closed loop, the A2 input that both fail, earlier latency passes")
S.rep("""code/models/experimental/pi0/               the model: the multi-config megakernel (from the tenstorrent/tt-metal PR branch
                                            changh95/pi05-megakernel-mc @ fae9cd03fa4, unchanged)""",
      """code/models/experimental/pi0/               the model: the multi-config megakernel (from the tt-metal PR branch
                                            changh95/pi05-megakernel-eth16 @ dd9431fa10f, unchanged)""")
S.rep("""    tt/          ttnn_pi05_model.py (PI05MegakernelTTNN) + tt/megakernel/: presets.py (the 32 presets), the vision /""",
      """    tt/          ttnn_pi05_model.py (PI05MegakernelTTNN, open_pi05_device) + tt/megakernel/: profile.py (PI05_DISPATCH),
                 presets.py (the 32 presets), the vision /""")
S.rep("- `code/models/` is byte-identical to `models/` of [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ `5edf139d458e9acabd31eea641cd364079962677`.",
      f"- `code/models/` is byte-identical to `models/` of [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) @ `{SRC_COMMIT}`.")
S.rep("""- `source.tt_metal` is a clean clone of tenstorrent/tt-metal `main` @ `f856a38a361939888f92d88f9e69b2f8a83fb713` (v0.80.0-dev20261001-17).
- The wire manifest shows `dirty: false` for this tree.
- The validation of the multi-config megakernel used this tree.
- The previous releases pinned `main` @ `668c2907575`.""",
      f"""- `source.tt_metal` is a clean clone of tt-metal `{TM['sha']}` ({TM['describe']}). It is tenstorrent/tt-metal `main` @ `f856a38a361939888f92d88f9e69b2f8a83fb713` plus two commits:
  - `a69a83df5ad`: tenstorrent/tt-metal PR #57142, squashed. Ethernet dispatch on harvested Blackhole Ethernet grids, and a 40 KiB idle-ERISC kernel budget.
  - `c718b5df9b9`: the fetch-queue command-size check stays in Release builds.
- The `non-scalable` profile needs these commits. The `scalable` profile does not.
- The wire manifest shows `dirty: false` for this tree.
- The validation of this release (both profiles) used this tree.
- The previous release pinned `main` @ `f856a38a361`. The releases before it pinned `main` @ `668c2907575`.""")
S.rep("""  - The 14 kernel sources and their digest (`1429d5bea05c31ad`).""",
      f"""  - The 14 kernel sources and their digest (`{DIGEST}`, the same for both profiles).
  - The two device profiles: `PI05_DISPATCH=eth` gives `non-scalable` 12 x 10, `tensix` gives `scalable` 11 x 10, and other values are refused.
  - N = 16 is accepted and N = 17 is refused.
  - The 40 KiB idle-ERISC kernel budget of PR #57142 in the device headers.""")
S.rep("1. Set `T` to a built tt-metal tree @ `f856a38a361`. Set `R` to this repo.",
      "1. Set `T` to a built tt-metal tree @ `c718b5df9b9` (the `scalable` profile, `PI05_DISPATCH=tensix`, does not need its two extra commits). Set `R` to this repo.")
S.rep("T=/path/to/tt-metal                      # a built tree @ f856a38a361",
      "T=/path/to/tt-metal                      # a built tree @ c718b5df9b9")
S.rep("export TT_MESH_SHAPE=1x1 TT_DEVICE_ID=0 PI05_NUM_IMAGES=2 PI05_ACTION_HORIZON=50 PI05_NUM_STEPS=10 PI05_TOKEN_LEN=224",
      "export TT_MESH_SHAPE=1x1 TT_DEVICE_ID=0 PI05_NUM_IMAGES=2 PI05_ACTION_HORIZON=50 PI05_NUM_STEPS=10 PI05_TOKEN_LEN=224\nexport PI05_DISPATCH=eth                  # non-scalable (default); tensix = scalable")
(nb, nw), (sb, sw) = BOOT["non-scalable"], BOOT["scalable"]
assert 30 < nb < 60 and 30 < sb < 60 and nw < 2 and sw < 2, BOOT
S.rep("""- Measured boot times of this image on the p150a (2026-10-03, wall time from `tt-model serve` to READY):
  - Default configuration, cold (empty JIT cache): 66.2 s. The four prompt buckets compile and capture in 11.8 s.
  - Default configuration, warm: 41.2 s. The four prompt buckets compile and capture in 0.8 s.
  - Other configurations (1, 3 or 4 cameras; H = 10; N = 5): 35-57 s. Each configuration compiles its own device programs at its first boot.""",
      f"""- Measured boot times of this image on the p150a (2026-10-05, default configuration, warm JIT cache; from the start of the server process to READY):
  - `non-scalable`: {nb:.1f} s. The four prompt buckets compile and capture in {nw:.1f} s.
  - `scalable`: {sb:.1f} s. The four prompt buckets compile and capture in {sw:.1f} s.
  - This release did not time a cold boot. Each configuration compiles its own device programs at its first boot.""")
S.rep("- To see the serve profile, use `tt-model profiles changh95/pi05-base-p150`.",
      "- To see the serve profiles, use `tt-model profiles changh95/pi05-base-p150`.\n- To use the `scalable` profile, add `--profile scalable` to `tt-model serve` (or `tt serve`).")
S.rep("  - `megakernel`: {backend `mc`, program {kernel_digest, cameras, action_horizon, suffix_rows, num_steps, prompt_buckets, kv_dram, device_ops_per_call}}.",
      "  - `megakernel`: {backend `mc`, profile {name, dispatch, grid}, program {kernel_digest, cameras, action_horizon, suffix_rows, num_steps, prompt_buckets, kv_dram, device_ops_per_call}}.")
S.rep("| `PI05_NUM_STEPS` | 1..10 (10). The adaRMS folds depend on the schedule. | at start |",
      "| `PI05_NUM_STEPS` | 1..16 (10). The adaRMS folds depend on the schedule. | at start |\n"
      "| `PI05_DISPATCH` | `eth` (`non-scalable`, default) / `tensix` (`scalable`). The serve profile sets it. | at import |")
S.rep("- The package has one serve profile, `p150`. Its default configuration is in `tt-model.yaml` (`serve.env`): 2 cameras, an action chunk of 50 actions and 10 flow-matching steps.",
      "- The package has two serve profiles, `non-scalable` (the default) and `scalable`. Both have the default configuration of `tt-model.yaml` (`serve.env`): 2 cameras, an action chunk of 50 actions and 10 flow-matching steps.")
S.rep("- This procedure is the tested method (2026-10-03, image `tt-model/pi05-base-p150:4dd06e9d6fd3`):",
      f"- This procedure is the tested method (2026-10-04, image `tt-model/pi05-base-p150:{TAG}`, both serve profiles):")
S.rep("1. Write the launch command of the serve profile to a variable with `tt-model serve ... --print`.",
      "1. Write the launch command of the serve profile to a variable with `tt-model serve ... --print` (add `--profile scalable` for the other profile).")
S.rep("docker rm -f tt-model-pi05-base-p150-p150      # stop the server",
      "docker rm -f tt-model-pi05-base-p150-non-scalable      # stop the server (tt-model-pi05-base-p150-scalable for the other profile)")
m5 = re.findall(r"exit 3: refused at startup: (pi0.5 megakernel refused: cameras = 5 .*)", VAL)
m17 = re.findall(r"exit 3: refused at startup: (pi0.5 megakernel refused: num_denoising_steps=17 .*)", VAL)
assert len(m5) == 2 and len(set(m5)) == 1 and len(m17) == 2 and len(set(m17)) == 1
for prof, disp in (("non-scalable", "eth"), ("scalable", "tensix")):
    for cam, h, n, ops in ((3, 10, 5, 4), (2, 50, 16, 3)):
        assert re.search(r'info \{"name": "%s", "dispatch": "%s".*\'cameras\': %d, \'action_horizon\': %d, \'num_steps\': %d, \'device_ops_per_call\': %d\}'
                         % (prof, disp, cam, h, n, ops), VAL), (prof, cam, h, n)
assert VAL.count("PASS pi05 actions=(10,32)") == 2 and VAL.count("PASS pi05 actions=(50,32)") == 4 and VAL.count("repeat_maxdiff=0.0000") == 6
S.rep("""- `/info` showed cameras 3, `action_horizon` 10, `num_steps` 5, `suffix_rows` 32 and 4 device programs for each call.
- `smoke_test.py` passed: actions (10, 32), repeat difference 0.0000, inference 65.93 / 64.71 ms.
- With `PI05_NUM_IMAGES=5`, the container stopped at start (exit code 3).
  - The message was `pi0.5 megakernel refused: cameras = 5 (compiled: 1, 2, 3, 4) [PI05_NUM_IMAGES=5, PI05_ACTION_HORIZON=50, PI05_NUM_STEPS=10]`.""",
      f"""- On both profiles, `/info` showed cameras 3, `action_horizon` 10, `num_steps` 5 and 4 device programs for each call.
- `smoke_test.py` passed: actions (10, 32), repeat difference 0.0000.
- The same test with 2 cameras, H = 50 and N = 16 also passed on both profiles (`/info` `num_steps` 16, 3 device programs for each call).
- With `PI05_NUM_IMAGES=5` or `PI05_NUM_STEPS=17`, the container stopped at start (exit code 3) on both profiles.
  - The messages were `{m5[0]}` and `{m17[0]}`.""")
rows = []
for p in ("non-scalable", "scalable"):
    st = BENCH[p]["stats"]
    rows.append(f"| `{p}` | 2 | 50 | 10 | 3 | 224 | {st['inference']['median']:.2f} | {st['inference']['p90']:.2f} | {st['total']['median']:.2f} | {BENCH[p]['n']} |")
S.rep("""Served latency of six configurations:

- The first row (the default configuration) is from image `4dd06e9d6fd3`. The other rows are from image `36f651704bf1`.

- The request is the request of the model card. Image `4dd06e9d6fd3` has the same packages and libraries as `36f651704bf1` (only the serve profile changed).
- With 3 or 4 cameras, the request repeated the two images of the card.
- The values are the medians of warm requests.


| Cameras | H | N | Device programs for each call | Prompt bucket | Inference ms | Total ms | Requests |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 50 | 10 | 3 | 224 | 56.40 | 57.63 | 100 |
| 1 | 50 | 10 | 3 | 224 | 44.33 | 45.08 | 30 |
| 3 | 50 | 10 | 4 | 224 | 76.16 | 77.72 | 30 |
| 4 | 50 | 10 | 4 | 224 | 96.28 | 98.37 | 30 |
| 2 | 10 | 10 | 3 | 224 | 53.43 | 54.58 | 30 |
| 2 | 10 | 5 | 3 | 224 | 45.86 | 47.00 | 30 |""",
      f"""Served latency of the default configuration (image `{TAG}`, 2026-10-05):

- The request is the request of the model card.
- The values are the medians of 100 warm requests, after 5 warm-up requests. The host had no other workload.
- For the device time of the other configurations, see [`PERF_PRESETS.md`](PERF_PRESETS.md). This release did not measure their served latency.

| Serve profile | Cameras | H | N | Device programs for each call | Prompt bucket | Inference ms | Inference p90 ms | Total ms | Requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows))
S.rep("""  - Single p150a (11 x 10 worker grid, Tensix dispatch).""",
      """  - Single p150a. The `non-scalable` profile (Ethernet dispatch) needs the tt-metal tree of this image and blocks a multi-chip fabric. The `scalable` profile (Tensix dispatch, 11 x 10) does not need the two extra commits.""")
S.rep("  - The device must open with the 64 KiB worker-L1 cut (`PI05_DEVICE_PARAMS`). The server does this.",
      "  - The device must open with `open_pi05_device`: the dispatch cores of the profile and the 64 KiB worker-L1 cut (`PI05_DEVICE_PARAMS`). The server does this.")
S.rep("""  - One input-sensitive trajectory of the 320-set matrix (2 cameras, a 224-token prompt, H = 64, N 5-10).
  - The expert-oracle residual at one action row.""",
      f"""  - One input of the 1,024-set matrix fails A2: 2 cameras, a 224-token prompt, H = 64, prompt 5, N 5-16. The GPU bf16 policy also fails it at N 6-16 (see [`GPU_COMPARISON.md`](GPU_COMPARISON.md)).
  - The expert-oracle check (A4): {len(a4n)} of 1,440 sets are below 0.999 at N >= 2. At N = 1, A4 is reported, not gated.""")
S.write()

# ---------------------------------------------------------------- GPU_COMPARISON.md: a current-image section on top
G = Doc("GPU_COMPARISON")
G.rep("- The p150a side has updates of 2026-10-01 and 2026-10-03 (the next sections).",
      "- The p150a side has updates of 2026-10-01, 2026-10-03 and 2026-10-05 (the next sections).")
lrows = []
for prof, tag in (("non-scalable", "nonscal"), ("scalable", "scal")):
    for n in (10, 5, 1, 16):
        t = json.load(open(f"{LD}/tt_dispatch_matrix/{tag}_c2_n{n}_summary.json")); g = json.load(open(f"{LD}/gpu_matrix/c2_n{n}_summary.json"))
        pv = PV[f"{tag}_c2_n{n}"]
        assert t["episodes"] == 100 and not t["errors"] and t["num_steps"] == n and t["paired_vs_gpu"] == pv
        assert pv["tt_success"] == t["success"] and pv["gpu_success"] == g["success"] and pv["mcnemar_exact_p"] == 1.0
        lrows.append(f"| `{prof}` | 2 | {n} | {t['success']} / 100 | {g['success']} / 100 | {len(pv['tt_only'])} / {len(pv['gpu_only'])} |")
arows = []
for n, mn, mean in a2f:
    assert mn < gpu[n]
    arows.append(f"| {n} | {mn:.3f} | {gpu[n]:.3f} | {'pass' if gpu[n] >= 0.95 else 'fail'} |")
st = {p: BENCH[p]["stats"] for p in BENCH}
new = f"""## Update 2026-10-05: two device profiles, N 1..16 (current image)

- Image: `tt-model/pi05-base-p150:{TAG}`.
- Code: tt-metal PR branch `changh95/pi05-megakernel-eth16` @ `dd9431fa10f`. Kernel digest `{DIGEST}`.
- tt-metal: `{TM['sha'][:11]}` (`main` @ `f856a38a361` plus PR #57142 and the fetch-queue check).
- Two device profiles: `non-scalable` (default, Ethernet dispatch, 12 x 10 for vision and prefix) and `scalable` (Tensix dispatch, 11 x 10). Their outputs are bit-identical.
- Served latency of the p150a (100 warm requests, the card's request: 2 cameras, H = 50, N = 10):
  - `non-scalable`: `timing_ms.inference` median **{st['non-scalable']['inference']['median']:.2f} ms** (p90 {st['non-scalable']['inference']['p90']:.2f}), `timing_ms.total` {st['non-scalable']['total']['median']:.2f} ms.
  - `scalable`: `timing_ms.inference` median **{st['scalable']['inference']['median']:.2f} ms** (p90 {st['scalable']['inference']['p90']:.2f}), `timing_ms.total` {st['scalable']['total']['median']:.2f} ms.
- This update did not measure the GPU latency again. It makes no latency comparison with the GPU.
  - The GPU latency rows in the sections below are from 2026-09-14. They compare with earlier p150a images.

Closed-loop comparison (LIBERO-spatial, `lerobot/pi05_libero`, 2 cameras, H = 10):

- Each row has 100 paired episodes: the same initial states and the same noise seeds for each call. openpi's client sends the requests.
- The GPU runs openpi's `PI0Pytorch` on the RTX 5090 of this host.
- This file does not report the latency of these runs: the host had other load during them.

| Profile | Cameras | N | p150a success | RTX 5090 success | Discordant pairs (only TT / only GPU) |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(lrows) + """

- The paired difference is not significant in any row (exact McNemar p = 1).

The input that fails A2 on both devices:

- A2 compares the full call with the fp32 reference. The gate is PCC min >= 0.95 and mean >= 0.98 over 6 prompts.
- One input fails: 2 cameras, a 224-token prompt, H = 64, prompt 5. The other 5 prompts of that preset stay at 0.998 or more.
- The table gives the A2 PCC of that input against the fp32 reference, for the p150a and for the GPU bf16 openpi policy (RTX 5090, bf16 weights).
- The GPU bf16 policy also fails at N = 6 to 16. The p150a is lower at every N.

| N | p150a | GPU bf16 | GPU bf16 against the 0.95 gate |
|---:|---:|---:|---|
""" + "\n".join(arows) + "\n\n"
G.rep("## Update 2026-10-03: the multi-config megakernel (current image)\n", new + "## Update 2026-10-03: the multi-config megakernel (earlier image `4dd06e9d6fd3`)\n")
G.write()
