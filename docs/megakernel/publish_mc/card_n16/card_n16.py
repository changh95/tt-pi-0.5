"""N16 card: the user's HF card (head 061677cf, README.hf_head_061677cf.md) with the N16 / 2-profile changes.
Keeps the user's structure and style; only the facts change. Pending inputs are marked PENDING[...]: a card with any
PENDING marker must not ship (`--final` asserts none is left). Every number comes from a file named in SRC.
usage: card_n16.py [--final]"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
head = open(f"{HERE}/README.hf_head_061677cf.md").read()
B = "/home/deepgadget/experiments/tt-models/build/pi05-base-p150/tt_kernel_manifest.json"
VAL = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc/results/n16/val"
TODO = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc/TODO_section_draft.md"
SRC = {}
man = json.load(open(B))["container"]; built = man["built"]; SRC["manifest"] = B
assert man["default_profile"] == "non-scalable" and [p["name"] for p in man["serve_profiles"]] == ["non-scalable", "scalable"]
src_commit = [l for l in open("/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc/tt-model.yaml") if "PI05_SOURCE_COMMIT:" in l][0].split('"')[1]
G = {p: json.load(open(f"{VAL}/img_check_{p}.json")) for p in ("non-scalable", "scalable")}
for p, g in G.items():
    assert g["golden"]["all_inputs_equal"] and g["refuse_all_ok"], p
gm, gn = G["non-scalable"]["golden"]["pcc7_mean"], G["non-scalable"]["golden"]["pcc7_min"]
assert abs(G["scalable"]["golden"]["pcc7_mean"] - gm) < 1e-12 and abs(G["scalable"]["golden"]["pcc7_min"] - gn) < 1e-12
SRC["golden"] = f"{VAL}/img_check_*.json"
# served bench (quiet slot 10-05 01:34-01:36, scratchpad n16/bench16.sh): 100 warm requests per profile, the card's curl request
BENCH = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc/n16/bench"
BN = {p: json.load(open(f"{BENCH}/bench-{p}.json")) for p in ("non-scalable", "scalable")}
for p, b in BN.items():
    assert b["n"] == 100 and b["identical_actions_all"] and b["info_profile"]["name"] == p
    assert all("uvicorn" in l or float(l.split()[0]) < 20 for h in b["host_load"].values() for l in h["top_cpu"]), p  # no foreign workload
SRC["bench"] = BENCH
# demo (pi05-libero-gpu-base, final 10-05 01:34; lead checked the poster): quiet-host recorded calls only
DM = json.load(open("/home/deepgadget/experiments/gr00t/libero_eval/pi05/tt_dispatch_matrix/demo/manifest.json"))
assert DM["profile"]["PI05_DISPATCH"] == "eth" and DM["code"]["commit"] == "dd9431fa10f" and "digest=26f0c46b7f1721c1" in DM["backend"]
assert DM["eval_score"] == "99/100" and DM["recorded_calls_condition"].startswith("quiet host") and DM["recorded_calls_n"] == 85
nt = {b["first_response"]["num_tokens"] for b in BN.values()}; assert nt == {142}, nt  # the card's default prompt, measured
med = {p: b["stats"]["inference"]["median"] for p, b in BN.items()}
load = max(float(h["loadavg"][0]) for b in BN.values() for h in b["host_load"].values())

def P(what):
    return f"PENDING[{what}]"

s = head
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:80])
    s = s.replace(a, b)

# --- intro
rep("  - 1 to 10 flow-matching steps (N).", "  - 1 to 16 flow-matching steps (N).")
rep("- The default profile with 2 cameras, action chunk of 50 actions and 10 steps, with 143 token-long prompt, the inference time is 56.4 ms.",
    "- The default configuration with 2 cameras, action chunk of 50 actions and 10 steps, with " + str(nt.pop()) + " token-long prompt, the inference time is "
    + f"{med['non-scalable']:.1f}" + " ms (non-scalable profile) and " + f"{med['scalable']:.1f}" + " ms (scalable profile).")
rep("- This package runs on **p150** (mesh `P150`, one p150a).",
    "- This package runs on **p150** (mesh `P150`, one p150a), with two serve profiles:\n"
    "  - `non-scalable` (default): Ethernet cores do the dispatch, so the vision and prefix programs get a 12 x 10 worker grid. The chip cannot join a multi-chip fabric in this mode.\n"
    "  - `scalable`: Tensix cores do the dispatch (11 x 10 worker grid). The Ethernet cores stay free for a multi-chip fabric.")
# --- demo (new recording)
rep("- Achieves 99/100 success.", "- Achieves " + DM["eval_score"] + " success (the libero_spatial run of the table below, `non-scalable`, N = 10).")
rep("- The mean policy latency on the p150a is **52.7 ms** for each call (server side, p90 53.0, 2,182 calls).",
    "- The median policy latency on the p150a is **%.1f ms** for each call (server side, p10 %.1f, p90 %.1f, %d calls of the recording)." % (DM["recorded_calls_infer_ms_median"], DM["recorded_calls_infer_ms_p10"], DM["recorded_calls_infer_ms_p90"], DM["recorded_calls_n"]))
rep("- The video shows the multi-config megakernel of this image in a LIBERO-Spatial closed-loop test in MuJoCo on one Tenstorrent Blackhole p150a.",
    "- The video shows the multi-config megakernel of this image (`non-scalable` profile) in a LIBERO-Spatial closed-loop test in MuJoCo on one Tenstorrent Blackhole p150a.")
# --- quickstart
rep("- The server uses port 20000. If that port is busy, the server uses the next free port.",
    "- The server uses port 20000. If that port is busy, the server uses the next free port.\n"
    "- The default serve profile is `non-scalable`. To use the other one: `tt-model serve changh95/pi05-base-p150 --profile scalable`.\n"
    "- [`SERVING.md`](SERVING.md) gives the request contract, the environment variables and the host validation procedure.")
# --- how to change the configuration
rep("""- The `p150` profile
  - Assumes batch=1""", """- The `non-scalable` (default) and `scalable` profiles
  - Assumes batch=1""")
rep("| Flow-matching (denoising) steps (N) | `PI05_NUM_STEPS` | `num_denoising_steps` | 1 to 10 | 10 | The server starts |",
    "| Flow-matching (denoising) steps (N) | `PI05_NUM_STEPS` | `num_denoising_steps` | 1 to 16 | 10 | The server starts |\n"
    "| Device profile (dispatch) | `PI05_DISPATCH` (set by the serve profile) | none (read at import; `open_pi05_device` uses it) | `eth` (non-scalable), `tensix` (scalable) | `eth` | The server starts |")
refs = {
 "| Number of cameras | `models/experimental/pi0_5/server/app.py:251` | `models/experimental/pi0/tt/ttnn_pi05_model.py:139`; the server check `models/experimental/pi0_5/server/mc_backend.py:48` | `models/experimental/pi0/common/configs.py:147` |":
 "| Number of cameras | `models/experimental/pi0_5/server/app.py:251` | `models/experimental/pi0/tt/ttnn_pi05_model.py:171`; the server check `models/experimental/pi0_5/server/mc_backend.py:51` | `models/experimental/pi0/common/configs.py:147` |",
 "| Action chunk length (H) | `models/experimental/pi0_5/server/app.py:264` | `models/experimental/pi0/tt/megakernel/geometry.py:391` (from `ttnn_pi05_model.py:138`) | `models/experimental/pi0/common/configs.py:132` |":
 "| Action chunk length (H) | `models/experimental/pi0_5/server/app.py:264` | `models/experimental/pi0/tt/megakernel/geometry.py:410` (from `ttnn_pi05_model.py:170`) | `models/experimental/pi0/common/configs.py:132` |",
 "| Flow-matching steps (N) | `models/experimental/pi0_5/server/app.py:252` | `models/experimental/pi0/tt/megakernel/geometry.py:391` (from `ttnn_pi05_model.py:138`) | `models/experimental/pi0/common/configs.py:141` |":
 "| Flow-matching steps (N) | `models/experimental/pi0_5/server/app.py:252` | `models/experimental/pi0/tt/megakernel/geometry.py:410` (from `ttnn_pi05_model.py:170`) | `models/experimental/pi0/common/configs.py:141` |",
 "| All three at server start | `models/experimental/pi0_5/server/app.py:274` | `models/experimental/pi0_5/server/mc_backend.py:34` | `models/experimental/pi0_5/server/mc_backend.py:72` |":
 "| All three at server start | `models/experimental/pi0_5/server/app.py:274` | `models/experimental/pi0_5/server/mc_backend.py:37` | `models/experimental/pi0_5/server/mc_backend.py:77` |\n"
 "| Device profile | `models/experimental/pi0/tt/megakernel/profile.py:18` | `models/experimental/pi0/tt/ttnn_pi05_model.py:301` (`device_refusal`, the worker grid) | `models/experimental/pi0/tt/ttnn_pi05_model.py:81` (`open_pi05_device`) |",
 "| Prompt bucket | `models/experimental/pi0_5/server/app.py:1050` (request field), `app.py:1277` | `models/experimental/pi0/tt/ttnn_pi05_model.py:305` | the compiled buckets: `models/experimental/pi0/tt/megakernel/presets.py:69` |":
 "| Prompt bucket | `models/experimental/pi0_5/server/app.py:1050` (request field), `app.py:1277` | `models/experimental/pi0/tt/ttnn_pi05_model.py:341` | the compiled buckets: `models/experimental/pi0/tt/megakernel/presets.py:71` |",
 "| Image count of a request | `models/experimental/pi0_5/server/app.py:1221` | `models/experimental/pi0/tt/ttnn_pi05_model.py:344` (masked cameras: line 339) | none |":
 "| Image count of a request | `models/experimental/pi0_5/server/app.py:1221` | `models/experimental/pi0/tt/ttnn_pi05_model.py:380` (masked cameras: line 375) | none |",
 "| Batch size, image size | none | `models/experimental/pi0/tt/ttnn_pi05_model.py:347`, `:363` | none |":
 "| Batch size, image size | none | `models/experimental/pi0/tt/ttnn_pi05_model.py:383`, `:399` | none |",
 "| One live model on each device | none | `models/experimental/pi0/tt/ttnn_pi05_model.py:133` | none |":
 "| One live model on each device | none | `models/experimental/pi0/tt/ttnn_pi05_model.py:165` | none |",
}
for a, b in refs.items():
    rep(a, b)
rep("2. Write the launch command of the serve profile to a variable. `tt-model serve` has no flag for the environment.",
    "2. Write the launch command of the serve profile to a variable (add `--profile scalable` for the other profile). `tt-model serve` has no flag for the environment.")
rep("docker rm -f tt-model-pi05-base-p150-p150      # stop the server", "docker rm -f tt-model-pi05-base-p150-non-scalable      # stop the server")
rep("1. Open the device with `PI05_DEVICE_PARAMS` (the 64 KiB worker-L1 cut is necessary).",
    "1. Open the device with `open_pi05_device(0)`: it uses the dispatch cores of `PI05_DISPATCH` (default `eth`) and the 64 KiB worker-L1 cut.")
rep("from models.experimental.pi0.tt.ttnn_pi05_model import PI05_DEVICE_PARAMS, PI05MegakernelTTNN\n\ndevice = ttnn.open_device(device_id=0, **PI05_DEVICE_PARAMS)  # the 64 KiB worker-L1 cut is required",
    "from models.experimental.pi0.tt.ttnn_pi05_model import PI05MegakernelTTNN, open_pi05_device\n\ndevice = open_pi05_device(0)  # PI05_DISPATCH=eth (default) or tensix; the 64 KiB worker-L1 cut")
# --- response example: refreshed after the bench
rep('"timing_ms": {"preprocess": 1.54, "inference": 56.55, "total": 58.09}}',
    '"timing_ms": {"preprocess": %.2f, "inference": %.2f, "total": %.2f}}' % tuple(BN["non-scalable"]["stats"][k]["median"] for k in ("preprocess", "inference", "total")))
# --- implementation implications
rep("- For 1 to 3 cameras, the KV cache lives in L1. For 4 cameras, the KV cache is in DRAM.",
    "- For 1 to 3 cameras, the KV cache lives in L1. For 4 cameras, the KV cache is in DRAM.\n"
    "- Both profiles give bit-identical outputs (checked on this image: c1-c4, N = 16 and the 8 openpi records); they differ only in speed and in what the chip can do next to the model (fabric or not).\n"
    "- With an action chunk of up to 32 actions and a prompt bucket of up to 128 tokens, the expert streams its weights through two-layer rings when the L1 allows it (automatic; outputs bit-identical).\n"
    "- With 4 cameras and 33-64 actions, the expert attention uses wider key chunks instead of a row loop (automatic).")
# --- accuracy
rep("| The device output against the openpi GPU policy. The test used `lerobot/pi05_libero`, 8 real LIBERO observations, 2 cameras, H = 10 and N = 10. The PCC is over the 7 action dims. This image did the measurement. | Mean **0.999981**, min **0.999958**. The host inputs of the image are bit-identical to the inputs of openpi (images, tokens, mask, noise). |",
    f"| The device output against the openpi GPU policy. The test used `lerobot/pi05_libero`, 8 real LIBERO observations, 2 cameras, H = 10 and N = 10. The PCC is over the 7 action dims. This image did the measurement, on both profiles. | Mean **{gm:.6f}**, min **{gn:.6f}** on each profile. The host inputs of the image are bit-identical to the inputs of openpi (images, tokens, mask, noise). |")
acc_rows = ["| A2: the full call against the fp32 reference on 6 prompts. The gate is PCC min ≥ 0.95 and mean ≥ 0.98. | **314/320** sets pass (see the limitation below). |",
            "| A4: the expert against an fp32 expert loop with the K / V caches of the device. The gate is ≥ 0.999 for N ≥ 2. | **286/288** sets pass. For N = 1, the card gives the values, but there is no gate. |",
            "| The K / V caches, the replay identity and the cross-check. | 320/320, 320/320, 320/320. |",
            "| Negative controls: inputs with a known error must fail the gates. | They fail as necessary on 320/320 (A2) and 320/320 (A4). |"]
# WP-V (impl, tt-metal-pr .val/mc_impl/wpv/summary.json): 32 presets x N 1..16 x 2 profiles = 1,024 sets
W = json.load(open("/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/wpv/summary.json")); SRC["wpv"] = "wpv/summary.json"
e = W["eth"]; assert W["profiles_identical"] and not e["new"] and not e["device_fail"] and not e["control_not_failing"]
a2f = [c for c in e["known"] + e["traj"] if c[1] == "A2"]; a4f = [c for c in e["known"] + e["traj"] if c[1] == "A4"]
assert len(a2f) == 12 and len(a4f) == 24 and all(c[2] == "L224_H64" and c[0].startswith("c2_S64") for c in a2f)
a4n = [c for c in a4f if not c[0].endswith("_N1")]; assert len(a4n) == 3
new_rows = [
 "| The matrix: 32 presets (cameras × prompt bucket × action-row bucket), N = 1 to 16, 3 action chunk lengths and 6 prompts for each set, on both profiles (1,024 sets). | The `scalable` outputs are bit-identical to the `non-scalable` outputs on 9,216/9,216 calls. So the rows below hold for both profiles. |",
 "| A2: the full call against the fp32 reference on 6 prompts. The gate is PCC min ≥ 0.95 and mean ≥ 0.98. | **1,524/1,536** sets pass. The 12 failures are one input (2 cameras, 224-token prompt, H = 64, prompt 5) at N = 5 to 16. The GPU bf16 policy also fails this input at N = 6 to 16. |",
 "| A4: the expert against an fp32 expert loop with the K / V caches of the device. The gate is ≥ 0.999 for N ≥ 2. | **1,437/1,440** sets pass for N ≥ 2 (min 0.9978). For N = 1, the card gives the values, but there is no gate (1,512/1,536 for all N). |",
 "| The K / V caches, the replay identity and the cross-check. | 96/96 prefixes (min PCC 0.9926), 1,536/1,536, 576/576. |",
 "| Negative controls: inputs with a known error must fail the gates. | They fail as necessary on 128/128 jobs (A2 and A4) and 96/96 prefixes (K / V). |",
]
old_rows = "\n".join(acc_rows) + "\n"
rep(old_rows, "\n".join(new_rows) + "\n")
# --- benchmarks
i = s.index("### Benchmarks"); j = s.index("### LIBERO closed loop (TT vs GPU)")
# impl's RELEASE_TABLE (wpv/perf/out/RELEASE_TABLE.md), recomputed from the raw sweep.json and asserted against the .md
PERF = "/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/wpv/perf"; SRC["perf"] = f"{PERF}/out_{{eth,tensix}}/sweep.json"
rt_md = open(f"{PERF}/out/RELEASE_TABLE.md").read()
assert "Commit dd9431fa10f" in rt_md and "c718b5df9b9" in rt_md
def perf_tables(tag, title):
    acc = {}
    for b in json.load(open(f"{PERF}/out_{tag}/sweep.json"))["builds"]:
        for L, v in b["presets"].items():
            acc.setdefault((b["cams"], b["S"], b["N"], int(L[1:])), []).append(v["replay_ms_median"])
    assert all(len(v) == 2 for v in acc.values()) and len(acc) == 32 + 2 * 4 * 3, len(acc)
    m = {k: sum(v) / 2 for k, v in acc.items()}
    sec = rt_md.split(f"\n## {title} (")[1].split("\n## ")[0]; assert sec.count("| cameras |") == 3  # N10 table, p10/p90 table, build log
    md_n10 = sec.split("\n\n")[1]; md_n = sec.split("1, 5 and 16 denoising steps")[1].split("\n\n")[1]
    def check(md, key_of_row):  # the .md rows: label | L32 S32 .. L224 S32 | L32 S64 .. L224 S64
        for row in md.splitlines()[2:]:
            c = [x.strip() for x in row.strip("|").split("|")]
            for j, cell in enumerate(c[1:]):
                k = key_of_row(int(c[0]), 32 if j < 4 else 64, (32, 64, 128, 224)[j % 4])
                assert abs(float(cell.split()[0]) - m[k]) < 0.0051, (tag, k, cell, m[k])
    check(md_n10, lambda r, S, L: (r, S, 10, L)); check(md_n, lambda r, S, L: (2, S, r, L))
    hcls = {32: "H ≤ 32", 64: "H 33-64"}
    head = "| Cameras | Action chunk (H) | N | inference time (ms) for prompt bucket 32 | ...for 64 | ...for 128 | ...for 224 |\n|---:|---|---:|---:|---:|---:|---:|\n"
    t1 = head + "\n".join(f"| {c} | {hcls[S]} | 10 | " + " | ".join(f"{m[(c, S, 10, L)]:.2f}" for L in (32, 64, 128, 224)) + " |" for c in (1, 2, 3, 4) for S in (32, 64))
    t2 = head + "\n".join(f"| 2 | {hcls[S]} | {n} | " + " | ".join(f"{m[(2, S, n, L)]:.2f}" for L in (32, 64, 128, 224)) + " |" for n in (1, 5, 16) for S in (32, 64))
    return t1, t2
pe1, pe2 = perf_tables("eth", "non-scalable"); pt1, pt2 = perf_tables("tensix", "scalable")
bench_tables = ("Device time of one trace replay in ms for each preset (the mean of 2 builds; each build gives the median of 60 trace replays):\n\n"
  "#### `non-scalable` (default)\n\n" + pe1 + "\n\n" + pe2 + "\n\n#### `scalable`\n\n" + pt1 + "\n\n" + pt2 + "\n\n"
  "- The host had no other workload during these measurements. The only load was the benchmark itself, while it built each model.\n"
  "- [`PERF_PRESETS.md`](PERF_PRESETS.md) gives the standard errors, p10 / p90 and the build logs of these measurements.\n")
s = s[:i] + "### Benchmarks\n\n" + bench_tables + "- " \
    + "Served over HTTP with the default configuration (2 cameras, H = 50, N = 10), the median inference time was **" + f"{med['non-scalable']:.2f}" + " ms** (non-scalable) and **" \
    + f"{med['scalable']:.2f}" + " ms** (scalable) for 100 warm requests. The only other load on the host was the model server itself " \
    + f"(1-min load average {load:.1f} or less).\n\n\n" + s[j:]
# --- LIBERO
i = s.index("| Cameras | N | p150a successes |"); j = s.index("### Limitations")
# LIBERO (pi05-libero-gpu-base, final 10-05): tt_dispatch_matrix/<scal|nonscal>_c2_n<N>_summary.json + paired_vs_gpu.json,
# GPU gpu_matrix/c2_n<N>_summary.json. Latencies were taken on a loaded host: not quoted (the column is dropped).
LD = "/home/deepgadget/experiments/gr00t/libero_eval/pi05"; SRC["libero"] = f"{LD}/tt_dispatch_matrix"
PV = json.load(open(f"{LD}/tt_dispatch_matrix/paired_vs_gpu.json"))
_an = open(f"{LD}/tt_dispatch_matrix/replay/analysis.txt").read()
assert _an.count("\n== N=") + _an.startswith("== N=") == 3 and _an.count("10/10 episodes identical across all four runs (obs+act hashes per call)") == 3
assert _an.count("first divergence at call/t/kind = (None, None, 'identical', {})") == 12 and "differ" not in _an.replace("identical", "")
lrows = []
for prof, tag in (("non-scalable", "nonscal"), ("scalable", "scal")):
    for n in (10, 5, 1, 16):
        t = json.load(open(f"{LD}/tt_dispatch_matrix/{tag}_c2_n{n}_summary.json"))
        g = json.load(open(f"{LD}/gpu_matrix/c2_n{n}_summary.json"))
        pv = PV[f"{tag}_c2_n{n}"]
        assert t["episodes"] == t["unique_episodes"] == 100 and not t["errors"] and t["cameras"] == 2 and t["num_steps"] == n
        assert t["suite"] == "libero_spatial" and t["paired_vs_gpu"] == pv and pv["n_paired"] == 100
        assert pv["tt_success"] == t["success"] and pv["gpu_success"] == g["success"] and pv["mcnemar_exact_p"] == 1.0
        lrows.append(f"| `{prof}` | 2 | {n} | **{t['success']} / 100** | {g['success']} / 100 | {len(pv['tt_only'])} / {len(pv['gpu_only'])} |")
s = s[:i] + ("| Profile | Cameras | N | p150a successes | RTX 5090 successes | Discordant pairs (only TT / only GPU) |\n"
             "|---|---:|---:|---:|---:|---:|\n" + "\n".join(lrows) + "\n\n"
             "- The paired difference is not significant in any row (exact McNemar p = 1).\n"
             "- Both device profiles produce identical closed-loop trajectories (per-call hash-identical replays). Three single-episode step-count differences in the 800-episode matrix did not reproduce: they are run-to-run variation, not a profile difference.\n"
             "- [`GPU_COMPARISON.md`](GPU_COMPARISON.md) has these results, the A2 input that both devices fail, and the earlier GPU comparisons.\n\n\n") + s[j:]
rep("- The p150a latency is the time of one policy call on the server, with host input preparation, device time and readback.\n"
    "- This card does not give the GPU latency, because its measurement was different (openpi model time only, on a shared host with load).\n",
    "- Both profiles ran the same 100 episodes.\n"
    "- This table does not give latency, because the host had other load during these runs. For latency, see the Benchmarks section.\n")
# --- limitations
rep("  - H must be 64 or less, and N must be 10 or less.", "  - H must be 64 or less, and N must be 16 or less.")
rep("- **Worker grid.** The worker grid is 11 × 10 (110 cores) with Tensix dispatch. A later release will use the 12 × 10 grid with Ethernet dispatch.",
    "- **Profiles.** `non-scalable` (the default) uses Ethernet dispatch and a 12 × 10 grid for vision and prefix. It needs the tt-metal runtime of this image (PR #57142 patches). The chip cannot join a multi-chip fabric in this mode. `scalable` uses Tensix dispatch (11 × 10) and does not need these patches.")
# --- accuracy limitations (lead 10-05; values from WP-V summary.json + the GPU incumbent JSONs, bf16 arm, seed 5)
WP6 = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/wp6"
gpu = {}
for f in ("gpu_inc_c2_L224_H64.json", "gpu_inc_c2_L224_H64_n16.json"):
    for n, d in json.load(open(f"{WP6}/{f}"))["arms"]["bf16"]["per_N"].items():
        gpu[int(n)] = [x["pcc"] for x in d["seeds"] if x["seed"] == 5][0]
SRC["gpu_incumbent"] = f"{WP6}/gpu_inc_c2_L224_H64*.json (bf16)"
dev = {int(c[0].split("_N")[1]): (c[3], c[4]) for c in a2f}
SEEDS_OURS = json.load(open("/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc/results/a2_cell_seeds.json"))
SEEDS_IMPL = json.load(open("/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/wpv/a2_cell_seeds_n5_16.json"))
assert SEEDS_IMPL["commit"] == "dd9431fa10f"; SRC["a2_seeds"] = "results/a2_cell_seeds.json + wpv/a2_cell_seeds_n5_16.json"
assert sorted(dev) == list(range(5, 17))
lim_rows = []
for n in range(5, 17):
    mn, mean = dev[n]
    assert mn < gpu[n], n                       # the device is lower at every N
    # per-seed: our a2_cell_seeds.json (10-03, N <= 10) and impl's a2_cell_seeds_n5_16.json (WP-V); seed 5 is the summary min
    sd = SEEDS_IMPL["per_N"][str(n)]; seeds = sorted(sd["seeds"], key=lambda x: x["seed"])
    assert sd["profiles_identical"] and seeds[5]["seed"] == 5 and abs(seeds[5]["a2"] - mn) < 5e-6, n
    assert all(x["a2"] >= 0.998 for x in seeds[:5]), n
    if n <= 10:
        assert all(abs(x["a2"] - y) < 1e-9 for x, y in zip(seeds, SEEDS_OURS[str(n)])), n
    lim_rows.append(f"| {n} | {mn:.3f} | {gpu[n]:.3f} |")
assert all(gpu[n] < 0.95 for n in range(6, 17)) and gpu[5] >= 0.95
a4n1 = [c for c in a4f if c[0].endswith("_N1")]
a4n_txt = {c[0]: c for c in a4n}
assert set(a4n_txt) == {"c1_S32_N5", "c2_S32_N3", "c2_S64_N16"}
lim = ("- **Accuracy.**\n"
 "  - A2 fails on one input: 2 cameras, a 224-token prompt, H = 64 and prompt (seed) 5, at N = 5 to 16. The other 5 prompts of that preset stay at 0.998 or more.\n"
 "  - The GPU bf16 openpi policy also fails this input at N = 6 to 16, but the device is lower at every N (A2 min PCC against fp32):\n\n"
 "    | N | p150a | GPU bf16 |\n    |---|---|---|\n" + "\n".join("    " + r for r in lim_rows) + "\n\n"
 f"  - A4 at N ≥ 2: 3 of 1,440 sets are below 0.999: 1 camera / 64-token prompt / N = 5 / H = 1 ({a4n_txt['c1_S32_N5'][3]:.4f}), "
 f"2 cameras / 128 / N = 3 / H = 1 ({a4n_txt['c2_S32_N3'][3]:.4f}), and 2 cameras / 128 / N = 16 / H = 50, prompt 5 ({a4n_txt['c2_S64_N16'][3]:.4f}, the same as before the 64-row change).\n"
 f"  - At N = 1, A4 is reported, not gated: {len(a4n1)} of 96 sets are below 0.999.\n")
rep("### Limitations\n\n", "### Limitations\n\n" + lim)
# --- TODO section before License
todo = open(TODO).read().strip()
# TODO item (user request 10-05, via the lead): the 12th column of the non-scalable profile. Numbers: the release sweep
# (sweep.json, mean of 2 builds), impl's E1a VISION / PREFIX spans (4 runs per profile), the repartition design note.
import statistics as _st
def _sweep(tag):
    acc = {}
    for b_ in json.load(open(f"{PERF}/out_{tag}/sweep.json"))["builds"]:
        for L, v in b_["presets"].items():
            acc.setdefault((b_["cams"], b_["S"], b_["N"], int(L[1:])), []).append(v["replay_ms_median"])
    return {k: sum(v) / 2 for k, v in acc.items()}
_e, _t = _sweep("eth"), _sweep("tensix")
_g2 = [100 * (_t[k] - _e[k]) / _t[k] for k in _e if k[0] == 2 and k[2] == 10]
_ex = (_e[(2, 32, 10, 64)], _t[(2, 32, 10, 64)])
_rows = [json.loads(l) for l in open("/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/e1a/out_time/timeprof.jsonl")]
_sp = {}
for r in _rows:
    _sp.setdefault((r["profile"], tuple(r["preset"])), []).append(r["spans_us"])
_vg, _pg, _gap = [], [], []
for pre in {k[1] for k in _sp if k[1][0] == 2}:
    sv = _st.mean(x["vision"] for x in _sp[("scalable", pre)]); nv = _st.mean(x["vision"] for x in _sp[("non-scalable", pre)])
    sq = _st.mean(x["prefix"] for x in _sp[("scalable", pre)]); nq = _st.mean(x["prefix"] for x in _sp[("non-scalable", pre)])
    assert len(_sp[("scalable", pre)]) == len(_sp[("non-scalable", pre)]) == 4
    _vg.append(100 * (sv - nv) / sv); _pg.append(100 * (sq - nq) / sq); _gap.append((nv + nq - (sv + sq) * 11 / 12) / 1000)
_dn = open("/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/repart/DESIGN_NOTE.md").read()
assert "19 (NU 10)\n  to 35 (NU <= 6) cores are IDLE on 11 x 10" in _dn and "8 x 8 MLP" in _dn
assert abs(_ex[0] - 47.66) < 0.005 and abs(_ex[1] - 48.72) < 0.005
lo_v, hi_v, lo_p, hi_p = min(_vg), max(_vg), min(_pg), max(_pg)
assert 3.3 < lo_p and hi_v < 4.1 and 1.6 < min(_gap) and max(_gap) < 1.8, (_vg, _pg, _gap)
todo += ("\n- **Use the 12th column fully (non-scalable)**\n"
  f"  - The non-scalable profile has 120 worker cores against 110, an ideal of about 8%. At 2 cameras and N = 10 it is only {min(_g2):.1f}-{max(_g2):.1f}% faster than scalable (L64 S32: {_ex[0]:.2f} vs {_ex[1]:.2f} ms).\n"
  f"  - VISION and PREFIX gain about {min(lo_v, lo_p):.1f}-{max(hi_v, hi_p):.1f}% at 2 cameras (less than half of ideal). Their work splits were parameterized for 12 columns, not re-tuned. Uneven 12-column splits and fixed per-op sync latency are the likely causes, not yet measured.\n"
  "  - EXPERT does not use the 12th column: its core map is fixed by the model (8 heads, 8 × 8 MLP), and 19-35 cores are already idle on 11 × 10. Using more cores needs a re-partition (see the blocked-matmul item).\n"
  f"  - Plan: per-phase timing of VISION / PREFIX on 12 × 10, then re-tune the splits. Upper bound: about {max(_gap):.1f} ms per call at 2 cameras (the gap of VISION + PREFIX to the ideal 11 / 12 time).")
rep("### License", todo + "\n\n### License")
# --- license / provenance
rep("@ [`5edf139`](https://github.com/changh95/tt-pi-0.5/commit/5edf139d458e9acabd31eea641cd364079962677).",
    f"@ [`{src_commit[:7]}`](https://github.com/changh95/tt-pi-0.5/commit/{src_commit}).")
tm = built["tt_metal"]; assert tm["sha"].startswith("c718b5df9b9") and not tm["dirty"]
rep("| tt-metal | [`f856a38a361939888f92d88f9e69b2f8a83fb713`](https://github.com/tenstorrent/tt-metal/commit/f856a38a361939888f92d88f9e69b2f8a83fb713) |",
    f"| tt-metal | `{tm['sha']}` = [`f856a38a361`](https://github.com/tenstorrent/tt-metal/commit/f856a38a361939888f92d88f9e69b2f8a83fb713) + `a69a83df5ad` (PR #57142, squashed: Ethernet dispatch on harvested Blackhole ETH grids) + `c718b5df9b9` (the fetch-queue command-size check kept in Release builds); describe `{tm['describe']}` |")
rep("| `code/` digest | `184c58b636fb7621` (sha256, first 16 hex digits) |", f"| `code/` digest | `{built['code_sha256'][:16]}` (sha256, first 16 hex digits) |")
rep("| built | 2026-10-03T06:47:04+00:00 by tt-model 0.1.0 |", f"| built | {built['created_at']} by tt-model 0.1.0 |")
open(f"{HERE}/README.md", "w").write(s)
pend = re.findall(r"PENDING\[[^\]]*\]", s)
print(f"written {HERE}/README.md; PENDING markers: {len(pend)}")
for p_ in pend: print("  ", p_[:140])
if "--final" in sys.argv:
    assert not pend, "PENDING markers left"
