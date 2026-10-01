"""demo/libero_eval.json for the phase-2 whole-model megakernel eval, every value read from the eval files."""
import json, copy
D = "/home/deepgadget/experiments/gr00t/libero_eval/pi05"; ST = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"
S = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pub2"
old = json.load(open(f"{S}/hf_head/demo/libero_eval.json"))   # = HF cf08fb95 (phase-1 demo)
summ = json.load(open(f"{D}/megakernel-p2/tt_spatial_summary.json"))
man = json.load(open(f"{D}/megakernel-p2/demo/manifest.json"))
ol = json.load(open(f"{D}/megakernel-p2/openloop_pcc_mkp2.json"))["summary"]
tt = [json.loads(l) for l in open(f"{D}/megakernel-p2/tt_spatial.jsonl")]
gpu = {(r["task_id"], r["init_state_idx"]): r for r in map(json.loads, open(f"{D}/ref_gpu_spatial.jsonl"))}
ttd = {(r["task_id"], r["init_state_idx"]): r for r in tt}
assert len(ttd) == 100 and sum(r["success"] for r in ttd.values()) == summ["successes"] == 99
stamps = sorted({r["backend"] for r in tt}); assert stamps == summ["backend_stamps"] and len(stamps) == 1
assert "megakernel=whole" in stamps[0] and "mk_digest=4aa02cdf21ed0c94" in stamps[0] and "code=7f32fccbe6b4(" in stamps[0]
MERGED = "821e8c528dfffa0d1d6e73ad6abf181a749b39d9"
new = copy.deepcopy(old)
new["date"] = "2026-10-01"
t = new["tt"]
t["model"] = ("models.experimental.pi0_5.tt.ttnn_pi0_model.PI0ModelTTNN.sample_actions_fused, default PI05_MEGAKERNEL=whole "
              "(the same code this repo's container serves): one Metal trace whose replay is ONE device op, a persistent "
              "ttnn.generic_op on 110 cores running SigLIP x2, the projector, the language embedding, the VLM prefill -> K/V, "
              "the 10-step x 18-layer action-expert loop, the action in / out projections and the Euler updates")
t["backend"] = stamps[0]
t["code"] = {"repo": "https://github.com/changh95/tt-pi-0.5", "commit": "7f32fccbe6b4",
             "note": ("branch megakernel-2026-09-29 at 7f32fccbe6b4 (clean tree; kernel digest 4aa02cdf21ed0c94). Between it and "
                      f"main @ {MERGED[:7]} (the merge of this branch, which is this repo's code/) models/ differs only in the package "
                      "README (models/experimental/pi0_5/README.md); every .py / .cpp / .hpp file is identical.")}
t["successes"] = summ["successes"]; t["rate"] = summ["success_rate"]; t["n"] = summ["n_episodes_recorded"]
t["errors"] = summ["n_errors"]; t["timeouts"] = summ["n_timeouts"]
t["failures_step_cap"] = [{"task_id": a, "init_state_idx": b, "steps": ttd[(a, b)]["steps"], "gpu_steps": gpu[(a, b)]["steps"]} for a, b in summ["failures_step_cap"]]
t["per_task"] = summ["per_task"]
t["server_infer_ms"] = summ["server_infer_ms"]
p = summ["paired_vs_gpu_init0_4"]
new["paired_subset_init0_4"] = {"n": 50, "tt_successes": p["tt_success_init0_4"], "gpu_successes": p["gpu_success"],
    "tt_rate": p["tt_success_init0_4"] / 50, "gpu_rate": p["gpu_success"] / 50, "counts": p["counts"], "only_gpu": p["episodes"]["only_gpu"]}
new["tt_init5_9"] = {"n": 50, "tt_successes": summ["tt_success_init5_9"]}
new["openloop_vs_openpi_golden"] = {"records": 8, "lang_len": 32, "H": 10, "mean_pcc_7": round(ol["mean_pcc_7"], 6),
    "min_pcc_7": round(ol["min_pcc_7"], 6), "deterministic": ol["all_deterministic"], "file": "megakernel-p2/openloop_pcc_mkp2.json (author's tree)"}
prev = old["tt"]
new["previous_tt_run_2026_09_30"] = {"path": "phase-1 expert megakernel (PI05_MEGAKERNEL=expert today: traced stock-op prefix + the expert loop as one generic_op)",
    "successes": summ["vs_phase1_expert_tt"]["old_successes"], "paired_init0_4": old["paired_subset_init0_4"]["tt_successes"],
    "server_infer_ms_median": prev["server_infer_ms"]["median"],
    "episodes_identical_steps_and_success": summ["vs_phase1_expert_tt"]["episodes_identical_steps_and_success"]}
new.pop("previous_tt_run_2026_09_29", None)
new["previous_tt_run_2026_09_29"] = dict(old["previous_tt_run_2026_09_29"], path="fused traced graph: stock TT-NN ops plus the 3 custom programs fused attention / row_rsqrt / geglu_rc (PI05_MEGAKERNEL=off today)")
eps = []
for e in man["episodes"]:
    assert e["recorded_success"] and e["recorded_steps"] == e["eval_steps"] == ttd[(e["task_id"], e["init"])]["steps"]
    eps.append({"task_id": e["task_id"], "init_state_idx": e["init"], "task": e["task"], "success": e["recorded_success"], "steps": e["recorded_steps"],
                "eval_steps": e["eval_steps"], "infer_ms_median": e["infer_ms_median"], "n_calls": e["n_calls"], "clip": "demo/" + e["clip"]})
new["demo_recording"]["episodes"] = eps
new["demo_recording"]["capture"] = man["capture"]
new["demo_recording"]["note"] = "Recorded runs reproduce the eval steps exactly (init 0 = first episode of the task with env.seed(7)). Same four episodes as the 2026-09-29 and 2026-09-30 demos."
new["supersedes"] = ("the 2026-09-30 demo (phase-1 expert megakernel, code a46beb5: 99/100, paired 49/50, 66.4 ms per call), which superseded "
                     "the 2026-09-29 demo (fused traced graph of stock TT-NN ops plus 3 custom programs, code 69b0969: 98/100, paired 48/50, 77.6 ms per call)")
for k in list(new["demo_recording"]):
    if k not in ("capture", "episodes"): print("demo_recording kept key:", k, str(new["demo_recording"][k])[:300])
for k in new:
    if k not in ("tt", "paired_subset_init0_4", "tt_init5_9", "openloop_vs_openpi_golden", "previous_tt_run_2026_09_30", "previous_tt_run_2026_09_29", "demo_recording", "supersedes", "date"):
        print("top-level kept key:", k, str(new[k])[:300])
for k in t:
    if k not in ("model", "backend", "code", "successes", "rate", "n", "errors", "timeouts", "failures_step_cap", "per_task", "server_infer_ms"):
        print("tt kept key:", k, str(t[k])[:300])
json.dump(new, open(f"{ST}/demo/libero_eval.json", "w"), indent=1, ensure_ascii=False)
print(t["successes"], t["failures_step_cap"], new["paired_subset_init0_4"]["counts"], t["server_infer_ms"]["median"])
print([(e["task_id"], e["steps"], e["n_calls"], e["infer_ms_median"]) for e in eps])
