"""demo/libero_eval.json for the phase-1 megakernel eval, every value read from the eval files."""
import json, copy
D = "/home/deepgadget/experiments/gr00t/libero_eval/pi05"; ST = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"
S = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pub1"
old = json.load(open(f"{S}/bak/demo/libero_eval.json"))
summ = json.load(open(f"{D}/megakernel-p1/tt_spatial_summary.json"))
man = json.load(open(f"{D}/megakernel-p1/demo/manifest.json"))
ol = json.load(open(f"{D}/megakernel-p1/openloop_pcc_mkp1.json"))["summary"]
tt = [json.loads(l) for l in open(f"{D}/megakernel-p1/tt_spatial.jsonl")]
gpu = {(r["task_id"], r["init_state_idx"]): r for r in map(json.loads, open(f"{D}/ref_gpu_spatial.jsonl"))}
ttd = {(r["task_id"], r["init_state_idx"]): r for r in tt}
assert len(ttd) == 100 and sum(r["success"] for r in ttd.values()) == summ["successes"] == 99
stamps = sorted({r["backend"] for r in tt}); assert stamps == summ["backend_stamps"] and len(stamps) == 1
MERGED = "f7f173bd61241f09b286e56a9eb4038f76ee0782"
new = copy.deepcopy(old)
new["date"] = "2026-09-30"
t = new["tt"]
t["model"] = ("models.experimental.pi0_5.tt.ttnn_pi0_model.PI0ModelTTNN.sample_actions_fused, default PI05_MEGAKERNEL=expert "
              "(the same code this repo's container serves): one Metal trace; SigLIP / VLM prefix = traced stock TT-NN ops; the whole "
              "10-step x 18-layer action-expert loop + action in / out + Euler = ONE persistent ttnn.generic_op on 110 cores")
t["backend"] = stamps[0]
t["code"] = {"repo": "https://github.com/changh95/tt-pi-0.5", "commit": "a46beb54a24f",
             "note": ("branch megakernel-2026-09-29 at a46beb54a24f (+ uncommitted README / docs edits only; device-path source md5 "
                      "403253dd516ba74bd937e6ac4a329009, kernel digest 328761c8a1ce3fd9). models/ at a46beb54a24f is byte-identical to "
                      f"main @ {MERGED[:7]} (the merge of this branch), which is this repo's code/.")}
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
    "min_pcc_7": round(ol["min_pcc_7"], 6), "deterministic": ol["all_deterministic"], "file": "megakernel-p1/openloop_pcc_mkp1.json (author's tree)"}
new["previous_tt_run_2026_09_29"] = {"path": "fused traced graph with the stock-op expert (PI05_MEGAKERNEL=off today)", "successes": summ["vs_previous_fused_tt"]["old_successes"],
    "paired_init0_4": old["paired_subset_init0_4"]["tt_successes"], "server_infer_ms_median": old["tt"]["server_infer_ms"]["median"]}
eps = []
for e in man["episodes"]:
    assert e["recorded_success"] and e["recorded_steps"] == e["eval_steps"] == ttd[(e["task_id"], e["init"])]["steps"]
    eps.append({"task_id": e["task_id"], "init_state_idx": e["init"], "task": e["task"], "success": e["recorded_success"], "steps": e["recorded_steps"],
                "eval_steps": e["eval_steps"], "infer_ms_median": e["infer_ms_median"], "n_calls": e["n_calls"], "clip": "demo/" + e["clip"]})
new["demo_recording"]["episodes"] = eps
new["demo_recording"]["capture"] = man["capture"]
new["demo_recording"]["note"] = "Recorded runs reproduce the eval steps exactly (init 0 = first episode of the task with env.seed(7)). Same four episodes as the 2026-09-29 demo."
new["supersedes"] = ("the 2026-09-29 demo (fused traced graph with the stock-op expert, code 69b0969: 98/100, paired 48/50, 77.6 ms per call), which "
                     "superseded the 2026-09-28 demo (eager PI0ModelTTNN on tt-metal changh95/pi05 @ 4c9fbfcc: 98/100, paired 49/50, 215.3 ms per call)")
for k in list(new["demo_recording"]):
    if k not in ("capture", "episodes"): print("demo_recording kept key:", k, str(new["demo_recording"][k])[:200])
json.dump(new, open(f"{ST}/demo/libero_eval.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: new[k] for k in ("date",)}), t["successes"], t["failures_step_cap"], new["paired_subset_init0_4"]["counts"], t["server_infer_ms"]["median"])
print([ (e["task_id"], e["steps"], e["n_calls"], e["infer_ms_median"]) for e in eps])
