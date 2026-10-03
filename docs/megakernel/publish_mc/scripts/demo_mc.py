"""Write the HF demo/README.md and the card's LIBERO demo section from the demo manifest + matrix files."""
import json
L = "/home/deepgadget/experiments/gr00t/libero_eval/pi05/tt_mc_matrix"
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-mc"
SP = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pubmc"
m = json.load(open(f"{L}/demo/manifest.json")); P = json.load(open(f"{L}/paired_vs_gpu.json"))["c2_n10"]
s = json.load(open(f"{L}/c2_n10_summary.json")); lat = s["policy_latency"]
assert m["eval_score"] == "99/100" and P["tt_success"] == 99 and P["gpu_success"] == 100 and P["gpu_only"] == ["t9/i4"]
assert all(e["recorded_success"] and e["recorded_steps"] == e["eval_steps"] for e in m["episodes"])
U = "https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo"
rows = "\n".join(f"| [`{e['clip']}`]({U}/{e['clip']}) | {e['task']} | {e['recorded_steps']} | {e['n_calls']} | {e['infer_ms_median']:.1f} |" for e in m["episodes"])
video = f'<video controls width="480" src="{U}/pi05_libero_spatial.mp4" poster="{U}/pi05_libero_spatial_poster.png"></video>'
f1 = lambda x: f"{x:.1f}"
readme = f"""# LIBERO closed-loop demo: pi0.5 on one Tenstorrent Blackhole p150a (multi-config megakernel)

{video}

[pi05_libero_spatial.mp4]({U}/pi05_libero_spatial.mp4) combines the four clips below (title card, four episodes, end card).

## Swapped weights: read this first

This demo runs the **same model code** this repository's image serves, `PI05MegakernelTTNN` (`code/models/experimental/pi0/tt/ttnn_pi05_model.py`, kernel digest `1429d5bea05c31ad`): SigLIP + projector, the Gemma-2B prefill writing the K / V caches, and the N-step x 18-layer action expert as three persistent `ttnn.generic_op` programs per call, replayed from one Metal trace per prompt bucket. It does **not** use the same weights or shape:

- weights: the LIBERO fine-tune [`lerobot/pi05_libero`](https://huggingface.co/lerobot/pi05_libero) @ `a217bfd3b14673cf2ce597e69997ab21866438dd`, with openpi's `pi05_libero` norm stats (sha256 `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`);
- shape: 2 cameras (agentview + wrist), H = 10 (5 actions executed per call), N = 10 steps, every LIBERO prompt in the 32-token bucket;
- server: `code/models/experimental/pi0_5/server/serve_pi05_libero.py` (openpi's websocket protocol, openpi-exact image normalisation `x * float32(1/255) * 2 - 1`), which ships in `code/` but is not the image's HTTP entry; openpi's LIBERO client and the norm stats live outside this repository. `tt-model serve` does not reproduce this demo.

## Result (libero_spatial, 10 tasks x official init states 0-9)

| policy | device | success | policy latency per call |
|---|---|---:|---:|
| `lerobot/pi05_libero` on `PI05MegakernelTTNN` (2 cameras, N = 10) | Tenstorrent p150a | **{P['tt_success']} / 100** (0 errors) | mean {f1(lat['mean_ms'])} ms, median {f1(lat['median_ms'])}, p90 {f1(lat['p90_ms'])} ({lat['n_calls']:,} calls; host inputs + device + readback) |
| openpi `PI0Pytorch`, same weights, client, init states and per-call noise seeds | RTX 5090 | {P['gpu_success']} / 100 | mean {f1(P['gpu_latency_mean_ms'])} ms |

- Paired episode by episode: {P['both_success']} both succeeded, 0 TT only, 1 GPU only (t9/i4: no error, it ran to the 230-step cap on the p150a); exact McNemar p = {P['mcnemar_exact_p']:g}.
- The full TT-vs-GPU matrix (cameras 1 and 2 x N 1 / 5 / 10) is on the model card and in [`libero_eval.json`](libero_eval.json).

## Clips (recorded {m["created"][:10]})

| clip | task | steps | calls | median ms / call |
|---|---|---:|---:|---:|
{rows}

Each clip is a fresh episode served live by the p150a (an on-screen MuJoCo viewer captured in real time at 30 fps; observations rendered offscreen exactly as in the evaluation), and each reproduced its step count from the 100-episode evaluation. Same tasks, init state and order as the previous demo. Just before recording, the packaged server's host inputs were `torch.equal` to openpi's on 8 / 8 golden observations (PCC over the 7 action dims, min 0.999958).
"""
# demo/README.md is pi05-libero-gpu-base's own (lead 2026-10-03); only the card section is written here

section = f"""## LIBERO closed-loop demo

{video}

The multi-config megakernel this image serves, running LIBERO-Spatial closed loop in MuJoCo on one p150a, captured on screen in real time ([video]({U}/pi05_libero_spatial.mp4)). How it was made and all four clips: [`demo/README.md`](demo/README.md); machine-readable record of the clips (probes, step counts, latency, backend stamps, eval score): [`demo/manifest.json`](demo/manifest.json).

| libero_spatial (10 tasks x init states 0-9) | device | success |
|---|---|---:|
| `lerobot/pi05_libero` on `PI05MegakernelTTNN`, 2 cameras, N = 10 | p150a | **{P['tt_success']} / 100** (0 errors; t9/i4 hit the step cap) |
| openpi `PI0Pytorch`, same weights, client and noise seeds | RTX 5090 | {P['gpu_success']} / 100 |

Policy latency on the p150a: mean **{f1(lat['mean_ms'])} ms** per call (server side, p90 {f1(lat['p90_ms'])}; {lat['n_calls']:,} calls). Code: tt-metal-pr `changh95/pi05-megakernel-mc` @ `{m['code']['commit']}` (the model code of `code/`, identical to `fae9cd03fa4`) on tt-metal `f856a38`, kernel digest `1429d5bea05c31ad`.

**Caveat: swapped weights.** The demo uses the LIBERO fine-tune `lerobot/pi05_libero` @ `a217bfd3b146` with openpi's `pi05_libero` norm stats, at openpi's LIBERO shape (H = 10, 5 actions executed per call, 32-token prompts). The image serves `lerobot/pi05_base`. The model code is the same; the LIBERO websocket server is `code/models/experimental/pi0_5/server/serve_pi05_libero.py`, and the client and norm stats live outside this repository.
"""
open(f"{SP}/demo_section_mc.md", "w").write(section)
print("ok")
