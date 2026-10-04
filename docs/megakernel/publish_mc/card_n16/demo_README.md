# LIBERO closed-loop demo: pi0.5 on one Tenstorrent Blackhole p150a

<video controls width="480" src="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4" poster="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial_poster.png"></video>

[pi05_libero_spatial.mp4](https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4) is the four clips below joined, with a title card at the start and a summary card at the end.

## Weights: read this first

This demo runs the multi-config pi0.5 megakernel that this repository serves, **with the LIBERO fine-tune instead of the base weights**:

- **Weights:** [`lerobot/pi05_libero`](https://huggingface.co/lerobot/pi05_libero) @ `a217bfd3b14673cf2ce597e69997ab21866438dd` (`model.safetensors`).
- **Normalization:** openpi's `pi05_libero` norm stats (`gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json`, sha256 `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`).
- **Configuration:** openpi's `pi05_libero` shape with 2 cameras (agentview + wrist), action horizon 10 and 10 flow-matching steps.

## What runs on the device

- **Model:** `PI05MegakernelTTNN`. Each policy call is one Metal trace replay of three `ttnn.generic_op` programs:
  - SigLIP on both cameras;
  - the Gemma-2B VLM prefill, which writes the K/V caches;
  - the 10-step × 18-layer action expert, with the action in/out projections and the Euler updates.
- **Device profile: non-scalable** (`PI05_DISPATCH=eth`, the default): ethernet dispatch cores and a 12 × 10 worker grid. Kernel digest `26f0c46b7f1721c1`.
- **Code:** tt-metal-pr branch `changh95/pi05-megakernel-eth16` @ `dd9431fa10f`. Runtime: tt-metal `c718b5df9b9` (`f856a38a` plus the ETH-dispatch patches).
- **Server:** the policy server shipped in this repository, run as `serve_pi05_libero.py --dispatch eth --cameras 2 --num-steps 10`.
  - It speaks openpi's websocket protocol, so openpi's LIBERO client connects unchanged.
  - Its image normalization is bit-exact to openpi's.
  - Just before recording, its inputs were `torch.equal` to openpi's GPU model inputs on 8 real LIBERO observations, with PCC over the 7 action dims of min 0.999958 and mean 0.999974.

## Result shown in the captions (libero_spatial, non-scalable profile, N = 10)

| policy | device | episodes | success |
|---|---|---:|---:|
| pi0.5 `lerobot/pi05_libero`, `PI05MegakernelTTNN` | Tenstorrent p150a | 100 (10 tasks × official init states 0-9) | **99 / 100** |
| openpi `PI0Pytorch`, same weights (reference) | RTX 5090 | 100 (same episodes, same per-call noise seeds) | 100 / 100 |

The one TT failure is task 9 / init 4: no error, it ran to the step cap. Paired by episode with the GPU, it is the only discordant pair, with exact McNemar p = 1. There were 0 errors and 0 timeouts.

## Clips

The clips are on-screen captures of the MuJoCo passive viewer (frontview camera), grabbed at 30 fps and played back in real time. Each is a fresh episode served live by the p150a, not a replay. Each one reproduced the step count of its evaluation episode exactly.

| clip | task | steps | calls | median ms / call |
|---|---|---:|---:|---:|
| [pi05_tt_libero_spatial_t3_i0.mp4](https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_tt_libero_spatial_t3_i0.mp4) | pick up the black bowl on the cookie box and place it on the plate | 97 | 18 | 49.4 |
| [pi05_tt_libero_spatial_t7_i0.mp4](https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_tt_libero_spatial_t7_i0.mp4) | pick up the black bowl on the stove and place it on the plate | 130 | 25 | 49.3 |
| [pi05_tt_libero_spatial_t0_i0.mp4](https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_tt_libero_spatial_t0_i0.mp4) | pick up the black bowl between the plate and the ramekin and place it on the plate | 116 | 22 | 49.3 |
| [pi05_tt_libero_spatial_t5_i0.mp4](https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_tt_libero_spatial_t5_i0.mp4) | pick up the black bowl on the ramekin and place it on the plate | 117 | 22 | 49.3 |

**Policy latency on the p150a, measured on an otherwise idle host while recording:** median **49.3 ms** per call over 85 calls (p10 49.1, p90 49.8, max 64.6; the first 2 calls are excluded).

Each call covers tokenization, building the host tensors, the trace replay, the readback and unnormalization. It excludes the websocket round trip and the simulator.

The poster frame is [`pi05_libero_spatial_poster.png`](pi05_libero_spatial_poster.png). [`manifest.json`](manifest.json) records, per clip, the ffmpeg probe, bytes, recorded and evaluation steps, and latency, plus the backend stamp, code, runtime and eval summary. Its relative paths (`eval_summary`, `raw_window`) refer to the evaluation workspace and are not part of this folder.

## How it was produced

- **Evaluation loop:** a copy of the loop in openpi's `examples/libero/main.py`. The GPU and TT runs used the same client. The loop:
  - calls `env.seed(7)` once per task, then `set_init_state(init_states[i])`;
  - takes 10 no-op wait steps;
  - flips the images 180° and applies `resize_with_pad` to 224;
  - sends openpi's `pi05_libero` prompt (no state tokens);
  - executes 5 actions of each 10-action chunk, for at most 220 steps.

  The noise for each call is `default_rng(100000*task + 1000*init + call).standard_normal((1,10,32))`, the same on the GPU and the TT server, so the two runs pair by episode.
- **GPU reference:** openpi @ `fdc03f52`, `PI0Pytorch` with config `pi05_libero`, bf16, `torch.compile`, loading the same safetensors strictly.
- **Recording:** the server was launched with the settings above and the recording client opened a MuJoCo passive viewer after reset. Observations were still rendered offscreen (EGL), exactly as in the evaluation. Captions were composited afterwards.

Date: 2026-10-05.
