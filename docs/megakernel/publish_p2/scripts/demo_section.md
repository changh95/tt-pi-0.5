## LIBERO closed-loop demo

<video controls width="480" src="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4" poster="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial_poster.png"></video>

This is the code path this image serves, the whole-model megakernel (one fused op per call), running LIBERO-Spatial closed loop in MuJoCo on one p150a. It was captured on screen in real time ([video](https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4)). How it was made and all four clips: [`demo/README.md`](demo/README.md). Machine-readable results: [`demo/libero_eval.json`](demo/libero_eval.json).

| libero_spatial (10 tasks) | device | success |
|---|---|---:|
| `lerobot/pi05_libero`, `PI0ModelTTNN.sample_actions_fused` with the whole-model megakernel, init states 0-9 | p150a | **99 / 100** (0 errors, 0 timeouts; t9/i4 hit the step cap) |
| same, paired subset init states 0-4 | p150a | 49 / 50 |
| openpi `PI0Pytorch` reference, same weights and client, init states 0-4 | RTX 5090 | 50 / 50 |

Policy latency on the p150a: median **53.6 ms** per call (server side, p90 54.1; 2,189 calls). The previous path (phase-1 expert megakernel) scored 99 / 100 at 66.4 ms.

Code: [changh95/tt-pi-0.5](https://github.com/changh95/tt-pi-0.5) branch `megakernel-2026-09-29` @ `7f32fcc` (clean tree). Its `models/` differs from `main` @ `821e8c5`, which is this repo's `code/`, only in the package README. tt-metal `668c2907575`.

**Caveat: swapped weights.** The demo uses the LIBERO fine-tune `lerobot/pi05_libero` @ `a217bfd3b146` with openpi's `pi05_libero` norm stats, at openpi's LIBERO shape (32 prompt tokens, H = 10, 5 actions executed per call). The model this image serves is `lerobot/pi05_base` at 224 tokens and H = 50. The code is the same, but the weights and shape are not. The LIBERO server wrapper, client and norm-stats handling live outside the published image, so `tt-model serve` does not reproduce this demo.
