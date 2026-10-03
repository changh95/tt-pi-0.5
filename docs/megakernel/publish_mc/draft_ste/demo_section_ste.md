## LIBERO closed-loop demo

<video controls width="480" src="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4" poster="https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial_poster.png"></video>

- The video shows the multi-config megakernel of this image in a LIBERO-Spatial closed-loop test in MuJoCo on one p150a.
- A screen capture recorded the test in real time ([video](https://huggingface.co/changh95/pi05-base-p150/resolve/main/demo/pi05_libero_spatial.mp4)).
- [`demo/README.md`](demo/README.md) gives the procedure for the demo and shows the four clips.
- [`demo/manifest.json`](demo/manifest.json) is the machine-readable record of the clips (probes, step counts, latency, backend stamps, eval score).

| libero_spatial (10 tasks x initial states 0-9) | Device | Successes |
|---|---|---:|
| `lerobot/pi05_libero` on `PI05MegakernelTTNN`, 2 cameras, N = 10. | p150a | **99 / 100** (0 errors). The t9/i4 episode stopped at the step limit. |
| openpi `PI0Pytorch` with the same weights, client and noise seeds. | RTX 5090 | 100 / 100 |

- The mean policy latency on the p150a is **52.7 ms** for each call (server side, p90 53.0, 2,182 calls).
- The code is tt-metal-pr `changh95/pi05-megakernel-mc` @ `7a622a3a570`. This is the model code of `code/`, identical to `fae9cd03fa4`.
- The tt-metal version is `f856a38`, and the kernel digest is `1429d5bea05c31ad`.

**Caution:** Do not use this demo to estimate the results of `lerobot/pi05_base`. The demo uses different weights and a different configuration:

- The weights are the LIBERO fine-tuned `lerobot/pi05_libero` @ `a217bfd3b146` with the `pi05_libero` normalization statistics of openpi.
- The configuration is the LIBERO configuration of openpi.
  - The action chunk has H = 10 actions. The robot executes 5 actions of each chunk.
  - The prompts have 32 tokens.
- The image serves `lerobot/pi05_base` with the same model code.
- The LIBERO websocket server is `code/models/experimental/pi0_5/server/serve_pi05_libero.py`.
- The client and the normalization statistics are not in this repository.
