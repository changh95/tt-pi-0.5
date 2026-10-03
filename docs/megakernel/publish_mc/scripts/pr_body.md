## pi0.5 multi-config megakernel as the served default

- `models/experimental/pi0`: the multi-config pi0.5 megakernel, vendored unchanged from the tenstorrent/tt-metal PR branch
  `changh95/pi05-megakernel-mc` @ `fae9cd03fa4` (kernel digest `1429d5bea05c31ad`). `PI05MegakernelTTNN` runs the whole
  `sample_actions` as three persistent `ttnn.generic_op` programs per call (vision | prefix | expert), four with 3-4
  cameras: cameras 1-4, prompt buckets 32 / 64 / 128 / 224, H 1..64, N 1..10, batch 1, 224 x 224, one live model per device.
- Server: `PI05_MEGAKERNEL` unset / `mc` on one chip serves it (`server/mc_backend.py`); `PI05_NUM_IMAGES`,
  `PI05_ACTION_HORIZON`, `PI05_NUM_STEPS` fix the configuration at start, out-of-range values are refused with the
  model's own messages; a request sends exactly `PI05_NUM_IMAGES` images; optional `prompt_bucket`; openpi-exact image
  normalisation. `whole` / `expert` / `off` stay as comparators.
- `server/serve_pi05_libero.py`: openpi websocket LIBERO server on the same model.
- tt-metal pin: `main` @ `f856a38a361` (was `668c2907575`).
- Validation (docs/megakernel/publish_mc/): image `tt-model/pi05-base-p150:36f651704bf1` outputs bit-identical to the
  tt-metal-pr build at one preset per camera count; golden PCC7 mean 0.999981 / min 0.999958; served c2 / H 50 / N 10
  56.44 ms; LIBERO c2 N10 99 / 100 (GPU 100 / 100).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
