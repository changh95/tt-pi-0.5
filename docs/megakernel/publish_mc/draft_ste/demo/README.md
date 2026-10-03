# LIBERO closed-loop demo: pi0.5 on one Tenstorrent Blackhole p150a (multi-config megakernel)

- Date and time of the demo capture: 2026-10-03 11:35 KST.
- Host of the demo capture: the host that built the image of this repository.
- Files marked *not published* below stayed on that host.
- All other files in this folder are on the Hub.

## Files

| file | what |
|---|---|
| `pi05_libero_spatial.mp4` | The combined clip (30.1 s, 960x1080, 30 fps, H.264). It contains a title card, the four episodes and an end card. |
| `pi05_libero_spatial_poster.png` | The poster frame (t3/i0, SUCCESS). |
| `pi05_tt_libero_spatial_t{3,7,0,5}_i0.mp4` | One clip with captions for each episode. |
| `manifest.json` | Each clip with its ffmpeg probe and its size in bytes. Also the recorded and eval step counts, the latency, the backend stamps, the code commit and the eval score. Also the result of the paired comparison against the GPU. |
| `recorded_runs.jsonl`, `record_calls.jsonl` (*not published*) | The results of the recorded runs, and one line for each policy call. |
| `record_client.log`, `record_server.log`, `run_demo.log` (*not published*) | The logs. |
| `package_golden_control.json` / `.log` (*not published*) | The device golden check of the packaged server. This check ran just before the demo capture. |
| `raw/` (*not published*) | The viewer captures without captions (`*_window_raw.mp4`) and the offscreen agentview\|wrist side-by-side images. |
| `tools/` (*not published*) | `run_demo.sh` (the whole pipeline), `record.sh`, `render_captions.py`, `make_manifest.py` |

## Clips

| clip | task | steps (recorded = eval) | calls | median ms / call |
|---|---|---:|---:|---:|
| `pi05_tt_libero_spatial_t3_i0.mp4` | pick up the black bowl on the cookie box and place it on the plate | 97 | 18 | 53.3 |
| `pi05_tt_libero_spatial_t7_i0.mp4` | pick up the black bowl on the stove and place it on the plate | 130 | 25 | 53.2 |
| `pi05_tt_libero_spatial_t0_i0.mp4` | pick up the black bowl between the plate and the ramekin and place it on the plate | 116 | 22 | 53.0 |
| `pi05_tt_libero_spatial_t5_i0.mp4` | pick up the black bowl on the ramekin and place it on the plate | 117 | 22 | 53.2 |

- Each clip is a new episode that the p150a served live. It is not a replay.
- Each clip reproduced its step count from the 100-episode `c2_n10` evaluation exactly.
- The tasks, the init state and the order are the same as in the previous HF demo (megakernel-p2, 2026-10-01).

## What runs

- **Model:** `PI05MegakernelTTNN` ([`code/models/experimental/pi0/tt/ttnn_pi05_model.py`](../code/models/experimental/pi0/tt/ttnn_pi05_model.py)).
  - A device program is one persistent `ttnn.generic_op` megakernel.
  - Three device programs run SigLIP, the VLM prefill (it writes K / V) and the action expert (10 steps x 18 layers).
  - A prompt bucket is a fixed prompt length that the device programs use.
  - One Metal trace for each prompt bucket holds the device programs. Each policy call is one trace replay of that trace.
  - Every LIBERO prompt runs in the 32-token prompt bucket.
  - An action chunk is the set of actions that one policy call returns. H is the number of actions in the action chunk.
  - Settings: cameras = 2 (agentview + wrist), H = 10, N = 10 flow-matching steps.
  - The kernel digest is `1429d5bea05c31ad`, the same as in the matrix run.
- **Code:** tt-metal-pr branch `changh95/pi05-megakernel-mc`, commit `7a622a3a570`.
  - The demo ran from a `git archive` of its `models/experimental/pi0` (*not published*).
  - The release-candidate HEAD `fae9cd03fa4` changes only the README.
  - Thus the code is identical to [`code/models/experimental/pi0`](../code/models/experimental/pi0) of this repository.
  - The ttnn build is tt-metal-main `f856a38`.
- **Weights:** `lerobot/pi05_libero` @ `a217bfd3b14673cf2ce597e69997ab21866438dd`.
  - `PI0WeightLoader` loads the weights.
  - The normalization uses the `pi05_libero` norm stats of openpi, sha256 `b3a44bb2...bd84`.
- **Server:** `serve_pi05_libero.py --cameras 2 --num-steps 10`, the packaged single-file server.
  - It is byte-identical (sha256 `96dff490370ad915...`) to [`code/models/experimental/pi0_5/server/serve_pi05_libero.py`](../code/models/experimental/pi0_5/server/serve_pi05_libero.py) of this repository.
  - It uses the openpi conventions. It normalizes the images exactly as openpi does: `x * f32(1/255) * 2 - 1`.
  - It held the device through the device lock of the host (*not published*) for its full lifetime.
  - Golden check just before the demo capture: the inputs are `torch.equal` to the GPU golden records of openpi (`openloop_golden.pt`, *not published*) on 8/8 records.
  - The golden check gave a PCC7 minimum of 0.999958 and a PCC7 mean of 0.999974.
  - These values are the same as the values of the server copy of the matrix, to the last digit.
- **Evaluation for the captions:** the `c2_n10` run.
  - The run has 100 episodes = 10 tasks x official init states 0-9. The score is 99/100.
  - The log for each episode is *not published*. The summary is in [`manifest.json`](manifest.json) and on the model card.
  - The only failure is t9/i4. This episode ran to the 230-step cap with no error.
  - Policy latency over 2,182 calls: mean 52.7 ms per call, median 52.7, p90 53.0.
  - The latency includes the host input build, the trace replay and the readback.
  - The latency source is `c2_n10_summary.json` (*not published*). [`manifest.json`](manifest.json) has the same figures.
  - openpi on an RTX 5090 scored 100/100.
  - The paired comparison with openpi has one discordant pair (t9/i4, GPU only), exact McNemar p = 1.

## The demo pipeline

`tools/run_demo.sh 3:0,7:0,0:0,5:0` runs these four steps (*not published*, like the other scripts in this section):

1. **Golden check.** `golden_control.py --package --configs c2_n10` runs the packaged policy class on the raw observations of the openpi GPU goldens. The pipeline stops if an input is not `torch.equal` or if PCC7 is less than 0.999.
2. **`record.sh`.** It starts the packaged server. It runs the same client (`record_client.py`) as the previous demos.
   - The client runs the loop of the eval: `env.seed(7)`, 10 wait steps, a 180° flip plus `resize_with_pad` to 224, replan 5, 220 max steps.
   - The noise seed is `100000*task + 1000*init + call`.
   - After the reset, the client opens a MuJoCo passive viewer on the frontview camera.
   - ffmpeg x11grab captures that window on DISPLAY=:1 at 30 fps, in real time.
   - The client still renders the observations offscreen (EGL), exactly as in the eval.
3. **`render_captions.py 3,7,0,5`.** It writes the captions into the video frames. It builds the combined clip with its title and end cards. It takes the poster frame at 8 s.
   - Caption text, first part: `pi0.5 (LIBERO finetune) on Tenstorrent / Blackhole p150a`.
   - Caption text, second part (after the `—` character): `multi-config pi0.5 megakernel (7a622a3) / 2 cameras (agentview + wrist), N = 10 flow-matching steps`.
   - The captions also show the task, the median latency of the episode, the mean latency of the eval and the result.
4. **`make_manifest.py`** writes `manifest.json`.

A person checked these frames by eye:

- The poster.
- The title card (1 s).
- t3 during its run (5.5 s).
- t7 at its success (14 s).
- The end card (27.5 s).
- A 5-frame strip across the combined clip.

Result: no other window covered the viewer in these frames.
