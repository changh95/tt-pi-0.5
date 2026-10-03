sentences 76 (instructions 0, descriptive 76); over limit 0; paragraphs > 6 sentences 0; max length 24; mean 10.3
table prose cells: sentences 16, over 25 0, max 19
British spellings: []
list items 58; list items nested deeper than 2 levels 0; prose paragraphs (not lists) 3
  PROSE (1 sentence(s)): `tools/run_demo.sh 3:0,7:0,0:0,5:0` runs these four steps (not published, like the other scripts in this section):
  PROSE (1 sentence(s)): A person checked these frames by eye:
  PROSE (1 sentence(s)): Result: no other window covered the viewer in these frames.
 10/25 desc | Date and time of the demo capture: 2026-10-03 11:35 KST.
 14/25 desc | Host of the demo capture: the host that built the image of this repository.
  9/25 desc | Files marked not published below stayed on that host.
 10/25 desc | All other files in this folder are on the Hub.
 11/25 desc | Each clip is a new episode that the p150a served live.
  5/25 desc | It is not a replay.
 12/25 desc | Each clip reproduced its step count from the 100-episode `c2_n10` evaluation exactly.
 19/25 desc | The tasks, the init state and the order are the same as in the previous HF demo (megakernel-p2, 2026-10-01).
  3/25 desc | Model: `PI05MegakernelTTNN` (`code/models/experimental/pi0/tt/ttnn_pi05_model.py`).
  8/25 desc | A device program is one persistent `ttnn.generic_op` megakernel.
 22/25 desc | Three device programs run SigLIP, the VLM prefill (it writes K / V) and the action expert (10 steps x 18 layers).
 13/25 desc | A prompt bucket is a fixed prompt length that the device programs use.
 11/25 desc | One Metal trace for each prompt bucket holds the device programs.
 10/25 desc | Each policy call is one trace replay of that trace.
  9/25 desc | Every LIBERO prompt runs in the 32-token prompt bucket.
 13/25 desc | An action chunk is the set of actions that one policy call returns.
 10/25 desc | H is the number of actions in the action chunk.
 15/25 desc | Settings: cameras = 2 (agentview + wrist), H = 10, N = 10 flow-matching steps.
 12/25 desc | The kernel digest is `1429d5bea05c31ad`, the same as in the matrix run.
  6/25 desc | Code: tt-metal-pr branch `changh95/pi05-megakernel-mc`, commit `7a622a3a570`.
 12/25 desc | The demo ran from a `git archive` of its `models/experimental/pi0` (not published).
  8/25 desc | The release-candidate HEAD `fae9cd03fa4` changes only the README.
 10/25 desc | Thus the code is identical to `code/models/experimental/pi0` of this repository.
  6/25 desc | The ttnn build is tt-metal-main `f856a38`.
  4/25 desc | Weights: `lerobot/pi05_libero` @ `a217bfd3b14673cf2ce597e69997ab21866438dd`.
  4/25 desc | `PI0WeightLoader` loads the weights.
 11/25 desc | The normalization uses the `pi05_libero` norm stats of openpi, sha256 `b3a44bb2...bd84`.
 10/25 desc | Server: `serve_pi05_libero.py --cameras 2 --num-steps 10`, the packaged single-file server.
 10/25 desc | It is byte-identical (sha256 `96dff490370ad915...`) to `code/models/experimental/pi0_5/server/serve_pi05_libero.py` of this repository.
  5/25 desc | It uses the openpi conventions.
 13/25 desc | It normalizes the images exactly as openpi does: `x  f32(1/255)  2 - 1`.
 17/25 desc | It held the device through the device lock of the host (not published) for its full lifetime.
 24/25 desc | Golden check just before the demo capture: the inputs are `torch.equal` to the GPU golden records of openpi (`openloop_golden.pt`, not published) on 8/8 records.
 15/25 desc | The golden check gave a PCC7 minimum of 0.999958 and a PCC7 mean of 0.999974.
 19/25 desc | These values are the same as the values of the server copy of the matrix, to the last digit.
  7/25 desc | Evaluation for the captions: the `c2_n10` run.
 13/25 desc | The run has 100 episodes = 10 tasks x official init states 0-9.
  4/25 desc | The score is 99/100.
  8/25 desc | The log for each episode is not published.
 10/25 desc | The summary is in `manifest.json` and on the model card.
  5/25 desc | The only failure is t9/i4.
 10/25 desc | This episode ran to the 230-step cap with no error.
 14/25 desc | Policy latency over 2,182 calls: mean 52.7 ms per call, median 52.7, p90 53.0.
 13/25 desc | The latency includes the host input build, the trace replay and the readback.
  7/25 desc | The latency source is `c2_n10_summary.json` (not published).
  5/25 desc | `manifest.json` has the same figures.
  7/25 desc | openpi on an RTX 5090 scored 100/100.
 17/25 desc | The paired comparison with openpi has one discordant pair (t9/i4, GPU only), exact McNemar p = 1.
 15/25 desc | `tools/run_demo.sh 3:0,7:0,0:0,5:0` runs these four steps (not published, like the other scripts in this section):
  2/25 desc | Golden check.
 18/25 desc | `golden_control.py --package --configs c2_n10` runs the packaged policy class on the raw observations of the openpi GPU goldens.
 16/25 desc | The pipeline stops if an input is not `torch.equal` or if PCC7 is less than 0.999.
  1/25 desc | `record.sh`.
  5/25 desc | It starts the packaged server.
 10/25 desc | It runs the same client (`record_client.py`) as the previous demos.
 24/25 desc | The client runs the loop of the eval: `env.seed(7)`, 10 wait steps, a 180° flip plus `resize_with_pad` to 224, replan 5, 220 max steps.
  9/25 desc | The noise seed is `100000task + 1000init + call`.
 14/25 desc | After the reset, the client opens a MuJoCo passive viewer on the frontview camera.
 13/25 desc | ffmpeg x11grab captures that window on DISPLAY=:1 at 30 fps, in real time.
 13/25 desc | The client still renders the observations offscreen (EGL), exactly as in the eval.
  2/25 desc | `render_captions.py 3,7,0,5`.
  8/25 desc | It writes the captions into the video frames.
 11/25 desc | It builds the combined clip with its title and end cards.
  8/25 desc | It takes the poster frame at 8 s.
 12/25 desc | Caption text, first part: `pi0.5 (LIBERO finetune) on Tenstorrent / Blackhole p150a`.
 23/25 desc | Caption text, second part (after the `—` character): `multi-config pi0.5 megakernel (7a622a3) / 2 cameras (agentview + wrist), N = 10 flow-matching steps`.
 21/25 desc | The captions also show the task, the median latency of the episode, the mean latency of the eval and the result.
  3/25 desc | `make_manifest.py` writes `manifest.json`.
  7/25 desc | A person checked these frames by eye:
  2/25 desc | The poster.
  5/25 desc | The title card (1 s).
  6/25 desc | t3 during its run (5.5 s).
  6/25 desc | t7 at its success (14 s).
  5/25 desc | The end card (27.5 s).
  7/25 desc | A 5-frame strip across the combined clip.
 10/25 desc | Result: no other window covered the viewer in these frames.
