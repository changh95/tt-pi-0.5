# Carry into the phase-2 publish (written 2026-09-30 by the orchestrator)

HF changh95/pi05-base-p150 commit cf08fb95 fixed these labels on the Hub only (README.md, tt-model.yaml,
GPU_COMPARISON.md, SERVING.md). The GitHub copies (README, docs, tt-model.yaml in the staging dir) still have them.
Do not reintroduce them in the phase-2 release:
1. The in-process trace replay is the median of 30 replays (A_*.json), not 60.
2. For expert-loop device time, the megakernel figure is the median of 21 profiled sessions. The previous path's 30.31 ms is
   post_cache_device_ms_first, the first of 5 profiled sessions, not a median of 21.
3. `off` is NOT "all stock ops": it is stock TT-NN ops plus the 3 custom programs (fused attention, row_rsqrt, geglu_rc),
   1,660 ops after the last K/V-cache write.
4. docs/megakernel/integrate_p1/results/hold4_serve_invalid/S_struct.json is a valid profiler result (from hold2) filed in
   the folder of a discarded served run. Move it to integrate_p1/results/ and fix the JOURNAL citation.
5. Quote the previous image's served figures from the same benchmark cycle as the current ones (c2: 84.05 / 85.30, not
   c1's 84.02 / 85.35).
