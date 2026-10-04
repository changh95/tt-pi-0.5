# pi0.5 megakernel: trace replay latency for each preset (release build fae9cd03fa4)

- Hardware: one Blackhole p150a.
- Each value is the time of one trace replay in ms. A trace replay is one replay of the Metal trace that holds the device programs of one request.
- The host waits for the end of each trace replay.
- Each value is the mean of 2 builds. Each build gives the median of 60 trace replays.
- The ± value is the standard error (se) of the build means.
- The batch size is 1.
- H is the number of actions in the action chunk (the actions that one request returns).
- H = 10 for the action-row bucket of 32 rows. H = 50 for the action-row bucket of 64 rows.
- The device programs depend on the action-row bucket, not on H.
- In the column names, L is the prompt bucket in tokens and S is the action-row bucket in rows.
- A preset is one combination of a camera count, a prompt bucket and an action-row bucket.

## 10 flow-matching steps (all 32 presets)

| cameras | L32 S32 | L64 S32 | L128 S32 | L224 S32 | L32 S64 | L64 S64 | L128 S64 | L224 S64 |
|---|---|---|---|---|---|---|---|---|
| 1 | 38.93 ± 0.01† | 39.56 ± 0.00† | 39.87 ± 0.01† | 41.13 ± 0.01† | 41.26 ± 0.00† | 41.34 ± 0.01† | 42.30 ± 0.01† | 43.75 ± 0.02† |
| 2 | 51.05 ± 0.01† | 50.23 ± 0.01† | 51.89 ± 0.02† | 52.49 ± 0.00† | 53.68 ± 0.02† | 53.72 ± 0.04† | 54.85 ± 0.02† | 55.57 ± 0.03† |
| 3 | 67.93 ± 0.01 | 68.12 ± 0.01 | 68.45 ± 0.01 | 69.87 ± 0.02 | 72.60 ± 0.04 | 72.76 ± 0.03 | 73.15 ± 0.03 | 74.91 ± 0.02 |
| 4 | 84.06 ± 0.01 | 84.17 ± 0.00 | 85.73 ± 0.01 | 86.77 ± 0.00 | 86.03 ± 0.01 | 91.75 ± 0.01 | 93.63 ± 0.01 | 94.65 ± 0.02 |

## 2 cameras with 1 and 5 flow-matching steps

| steps | L32 S32 | L64 S32 | L128 S32 | L224 S32 | L32 S64 | L64 S64 | L128 S64 | L224 S64 |
|---|---|---|---|---|---|---|---|---|
| 1 | 37.40 ± 0.02 | 37.42 ± 0.01 | 38.49 ± 0.05 | 40.11 ± 0.18 | 37.70 ± 0.03† | 37.90 ± 0.16† | 39.02 ± 0.19† | 40.35 ± 0.07† |
| 5 | 43.43 ± 0.01† | 43.17 ± 0.09† | 44.34 ± 0.03† | 45.52 ± 0.08† | 44.74 ± 0.03† | 44.86 ± 0.03† | 45.78 ± 0.09† | 46.98 ± 0.07† |

Footnote: trace replay, median of 60 per build, mean of 2 builds; measured at host 1-min load 2-9; the 4 presets also timed on a quiet host agree within 0.09 ms (c2 L64 S32 +0.027, c3 L224 S32 +0.090, c4 L224 S32 +0.057, c4 L224 S64 +0.081 ms here vs the quiet ABAB holds of 2026-10-03 09:38 / 10:00).

† one of the two builds ran while a foreign CPU process was active (windows below). Build-to-build se is <= 0.042 ms at 10 steps; at 1 / 5 steps it reaches 0.19 ms.

## Build log (host load, foreign activity)

| # | rep | cameras | S | N | window (KST) | 1-min load start -> end | foreign CPU activity in the window |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1 | 32 | 10 | 11:11:27-11:12:12 | 1.06 -> 4.67 | - |
| 2 | 0 | 1 | 64 | 10 | 11:12:12-11:12:57 | 4.67 -> 3.47 | - |
| 3 | 0 | 2 | 32 | 10 | 11:12:57-11:13:45 | 3.47 -> 3.37 | pid 2847144 (LIBERO policy import, ~220 % CPU; seen at 11:13:09, 1 s old; end not observed) |
| 4 | 0 | 2 | 64 | 10 | 11:13:45-11:14:36 | 3.37 -> 3.54 | - |
| 5 | 0 | 3 | 32 | 10 | 11:14:36-11:15:32 | 3.54 -> 4.00 | - |
| 6 | 0 | 3 | 64 | 10 | 11:15:32-11:16:26 | 4.00 -> 5.10 | - |
| 7 | 0 | 4 | 32 | 10 | 11:16:26-11:17:22 | 5.10 -> 4.72 | - |
| 8 | 0 | 4 | 64 | 10 | 11:17:22-11:18:22 | 4.72 -> 3.25 | - |
| 9 | 0 | 2 | 32 | 1 | 11:18:22-11:18:50 | 3.25 -> 2.42 | - |
| 10 | 0 | 2 | 64 | 1 | 11:18:50-11:19:19 | 2.42 -> 5.24 | pi05-mc-ship CPU pytest (~62 s) |
| 11 | 0 | 2 | 32 | 5 | 11:19:19-11:20:03 | 5.24 -> 9.05 | pi05-mc-ship CPU pytest (~62 s) |
| 12 | 0 | 2 | 64 | 5 | 11:20:03-11:20:40 | 9.05 -> 6.59 | pi05-mc-ship CPU pytest (~62 s) |
| 13 | 1 | 1 | 32 | 10 | 11:20:40-11:21:17 | 6.59 -> 5.97 | pi05-mc-ship pytest (2 s) + git clone / checkout (I/O) |
| 14 | 1 | 1 | 64 | 10 | 11:21:17-11:21:55 | 5.97 -> 6.04 | pi05-mc-ship pytest (2 s) + git clone / checkout (I/O) |
| 15 | 1 | 2 | 32 | 10 | 11:21:55-11:22:37 | 6.04 -> 6.18 | pi05-mc-ship pytest (2 s) + git clone / checkout (I/O) |
| 16 | 1 | 2 | 64 | 10 | 11:22:37-11:23:20 | 6.18 -> 6.88 | pi05-mc-ship pytest (2 s) + git clone / checkout (I/O) |
| 17 | 1 | 3 | 32 | 10 | 11:23:20-11:24:07 | 6.88 -> 4.78 | - |
| 18 | 1 | 3 | 64 | 10 | 11:24:07-11:24:55 | 4.78 -> 4.05 | - |
| 19 | 1 | 4 | 32 | 10 | 11:24:55-11:25:46 | 4.05 -> 6.65 | - |
| 20 | 1 | 4 | 64 | 10 | 11:25:46-11:26:37 | 6.65 -> 4.85 | - |
| 21 | 1 | 2 | 32 | 1 | 11:26:37-11:27:04 | 4.85 -> 5.33 | - |
| 22 | 1 | 2 | 64 | 1 | 11:27:04-11:27:31 | 5.33 -> 4.47 | - |
| 23 | 1 | 2 | 32 | 5 | 11:27:31-11:28:05 | 4.47 -> 4.93 | - |
| 24 | 1 | 2 | 64 | 5 | 11:28:05-11:28:39 | 4.93 -> 5.24 | - |

Foreign windows: pid 2847144 seen at 11:13:09 (1 s old; how long it ran is unknown, so only the build spanning 11:13 is marked); pi05-mc-ship pytest 11:19:10-11:20:15 and pytest + git clone 11:21-11:22 (its own report). Every build's load also includes the sweep's own weight loading (~500 % CPU for a few s per build). Source: .val/mc_impl/wp6/perf/out/sweep.json, PROGRESS.log; hold 11:11-11:29, 24 / 24 builds, rc 0.
