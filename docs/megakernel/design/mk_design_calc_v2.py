# DESIGN.md v2 arithmetic (review resolution, 2026-09-30). Every number printed here is quoted in DESIGN.md v2.
# Inputs: M = measured (file named in DESIGN.md), A = assumed (a WP-P1-2 / P1-3 gate measures it).
# The ring-constrained stream timeline uses the review's fluid model (review_stream_sim.py, validated there:
# unbounded rings reproduce the v1 chain / stream figures) with v2's phase list and ring sizes.
import math
import review_stream_sim as sim

BF16, BFP8, FP32 = 2048, 1088, 4096

# ---------------- rates (A unless noted) ----------------
T_MM = 0.029           # us per tile-matmul, GR00T K4b mk_P2_mm 5.5 us / 192 (M there, D here)
T_E = 0.125            # us per eltwise / SFPU tile-op, A: GR00T "part send 1.5 + two merges ~7" (mk_k4b §3.5)
                       # = 3.5 us per merge of a 2-row x 2-dh part = 28 tile-ops (2 rows x (max 1, sub 2, exp 2, l 3)
                       # + 4 O tiles x 3); the shipped fused attention bounds it below: 1.72 us / key tile / ~20 ops = 0.09
T_KEY = (43.00 - 29.90) / (25 - 18)   # us per key tile, slope of the shipped fused_attn (PROFILE.md :223 / :401, M)
T_GELU = 1.5           # us per GELU tile on PACK (F0), A
print("rates: T_MM %.3f  T_E %.3f  T_KEY %.3f (slope; the average 43.0/25 = %.2f)" % (T_MM, T_E, T_KEY, 43.0 / 25))

def merge_fold_ops(o_tiles, rows=1):
    # one pairwise fold of a remote (m, l, O) part into the local one, per row tile:
    # max(m) 1, m - M twice 2, exp twice 2, l = a*l1 + b*l2 3, O: per tile 2 bcast-mul + 1 add
    return rows * (1 + 2 + 2 + 3) + rows * o_tiles * 3

def chain(shape):
    rows = 2 if shape == "base" else 1
    key_tiles_chunk = 5 if shape == "base" else 3
    n_remote = 4 if shape == "base" else 5     # remote parts per slice merger (5 / 6 chunks per (h, r))
    ph = {}
    ph["R1 x round"] = 6.5                                    # A (GR00T hub round, 192 KB; not scaled at LIBERO)
    ph["C1 qkv"] = 32 * rows * T_MM                            # D
    rope_ops = rows * 3                                       # own tile d, both rows: 2 mul + 1 add
    ph["R2 pair+RoPE+Q/KV dist"] = 1.0 + rope_ops * T_E + 3.5  # A pair exchange 1.0 + D RoPE + A gather/mcast 3.5
    ph["C2 attention chunk"] = key_tiles_chunk * T_KEY + 1.0   # M slope + A 1.0 (mask/finalise/part pack)
    ph["R3 dh-split merge"] = 2.0 + n_remote * merge_fold_ops(2) * T_E   # A send 2.0 + A-rate folds
    ph["R4 ctx round"] = 6.5                                   # A
    ph["C3 o_proj"] = 64 * rows * T_MM + 0.5                   # D + A residual
    ph["R5 x_mid round"] = 6.5                                 # A
    ph["C4 up|gate + GeGLU"] = 4 * 32 * rows * T_MM + 2 * rows * T_GELU   # D + A
    ph["R6 h row exchange"] = 4.0                              # A
    ph["C5 down partial"] = 16 * 4 * rows * T_MM               # D
    red_ops = rows * (7 + 7 + 1 + 1)                          # 7 landed copies + 7 adds + gate mul + residual add
    ph["R7 reduce to owners"] = 1.5 + red_ops * T_E            # A transfer + A-rate compute
    return ph

res = {}
for shape in ("base", "libero"):
    ph = chain(shape)
    c = sum(ph.values())
    print("\n%s chain per layer %.1f us" % (shape, c))
    for k, v in ph.items():
        print("  %-26s %6.2f" % (k, v))
    # ring-constrained per-core stream timeline (review model), v2 rings: w8 16 x 8,704, w16 8 x 16,384
    phases = [(k.split()[0], v) for k, v in ph.items()]
    out = {}
    for cap in (414.0, 440.0, 464.0):
        out[cap] = sim.run(phases, cap, None, w8=16 * 8704, w16=131072)
    out["v1rings414"] = sim.run(phases, 414.0, None, w8=87040, w16=131072)
    print("  timeline layer us: 414 %.1f | 440 %.1f | 464 %.1f | v1 rings @414 %.1f"
          % (out[414.0], out[440.0], out[464.0], out["v1rings414"]))
    lay = out[414.0]
    step = 18 * lay + 8.0
    loop = 10 * step / 1000
    res[shape] = (c, lay, loop)
    print("  layer %.1f  step %.1f us  loop %.2f ms  x1.36 %.2f  x1.68 %.2f" % (lay, step, loop, loop * 1.36, loop * 1.68))

# whole call: today (unprofiled) minus the profiled expert + action-io stage plus the prediction
for shape, today, stage in (("base", 84.21, 31.263 + 0.167), ("libero", 76.77, 27.845 + 0.163)):
    loop = res[shape][2]
    print("%s whole call: %.1f / %.1f / %.1f ms (today %.2f; stage subtracted %.3f, profiled basis)"
          % (shape, today - stage + loop, today - stage + 1.36 * loop, today - stage + 1.68 * loop, today, stage))
print("break-even realisation base: %.2fx of the timeline layer (173.1 us today)" % (173.1 / res["base"][1]))
print("break-even realisation libero: %.2fx (153.8 us today)" % (153.8 / res["libero"][1]))

# ---------------- L1 (union layout) ----------------
def l1(shape):
    r = 2 if shape == "base" else 1
    kt = 5 if shape == "base" else 3
    nrem = 4 if shape == "base" else 5
    in0 = max(32 * r * BF16, 64 * r * BFP8)   # x / x_mid bf16 [rows x 1024] and ctx bfp8 [rows x 2048]
    assert in0 % BF16 == 0 and in0 % BFP8 == 0
    cbs = [
        (0, "cb_in0 bf16 x/x_mid + id 29 bfp8 ctx (one CBDescriptor, two formats)", in0),
        (1, "cb_w8 16 x 8,704", 16 * 8704),
        (2, "cb_w16 8 x 16,384", 8 * 16384),
        (3, "cb_acc fp32", 4 * r * FP32),
        (4, "cb_out bf16 (qkv out; O part staging 8 dh)", 8 * BF16),
        (5, "cb_q Q[h] landing (roped)", 8 * r * BF16),
        (6, "cb_rpart RoPE partner tile landing", r * BF16),
        (7, "cb_rtmp RoPE scratch", r * BF16),
        (8, "cb_rope cos/sin own tile, own rows", 2 * r * BF16),
        (9, "cb_kv chunk double buffer bfp8", 2 * kt * 8 * 2 * BFP8),
        (10, "cb_ksuf roped K_s | V_s", 2 * 8 * r * BF16),
        (11, "cb_mask", kt * BF16),
        (12, "cb_s scores fp32", kt * FP32),
        (13, "cb_oacc fp32", 8 * FP32),
        (14, "cb_ml m, l fp32 (+ ping-pong)", 4 * FP32),
        (15, "cb_part slice-merger landing", nrem * (2 * BF16 + 2 * FP32)),
        (16, "cb_ctx 2 dh slice bfp8", 2 * BFP8),
        (17, "cb_r", r * BF16),
        (18, "cb_bias c / gate tiles", 6 * BF16),
        (19, "cb_ug fp32", 2 * r * FP32),
        (20, "cb_h", 2 * r * BF16),
        (21, "cb_hg row h slice", 16 * r * BF16),
        (22, "cb_dpart fp32", 4 * r * FP32),
        (23, "cb_red owner landing fp32", 7 * r * FP32),
        (24, "cb_x fp32 residual", r * FP32),
        (25, "cb_xs send staging", r * BF16),
        (26, "cb_const", 4 * BF16),
        (27, "cb_fence", FP32),
        (28, "cb_sync 128 words x 16 B", 2048),
    ]
    return cbs

ALLOC = 1395712 - 24576          # worker_l1_size (64 KiB cut) - l1_small_size 24,576 (test_pcc_pi05_fused.py:68)
for shape, cache_len in (("base", 800), ("libero", 640)):
    cbs = l1(shape)
    tot = sum(b for _, _, b in cbs)
    kv_tiles = cache_len // 32 * 8
    kv_bank = 36 * math.ceil(kv_tiles / 110) * BFP8    # interleaved: whole pages per bank, lockstep addresses
    print("\n%s L1 union %d B (%d CBDescriptors; ids 0..28 plus the bfp8 alias id 29 on cb_in0 = 30 ids)" % (shape, tot, len(cbs)))
    for i, n, b in cbs:
        print("  %2d %-66s %8d" % (i, n, b))
    print("  allocatable %d, KV caches per bank %d (%d tiles each, ceil/110 = %d pages), other co-tenants 20,000"
          % (ALLOC, kv_bank, kv_tiles, math.ceil(kv_tiles / 110)))
    print("  headroom %d" % (ALLOC - kv_bank - 20000 - tot))
print("v1 separate-ctx-CB check: 1,154,560 + 139,264 = %d" % (1154560 + 139264))

# ---------------- phase 2 corrections ----------------
F = lambda m, k, n: 2 * m * k * n
m = 736
vf = dict(qkv=F(m, 2048, 2560), o=F(m, 2048, 2048), gu=F(m, 2048, 32768), dn=F(m, 16384, 2048))
gf = sum(vf.values()) / 1e9
gelu_tiles = 24 * 512 / 88
lo = dict(qkv=32.93, o=26.23, gu=vf['gu'] / 168e6, gelu=gelu_tiles * 0.58, mul=gelu_tiles * 0.1, dn=200.0,
          attn=139.6 + 20, norms=2 * 15.3)
lo['exch'] = 0.1 * (lo['qkv'] + lo['o'] + lo['gu'] + lo['dn'])
print("\nphase 2 VLM layer %.2f GFLOP; GELU tiles per core (88 cores) %.1f" % (gf, gelu_tiles))
print("  lower (v2 geometry) %.0f us: %s" % (sum(lo.values()), {k: round(v, 1) for k, v in lo.items()}))
peak = 42.6e12 / 0.07
for pct in (0.07, 0.15):
    mm = sum(vf.values()) / (peak * pct) * 1e6
    other = lo['gelu'] + lo['mul'] + lo['attn'] + lo['norms']
    exch = 0.26 * mm
    print("  GR00T-realised %2.0f %% of peak (%.1f TFLOP/s): matmuls %.0f + other %.0f + exchanges 26 %% %.0f = %.0f us"
          % (pct * 100, peak * pct / 1e12, mm, other, exch, mm + other + exch))
print("  gate+up alone at 7 %%: %.0f us (TTNN layer 2,437 M)" % (vf['gu'] / 42.6e6))
vlo = sum(lo.values())
print("  VLM stack (17 full layers + 0.1 ms KV-only, as v1): lower %.1f ms" % (17 * vlo / 1e3 + 0.1))
for pct in (0.07, 0.15):
    mm = sum(vf.values()) / (peak * pct) * 1e6
    lay = mm * 1.26 + lo['gelu'] + lo['mul'] + lo['attn'] + lo['norms']
    print("  VLM stack at %2.0f %% of peak: %.1f ms (TTNN 41.54 M)" % (pct * 100, 17 * lay / 1e3 + 0.1))
# phase-2 whole-call low with the corrected GELU geometry (v1 formula: SigLIP lower 6.40 + 0.115 + VLM + 0.05 + expert + host 1.40)
print("  phase-2 whole call base, low: %.1f ms" % (6.40 + 0.115 + 17 * vlo / 1e3 + 0.1 + 0.05 + res["base"][2] + 1.40))
# phase-2 resident KV region: design-owned layout, rows 0..P-1 only (no cache_len padding), spread over 110 cores
for shape, ptiles in (("base", 23), ("libero", 17)):
    tiles = 36 * ptiles * 8
    per = math.ceil(tiles / 110) * BFP8
    print("  %s resident KV: %d tiles, %d per core -> %d B per core" % (shape, tiles, math.ceil(tiles / 110), per))
mlp = 393216 + 208896 + 147456 + 288768 + 73728 + 73728 + 40000
kvb = math.ceil(36 * 23 * 8 / 110) * BFP8
print("  VLM MLP phase L1 (base): %d + resident KV %d = %d of %d, headroom %d" % (mlp, kvb, mlp + kvb, ALLOC, ALLOC - mlp - kvb))
