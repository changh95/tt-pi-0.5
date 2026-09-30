// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pi0.5 whole-model megakernel (phase 2): constants of the PREFIX ENGINE (SigLIP x2 + projector + language embedding +
// VLM prefill writing the expert's K / V caches), shared by the three RISC kernels AND the host.
// pe_geometry.py parses every `constexpr uint32_t NAME = <int>;` line of this file (one definition per line, integer
// literals only), exactly like geometry.py parses ../kernels/mk_defs.hpp.
//
// Execution model (docs/megakernel/DESIGN.md §11): a fixed sequence of N_OPS ops; every op reads its inputs from DRAM
// (activations) / multicast feeders (weights, in0 bands) and writes its outputs to DRAM; consecutive ops are separated by
// ONE global barrier (every core arrives after its writes are acknowledged; the hub multicasts go). Inside an op the only
// cross-core protocol is the feeder -> receiver multicast ring with cumulative credits (ready) and flags (valid).
// Circular buffers 32.. are declared tiny on the host and RE-POINTED per op into the L1 arena (the phase-1 CB region,
// idle during the prefix) by every RISC that uses them (pe_common.hpp cb_point).
#pragma once
#include <cstdint>

namespace pe {

// ---------------------------------------------------------------- grid roles
constexpr uint32_t NCOL = 11;      // matmul columns (x)
constexpr uint32_t GRID_Y = 10;
constexpr uint32_t NCORES = 110;
constexpr uint32_t WF_Y = 8;       // weight feeder of column x = (x, WF_Y)
constexpr uint32_t IF_Y = 9;       // in0 feeder of band b = (b, IF_Y), b < 8
constexpr uint32_t HUB_X = 10;
constexpr uint32_t HUB_Y = 9;

// ---------------------------------------------------------------- model constants (tiles)
constexpr uint32_t S_M = 16;       // SigLIP rows: 2 images x 256 patches
constexpr uint32_t S_R = 8;        // SigLIP bands (2 row tiles each)
constexpr uint32_t S_D = 36;       // hidden 1152
constexpr uint32_t S_NQKV = 144;   // [q | k | v], 16 heads x 96 (head_dim 72 zero-padded to 96) each
constexpr uint32_t S_DCTX = 48;    // attention output width (16 heads x 3 tiles) = o_proj K
constexpr uint32_t S_I = 136;      // MLP 4304 zero-padded to 4352
constexpr uint32_t S_KP = 19;      // patch features 588 zero-padded to 608
constexpr uint32_t S_L = 27;
constexpr uint32_t S_HEADS = 16;
constexpr uint32_t S_DH = 3;
constexpr uint32_t S_IMG = 8;      // row tiles per image
constexpr uint32_t V_D = 64;       // VLM hidden 2048
constexpr uint32_t V_NQKV = 80;    // 8 q heads x 8 + K 8 + V 8
constexpr uint32_t V_I = 512;      // MLP 16384
constexpr uint32_t V_L = 18;
constexpr uint32_t V_H = 8;
constexpr uint32_t V_DH = 8;
constexpr uint32_t V_PIECE = 16;   // K tiles per weight page of every VLM op
constexpr uint32_t S_PIECE = 12;   // SigLIP qkv / fc1 / projector (K 36)
constexpr uint32_t S_PIECE_O = 16; // SigLIP o_proj (K 48)
constexpr uint32_t S_PIECE_2 = 17; // SigLIP fc2 (K 136)

// ---------------------------------------------------------------- op sequence
constexpr uint32_t K_NOP = 0;
constexpr uint32_t K_MM = 1;
constexpr uint32_t K_NORM = 2;
constexpr uint32_t K_ATTN = 3;
constexpr uint32_t K_EMBED = 4;

// op codes (what) -- the per-layer order
constexpr uint32_t W_PATCH = 1;
constexpr uint32_t W_SLN1 = 2;
constexpr uint32_t W_SQKV = 3;
constexpr uint32_t W_SATTN = 4;
constexpr uint32_t W_SO = 5;
constexpr uint32_t W_SLN2 = 6;
constexpr uint32_t W_SFC1 = 7;
constexpr uint32_t W_SFC2 = 8;
constexpr uint32_t W_POSTLN = 9;
constexpr uint32_t W_PROJ = 10;
constexpr uint32_t W_EMBED = 11;
constexpr uint32_t W_VRMS1 = 12;
constexpr uint32_t W_VQKV = 13;
constexpr uint32_t W_VATTN = 14;
constexpr uint32_t W_VO = 15;
constexpr uint32_t W_VRMS2 = 16;
constexpr uint32_t W_VGU = 17;
constexpr uint32_t W_VDOWN = 18;

constexpr uint32_t OPS_PER_LAYER = 7;
constexpr uint32_t OP_S0 = 1;      // first SigLIP layer op
constexpr uint32_t OP_POSTLN = 190;
constexpr uint32_t OP_PROJ = 191;
constexpr uint32_t OP_EMBED = 192;
constexpr uint32_t OP_V0 = 193;    // first VLM layer op
constexpr uint32_t N_OPS = 314;    // the last VLM layer runs RMS1 + QKV only (its K / V is all the expert reads)

// matmul modes: R = in0 band resident, N-outer, the whole K accumulated in DST (no reload);
//               S = in0 streamed in K blocks, K-outer, fp32 partials reloaded (VLM down: K 512)
constexpr uint32_t MM_R = 0;
constexpr uint32_t MM_S = 1;

// epilogues (per output pair of N tiles; DST tile (r, t) = r * 2 + t)
constexpr uint32_t E_NONE = 0;
constexpr uint32_t E_BIAS = 1;       // + row-broadcast bias tile per N tile
constexpr uint32_t E_BIAS_GELU = 2;  // + bias, gelu_tanh
constexpr uint32_t E_RES = 3;        // DST preloaded with the fp32 residual (out tensor, in place)
constexpr uint32_t E_RES_BIAS = 4;   // residual preload + bias
constexpr uint32_t E_ROPE = 5;       // VLM qkv: RoPE on q / k pairs (d, d + 4), none on v pairs
constexpr uint32_t E_GEGLU = 6;      // h = up * gelu_tanh(gate), one tile per row
constexpr uint32_t E_POS = 7;        // DST preloaded with the bf16 (position + bias) table

// norm kinds
constexpr uint32_t N_RMS = 0;
constexpr uint32_t N_LN = 1;
// norm column groups: an item is (row tile, column group); statistics over the whole row (redundant per group), the
// apply and the write of one group only (SigLIP 16 x 6 = 96 items, VLM mt x 4 <= 96 items: <= one item per core)
constexpr uint32_t NCG_S = 6;
constexpr uint32_t NCG_V = 4;
static_assert(S_M * NCG_S <= NCORES, "norm items: <= one per core (the stats exchange assumes item == core)");

// ---------------------------------------------------------------- circular buffers (ids 32..; 0..31 = phase 1)
constexpr uint32_t P_IN0 = 32;     // bf16 in0 (resident band / streamed K blocks)
constexpr uint32_t P_IN08 = 33;    // bfp8 in0 (VLM down: h)
constexpr uint32_t P_W8 = 34;      // bfp8 weight ring
constexpr uint32_t P_W16 = 35;     // bf16 weight ring (patch embed)
constexpr uint32_t P_S16 = 36;     // bf16 side tiles (bias / RoPE tables / pos / gamma / beta / mask)
constexpr uint32_t P_S32 = 37;     // fp32 side tiles, UnpackToDestFp32 (residual preload)
constexpr uint32_t P_PART = 38;    // fp32 partials, UnpackToDestFp32 (mode S)
constexpr uint32_t P_O16 = 39;     // bf16 out
constexpr uint32_t P_O8 = 40;      // bfp8 out
constexpr uint32_t P_O32 = 41;     // fp32 out
constexpr uint32_t P_X32 = 42;     // fp32 norm input (FPU operand: default unpack)
constexpr uint32_t P_SCR = 43;     // fp32 scratch (FPU operand)
constexpr uint32_t P_R = 44;       // fp32 per-row statistics tiles (FPU operand)
constexpr uint32_t P_Q = 45;       // bf16 attention q
constexpr uint32_t P_KV8 = 46;     // bfp8 K / V (VLM, from the caches)
constexpr uint32_t P_KV16 = 47;    // bf16 K / V (SigLIP)
constexpr uint32_t P_MSK = 48;     // bf16 key-mask tiles
constexpr uint32_t P_SS = 49;      // bf16 scores / P
constexpr uint32_t P_M = 50;       // bf16 row max (column vector)
constexpr uint32_t P_MF = 51;      // bf16 row max broadcast (full tile)
constexpr uint32_t P_L = 52;       // fp32 row sum (UnpackToDestFp32)
constexpr uint32_t P_OP = 53;      // bf16 parts O [part][d]
constexpr uint32_t P_PM = 54;      // bf16 parts m (full tiles)
constexpr uint32_t P_PL = 55;      // fp32 parts l (UnpackToDestFp32)
constexpr uint32_t P_D = 56;       // fp32 merge weights diag(w_i / L) (FPU operand)
constexpr uint32_t P_CONST = 57;   // bf16 constants: ONES, 1/2048, 1/1152, IDENT, ZERO
constexpr uint32_t P_RM = 58;      // bf16 row-major staging (tilize input)
constexpr uint32_t P_TOK = 59;     // raw: token ids / TRISC go tokens (1 page)
constexpr uint32_t P_SYNC = 60;    // raw: prefix sync words (static, never re-pointed)
constexpr uint32_t P_TAIL = 61;    // raw: arena extension after the phase-1 CBs (host-sized, may be one page)
constexpr uint32_t P_OPD = 62;     // raw: per-op descriptors NCRISC -> TRISC (static, 2 pages of OPD_BYTES)
constexpr uint32_t P_FIRST = 32;
constexpr uint32_t P_LAST = 62;

// ---------------------------------------------------------------- op descriptor (P_OPD page, uint32 words)
// The NCRISC computes every op's geometry (pe_common.hpp describe / layout) and hands the TRISCs one page per op
// after the op's barrier go; the TRISCs read it with read_tile_value (UNPACK reads, mailboxes to MATH / PACK), so the
// geometry code lives in the two DM binaries only and a TRISC can never start an op before its barrier.
constexpr uint32_t OPD_BYTES = 256;
constexpr uint32_t OD_KIND = 0;      // K_NOP when this core has no part in the op
constexpr uint32_t OD_WHAT = 1;
constexpr uint32_t OD_MODE = 2;
constexpr uint32_t OD_RPB = 3;
constexpr uint32_t OD_KT = 4;
constexpr uint32_t OD_PIECE = 5;
constexpr uint32_t OD_NP = 6;        // this core's output pairs
constexpr uint32_t OD_P0 = 7;        // its first pair
constexpr uint32_t OD_EPI = 8;
constexpr uint32_t OD_FID = 9;
constexpr uint32_t OD_WBF16 = 10;
constexpr uint32_t OD_NKIND = 11;
constexpr uint32_t OD_NK = 12;
constexpr uint32_t OD_NITEMS = 13;   // this core's items (norm / attention / embedding)
constexpr uint32_t OD_IT0 = 14;      // its first item (= its linear index; items step by NCORES)
constexpr uint32_t OD_WSLOTS = 15;
constexpr uint32_t OD_A = 16;        // 16..: absolute L1 addresses of the op's buffers (OA_* order)
constexpr uint32_t OA_IN0 = 0;
constexpr uint32_t OA_W = 1;
constexpr uint32_t OA_S16 = 2;
constexpr uint32_t OA_S32 = 3;
constexpr uint32_t OA_O16 = 4;
constexpr uint32_t OA_O8 = 5;
constexpr uint32_t OA_O32 = 6;
constexpr uint32_t OA_PART = 7;
constexpr uint32_t OA_X32 = 8;
constexpr uint32_t OA_SCR = 9;
constexpr uint32_t OA_R = 10;
constexpr uint32_t OA_Q = 11;
constexpr uint32_t OA_KV = 12;
constexpr uint32_t OA_MSK = 13;
constexpr uint32_t OA_SS = 14;
constexpr uint32_t OA_M = 15;
constexpr uint32_t OA_OP = 16;
constexpr uint32_t OA_PM = 17;
constexpr uint32_t OA_PL = 18;
constexpr uint32_t OA_D = 19;
constexpr uint32_t OA_CST = 20;
constexpr uint32_t OA_RM = 21;
constexpr uint32_t OA_TOK = 22;
constexpr uint32_t OA_N = 23;
constexpr uint32_t OD_N = 39;

constexpr uint32_t PC_ONES = 0;
constexpr uint32_t PC_MEAN_V = 1;  // 1/2048
constexpr uint32_t PC_MEAN_S = 2;  // 1/1152
constexpr uint32_t PC_IDENT = 3;
constexpr uint32_t PC_ZERO = 4;
constexpr uint32_t PC_N = 5;

// ---------------------------------------------------------------- sync words (P_SYNC, 16 B apart, zeroed at boot)
constexpr uint32_t PSTRIDE = 16;
constexpr uint32_t PS_GO = 0;        // every core: barrier go (op index + 1), multicast by the hub
constexpr uint32_t PS_ARR = 1;       // hub: barrier arrivals (cumulative)
constexpr uint32_t PS_NCDONE = 2;    // local: ops the NCRISC finished (BRISC waits before it arrives)
constexpr uint32_t PS_OPK = 3;       // local: ops whose (Op, Lay) the NCRISC published at PS_SHARE (op index + 1)
constexpr uint32_t PS_NR = 5;        // norm item core: (rstd, mu) tiles landed in its P_R (cumulative count)
constexpr uint32_t PS_W_VAL = 4;     // receiver: weight pages valid ((op << 16) | pages)
constexpr uint32_t PS_I_VAL = 6;     // receiver: in0 pages valid ((op << 16) | pages)
constexpr uint32_t PS_SRC_GO = 7;    // hub-local source of the go multicast
constexpr uint32_t PS_SRC_W = 8;     // weight-feeder-local source of the valid multicast
constexpr uint32_t PS_SRC_I = 9;     // in0-feeder-local source of the valid multicast
constexpr uint32_t PS_DBG = 10;      // 10..13 NCRISC (op, phase, a, b), 14..15 BRISC (op, phase): last blocking step
// Ring credits are PER RECEIVER (cumulative pages each receiver has room for) and a feeder takes the MINIMUM: a
// summed credit lets a receiver that runs ahead cover for one that lags, and the feeder then overwrites a slot the
// laggard is still reading (ring depth > 1; reproduced 2026-09-30 on the VLM down op, bands far from the feeders).
constexpr uint32_t PS_W_RDY0 = 16;   // weight feeder of column x: 16 + y = credits of receiver (x, y), y < 8
constexpr uint32_t PS_I_RDY0 = 24;   // in0 feeder of band b: 24 + x = credits of receiver (x, b), x < 11
constexpr uint32_t PS_SRC_KV = 35;   // K / V feeder-local source of the valid multicast
constexpr uint32_t PS_KV_VAL0 = 36;  // 36..39: every core: VLM K / V quarter f landed ((op << 16) | 1)
constexpr uint32_t KVF_X0 = 7;       // the 4 VLM K / V feeders: (7..10, IF_Y) (linear index >= 106: never an attention item)
constexpr uint32_t PS_N = 40;
constexpr uint32_t PS_DIAG = 40;     // 40..43: diagnostics staging (8 words, written once at the end)
constexpr uint32_t PS_TSTAMP = 44;   // 44..47: the hub's time-stamp record staging (16 B)
constexpr uint32_t PS_SHARE = 48;    // 48..63: the NCRISC's (Op, Lay) of the current op, read by the BRISC (256 B)
constexpr uint32_t PS_IV0 = 64;      // 64..71: every compute core: mode-R in0 piece q landed ((op << 16) | 1)
constexpr uint32_t PS_IV_N = 8;      // (at most 8 K pieces per mode-R op)
constexpr uint32_t PS_WORDS = 72;    // P_SYNC = PS_WORDS x PSTRIDE bytes

// ---------------------------------------------------------------- common runtime args (appended after phase 1's)
// The TRISC reads only phase 1's and PA_OPFIRST..PA_REPS, so its list is cut at PA_TRISC_N (ring bytes). The VLM K / V
// cache addresses are phase 1's own C_K_ADDR / C_V_ADDR (the prefix writes the caches the expert reads).
constexpr uint32_t PA0 = 100;
constexpr uint32_t PA_OPFIRST = 100; // first op executed (0 = from the start)
constexpr uint32_t PA_DBGSTOP = 101; // first op NOT executed (N_OPS = everything)
constexpr uint32_t PA_REPS = 102;  // the op range [OPFIRST, DBGSTOP) runs this many times (timing; 1 in production)
constexpr uint32_t PA_TRISC_N = 103;  // the TRISC's common-arg list length
constexpr uint32_t PA_X_S = 103;   // fp32  [512, 1152]   SigLIP residual
constexpr uint32_t PA_XN_S = 104;  // bf16  [512, 1152]   normalised (LN out)
constexpr uint32_t PA_QKV_S = 105; // bf16  [512, 4608]
constexpr uint32_t PA_CTX_S = 106; // bf16  [512, 1536]
constexpr uint32_t PA_H_S = 107;   // bf16  [512, 4352]
constexpr uint32_t PA_X_V = 108;   // fp32  [MT*32, 2048] VLM residual
constexpr uint32_t PA_XN_V = 109;  // bf16  [MT*32, 2048]
constexpr uint32_t PA_Q_V = 110;   // bf16  [8 heads][MT][8] tiles
constexpr uint32_t PA_CTX_V = 111; // bf16  [MT*32, 2048]
constexpr uint32_t PA_H_V = 112;   // bfp8  [MT*32, 16384]
constexpr uint32_t PA_IM2COL = 113; // bf16  [512, 608] TILE (host im2col, the request's pixels)
constexpr uint32_t PA_TOK = 114;   // uint32 [1, NTOK] ROW_MAJOR (the request's token ids)
constexpr uint32_t PA_EMB = 115;   // bf16  [vocab, 2048] ROW_MAJOR (embedding table)
constexpr uint32_t PA_VMASK = 116; // bf16  [32, P] TILE: row-broadcast key bias of the VLM (the request's mask)
constexpr uint32_t PA_COSQ = 117;  // bf16  [MT*32, 256] TILE: VLM q RoPE cos (x 1/16)
constexpr uint32_t PA_SINQ = 118;  // signed sin (x 1/16)
constexpr uint32_t PA_COSK = 119;  
constexpr uint32_t PA_SINK = 120;  
constexpr uint32_t PA_POS = 121;   // bf16  [256, 1152] position table + patch bias
constexpr uint32_t PA_SVEC = 122;  // bf16  per SigLIP layer: row-broadcast vectors (see pe_common.hpp SV_*)
constexpr uint32_t PA_VVEC = 123;  // bf16  per VLM layer: 1 + w of both RMS norms (row-broadcast)
constexpr uint32_t PA_GVEC = 124;  // bf16  post-LN w, b, projector bias (row-broadcast)
constexpr uint32_t PA_CONST = 125; // bf16  [32, 32 * PC_N] constants
constexpr uint32_t PA_WPATCH = 126; // bf16  patch-embed weight arena (bank-striped pages)
constexpr uint32_t PA_WPROJ = 127; // bfp8  projector weight arena
constexpr uint32_t PA_DIAG = 128;  // uint32 [110 pages of 64 B] per-core diagnostics (arena bounds, ops run)
constexpr uint32_t PA_TIMES = 129; // uint32 [4096, 16] ROW_MAJOR: the hub stamps (wall clock, k, op) at every go
constexpr uint32_t PA_NOCX0 = 130;   // 130..140: NoC x of logical columns 0..10 (host-translated)
constexpr uint32_t PA_BRISC_N = 141; // the BRISC's common-arg list length (the weight arenas are NCRISC-only)
constexpr uint32_t PA_WS = 141;      // SigLIP layer weight arenas (27, bfp8, bank-striped pages)
constexpr uint32_t PA_WV = 168;      // VLM layer weight arenas (18)
constexpr uint32_t PA_N = 186;       // common args in total

// ---------------------------------------------------------------- per-core runtime args (appended after phase 1's)
constexpr uint32_t PR0 = 48;
constexpr uint32_t PR_X = 48;        // logical x
constexpr uint32_t PR_Y = 49;        // logical y
constexpr uint32_t PR_LIN = 50;      // y * 11 + x
constexpr uint32_t PR_WFX = 51;      // NoC x / y of this core's column weight feeder
constexpr uint32_t PR_WFY = 52;
constexpr uint32_t PR_IFX = 53;      // NoC x / y of this core's band in0 feeder (y < 8)
constexpr uint32_t PR_IFY = 54;
constexpr uint32_t PR_HUBX = 55;
constexpr uint32_t PR_HUBY = 56;
constexpr uint32_t PR_COLX = 57;     // weight feeder: its column rectangle x (NoC) ; y0 = row 0 ; y1 given per op
constexpr uint32_t PR_COLY0 = 58;    // NoC y of logical row 0 in this column
constexpr uint32_t PR_ROWY = 59;     // in0 feeder: NoC y of its band row
constexpr uint32_t PR_ROWX0 = 60;    // NoC x of logical column 0 / 10
constexpr uint32_t PR_ROWX1 = 61;
constexpr uint32_t PR_GX0 = 62;      // whole-grid rectangle (NoC)
constexpr uint32_t PR_GY0 = 63;
constexpr uint32_t PR_GX1 = 64;
constexpr uint32_t PR_GY1 = 65;
constexpr uint32_t PR_NOCY0 = 66;    // 66..75: NoC y of logical rows 0..9 (column rectangles end at row R - 1)
constexpr uint32_t PR_N = 76;

// ---------------------------------------------------------------- prefix compile-time args (at index PE_CT0 define)
constexpr uint32_t PT_MT = 0;        // VLM row tiles (24 base / 18 LIBERO)
constexpr uint32_t PT_PT = 1;        // VLM prefix key tiles (23 / 17)
constexpr uint32_t PT_RV = 2;        // VLM bands (8 / 6)
constexpr uint32_t PT_LT = 3;        // language row tiles (7 / 1)
constexpr uint32_t PT_NTOK = 4;      // prompt tokens (224 / 32)
constexpr uint32_t PT_EPS_V = 5;     // fp32 bits of the VLM RMS eps
constexpr uint32_t PT_EPS_S = 6;     // fp32 bits of the SigLIP LN eps
constexpr uint32_t PT_EMB_SCALE = 7; // fp32 bits of sqrt(2048)
constexpr uint32_t PT_ARENA = 8;     // arena bytes the host reserved (checked in-kernel)
constexpr uint32_t PT_ACC = 9;       // TensorAccessorArgs: DRAM interleaved, then the L1 cache (bfp8)
constexpr uint32_t PT_N = 9;

}  // namespace pe
