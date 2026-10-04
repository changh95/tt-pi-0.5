// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pi0.5 expert megakernel (phase 1): constants shared by the three RISC kernels AND by the host.
// geometry.py parses every `constexpr uint32_t NAME = <int>;` line of this file, so the host and the kernels can
// never disagree on a CB id, a sync-word slot or a runtime-argument index (test_cpu_defs_parse).
// Keep one definition per line, integer literals only.
#pragma once
#include <cstdint>

namespace mk {

// ---------------------------------------------------------------- model constants
constexpr uint32_t D_T = 32;     // expert width 1024 in tiles
constexpr uint32_t NH = 8;       // q heads
constexpr uint32_t GRID_ROWS = 10;  // worker grid rows (geometry.GRID[1]): a head column holds <= 10 units
constexpr uint32_t DH_T = 8;     // head_dim 256 in tiles
constexpr uint32_t MLP_T = 128;  // mlp_dim 4096 in tiles
constexpr uint32_t N_LAYERS = 18;
constexpr uint32_t N_STEPS = 16;  // the most denoising steps (per-step arena slots); a model runs N <= N_STEPS
constexpr uint32_t N_GEN = 288;   // N_STEPS * N_LAYERS

// ---------------------------------------------------------------- circular buffers (ids)
// Every CB exists on every core of the program with the same size, so a CB's base address is identical on every
// core: remote writers address a peer's CB by their own local base. Landing CBs are used in FULL-CAPACITY cycles
// (every push / pop moves exactly the CB capacity), so every pointer is at the base between transactions.
constexpr uint32_t CB_IN0 = 0;  // bf16 64*RT: x / x_mid (32*RT tiles) + r (RT tiles at 32*RT) | ctx (64*RT)
constexpr uint32_t CB_W8 = 1;   // bfp8 ring: qkv / up|gate weight pages (8 tiles per page)
// CB_W16: bf16 ring (8 pages): o_proj / down pages; H0 per step: 4 x (W_in, b_in), Wout', c_out, 3 pad
constexpr uint32_t CB_W16 = 2;
constexpr uint32_t CB_WC = 3;      // bf16 ring: the per-(step, layer) constant page (8 tiles)
constexpr uint32_t CB_RTOK = 4;    // token: r has landed in CB_IN0 (BRISC -> TRISC)
constexpr uint32_t CB_QKVO = 5;    // bf16 2*RT: pair core output [r][2] (dh tiles j, j+4)
constexpr uint32_t CB_TAB = 6;     // bf16 4*RT: pair core RoPE tables [r][cos j, cos j+4, sin j, sin j+4]
constexpr uint32_t CB_Q = 7;       // bf16 8*RT: roped Q of the column's head [r][d]
constexpr uint32_t CB_KSV = 8;     // bf16 16*RT: suffix K [r][d] then suffix V [r][d]
constexpr uint32_t CB_KV = 9;      // bfp8 2*CHT*8: this unit's prefix K [i][d] then V [i][d]
constexpr uint32_t CB_MASK = 10;   // bf16 CHT: key-mask tiles of this unit's chunk
constexpr uint32_t CB_S = 11;      // bf16 CHT: scores, then P in place
constexpr uint32_t CB_M = 12;      // bf16 1: row max (column vector, col 0)
constexpr uint32_t CB_MF = 13;     // bf16 1: row max broadcast over columns (full tile)
constexpr uint32_t CB_L = 14;      // fp32 1: row sum of P, full tile (UnpackToDestFp32)
constexpr uint32_t CB_OP = 15;     // bf16 8: unnormalised O part [d]
constexpr uint32_t CB_PART = 16;   // bf16 8*(NCH-1): merger landing O parts [i][d]
constexpr uint32_t CB_PM = 17;     // bf16 NCH-1: merger landing m (full tiles)
constexpr uint32_t CB_PL = 18;     // fp32 NCH-1: merger landing l (UnpackToDestFp32)
constexpr uint32_t CB_D = 19;      // fp32 NCH: merge weights diag(s_i) (FPU operand)
constexpr uint32_t CB_CTXO = 20;   // bf16 8: merger output ctx[r, h] [d]
constexpr uint32_t CB_XRES = 21;   // fp32 RT: owner residual x[:, n] (UnpackToDestFp32); H0: Euler state x_t
constexpr uint32_t CB_XS = 22;     // bf16 RT: owner x / x_mid staging (sent to H0)
constexpr uint32_t CB_H = 23;      // bf16 2*RT: MLP core h [r][2]
constexpr uint32_t CB_HG = 24;     // bf16 16*RT: row h slice [r][16] (down in0)
constexpr uint32_t CB_DP = 25;     // fp32 4*RT: down partial [n][r]
constexpr uint32_t CB_RED = 26;    // fp32 8*RT: owner reduce landing [kg][r] (FPU operand)
constexpr uint32_t CB_CONST = 27;  // bf16 4: ONES, ONES/1024, IDENT, ZERO
constexpr uint32_t CB_SCR = 28;    // fp32 2: scratch (FPU operand)
constexpr uint32_t CB_SCR16 = 29;  // bf16 RT: H0 scratch (noise, x_t in bf16, r_f)
constexpr uint32_t CB_ROUT = 30;   // bf16 RT: H0 r output / final output staging
constexpr uint32_t CB_SYNC = 31;   // raw: sync words, SYNC_STRIDE bytes each
constexpr uint32_t N_CBS = 32;

constexpr uint32_t CONST_ONES = 0;
constexpr uint32_t CONST_MEAN = 1;  // 1/1024 in every element (row mean over the 1024-wide residual)
constexpr uint32_t CONST_IDENT = 2;
constexpr uint32_t CONST_ZERO = 3;

// per-(step, layer) constant page slots (CB_WC)
constexpr uint32_t WC_CQKV = 0;   // 2 tiles: folded qkv bias of the pair core's two columns (row-broadcast)
constexpr uint32_t WC_CUG = 2;    // 4 tiles: folded up|gate bias [u j0, u j1, g j0, g j1]
constexpr uint32_t WC_GATTN = 6;  // 1 tile: attention gate of owner column n (row-broadcast)
constexpr uint32_t WC_GMLP = 7;   // 1 tile: MLP gate of owner column n

constexpr uint32_t PAGE_TILES = 8;
constexpr uint32_t W8_PAGES = 16;
constexpr uint32_t W16_PAGES = 8;
constexpr uint32_t WC_PAGES = 2;

// ---------------------------------------------------------------- sync words (index into CB_SYNC, 16 B apart)
// Every word is a cumulative counter that never resets inside a launch (zeroed once before the boot barrier).
// Flags (written by a sender's multicast of its own source word) carry g + 1 after generation g.
constexpr uint32_t SYNC_STRIDE = 16;
constexpr uint32_t S_X_FLAG = 0;     // receivers of H0's x round
constexpr uint32_t S_R_FLAG = 1;     // receivers of H0's r_in
constexpr uint32_t S_XM_FLAG = 2;    // receivers of H0's x_mid round
constexpr uint32_t S_RP_FLAG = 3;    // receivers of H0's r_post
constexpr uint32_t S_CTX_FLAG = 4;   // owners, H1's ctx round
constexpr uint32_t S_Q_FLAG = 5;     // column members, Q leader
constexpr uint32_t S_KV_FLAG = 6;    // last-chunk units, KL
constexpr uint32_t S_H_FLAG = 7;     // row members, row leader
constexpr uint32_t S_X_ARR = 8;      // H0: owners' x deposits (32 per gather)
constexpr uint32_t S_XM_ARR = 9;     // H0: owners' x_mid deposits (32 per gen)
constexpr uint32_t S_X_RDY = 10;     // H0: x-round receivers ready
constexpr uint32_t S_XM_RDY = 11;    // H0: x_mid-round receivers ready
constexpr uint32_t S_CTX_ARR = 12;   // H1: mergers' ctx deposits
constexpr uint32_t S_CTX_RDY = 13;   // H1: owners ready
constexpr uint32_t S_Q_ARR = 14;     // Q leader: 4 pair deposits per gen
constexpr uint32_t S_Q_RDY = 15;     // Q leader: column receivers ready
constexpr uint32_t S_KV_ARR = 16;    // KL: 8 K/V pair deposits per gen
constexpr uint32_t S_KV_QDONE = 17;  // KL: 8 Q leaders finished their multicast
constexpr uint32_t S_KV_RDY = 18;    // KL: last-chunk units ready
constexpr uint32_t S_PART_ARR = 19;  // merger: NCH-1 part deposits per gen
constexpr uint32_t S_H_ARR = 20;     // row leader: 8 h deposits per gen
constexpr uint32_t S_H_RDY = 21;     // row leader: 7 row receivers ready
constexpr uint32_t S_RED_ARR = 22;   // owner: 8 partial deposits per gen
constexpr uint32_t S_SRC0 = 24;      // 24..31: sender-local source words of the flag multicasts
constexpr uint32_t S_DBG = 32;       // 32..35: debug words (s, l, phase, waypoint)
constexpr uint32_t S_KVM_FLAG = 36;  // MK_KV_MCAST: row units, the chunk's K / V landed (prefetch count)
constexpr uint32_t S_KVM_RDY = 37;   // MK_KV_MCAST: the row's head-0 unit, the other 7 units' CB_KV free (cumulative)
constexpr uint32_t S_PRT_FLAG = 38;  // row loop: column units, the merger's part slots free for row r >= 1 (count)
constexpr uint32_t N_SYNC = 40;

// ---------------------------------------------------------------- role bits
constexpr uint32_t R_PAIR = 1;
constexpr uint32_t R_UNIT = 2;
constexpr uint32_t R_MERGER = 4;
constexpr uint32_t R_QLEAD = 8;
constexpr uint32_t R_KL = 16;
constexpr uint32_t R_OWNER = 32;
constexpr uint32_t R_MLP = 64;
constexpr uint32_t R_ROWLEAD = 128;
constexpr uint32_t R_H0 = 256;
constexpr uint32_t R_H1 = 512;
constexpr uint32_t R_XRECV = 1024;   // consumes the x round (pair cores; owners at layer 0)
constexpr uint32_t R_LASTCH = 2048;  // unit of the last key chunk (reads the suffix K / V)

constexpr uint32_t PK_Q = 0;
constexpr uint32_t PK_K = 1;
constexpr uint32_t PK_V = 2;

// ---------------------------------------------------------------- per-core runtime args (same on every RISC)
constexpr uint32_t A_ROLE = 0;
constexpr uint32_t A_PAIR_KIND = 1;
constexpr uint32_t A_PAIR_J = 2;
constexpr uint32_t A_HEAD = 3;
constexpr uint32_t A_PAIR_DX = 4;  // NoC x of the pair's destination (Q leader / KL)
constexpr uint32_t A_PAIR_DY = 5;
constexpr uint32_t A_PAIR_DT = 6;  // tile offset of (r=0, d=j) in the destination landing
constexpr uint32_t A_UNIT_R = 7;
constexpr uint32_t A_UNIT_KC = 8;
constexpr uint32_t A_UNIT_NPRE = 9;  // prefix key tiles in this unit's chunk
constexpr uint32_t A_MERGER_X = 10;
constexpr uint32_t A_MERGER_Y = 11;
constexpr uint32_t A_QLEAD_X = 12;
constexpr uint32_t A_QLEAD_Y = 13;
constexpr uint32_t A_COL_X0 = 14;  // Q leader: its column rectangle (NoC coords)
constexpr uint32_t A_COL_Y0 = 15;
constexpr uint32_t A_COL_X1 = 16;
constexpr uint32_t A_COL_Y1 = 17;
constexpr uint32_t A_OWNER_N = 18;
constexpr uint32_t A_ROWL_X = 19;
constexpr uint32_t A_ROWL_Y = 20;
constexpr uint32_t A_ROW_X0 = 21;  // row leader: its row rectangle
constexpr uint32_t A_ROW_Y0 = 22;
constexpr uint32_t A_ROW_X1 = 23;
constexpr uint32_t A_ROW_Y1 = 24;
constexpr uint32_t A_MLP_KG = 25;
constexpr uint32_t A_MLP_NG = 26;
constexpr uint32_t A_OWNX0 = 27;     // 27..30: NoC x of this MLP core's 4 owners
constexpr uint32_t A_OWNY0 = 31;     // 31..34: NoC y
constexpr uint32_t A_BANK = 35;      // DRAM bank of this core's streams
constexpr uint32_t A_W8_OFF = 36;    // byte offset of the w8 stream in the bank (every step tensor)
constexpr uint32_t A_W8_PAGES = 37;  // pages per step
constexpr uint32_t A_W16_OFF = 38;
constexpr uint32_t A_W16_PAGES = 39;
constexpr uint32_t A_WC_OFF = 40;
constexpr uint32_t A_WC_PAGES = 41;
constexpr uint32_t A_UNIT_KT0 = 42;  // first key tile of the chunk
constexpr uint32_t N_RT_ARGS = 43;
// MK_KV_MCAST only (appended by ExpertMegakernel.program when the define is on): the row's head-0 unit and the row
// rectangle (the prefix programs, which compile the expert code without the define, never read them)
constexpr uint32_t A_KVM_LX = 43;
constexpr uint32_t A_KVM_LY = 44;
constexpr uint32_t A_KVM_X0 = 45;
constexpr uint32_t A_KVM_Y0 = 46;
constexpr uint32_t A_KVM_X1 = 47;
constexpr uint32_t A_KVM_Y1 = 48;

// ---------------------------------------------------------------- common runtime args
constexpr uint32_t C_H0_X = 0;
constexpr uint32_t C_H0_Y = 1;
constexpr uint32_t C_H1_X = 2;
constexpr uint32_t C_H1_Y = 3;
constexpr uint32_t C_KL_X = 4;
constexpr uint32_t C_KL_Y = 5;
constexpr uint32_t C_RX_X0 = 6;  // x round rectangle (pair cores + owners)
constexpr uint32_t C_RX_Y0 = 7;
constexpr uint32_t C_RX_X1 = 8;
constexpr uint32_t C_RX_Y1 = 9;
constexpr uint32_t C_RM_X0 = 10;  // x_mid round rectangle (the 64 MLP cores)
constexpr uint32_t C_RM_Y0 = 11;
constexpr uint32_t C_RM_X1 = 12;
constexpr uint32_t C_RM_Y1 = 13;
constexpr uint32_t C_RO_X0 = 14;  // owners rectangle (ctx round)
constexpr uint32_t C_RO_Y0 = 15;
constexpr uint32_t C_RO_X1 = 16;
constexpr uint32_t C_RO_Y1 = 17;
constexpr uint32_t C_RK_X0 = 18;  // last-chunk units rectangle (KL round)
constexpr uint32_t C_RK_Y0 = 19;
constexpr uint32_t C_RK_X1 = 20;
constexpr uint32_t C_RK_Y1 = 21;
constexpr uint32_t C_RA_X0 = 22;  // whole grid (boot barrier)
constexpr uint32_t C_RA_Y0 = 23;
constexpr uint32_t C_RA_X1 = 24;
constexpr uint32_t C_RA_Y1 = 25;
constexpr uint32_t C_N_RX = 26;     // cores in the x rectangle (multicast destinations)
constexpr uint32_t C_N_XRDY = 27;   // x-round receivers that announce ready (pair cores + owners)
constexpr uint32_t C_N_CORES = 28;  // cores in the program (boot barrier)
constexpr uint32_t C_MASK = 29;
constexpr uint32_t C_COSQ = 30;
constexpr uint32_t C_SINQ = 31;
constexpr uint32_t C_COSK = 32;
constexpr uint32_t C_SINK = 33;
constexpr uint32_t C_NOISE = 34;
constexpr uint32_t C_OUT = 35;
constexpr uint32_t C_CONSTS = 36;
constexpr uint32_t C_DEBUG = 37;     // generations run: 18 N for N denoising steps (fewer = debug stop)
constexpr uint32_t C_W8_ADDR = 38;    // 38..53: per-step w8 arena tensors (the first N used)
constexpr uint32_t C_W16_ADDR = 54;   // 54..69: per-step w16 arena tensors (also hold the WC streams)
constexpr uint32_t C_K_ADDR = 70;     // 70..87: per-layer K caches
constexpr uint32_t C_V_ADDR = 88;     // 88..105: per-layer V caches
constexpr uint32_t C_DBGOUT = 106;    // debug output: the residual x [S, 1024] after the last generation
constexpr uint32_t C_DT_BITS = 107;   // Euler dt (-1 / N) as fp32 bits
constexpr uint32_t N_COMMON_ARGS = 108;

// ---------------------------------------------------------------- compile-time args (same list on every kernel)
constexpr uint32_t CT_RT = 0;        // suffix row tiles (2 base / 1 LIBERO)
constexpr uint32_t CT_PT = 1;        // prefix key tiles
constexpr uint32_t CT_CHT = 2;       // key tiles per chunk
constexpr uint32_t CT_NCH = 3;       // key chunks (units per (head, row))
constexpr uint32_t CT_EPS_BITS = 4;  // RMS eps as fp32 bits
constexpr uint32_t CT_ACC0 = 5;      // TensorAccessorArgs start (cache, mask, table, noise, out, consts)

}  // namespace mk
