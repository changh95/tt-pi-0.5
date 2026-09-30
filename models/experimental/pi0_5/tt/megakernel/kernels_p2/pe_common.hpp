// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefix engine: the op sequence and every per-op geometry quantity, as pure integer code compiled into all five RISC
// binaries (so the three kernels can never disagree on a shape, a split or an arena offset). Mirrored by the host in
// pe_geometry.py (tests/megakernel/test_cpu_pe.py compares the two on every op).
#pragma once
#include <cstdint>
#include "pe_defs.hpp"
#include "internal/circular_buffer_interface.h"

#ifndef NOINL
#define NOINL __attribute__((noinline, noclone))
#endif
// control code (op bookkeeping, addresses, loops over tiles): size over speed; hot inner loops keep the default
#define PE_OS __attribute__((noinline, noclone, optimize("Os")))

#ifndef PE_CT0
#error "PE_CT0 (index of the prefix compile-time args) must be defined by the host"
#endif

namespace pe {

constexpr uint32_t MT = get_compile_time_arg_val(PE_CT0 + PT_MT);
constexpr uint32_t PTV = get_compile_time_arg_val(PE_CT0 + PT_PT);
constexpr uint32_t RV = get_compile_time_arg_val(PE_CT0 + PT_RV);
constexpr uint32_t LT = get_compile_time_arg_val(PE_CT0 + PT_LT);
constexpr uint32_t NTOK = get_compile_time_arg_val(PE_CT0 + PT_NTOK);
constexpr uint32_t EPS_V = get_compile_time_arg_val(PE_CT0 + PT_EPS_V);
constexpr uint32_t EPS_S = get_compile_time_arg_val(PE_CT0 + PT_EPS_S);
constexpr uint32_t EMB_SCALE = get_compile_time_arg_val(PE_CT0 + PT_EMB_SCALE);
constexpr uint32_t ARENA_BYTES = get_compile_time_arg_val(PE_CT0 + PT_ARENA);
static_assert(MT % RV == 0, "VLM bands must split the row tiles evenly");
static_assert(MT == PTV + 1, "exactly one pad row tile (base 736 -> 768, LIBERO 544 -> 576)");

constexpr uint32_t T16 = 2048, T8 = 1088, T32 = 4096;
#ifdef PE_VLM_LOFI
constexpr uint32_t VLM_FID_HI = 0;  // A/B arm: VLM matmuls at LoFi (the shipped path's fidelity): fails the amended
#else                               // per-seed gate on 4 / 22 seeds (p2/results/seeds_lofi.json)
constexpr uint32_t VLM_FID_HI = 1;  // default: HiFi2 (22 / 22 seeds closer to the fp32 reference than shipped, seeds_hifi2.json)
#endif

// Re-point a CB at [addr, addr + pages * page_bytes) of the arena (byte address; every RISC keeps its own interface copy,
// DM RISCs in bytes, TRISCs in 16 B units: cb_addr_shift). Only while the CB is drained from this RISC's view (every op
// starts with its CBs drained: the global barrier orders it). The tile counters are left alone.
#if defined(TRISC_MATH)
// the MATH thread has no CB interface (it never touches L1 through a CB)
FORCE_INLINE void cb_point(uint32_t, uint32_t, uint32_t, uint32_t) {}
FORCE_INLINE uint32_t cbase(uint32_t) { return 0; }
FORCE_INLINE uint32_t cbytes(uint32_t) { return 0; }
#else
FORCE_INLINE void cb_point(uint32_t cb, uint32_t addr, uint32_t pages, uint32_t page_bytes) {
    LocalCBInterface& i = get_local_cb_interface(cb);
    const uint32_t a = addr >> cb_addr_shift;
    const uint32_t sz = (pages * page_bytes) >> cb_addr_shift;
    i.fifo_size = sz;
    i.fifo_limit = a + sz;
    i.fifo_rd_ptr = a;
    i.fifo_wr_ptr = a;
    i.fifo_num_pages = pages;
    i.fifo_page_size = page_bytes >> cb_addr_shift;
    i.fifo_wr_tile_ptr = 0;
}
FORCE_INLINE uint32_t cbase(uint32_t cb) {
    const auto& iface = get_local_cb_interface(cb);
    return (iface.fifo_limit - iface.fifo_size) << cb_addr_shift;
}
FORCE_INLINE uint32_t cbytes(uint32_t cb) { return get_local_cb_interface(cb).fifo_size << cb_addr_shift; }
#endif
// The arena: phase-1 CB 0 .. the end of P_TAIL (host descriptor order: P_SYNC, the 32.. placeholders, phase-1 CB_SYNC,
// phase-1 CBs 0..30, P_TAIL -> one contiguous range, checked in-kernel against PT_ARENA)
FORCE_INLINE uint32_t arena_lo() { return cbase(0); }
FORCE_INLINE uint32_t arena_hi() { return cbase(P_TAIL) + cbytes(P_TAIL); }

// per SigLIP layer row-broadcast vector tiles (PA_SVEC tensor: layer * SV_N + offset + n)
constexpr uint32_t SV_LN1W = 0, SV_LN1B = 36, SV_BQKV = 72, SV_BO = 216, SV_LN2W = 252, SV_LN2B = 288, SV_BFC1 = 324,
                   SV_BFC2 = 460, SV_N = 496;
constexpr uint32_t VV_G1 = 0, VV_G2 = 64, VV_N = 128;
constexpr uint32_t GV_PLNW = 0, GV_PLNB = 36, GV_BPROJ = 72, GV_N = 136;

// attention chunking
constexpr uint32_t S_NK = 8, S_CH = 8;  // SigLIP: all 8 key tiles in one chunk (no mask, DST holds 8)
constexpr uint32_t V_CH = 6;
constexpr uint32_t V_NP = (PTV + V_CH - 1) / V_CH;
static_assert(V_NP <= 6, "merge keeps the NP weights in DST 1..NP");

// ---------------------------------------------------------------- the op description
struct Op {
    uint32_t op, kind, what, layer;
    // matmul
    uint32_t mode, mt, nb, rpb, kt, np, piece, wbf16, epi, fid_hi;
    // norm
    uint32_t nkind, nk;
    // shared tile counts
    uint32_t items;
};

FORCE_INLINE uint32_t split_n(uint32_t total, uint32_t c) { return total / NCOL + (c < total % NCOL ? 1 : 0); }
FORCE_INLINE uint32_t split_0(uint32_t total, uint32_t c) {
    return c * (total / NCOL) + (c < total % NCOL ? c : total % NCOL);
}

#if !defined(COMPILE_FOR_TRISC)  // the geometry code runs on the DM RISCs only (the TRISCs read P_OPD)
PE_OS Op describe(uint32_t op) {
    Op o{};
    o.op = op;
    uint32_t what = 0, layer = 0;
    if (op == 0) {
        what = W_PATCH;
    } else if (op < OP_POSTLN) {
        layer = (op - OP_S0) / OPS_PER_LAYER;
        what = W_SLN1 + (op - OP_S0) % OPS_PER_LAYER;
    } else if (op == OP_POSTLN) {
        what = W_POSTLN;
    } else if (op == OP_PROJ) {
        what = W_PROJ;
    } else if (op == OP_EMBED) {
        what = W_EMBED;
    } else {
        layer = (op - OP_V0) / OPS_PER_LAYER;
        what = W_VRMS1 + (op - OP_V0) % OPS_PER_LAYER;
    }
    o.what = what;
    o.layer = layer;
    auto mm = [&](uint32_t mode, uint32_t mt, uint32_t nb, uint32_t kt, uint32_t np, uint32_t piece, uint32_t epi,
                  uint32_t wbf16, uint32_t fid_hi) {
        o.kind = K_MM;
        o.mode = mode;
        o.mt = mt;
        o.nb = nb;
        o.rpb = mt / nb;
        o.kt = kt;
        o.np = np;
        o.piece = piece;
        o.epi = epi;
        o.wbf16 = wbf16;
        o.fid_hi = fid_hi;
    };
    switch (what) {
        case W_PATCH: mm(MM_R, S_M, S_R, S_KP, S_D / 2, S_KP, E_POS, 1, 1); break;
        case W_SQKV: mm(MM_R, S_M, S_R, S_D, S_NQKV / 2, S_PIECE, E_BIAS, 0, 1); break;
        case W_SO: mm(MM_R, S_M, S_R, S_DCTX, S_D / 2, S_PIECE_O, E_RES_BIAS, 0, 1); break;
        case W_SFC1: mm(MM_R, S_M, S_R, S_D, S_I / 2, S_PIECE, E_BIAS_GELU, 0, 1); break;
        case W_SFC2: mm(MM_R, S_M, S_R, S_I, S_D / 2, S_PIECE_2, E_RES_BIAS, 0, 1); break;
        case W_PROJ: mm(MM_R, S_M, S_R, S_D, V_D / 2, S_PIECE, E_BIAS, 0, 1); break;
        case W_VQKV: mm(MM_R, MT, RV, V_D, V_NQKV / 2, V_PIECE, E_ROPE, 0, VLM_FID_HI); break;
        case W_VO: mm(MM_R, MT, RV, V_D, V_D / 2, V_PIECE, E_RES, 0, VLM_FID_HI); break;
        case W_VGU: mm(MM_R, MT, RV, V_D, V_I, V_PIECE, E_GEGLU, 0, VLM_FID_HI); break;
        case W_VDOWN: mm(MM_S, MT, RV, V_I, V_D / 2, V_PIECE, E_RES, 0, VLM_FID_HI); break;
        case W_SLN1:
        case W_SLN2:
        case W_POSTLN:
            o.kind = K_NORM;
            o.nkind = N_LN;
            o.nk = S_D;
            o.items = S_M;
            break;
        case W_VRMS1:
        case W_VRMS2:
            o.kind = K_NORM;
            o.nkind = N_RMS;
            o.nk = V_D;
            o.items = MT;
            break;
        case W_SATTN:
            o.kind = K_ATTN;
            o.items = 2 * S_HEADS * 3;  // (image, head, q row group {0..2, 3..5, 6..7}): <= one item per core
            break;
        case W_VATTN:
            o.kind = K_ATTN;
            o.items = MT * (V_H / 2);  // (q row tile, head pair)
            break;
        case W_EMBED:
            o.kind = K_EMBED;
            o.items = (LT + (MT - PTV)) * 4;  // (language or pad row tile, quarter of 16 tiles)
            break;
        default: o.kind = K_NOP; break;
    }
#ifdef PE_DBG_ALL_NOP
    o.kind = K_NOP;  // timing arm: barrier + descriptor cost per op
#endif
    return o;
}

#endif

// SigLIP attention item -> (image, head, first q row tile, row count)
FORCE_INLINE void s_attn_item(uint32_t it, uint32_t& img, uint32_t& h, uint32_t& r0, uint32_t& nr) {
    img = it / (S_HEADS * 3);
    const uint32_t rem = it % (S_HEADS * 3), g = rem % 3;
    h = rem / 3;
    r0 = img * S_IMG + g * 3;
    nr = g < 2 ? 3 : 2;
}

// matmul: this core's band / column shares
FORCE_INLINE uint32_t mm_pairs(const Op& o, uint32_t x) { return split_n(o.np, x); }
FORCE_INLINE uint32_t mm_pair0(const Op& o, uint32_t x) { return split_0(o.np, x); }
FORCE_INLINE uint32_t mm_nq(const Op& o) { return o.kt / o.piece; }            // weight pages per pair (R) / K blocks (S)
FORCE_INLINE uint32_t mm_wtile(const Op& o) { return o.wbf16 ? T16 : T8; }
FORCE_INLINE uint32_t mm_page_tiles(const Op& o) { return 2 * o.piece; }
FORCE_INLINE uint32_t mm_page_bytes(const Op& o) { return mm_page_tiles(o) * mm_wtile(o); }
FORCE_INLINE uint32_t mm_col_pages(const Op& o, uint32_t x) { return mm_pairs(o, x) * mm_nq(o); }
FORCE_INLINE uint32_t mm_in0_tile(const Op& o) { return o.mode == MM_S ? T8 : T16; }
// in0 transfer unit: R = the whole band in chunks of IN0_CHUNK tiles, S = one K block per page
constexpr uint32_t IN0_CHUNK = 16;
FORCE_INLINE uint32_t mm_in0_pages(const Op& o) { return o.mode == MM_S ? mm_nq(o) : 1; }
FORCE_INLINE uint32_t mm_in0_page_tiles(const Op& o) { return o.mode == MM_S ? o.rpb * o.piece : o.rpb * o.kt; }

// weight arena pages of a layer arena, in op order; each op's pages are striped over the 8 DRAM banks, and each op
// starts at a bank offset that is a multiple of its page size in every bank
constexpr uint32_t N_BANKS = 8;
FORCE_INLINE uint32_t ceil_div(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
FORCE_INLINE uint32_t op_pages_total(const Op& o) { return o.np * mm_nq(o); }
FORCE_INLINE uint32_t op_bank_bytes(const Op& o) { return ceil_div(op_pages_total(o), N_BANKS) * mm_page_bytes(o); }
// byte offset (same in every bank) of this op's region inside its layer arena
#if !defined(COMPILE_FOR_TRISC)
PE_OS uint32_t mm_arena_off(const Op& o) {
    if (o.what == W_PATCH || o.what == W_PROJ) {
        return 0;
    }
    uint32_t first = (o.what >= W_VRMS1) ? W_VQKV : W_SQKV;
    uint32_t off = 0;
    for (uint32_t w = first; w < o.what; ++w) {
        if (w == W_SATTN || w == W_SLN2 || w == W_VATTN || w == W_VRMS2) {
            continue;
        }
        Op p = describe(o.op - (o.what - w));
        off += op_bank_bytes(p);
    }
    return off;
}

#endif

// output N tiles of pair p (t = 0 / 1) = the weight column the pair's page holds
FORCE_INLINE uint32_t pair_col(const Op& o, uint32_t p, uint32_t t) {
    if (o.what == W_VQKV) {
        if (p < 32) {
            return (p / 4) * 8 + (p % 4) + 4 * t;
        }
        const uint32_t j = (p - 32) % 4;
        return (p < 36 ? 64 : 72) + j + 4 * t;
    }
    if (o.what == W_VGU) {
        return p + V_I * t;  // [up | gate] columns of the fused weight; the output h tile is p
    }
    return 2 * p + t;
}

// side tiles one output pair consumes (the NCRISC pushes exactly these, in this order)
template <typename O>
FORCE_INLINE uint32_t side16(const O& o, uint32_t p) {
    switch (o.epi) {
        case E_POS: return 2 * o.rpb;
        case E_BIAS:
        case E_BIAS_GELU:
        case E_RES_BIAS: return 2;
        case E_ROPE: return p < 36 ? 4 * o.rpb : 0;
        default: return 0;
    }
}
template <typename O>
FORCE_INLINE uint32_t side32(const O& o) { return (o.epi == E_RES || o.epi == E_RES_BIAS) ? 2 * o.rpb : 0; }
template <typename O>
FORCE_INLINE uint32_t out_cb(const O& o, uint32_t p) {
    switch (o.what) {
        case W_SQKV:
        case W_SFC1: return P_O16;
        case W_VQKV: return p < 32 ? P_O16 : P_O8;
        case W_VGU: return P_O8;
        default: return P_O32;
    }
}
template <typename O>
FORCE_INLINE uint32_t out_tiles(const O& o) { return o.epi == E_GEGLU ? o.rpb : 2 * o.rpb; }

// ---------------------------------------------------------------- L1 arena layouts (byte offsets from the arena base)
struct Lay {
    uint32_t in0, in0_bytes, w, w_slots, s16, s32, o16, o8, o32, part, x32, scr, r, q, kv, msk, ss, m, mf, l, op_, pm, pl,
        d, cst, rm, tok, stage, stage_bytes, end;
};
constexpr uint32_t W_BUDGET = 8 * 34816;
#ifdef PE_DBG_IN0_SLOTS
constexpr uint32_t IN0_SLOTS = PE_DBG_IN0_SLOTS;
#else
constexpr uint32_t IN0_SLOTS = 4;
#endif

#if !defined(COMPILE_FOR_TRISC)
PE_OS Lay layout(const Op& o) {
    Lay y{};
    uint32_t a = 0;
    auto take = [&](uint32_t bytes) {
        const uint32_t r = a;
        a += (bytes + 63) & ~63u;
        return r;
    };
    if (o.kind == K_MM) {
        const uint32_t rp = o.rpb;
        y.in0_bytes = o.mode == MM_S ? IN0_SLOTS * mm_in0_page_tiles(o) * T8 : rp * o.kt * T16;
        y.in0 = take(y.in0_bytes);
        y.w_slots = W_BUDGET / mm_page_bytes(o);
        if (y.w_slots > 8) {
            y.w_slots = 8;
        }
        y.w = take(y.w_slots * mm_page_bytes(o));
        y.s16 = take(24 * T16);
        y.s32 = take(4 * rp * T32);
        y.o16 = take(4 * rp * T16);
        y.o8 = take(4 * rp * T8);
        y.o32 = take(4 * rp * T32);
        y.part = o.mode == MM_S ? take(3 * 2 * rp * T32) : 0;
        // feeders stage in their own arena (they compute nothing in a matmul op)
        y.stage_bytes = mm_page_bytes(o) > rp * o.piece * T16 ? mm_page_bytes(o) : rp * o.piece * T16;
        if (o.mode == MM_S && mm_in0_page_tiles(o) * T8 > y.stage_bytes) {
            y.stage_bytes = mm_in0_page_tiles(o) * T8;
        }
        y.stage = 0;
    } else if (o.kind == K_NORM) {
        y.x32 = take(o.nk * T32);
        y.s16 = take(2 * o.nk * T16);
        y.o16 = take(8 * T16);
        y.scr = take(2 * T32);
        y.r = take(2 * T32);
        y.cst = take(PC_N * T16);
    } else if (o.kind == K_ATTN) {
        const bool v = o.what == W_VATTN;
        y.q = take((v ? 2 * V_DH : 3 * S_DH) * T16);
        y.kv = v ? take(2 * PTV * V_DH * T8) : take(2 * S_NK * S_DH * T16);
        y.msk = take((v ? PTV : 1) * T16);
        y.ss = take(8 * T16);
        y.m = take(T16);
        y.mf = take(T16);
        y.l = take(T32);
        const uint32_t np = v ? V_NP : (S_NK + S_CH - 1) / S_CH;
        const uint32_t dh = v ? V_DH : S_DH;
        y.op_ = take(2 * np * dh * T16);
        y.pm = take(2 * np * T16);
        y.pl = take(2 * np * T32);
        y.d = take(np * T32);
        y.o16 = take(2 * dh * T16);
        y.cst = take(PC_N * T16);
    } else if (o.kind == K_EMBED) {
        y.tok = take(NTOK * 4 > 64 ? NTOK * 4 : 64);
        y.rm = take(16 * T16);
        y.s16 = take(16 * T16);
        y.o32 = take(16 * T32);
        y.cst = take(PC_N * T16);
    }
    y.end = a;
    return y;
}

#endif

}  // namespace pe
