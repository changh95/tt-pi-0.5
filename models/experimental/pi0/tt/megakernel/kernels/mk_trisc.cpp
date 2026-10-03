// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pi0.5 expert megakernel, TRISC: every piece of arithmetic of the 10 x 18 expert loop plus the action in / out
// projections and the Euler steps.
//
// Per generation (step s, layer l) a compute core runs, in this order, the ops its role bits select:
//   pair   : qkv (bfp8 LoFi, 2 columns x 32 K, both rows) + r / c epilogue + RoPE -> CB_QKVO
//   owner  : (l == 0) copy its x column of the new step's input into the fp32 residual
//   unit   : one key chunk of flash attention -> (O part, m, l)
//   merger : merge the NCH parts with diag(w_i / L) matmuls -> ctx[r, head]
//   owner  : o_proj column (bf16 HiFi2) + x_mid = x + g_attn * acc (fp32 residual)
//   MLP    : up|gate (bfp8 LoFi) + r / c epilogue + GeGLU (tanh) -> h; down partial (bf16 HiFi2)
//   owner  : sum of the 8 partials (IDENT matmuls, HiFi4) + x = x_mid + g_mlp * sum
// H0: the in-projection at every step start, r = rsqrt(mean(x^2) + eps) for the x and x_mid rounds, the final adaRMS +
// out-projection + Euler (fp32 state) at every step end.
// DST: fp32_dest_acc_en + dst_full_sync_en = 8 fp32 tiles (F0 device fact 8). Every multiply-called helper is
// noinline, noclone (IPA-CP clones otherwise: trisc-ipa-constprop-clones).
#include <cstdint>
#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "mk_defs.hpp"
#ifdef MK_TRACE
#include "api/debug/dprint.h"
#define TR(tag, v)                                      \
    do {                                                \
        DPRINT_UNPACK("U " tag " {}\n", (uint32_t)(v)); \
        DPRINT_MATH("M " tag " {}\n", (uint32_t)(v));   \
        DPRINT_PACK("P " tag " {}\n", (uint32_t)(v));   \
    } while (0)
#else
#define TR(tag, v) \
    do {           \
    } while (0)
#endif

#define NOINL __attribute__((noinline, noclone))

using namespace mk;

namespace {

constexpr uint32_t RT = get_compile_time_arg_val(CT_RT);
constexpr uint32_t PT = get_compile_time_arg_val(CT_PT);
constexpr uint32_t CHT = get_compile_time_arg_val(CT_CHT);
constexpr uint32_t NCH = get_compile_time_arg_val(CT_NCH);
constexpr bool ROWLOOP = NCH * RT > GRID_ROWS;  // unit = (head, chunk) running both query rows (mk_brisc.cpp)
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(CT_EPS_BITS);
constexpr uint32_t IN0_PAGES = 64 * RT;
constexpr uint32_t X_TILES = 32 * RT;

#if defined(MK_FID)
// MK_FID (2 / 3 / 4): the fidelity of every HiFi2 matmul (bfp8 qkv / up|gate, q.K, P.V, o_proj, down). HiFi2 drops
// the in0 operand's last mantissa bit; HiFi3 / HiFi4 keep it. The expert program sets MK_FID = 3 (program.py); the
// prefix-engine programs, which compile this code without running it, leave it unset.
constexpr MathFidelity LOFI = static_cast<MathFidelity>(MK_FID);
constexpr MathFidelity HIFI2 = static_cast<MathFidelity>(MK_FID);
#else
#ifdef MK_FID8_HIFI2
constexpr MathFidelity LOFI = MathFidelity::HiFi2;  // the bfp8 matmuls at HiFi2 (the host sets MK_FID8_HIFI2)
#else
constexpr MathFidelity LOFI = MathFidelity::LoFi;
#endif
#ifdef MK_FID2  // the fidelity of q.K, P.V, o_proj and down (HiFi2 default)
constexpr MathFidelity HIFI2 = static_cast<MathFidelity>(MK_FID2);
#else
constexpr MathFidelity HIFI2 = MathFidelity::HiFi2;
#endif
#endif
constexpr MathFidelity HIFI4 = MathFidelity::HiFi4;

// ---------------------------------------------------------------- LLK wrappers with an explicit fidelity
template <MathFidelity FID>
ALWI void mm_init_f(uint32_t in0, uint32_t in1, uint32_t transpose = 0) {
    reconfig_data_format(in1, in0);  // matmul: SrcA <- in1, SrcB <- in0 (tensix-operand-format-rules rule 2)
    MATH((llk_math_matmul_init<FID, MM_THROTTLE>(in0, in1, transpose)));
    UNPACK((llk_unpack_AB_matmul_init(in0, in1, transpose)));
}
template <MathFidelity FID>
ALWI void mm_f(uint32_t in0, uint32_t in1, uint32_t t0, uint32_t t1, uint32_t dst) {
    UNPACK((llk_unpack_AB_matmul(in0, in1, t0, t1)));
    MATH((llk_math_matmul<FID, MM_THROTTLE>(dst)));
}
// DST[dst] += A * B elementwise (fp32 DEST not cleared: accumulates; row_layernorm.hpp ln_llk::mul_tiles_acc)
ALWI void mul_acc_init(uint32_t icb0, uint32_t icb1) {
    reconfig_data_format(icb0, icb1);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, HIFI4>(icb0, icb1, 0)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1)));
}
ALWI void mul_acc(uint32_t icb0, uint32_t icb1, uint32_t t0, uint32_t t1, uint32_t dst) {
    UNPACK((llk_unpack_AB(icb0, icb1, t0, t1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          HIFI4,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, dst, false)));
}

// ---------------------------------------------------------------- small DST helpers
NOINL void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
NOINL void fmul(uint32_t a, uint32_t b, uint32_t o) {
    mul_binary_tile_init();
    mul_binary_tile(a, b, o);
}
NOINL void fadd(uint32_t a, uint32_t b, uint32_t o) {
    add_binary_tile_init();
    add_binary_tile(a, b, o);
}
NOINL void fsub(uint32_t a, uint32_t b, uint32_t o) {
    sub_binary_tile_init();
    sub_binary_tile(a, b, o);
}
NOINL void fmax(uint32_t a, uint32_t b, uint32_t o) {
    binary_max_tile_init();
    binary_max_tile(a, b, o);
}
NOINL void fexp(uint32_t a) {
    exp_tile_init<false>();
    exp_tile<false>(a);
}
NOINL void frecip(uint32_t a) {
    recip_tile_init();
    recip_tile(a);
}
NOINL void fscale(uint32_t a, uint32_t bits) {
    binop_with_scalar_tile_init();
    mul_unary_tile(a, bits);
}
// a = a * b + c   (the r / c epilogue: b = the full r tile, c = the row-broadcast bias tile)
NOINL void fma3(uint32_t a, uint32_t b, uint32_t c) {
    fmul(a, b, a);
    fadd(a, c, a);
}
NOINL void pack_to(uint32_t dst, uint32_t cb, uint32_t idx) {
    pack_reconfig_data_format(cb);
    pack_tile<true>(dst, cb, idx);
}

// ---------------------------------------------------------------- role state
struct Me {
    uint32_t bits, kind, head, ur, kc, npre, kt0, n;
    bool has(uint32_t b) const { return (bits & b) != 0; }
};
Me me;

// ================================================================ pair: qkv + epilogue + RoPE
NOINL void pair_qkv() {
    TR("pair_qkv", 0);
    cb_wait_front(CB_IN0, IN0_PAGES);
    cb_wait_front(CB_W8, 8 * PAGE_TILES);
    cb_reserve_back(CB_QKVO, 2 * RT);
    for (uint32_t r = 0; r < RT; ++r) {
        tile_regs_acquire();
        mm_init_f<LOFI>(CB_IN0, CB_W8);
        for (uint32_t kb = 0; kb < 4; ++kb) {
            for (uint32_t k = 0; k < 8; ++k) {
                const uint32_t a = r * 32 + kb * 8 + k;
                mm_f<LOFI>(CB_IN0, CB_W8, a, (2 * kb) * 8 + k, 0);
                mm_f<LOFI>(CB_IN0, CB_W8, a, (2 * kb + 1) * 8 + k, 1);
            }
        }
        if (r == 0) {
            cb_wait_front(CB_RTOK, 1);
        }
        load(CB_IN0, X_TILES + r, 2);
        load(CB_WC, WC_CQKV + 0, 3);
        load(CB_WC, WC_CQKV + 1, 4);
        fma3(0, 2, 3);
        fma3(1, 2, 4);
        uint32_t o0 = 0, o1 = 1;
        if (me.kind != PK_V) {
            load(CB_TAB, 4 * r + 0, 2);  // cos j
            load(CB_TAB, 4 * r + 2, 3);  // sin j (signed)
            load(CB_TAB, 4 * r + 1, 4);  // cos j+4
            load(CB_TAB, 4 * r + 3, 5);  // sin j+4 (signed)
            fmul(0, 2, 6);
            fmul(1, 3, 7);
            fadd(6, 7, 6);  // out_j = x_j cos_j + x_{j+4} sin_j
            fmul(1, 4, 7);
            fmul(0, 5, 2);
            fadd(7, 2, 7);  // out_{j+4} = x_{j+4} cos_{j+4} + x_j sin_{j+4}
            o0 = 6;
            o1 = 7;
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(o0, CB_QKVO, 2 * r);
        pack_to(o1, CB_QKVO, 2 * r + 1);
        tile_regs_release();
    }
    cb_push_back(CB_QKVO, 2 * RT);
    cb_pop_front(CB_W8, 8 * PAGE_TILES);
    cb_pop_front(CB_IN0, IN0_PAGES);
    cb_pop_front(CB_RTOK, 1);
}

// ================================================================ owner: step input -> fp32 residual
NOINL void owner_load_x(bool has_res) {
    TR("owner_load_x", 0);
    cb_wait_front(CB_IN0, IN0_PAGES);
    tile_regs_acquire();
    for (uint32_t r = 0; r < RT; ++r) {
        load(CB_IN0, r * 32 + me.n, r);
    }
    tile_regs_commit();
    if (has_res) {
        cb_pop_front(CB_XRES, RT);
    }
    cb_reserve_back(CB_XRES, RT);
    tile_regs_wait();
    for (uint32_t r = 0; r < RT; ++r) {
        pack_to(r, CB_XRES, r);
    }
    tile_regs_release();
    cb_push_back(CB_XRES, RT);
    cb_pop_front(CB_IN0, IN0_PAGES);
}

// ================================================================ unit: one key chunk
NOINL void scores_into(uint32_t i, uint32_t t) {
    if (t < PT) {
        mm_init_f<HIFI2>(CB_Q, CB_KV, 1);
        for (uint32_t d = 0; d < DH_T; ++d) {
            mm_f<HIFI2>(CB_Q, CB_KV, me.ur * DH_T + d, i * DH_T + d, i);
        }
    } else {
        mm_init_f<HIFI2>(CB_Q, CB_KSV, 1);
        for (uint32_t d = 0; d < DH_T; ++d) {
            mm_f<HIFI2>(CB_Q, CB_KSV, me.ur * DH_T + d, (t - PT) * DH_T + d, i);
        }
    }
}

NOINL void unit_attention() {
    TR("unit_attention", 0);
    cb_wait_front(CB_Q, DH_T * RT);
    if (me.npre) {
        cb_wait_front(CB_KV, 2 * CHT * DH_T);
    }
    if (me.has(R_LASTCH)) {
        cb_wait_front(CB_KSV, 2 * DH_T * RT);
    }
    // S = q K^T + mask
    cb_reserve_back(CB_S, CHT);
    tile_regs_acquire();
    for (uint32_t i = 0; i < CHT; ++i) {
        scores_into(i, me.kt0 + i);
    }
    for (uint32_t i = 0; i < CHT; ++i) {
        load(CB_MASK, i, 7);
        fadd(i, 7, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < CHT; ++i) {
        pack_to(i, CB_S, i);
    }
    tile_regs_release();
    cb_push_back(CB_S, CHT);
    cb_wait_front(CB_S, CHT);
    // m = rowmax(S)  (column vector)
    reconfig_data_format(CB_S, CB_CONST);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_CONST, CB_M);
    cb_reserve_back(CB_M, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < CHT; ++i) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_CONST, i, CONST_ONES, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_M, 0);
    tile_regs_release();
    reduce_uninit(CB_S);
    cb_push_back(CB_M, 1);
    cb_wait_front(CB_M, 1);
    // MF = m broadcast over columns (full tile, for the merge)
    reconfig_data_format(CB_CONST, CB_M);
    mul_bcast_cols_init(CB_CONST, CB_M);
    cb_reserve_back(CB_MF, 1);
    tile_regs_acquire();
    mul_tiles_bcast_cols(CB_CONST, CB_M, CONST_ONES, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_MF, 0);
    tile_regs_release();
    cb_push_back(CB_MF, 1);
    // P = exp(S - m), in place
    reconfig_data_format(CB_S, CB_M);
    sub_bcast_cols_init(CB_S, CB_M);
    exp_tile_init<false>();
    tile_regs_acquire();
    for (uint32_t i = 0; i < CHT; ++i) {
        sub_tiles_bcast_cols(CB_S, CB_M, i, 0, i);
        exp_tile<false>(i);
    }
    tile_regs_commit();
    cb_pop_front(CB_S, CHT);
    cb_reserve_back(CB_S, CHT);
    tile_regs_wait();
    for (uint32_t i = 0; i < CHT; ++i) {
        pack_to(i, CB_S, i);
    }
    tile_regs_release();
    cb_push_back(CB_S, CHT);
    cb_wait_front(CB_S, CHT);
    // l = sum_i P_i @ ONES (row sums in every column, fp32)
    cb_reserve_back(CB_L, 1);
    tile_regs_acquire();
    mm_init_f<HIFI4>(CB_S, CB_CONST);
    for (uint32_t i = 0; i < CHT; ++i) {
        mm_f<HIFI4>(CB_S, CB_CONST, i, CONST_ONES, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_L, 0);
    tile_regs_release();
    cb_push_back(CB_L, 1);
    // O = sum_i P_i @ V_i (unnormalised)
    cb_reserve_back(CB_OP, DH_T);
    tile_regs_acquire();
    for (uint32_t i = 0; i < CHT; ++i) {
        const uint32_t t = me.kt0 + i;
        if (t < PT) {
            mm_init_f<HIFI2>(CB_S, CB_KV);
            for (uint32_t d = 0; d < DH_T; ++d) {
                mm_f<HIFI2>(CB_S, CB_KV, i, (CHT + i) * DH_T + d, d);
            }
        } else {
            mm_init_f<HIFI2>(CB_S, CB_KSV);
            for (uint32_t d = 0; d < DH_T; ++d) {
                mm_f<HIFI2>(CB_S, CB_KSV, i, DH_T * RT + (t - PT) * DH_T + d, d);
            }
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < DH_T; ++d) {
        pack_to(d, CB_OP, d);
    }
    tile_regs_release();
    cb_push_back(CB_OP, DH_T);
    cb_pop_front(CB_S, CHT);
    cb_pop_front(CB_M, 1);
    cb_pop_front(CB_Q, DH_T * RT);
    if (me.npre) {
        cb_pop_front(CB_KV, 2 * CHT * DH_T);
    }
    if (me.has(R_LASTCH)) {
        cb_pop_front(CB_KSV, 2 * DH_T * RT);
    }
}

// ================================================================ merger: NCH parts -> ctx[r, head]
NOINL void merge() {
    TR("merge", 0);
    constexpr uint32_t NP = NCH;
    cb_wait_front(CB_PART, DH_T * (NCH - 1));
    cb_wait_front(CB_PM, NCH - 1);
    cb_wait_front(CB_PL, NCH - 1);
    cb_wait_front(CB_OP, DH_T);
    cb_wait_front(CB_MF, 1);
    cb_wait_front(CB_L, 1);
    // weights: M = max_i m_i; w_i = exp(m_i - M); L = sum w_i l_i; D_i = diag(w_i / L)
    cb_reserve_back(CB_D, NP);
    tile_regs_acquire();
    load(CB_MF, 0, 0);
    for (uint32_t i = 1; i < NP; ++i) {
        load(CB_PM, i - 1, 1);
        fmax(0, 1, 0);
    }
    for (uint32_t i = 0; i < NP; ++i) {
        if (i == 0) {
            load(CB_MF, 0, 1);
        } else {
            load(CB_PM, i - 1, 1 + i);
        }
        fsub(1 + i, 0, 1 + i);
        fexp(1 + i);
    }
    load(CB_L, 0, 0);
    fmul(0, 1, 0);
    for (uint32_t i = 1; i < NP; ++i) {
        load(CB_PL, i - 1, 7);
        fmul(7, 1 + i, 7);
        fadd(0, 7, 0);
    }
    frecip(0);
    for (uint32_t i = 0; i < NP; ++i) {
        fmul(1 + i, 0, 1 + i);
    }
    load(CB_CONST, CONST_IDENT, 0);
    for (uint32_t i = 0; i < NP; ++i) {
        fmul(1 + i, 0, 1 + i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < NP; ++i) {
        pack_to(1 + i, CB_D, i);
    }
    tile_regs_release();
    cb_push_back(CB_D, NP);
    cb_wait_front(CB_D, NP);
    // ctx[d] = sum_i D_i @ O_i[d]
    cb_reserve_back(CB_CTXO, DH_T);
    tile_regs_acquire();
    mm_init_f<HIFI4>(CB_D, CB_OP);
    for (uint32_t d = 0; d < DH_T; ++d) {
        mm_f<HIFI4>(CB_D, CB_OP, 0, d, d);
    }
    mm_init_f<HIFI4>(CB_D, CB_PART);
    for (uint32_t i = 1; i < NP; ++i) {
        for (uint32_t d = 0; d < DH_T; ++d) {
            mm_f<HIFI4>(CB_D, CB_PART, i, (i - 1) * DH_T + d, d);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < DH_T; ++d) {
        pack_to(d, CB_CTXO, d);
    }
    tile_regs_release();
    cb_push_back(CB_CTXO, DH_T);
    cb_pop_front(CB_D, NP);
    cb_pop_front(CB_PART, DH_T * (NCH - 1));
    cb_pop_front(CB_PM, NCH - 1);
    cb_pop_front(CB_PL, NCH - 1);
    cb_pop_front(CB_OP, DH_T);
    cb_pop_front(CB_MF, 1);
    cb_pop_front(CB_L, 1);
}

// ================================================================ row loop (NCH x RT > GRID_ROWS): unit = (head, chunk)
// Every unit runs both query rows over its chunk (Q, K / V, K_s / V_s land once per generation); the merger (kc = 0)
// merges row r right after its own row-r part (its BRISC delivers every row's remote parts at once, row-major).
NOINL void unit_row_rl() {
    TR("unit_row_rl", me.ur);
    // S = q K^T + mask
    cb_reserve_back(CB_S, CHT);
    tile_regs_acquire();
    for (uint32_t i = 0; i < CHT; ++i) {
        scores_into(i, me.kt0 + i);
    }
    for (uint32_t i = 0; i < CHT; ++i) {
        load(CB_MASK, i, 7);
        fadd(i, 7, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < CHT; ++i) {
        pack_to(i, CB_S, i);
    }
    tile_regs_release();
    cb_push_back(CB_S, CHT);
    cb_wait_front(CB_S, CHT);
    // m = rowmax(S)  (column vector)
    reconfig_data_format(CB_S, CB_CONST);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_CONST, CB_M);
    cb_reserve_back(CB_M, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < CHT; ++i) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(CB_S, CB_CONST, i, CONST_ONES, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_M, 0);
    tile_regs_release();
    reduce_uninit(CB_S);
    cb_push_back(CB_M, 1);
    cb_wait_front(CB_M, 1);
    // MF = m broadcast over columns (full tile, for the merge)
    reconfig_data_format(CB_CONST, CB_M);
    mul_bcast_cols_init(CB_CONST, CB_M);
    cb_reserve_back(CB_MF, 1);
    tile_regs_acquire();
    mul_tiles_bcast_cols(CB_CONST, CB_M, CONST_ONES, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_MF, 0);
    tile_regs_release();
    cb_push_back(CB_MF, 1);
    // P = exp(S - m), in place
    reconfig_data_format(CB_S, CB_M);
    sub_bcast_cols_init(CB_S, CB_M);
    exp_tile_init<false>();
    tile_regs_acquire();
    for (uint32_t i = 0; i < CHT; ++i) {
        sub_tiles_bcast_cols(CB_S, CB_M, i, 0, i);
        exp_tile<false>(i);
    }
    tile_regs_commit();
    cb_pop_front(CB_S, CHT);
    cb_reserve_back(CB_S, CHT);
    tile_regs_wait();
    for (uint32_t i = 0; i < CHT; ++i) {
        pack_to(i, CB_S, i);
    }
    tile_regs_release();
    cb_push_back(CB_S, CHT);
    cb_wait_front(CB_S, CHT);
    // l = sum_i P_i @ ONES (row sums in every column, fp32)
    cb_reserve_back(CB_L, 1);
    tile_regs_acquire();
    mm_init_f<HIFI4>(CB_S, CB_CONST);
    for (uint32_t i = 0; i < CHT; ++i) {
        mm_f<HIFI4>(CB_S, CB_CONST, i, CONST_ONES, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, CB_L, 0);
    tile_regs_release();
    cb_push_back(CB_L, 1);
    // O = sum_i P_i @ V_i (unnormalised)
    cb_reserve_back(CB_OP, DH_T);
    tile_regs_acquire();
    for (uint32_t i = 0; i < CHT; ++i) {
        const uint32_t t = me.kt0 + i;
        if (t < PT) {
            mm_init_f<HIFI2>(CB_S, CB_KV);
            for (uint32_t d = 0; d < DH_T; ++d) {
                mm_f<HIFI2>(CB_S, CB_KV, i, (CHT + i) * DH_T + d, d);
            }
        } else {
            mm_init_f<HIFI2>(CB_S, CB_KSV);
            for (uint32_t d = 0; d < DH_T; ++d) {
                mm_f<HIFI2>(CB_S, CB_KSV, i, DH_T * RT + (t - PT) * DH_T + d, d);
            }
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < DH_T; ++d) {
        pack_to(d, CB_OP, d);
    }
    tile_regs_release();
    cb_push_back(CB_OP, DH_T);
    cb_pop_front(CB_S, CHT);
    cb_pop_front(CB_M, 1);
}

NOINL void merge_rl() {
    TR("merge_rl", me.ur);
    constexpr uint32_t NP = NCH;
    cb_wait_front(CB_PART, DH_T * (NCH - 1));
    cb_wait_front(CB_PM, NCH - 1);
    cb_wait_front(CB_PL, NCH - 1);
    cb_wait_front(CB_OP, DH_T);
    cb_wait_front(CB_MF, 1);
    cb_wait_front(CB_L, 1);
    // weights: M = max_i m_i; w_i = exp(m_i - M); L = sum w_i l_i; D_i = diag(w_i / L)
    cb_reserve_back(CB_D, NP);
    tile_regs_acquire();
    load(CB_MF, 0, 0);
    for (uint32_t i = 1; i < NP; ++i) {
        load(CB_PM, i - 1, 1);
        fmax(0, 1, 0);
    }
    for (uint32_t i = 0; i < NP; ++i) {
        if (i == 0) {
            load(CB_MF, 0, 1);
        } else {
            load(CB_PM, i - 1, 1 + i);
        }
        fsub(1 + i, 0, 1 + i);
        fexp(1 + i);
    }
    load(CB_L, 0, 0);
    fmul(0, 1, 0);
    for (uint32_t i = 1; i < NP; ++i) {
        load(CB_PL, i - 1, 7);
        fmul(7, 1 + i, 7);
        fadd(0, 7, 0);
    }
    frecip(0);
    for (uint32_t i = 0; i < NP; ++i) {
        fmul(1 + i, 0, 1 + i);
    }
    load(CB_CONST, CONST_IDENT, 0);
    for (uint32_t i = 0; i < NP; ++i) {
        fmul(1 + i, 0, 1 + i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < NP; ++i) {
        pack_to(1 + i, CB_D, i);
    }
    tile_regs_release();
    cb_push_back(CB_D, NP);
    cb_wait_front(CB_D, NP);
    // ctx[d] = sum_i D_i @ O_i[d]
    cb_reserve_back(CB_CTXO, DH_T);
    tile_regs_acquire();
    mm_init_f<HIFI4>(CB_D, CB_OP);
    for (uint32_t d = 0; d < DH_T; ++d) {
        mm_f<HIFI4>(CB_D, CB_OP, 0, d, d);
    }
    mm_init_f<HIFI4>(CB_D, CB_PART);
    for (uint32_t i = 1; i < NP; ++i) {
        for (uint32_t d = 0; d < DH_T; ++d) {
            mm_f<HIFI4>(CB_D, CB_PART, i, (i - 1) * DH_T + d, d);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < DH_T; ++d) {
        pack_to(d, CB_CTXO, d);
    }
    tile_regs_release();
    cb_push_back(CB_CTXO, DH_T);
    cb_pop_front(CB_D, NP);
    cb_pop_front(CB_PART, DH_T * (NCH - 1));
    cb_pop_front(CB_PM, NCH - 1);
    cb_pop_front(CB_PL, NCH - 1);
    cb_pop_front(CB_OP, DH_T);
    cb_pop_front(CB_MF, 1);
    cb_pop_front(CB_L, 1);
}

NOINL void unit_attention_rl() {
    cb_wait_front(CB_Q, DH_T * RT);
    if (me.npre) {
        cb_wait_front(CB_KV, 2 * CHT * DH_T);
    }
    if (me.has(R_LASTCH)) {
        cb_wait_front(CB_KSV, 2 * DH_T * RT);
    }
    for (uint32_t r = 0; r < RT; ++r) {
        me.ur = r;
        unit_row_rl();
        if (me.has(R_MERGER)) {
            merge_rl();
        }
    }
    cb_pop_front(CB_Q, DH_T * RT);
    if (me.npre) {
        cb_pop_front(CB_KV, 2 * CHT * DH_T);
    }
    if (me.has(R_LASTCH)) {
        cb_pop_front(CB_KSV, 2 * DH_T * RT);
    }
}

// ================================================================ owner residual epilogue
// DST[0..RT) hold the column's update (acc); x = x + gate * acc -> CB_XRES (fp32) and CB_XS (bf16)
NOINL void owner_residual(uint32_t gate_slot) {
    cb_wait_front(CB_XRES, RT);
    for (uint32_t r = 0; r < RT; ++r) {
        load(CB_XRES, r, RT + r);
    }
    load(CB_WC, gate_slot, 7);
    for (uint32_t r = 0; r < RT; ++r) {
        fmul(r, 7, r);
        fadd(r, RT + r, r);
    }
    tile_regs_commit();
    cb_pop_front(CB_XRES, RT);
    cb_reserve_back(CB_XRES, RT);
    cb_reserve_back(CB_XS, RT);
    tile_regs_wait();
    for (uint32_t r = 0; r < RT; ++r) {
        pack_to(r, CB_XRES, r);
    }
    for (uint32_t r = 0; r < RT; ++r) {
        pack_to(r, CB_XS, r);
    }
    tile_regs_release();
    cb_push_back(CB_XRES, RT);
    cb_push_back(CB_XS, RT);
}

NOINL void owner_oproj() {
    TR("owner_oproj", 0);
    cb_wait_front(CB_IN0, IN0_PAGES);
    tile_regs_acquire();
    mm_init_f<HIFI2>(CB_IN0, CB_W16);
    for (uint32_t kb = 0; kb < 8; ++kb) {
        cb_wait_front(CB_W16, PAGE_TILES);
        for (uint32_t r = 0; r < RT; ++r) {
            for (uint32_t k = 0; k < 8; ++k) {
                mm_f<HIFI2>(CB_IN0, CB_W16, r * 64 + kb * 8 + k, k, r);
            }
        }
        cb_pop_front(CB_W16, PAGE_TILES);
    }
    owner_residual(WC_GATTN);
    cb_pop_front(CB_IN0, IN0_PAGES);
}

NOINL void owner_reduce() {
    TR("owner_reduce", 0);
    cb_wait_front(CB_RED, 8 * RT);
    tile_regs_acquire();
    mm_init_f<HIFI4>(CB_CONST, CB_RED);
    for (uint32_t kg = 0; kg < 8; ++kg) {
        for (uint32_t r = 0; r < RT; ++r) {
            mm_f<HIFI4>(CB_CONST, CB_RED, CONST_IDENT, kg * RT + r, r);
        }
    }
    owner_residual(WC_GMLP);
    cb_pop_front(CB_RED, 8 * RT);
}

// ================================================================ MLP: up|gate + GeGLU, down partial
NOINL void mlp_upgate() {
    TR("mlp_upgate", 0);
    cb_wait_front(CB_IN0, IN0_PAGES);
    cb_reserve_back(CB_H, 2 * RT);
    constexpr uint32_t RB = 2 * RT;  // r tiles at RB.., then c_u, c_g
    constexpr uint32_t CU = 3 * RT, CG = 3 * RT + 1;
    for (uint32_t jj = 0; jj < 2; ++jj) {
        tile_regs_acquire();
        mm_init_f<LOFI>(CB_IN0, CB_W8);
        for (uint32_t kb = 0; kb < 4; ++kb) {
            cb_wait_front(CB_W8, 2 * PAGE_TILES);
            for (uint32_t r = 0; r < RT; ++r) {
                for (uint32_t k = 0; k < 8; ++k) {
                    const uint32_t a = r * 32 + kb * 8 + k;
                    mm_f<LOFI>(CB_IN0, CB_W8, a, k, 2 * r);
                    mm_f<LOFI>(CB_IN0, CB_W8, a, 8 + k, 2 * r + 1);
                }
            }
            cb_pop_front(CB_W8, 2 * PAGE_TILES);
        }
        if (jj == 0) {
            cb_wait_front(CB_RTOK, 1);
        }
        for (uint32_t r = 0; r < RT; ++r) {
            load(CB_IN0, X_TILES + r, RB + r);
        }
        load(CB_WC, WC_CUG + jj, CU);
        load(CB_WC, WC_CUG + 2 + jj, CG);
        for (uint32_t r = 0; r < RT; ++r) {
            fma3(2 * r, RB + r, CU);
            fma3(2 * r + 1, RB + r, CG);
        }
        gelu_tanh_tile_init();
        for (uint32_t r = 0; r < RT; ++r) {
            gelu_tanh_tile(2 * r + 1);
        }
        for (uint32_t r = 0; r < RT; ++r) {
            fmul(2 * r, 2 * r + 1, 2 * r);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t r = 0; r < RT; ++r) {
            pack_to(2 * r, CB_H, 2 * r + jj);
        }
        tile_regs_release();
    }
    cb_push_back(CB_H, 2 * RT);
    cb_pop_front(CB_IN0, IN0_PAGES);
    cb_pop_front(CB_RTOK, 1);
}

NOINL void mlp_down() {
    TR("mlp_down", 0);
    cb_wait_front(CB_HG, 16 * RT);
    cb_reserve_back(CB_DP, 4 * RT);
    tile_regs_acquire();
    mm_init_f<HIFI2>(CB_HG, CB_W16);
    for (uint32_t kb = 0; kb < 2; ++kb) {
        for (uint32_t n = 0; n < 4; ++n) {
            cb_wait_front(CB_W16, PAGE_TILES);
            for (uint32_t r = 0; r < RT; ++r) {
                for (uint32_t k = 0; k < 8; ++k) {
                    mm_f<HIFI2>(CB_HG, CB_W16, r * 16 + kb * 8 + k, k, n * RT + r);
                }
            }
            cb_pop_front(CB_W16, PAGE_TILES);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < 4 * RT; ++i) {
        pack_to(i, CB_DP, i);
    }
    tile_regs_release();
    cb_push_back(CB_DP, 4 * RT);
    cb_pop_front(CB_HG, 16 * RT);
}

// ================================================================ H0
// r = rsqrt(mean(x^2) + eps) per row of CB_IN0's x (full tiles) -> `out_cb` tiles [0, RT)
NOINL void h0_rms(uint32_t out_cb) {
    TR("h0_rms", 0);
    cb_reserve_back(out_cb, RT);
    for (uint32_t r = 0; r < RT; ++r) {
        cb_reserve_back(CB_SCR, 1);
        tile_regs_acquire();
        mul_acc_init(CB_IN0, CB_IN0);
        for (uint32_t k = 0; k < 32; ++k) {
            mul_acc(CB_IN0, CB_IN0, r * 32 + k, r * 32 + k, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, CB_SCR, 0);
        tile_regs_release();
        cb_push_back(CB_SCR, 1);
        TR("rms_scr", r);
        cb_wait_front(CB_SCR, 1);
        tile_regs_acquire();
        mm_init_f<HIFI4>(CB_SCR, CB_CONST);
        mm_f<HIFI4>(CB_SCR, CB_CONST, 0, CONST_MEAN, 0);
        binop_with_scalar_tile_init();
        add_unary_tile(0, EPS_BITS);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_to(0, out_cb, r);
        tile_regs_release();
        cb_pop_front(CB_SCR, 1);
    }
    cb_push_back(out_cb, RT);
}

// x_new = x_t @ W_in + b_in -> row 0 in CB_PART, row 1 in CB_HG (TRISC -> BRISC; the BRISC multicasts them and copies
// them into its own CB_IN0, which it pushes as for a gathered x), then r_in from CB_IN0 -> CB_ROUT
NOINL void h0_inproj() {
    TR("h0_inproj", 0);
    // x_t (fp32 state) -> bf16 in0
    cb_reserve_back(CB_SCR16, RT);
    cb_wait_front(CB_XRES, RT);
    tile_regs_acquire();
    for (uint32_t r = 0; r < RT; ++r) {
        load(CB_XRES, r, r);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t r = 0; r < RT; ++r) {
        pack_to(r, CB_SCR16, r);
    }
    tile_regs_release();
    cb_push_back(CB_SCR16, RT);
    cb_wait_front(CB_SCR16, RT);
    cb_reserve_back(CB_PART, DH_T * (NCH - 1));
    if (RT == 2) {
        cb_reserve_back(CB_HG, 16 * RT);
    }
    for (uint32_t nb = 0; nb < 4; ++nb) {
        TR("inproj_nb", nb);
        cb_wait_front(CB_W16, 2 * PAGE_TILES);  // W_in tiles nb*8..+8 (one K tile each), then their b_in tiles
        for (uint32_t r = 0; r < RT; ++r) {
            for (uint32_t n0 = 0; n0 < 8; n0 += 4) {
                tile_regs_acquire();
                mm_init_f<HIFI4>(CB_SCR16, CB_W16);
                for (uint32_t q = 0; q < 4; ++q) {
                    mm_f<HIFI4>(CB_SCR16, CB_W16, r, n0 + q, q);
                }
                for (uint32_t q = 0; q < 4; ++q) {
                    load(CB_W16, 8 + n0 + q, 4 + q);
                }
                for (uint32_t q = 0; q < 4; ++q) {
                    fadd(q, 4 + q, q);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t q = 0; q < 4; ++q) {
                    pack_to(q, r == 0 ? CB_PART : CB_HG, nb * 8 + n0 + q);
                }
                tile_regs_release();
            }
        }
        cb_pop_front(CB_W16, 2 * PAGE_TILES);
    }
    cb_push_back(CB_PART, DH_T * (NCH - 1));
    if (RT == 2) {
        cb_push_back(CB_HG, 16 * RT);
    }
    cb_pop_front(CB_SCR16, RT);
    cb_wait_front(CB_IN0, IN0_PAGES);  // the BRISC's copy of x_new
    h0_rms(CB_ROUT);
    cb_pop_front(CB_IN0, IN0_PAGES);
}

// final adaRMS + out-projection + Euler on the gathered x (CB_IN0); `last` also packs x_t (bf16) -> CB_ROUT
NOINL void h0_tail(bool last) {
    TR("h0_tail", 0);
    cb_wait_front(CB_IN0, IN0_PAGES);
    h0_rms(CB_SCR16);
    cb_wait_front(CB_SCR16, RT);
    cb_wait_front(CB_W16, 40);  // Wout' (32 K tiles x 1 N), c_out page
    cb_wait_front(CB_XRES, RT);
    tile_regs_acquire();
    mm_init_f<HIFI4>(CB_IN0, CB_W16);
    for (uint32_t r = 0; r < RT; ++r) {
        for (uint32_t k = 0; k < 32; ++k) {
            mm_f<HIFI4>(CB_IN0, CB_W16, r * 32 + k, k, r);
        }
    }
    for (uint32_t r = 0; r < RT; ++r) {
        load(CB_SCR16, r, RT + r);
    }
    load(CB_W16, 32, 2 * RT);
    for (uint32_t r = 0; r < RT; ++r) {
        load(CB_XRES, r, 2 * RT + 1 + r);
    }
    for (uint32_t r = 0; r < RT; ++r) {
        fma3(r, RT + r, 2 * RT);                             // v = r_f * (x @ Wout') + c_out
        fscale(r, get_common_arg_val<uint32_t>(C_DT_BITS));  // dt * v
        fadd(r, 2 * RT + 1 + r, r);                          // x_t + dt * v
    }
    tile_regs_commit();
    cb_pop_front(CB_XRES, RT);
    cb_reserve_back(CB_XRES, RT);
    if (last) {
        cb_reserve_back(CB_ROUT, RT);
    }
    tile_regs_wait();
    for (uint32_t r = 0; r < RT; ++r) {
        pack_to(r, CB_XRES, r);
    }
    if (last) {
        for (uint32_t r = 0; r < RT; ++r) {
            pack_to(r, CB_ROUT, r);
        }
    }
    tile_regs_release();
    cb_push_back(CB_XRES, RT);
    if (last) {
        cb_push_back(CB_ROUT, RT);
    }
    cb_pop_front(CB_W16, 40);
    cb_wait_front(CB_W16, 24);  // the step's 3 pad pages (keep every step at ring page 0)
    cb_pop_front(CB_W16, 24);
    cb_pop_front(CB_SCR16, RT);
    cb_pop_front(CB_IN0, IN0_PAGES);
}

// x_t (bf16) -> CB_ROUT without an Euler step (debug stop in the middle of a step)
NOINL void h0_emit_xt() {
    TR("h0_emit_xt", 0);
    cb_wait_front(CB_IN0, IN0_PAGES);
    cb_wait_front(CB_W16, 40);
    cb_pop_front(CB_W16, 40);
    cb_wait_front(CB_W16, 24);
    cb_pop_front(CB_W16, 24);
    cb_wait_front(CB_XRES, RT);
    cb_reserve_back(CB_ROUT, RT);
    tile_regs_acquire();
    for (uint32_t r = 0; r < RT; ++r) {
        load(CB_XRES, r, r);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t r = 0; r < RT; ++r) {
        pack_to(r, CB_ROUT, r);
    }
    tile_regs_release();
    cb_push_back(CB_ROUT, RT);
    cb_pop_front(CB_IN0, IN0_PAGES);
}

void run_h0(uint32_t ngen) {
    // noise (bf16, loaded by the BRISC into CB_Q: one producer RISC per CB) -> x_t (fp32)
    cb_wait_front(CB_Q, RT);
    cb_reserve_back(CB_XRES, RT);
    tile_regs_acquire();
    for (uint32_t r = 0; r < RT; ++r) {
        load(CB_Q, r, r);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t r = 0; r < RT; ++r) {
        pack_to(r, CB_XRES, r);
    }
    tile_regs_release();
    cb_push_back(CB_XRES, RT);
    cb_pop_front(CB_Q, RT);
    for (uint32_t g = 0; g < ngen; ++g) {
        TR("h0_gen", g);
        const uint32_t l = g % N_LAYERS;
        if (l == 0) {
            if (g > 0) {
                h0_tail(false);
            }
            h0_inproj();
        } else {
            cb_wait_front(CB_IN0, IN0_PAGES);
            h0_rms(CB_ROUT);
            cb_pop_front(CB_IN0, IN0_PAGES);
        }
        cb_wait_front(CB_IN0, IN0_PAGES);  // x_mid
        h0_rms(CB_ROUT);
        cb_pop_front(CB_IN0, IN0_PAGES);
    }
    if (ngen % N_LAYERS == 0) {
        h0_tail(true);
    } else {
        h0_emit_xt();
    }
}

}  // namespace

void kernel_main() {
    me.bits = get_arg_val<uint32_t>(A_ROLE);
    me.kind = get_arg_val<uint32_t>(A_PAIR_KIND);
    me.head = get_arg_val<uint32_t>(A_HEAD);
    me.ur = get_arg_val<uint32_t>(A_UNIT_R);
    me.kc = get_arg_val<uint32_t>(A_UNIT_KC);
    me.npre = get_arg_val<uint32_t>(A_UNIT_NPRE);
    me.kt0 = get_arg_val<uint32_t>(A_UNIT_KT0);
    me.n = get_arg_val<uint32_t>(A_OWNER_N);
    const uint32_t ngen = get_common_arg_val<uint32_t>(C_DEBUG);
    if ((me.bits & (R_PAIR | R_UNIT | R_OWNER | R_MLP | R_H0)) == 0) {
        return;
    }
    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_IN0, CB_W8, CB_QKVO);
    cb_wait_front(CB_CONST, 4);
    if (me.has(R_PAIR) && me.kind != PK_V) {
        cb_wait_front(CB_TAB, 4 * RT);
    }
    if (me.has(R_UNIT)) {
        cb_wait_front(CB_MASK, CHT);
    }
    if (me.has(R_H0)) {
        run_h0(ngen);
        return;
    }
    const bool has_wc = (me.bits & (R_PAIR | R_MLP | R_OWNER)) != 0;
    for (uint32_t g = 0; g < ngen; ++g) {
        TR("gen", g);
        const uint32_t l = g % N_LAYERS;
        if (has_wc) {
            cb_wait_front(CB_WC, PAGE_TILES);
        }
        if (me.has(R_PAIR)) {
            pair_qkv();
        }
        if (me.has(R_OWNER) && l == 0) {
            owner_load_x(g > 0);
        }
        if constexpr (ROWLOOP) {
            if (me.has(R_UNIT)) {
                unit_attention_rl();
            }
        } else {
            if (me.has(R_UNIT)) {
                unit_attention();
            }
            if (me.has(R_MERGER)) {
                merge();
            }
        }
        if (me.has(R_OWNER)) {
            owner_oproj();
        }
        if (me.has(R_MLP)) {
            mlp_upgate();
            mlp_down();
        }
        if (me.has(R_OWNER)) {
            owner_reduce();
        }
        if (has_wc) {
            cb_pop_front(CB_WC, PAGE_TILES);
        }
    }
}
