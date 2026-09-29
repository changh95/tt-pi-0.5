// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused expert attention, compute (one core per (head, query tile-row)):
//   q_rot = RoPE(q) (scale folded into the tables), k_rot = RoPE(k suffix rows)
//   S = q_rot @ [K_prefix ; k_rot]^T + mask   (NKt key tiles; the additive mask hides prefix pad keys and the
//                                             tile-padding suffix rows)
//   P = softmax_rows(S)                (row max, exp, row sum, reciprocal)
//   out = (P @ [V_prefix ; v]) * (1/rowsum) -> DHt tiles (bf8) for the writer
// Small-footprint variant: K/V prefix rings of 3 x 4 rows, P in place of S, per-row k RoPE tables (~440 KB of L1 per core).
// Replaces nlp_create_qkv_heads + rotary_embedding x2 + cache fills + SDPA + nlp_concat_heads.
#include <cstdint>
#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_cosq = 3, cb_sinq = 4, cb_cosk = 5, cb_sink = 6, cb_mask = 7;
constexpr uint32_t cb_scaler = 8, cb_kpre = 9, cb_krot = 10, cb_vpre = 11;
constexpr uint32_t cb_qrot = 13, cb_tmp1 = 14, cb_tmp2 = 15, cb_out = 16, cb_s = 17, cb_max = 18, cb_p = 17;  // P overwrites S
constexpr uint32_t cb_sum = 20, cb_o = 21, cb_rsum = 23;
constexpr uint32_t DST = 8;  // dest tiles per acquire (16-bit, half sync)
#ifdef RC_PROLOGUE
// adaRMS folded into the qkv weights: q/k/v tiles arrive as x @ (diag(scale) W) and get r (row rsqrt, bcast over
// columns) and c (shift @ W, bcast over rows) applied here -> bf16 copies that the RoPE / PV consume.
constexpr uint32_t cb_r = 19, cb_cq = 24, cb_ck = 25, cb_cv = 26, cb_q2 = 27, cb_k2 = 28, cb_v2 = 29;
constexpr uint32_t cb_qin = cb_q2, cb_kin = cb_k2, cb_vin = cb_v2;
#else
constexpr uint32_t cb_qin = cb_q, cb_kin = cb_k, cb_vin = cb_v;
#endif

// out = x*cos + rotate_half(x)*sin_signed for DHt tiles of one tile-row (x tiles at x_off.., tables at cs_off..)
template <uint32_t DHt>
void rope_row(uint32_t cb_x, uint32_t x_off, uint32_t cb_cos, uint32_t cb_sin, uint32_t cs_off, uint32_t cb_o_) {
    constexpr uint32_t half = DHt / 2;
    reconfig_data_format(cb_x, cb_cos);
    mul_init(cb_x, cb_cos);
    pack_reconfig_data_format(cb_tmp1);
    cb_reserve_back(cb_tmp1, DHt);
    tile_regs_acquire();
    for (uint32_t t = 0; t < DHt; ++t) mul_tiles(cb_x, cb_cos, x_off + t, cs_off + t, t);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < DHt; ++t) pack_tile(t, cb_tmp1);
    tile_regs_release();
    cb_push_back(cb_tmp1, DHt);

    reconfig_data_format(cb_x, cb_sin);
    mul_init(cb_x, cb_sin);
    pack_reconfig_data_format(cb_tmp2);
    cb_reserve_back(cb_tmp2, DHt);
    tile_regs_acquire();
    for (uint32_t t = 0; t < DHt; ++t) mul_tiles(cb_x, cb_sin, x_off + ((t + half) % DHt), cs_off + t, t);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < DHt; ++t) pack_tile(t, cb_tmp2);
    tile_regs_release();
    cb_push_back(cb_tmp2, DHt);

    cb_wait_front(cb_tmp1, DHt);
    cb_wait_front(cb_tmp2, DHt);
    reconfig_data_format(cb_tmp1, cb_tmp2);
    add_init(cb_tmp1, cb_tmp2);
    pack_reconfig_data_format(cb_o_);
    cb_reserve_back(cb_o_, DHt);
    tile_regs_acquire();
    for (uint32_t t = 0; t < DHt; ++t) add_tiles(cb_tmp1, cb_tmp2, t, t, t);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < DHt; ++t) pack_tile(t, cb_o_);
    tile_regs_release();
    cb_push_back(cb_o_, DHt);
    cb_pop_front(cb_tmp1, DHt);
    cb_pop_front(cb_tmp2, DHt);
}
#ifdef RC_PROLOGUE
// out[t] = x[x_off + t] * r[r_idx] (bcast cols) + c[t] (bcast rows), n tiles, via cb_tmp1
void rc_apply(uint32_t cb_x, uint32_t x_off, uint32_t r_idx, uint32_t cb_c, uint32_t cb_o_, uint32_t n) {
    reconfig_data_format(cb_x, cb_r);
    mul_bcast_cols_init(cb_x, cb_r);
    pack_reconfig_data_format(cb_tmp1);
    cb_reserve_back(cb_tmp1, n);
    tile_regs_acquire();
    for (uint32_t t = 0; t < n; ++t) mul_tiles_bcast_cols(cb_x, cb_r, x_off + t, r_idx, t);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < n; ++t) pack_tile(t, cb_tmp1);
    tile_regs_release();
    cb_push_back(cb_tmp1, n);
    cb_wait_front(cb_tmp1, n);
    reconfig_data_format(cb_tmp1, cb_c);
    add_bcast_rows_init(cb_tmp1, cb_c);
    pack_reconfig_data_format(cb_o_);
    cb_reserve_back(cb_o_, n);
    tile_regs_acquire();
    for (uint32_t t = 0; t < n; ++t) add_tiles_bcast_rows(cb_tmp1, cb_c, t, t, t);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < n; ++t) pack_tile(t, cb_o_);
    tile_regs_release();
    cb_push_back(cb_o_, n);
    cb_pop_front(cb_tmp1, n);
}
#endif
}  // namespace

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t Pt = get_compile_time_arg_val(2);
    constexpr uint32_t NKt = Pt + St;
#ifdef RC_PROLOGUE
    const uint32_t R = get_arg_val<uint32_t>(0);  // this core's query tile-row (index into the r tiles)
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_cosq, cb_out);
    cb_wait_front(cb_q, DHt);
    cb_wait_front(cb_k, St * DHt);
    cb_wait_front(cb_v, St * DHt);
    cb_wait_front(cb_cosq, DHt);
    cb_wait_front(cb_sinq, DHt);
    cb_wait_front(cb_mask, NKt);
    cb_wait_front(cb_scaler, 1);
#ifdef RC_PROLOGUE
    // 0. r / c on q (this core's query tile-row R), k and v (all St suffix rows of the KV head)
    cb_wait_front(cb_r, St);
    cb_wait_front(cb_cq, DHt);
    cb_wait_front(cb_ck, DHt);
    cb_wait_front(cb_cv, DHt);
    rc_apply(cb_q, 0, R, cb_cq, cb_q2, DHt);
    for (uint32_t s = 0; s < St; ++s) rc_apply(cb_k, s * DHt, s, cb_ck, cb_k2, DHt);
    for (uint32_t s = 0; s < St; ++s) rc_apply(cb_v, s * DHt, s, cb_cv, cb_v2, DHt);
    cb_wait_front(cb_q2, DHt);
    cb_wait_front(cb_k2, St * DHt);
    cb_wait_front(cb_v2, St * DHt);
#endif

    // 1. RoPE
    rope_row<DHt>(cb_qin, 0, cb_cosq, cb_sinq, 0, cb_qrot);
    for (uint32_t s = 0; s < St; ++s) {
        cb_wait_front(cb_cosk, DHt);
        cb_wait_front(cb_sink, DHt);
        rope_row<DHt>(cb_kin, s * DHt, cb_cosk, cb_sink, 0, cb_krot);
        cb_pop_front(cb_cosk, DHt);
        cb_pop_front(cb_sink, DHt);
    }
    cb_wait_front(cb_qrot, DHt);
    cb_wait_front(cb_krot, St * DHt);

    // 2. S = q_rot @ K^T  (in1 tiles transposed). The K prefix streams through a ring of CH-row chunks; each DEST
    //    round handles CH key tiles (one chunk, indexed relative to the CB front, popped after use); the suffix rows
    //    come from cb_krot.
    constexpr uint32_t CH = 4;
    reconfig_data_format(cb_kpre, cb_qrot);
    matmul_init(cb_qrot, cb_kpre, 1);
    pack_reconfig_data_format(cb_s);
    cb_reserve_back(cb_s, NKt);
    for (uint32_t n0 = 0; n0 < NKt; n0 += CH) {
        const uint32_t cnt = (NKt - n0 < CH) ? (NKt - n0) : CH;
        const uint32_t pre_rows = (n0 >= Pt) ? 0 : ((Pt - n0 < CH) ? (Pt - n0) : CH);  // valid prefix rows this round
        if (pre_rows > 0) cb_wait_front(cb_kpre, CH * DHt);  // chunks are always CH rows (see reader)
        tile_regs_acquire();
        for (uint32_t j = 0; j < cnt; ++j) {
            const uint32_t n = n0 + j;
            if (n < Pt) {
                for (uint32_t t = 0; t < DHt; ++t) matmul_tiles(cb_qrot, cb_kpre, t, j * DHt + t, j);
            } else {
                for (uint32_t t = 0; t < DHt; ++t) matmul_tiles(cb_qrot, cb_krot, t, (n - Pt) * DHt + t, j);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) pack_tile(j, cb_s);
        tile_regs_release();
        if (pre_rows > 0) cb_pop_front(cb_kpre, CH * DHt);
    }
    cb_push_back(cb_s, NKt);
    cb_wait_front(cb_s, NKt);

    // 3. S += mask (request b's key mask row, one tile per key tile), in place like 4b: pop the consumed S tiles and
    //    push the masked ones into the same CB (it holds exactly NKt tiles)
    reconfig_data_format(cb_s, cb_mask);
    add_init(cb_s, cb_mask);
    pack_reconfig_data_format(cb_s);
    for (uint32_t n0 = 0; n0 < NKt; n0 += DST) {
        const uint32_t cnt = (NKt - n0 < DST) ? (NKt - n0) : DST;
        tile_regs_acquire();
        for (uint32_t j = 0; j < cnt; ++j) add_tiles(cb_s, cb_mask, j, n0 + j, j);
        tile_regs_commit();
        cb_pop_front(cb_s, cnt);
        cb_reserve_back(cb_s, cnt);
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) pack_tile(j, cb_s);
        tile_regs_release();
        cb_push_back(cb_s, cnt);
    }
    cb_wait_front(cb_s, NKt);

    // 4a. row max
    reconfig_data_format(cb_s, cb_scaler);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scaler, cb_max);
    pack_reconfig_data_format(cb_max);
    cb_reserve_back(cb_max, 1);
    tile_regs_acquire();
    for (uint32_t n = 0; n < NKt; ++n) reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_s, cb_scaler, n, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_max);
    tile_regs_release();
    reduce_uninit(cb_s);
    cb_push_back(cb_max, 1);
    cb_wait_front(cb_max, 1);

    // 4b. P = exp(S - rowmax), written in place of S (pop the consumed S tiles, push the P tiles into the same CB;
    //     the CB holds exactly NKt tiles so each round's P lands where its S was)
    reconfig_data_format(cb_s, cb_max);
    sub_bcast_cols_init(cb_s, cb_max);
    exp_tile_init<false>();
    pack_reconfig_data_format(cb_s);
    for (uint32_t n0 = 0; n0 < NKt; n0 += DST) {
        const uint32_t cnt = (NKt - n0 < DST) ? (NKt - n0) : DST;
        tile_regs_acquire();
        for (uint32_t j = 0; j < cnt; ++j) {
            sub_tiles_bcast_cols(cb_s, cb_max, j, 0, j);  // S tiles of this round sit at the CB front
            exp_tile(j);
        }
        tile_regs_commit();
        cb_pop_front(cb_s, cnt);
        cb_reserve_back(cb_s, cnt);
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) pack_tile(j, cb_s);
        tile_regs_release();
        cb_push_back(cb_s, cnt);
    }
    cb_wait_front(cb_p, NKt);

    // 4c. row sum (REDUCE_ROW SUM wants the scaler as srcA)
    reconfig_data_format(cb_scaler, cb_p);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_scaler, cb_sum);
    pack_reconfig_data_format(cb_sum);
    cb_reserve_back(cb_sum, 1);
    tile_regs_acquire();
    for (uint32_t n = 0; n < NKt; ++n) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_p, cb_scaler, n, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sum);
    tile_regs_release();
    reduce_uninit(cb_p);
    cb_push_back(cb_sum, 1);
    cb_wait_front(cb_sum, 1);

    // 4d. 1 / rowsum
    reconfig_data_format_srca(cb_sum);
    copy_init(cb_sum);
    recip_tile_init();
    pack_reconfig_data_format(cb_rsum);
    cb_reserve_back(cb_rsum, 1);
    tile_regs_acquire();
    copy_tile(cb_sum, 0, 0);
    recip_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_rsum);
    tile_regs_release();
    cb_push_back(cb_rsum, 1);
    cb_wait_front(cb_rsum, 1);

    // 5. O = P @ V: the V prefix streams through the same kind of ring (chunk = CH rows, consumed and popped per
    //    chunk); DEST accumulates the DHt output tiles across all key rows; suffix rows = raw v tiles.
    reconfig_data_format(cb_vpre, cb_p);
    matmul_init(cb_p, cb_vpre, 0);
    pack_reconfig_data_format(cb_o);
    cb_reserve_back(cb_o, DHt);
    tile_regs_acquire();
    for (uint32_t n0 = 0; n0 < Pt; n0 += CH) {
        const uint32_t rows = (Pt - n0 < CH) ? (Pt - n0) : CH;  // valid rows; the chunk itself is always CH rows
        cb_wait_front(cb_vpre, CH * DHt);
        for (uint32_t j = 0; j < rows; ++j) {
            for (uint32_t t = 0; t < DHt; ++t) matmul_tiles(cb_p, cb_vpre, n0 + j, j * DHt + t, t);
        }
        cb_pop_front(cb_vpre, CH * DHt);
    }
#ifdef RC_PROLOGUE
    reconfig_data_format(cb_vin, cb_p);  // the suffix V tiles are bf16 (r/c applied), the prefix ones the cache dtype
    matmul_init(cb_p, cb_vin, 0);
#endif
    for (uint32_t s = 0; s < St; ++s) {
        for (uint32_t t = 0; t < DHt; ++t) matmul_tiles(cb_p, cb_vin, Pt + s, s * DHt + t, t);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < DHt; ++t) pack_tile(t, cb_o);
    tile_regs_release();
    cb_push_back(cb_o, DHt);
    cb_wait_front(cb_o, DHt);

    // 6. out = O * (1/rowsum)
    reconfig_data_format(cb_o, cb_rsum);
    mul_bcast_cols_init(cb_o, cb_rsum);
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, DHt);
    tile_regs_acquire();
    for (uint32_t t = 0; t < DHt; ++t) mul_tiles_bcast_cols(cb_o, cb_rsum, t, 0, t);
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < DHt; ++t) pack_tile(t, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, DHt);
}
