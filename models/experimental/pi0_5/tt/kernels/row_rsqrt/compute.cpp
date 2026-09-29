// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// row_rsqrt compute: r = rsqrt(sum_k(x_k^2) * scaler + eps) per row of one tile-row (Kt tiles). The RMS-norm of
// the adaRMS layers reduces to this once scale/shift are folded into the following matmul's weights.
#include <cstdint>
#define REDUCE_OP (PoolType::SUM)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t cb_x = 0, cb_sc = 1, cb_eps = 2, cb_sq = 3, cb_sum = 4, cb_out = 16;
    constexpr uint32_t DST = 4;  // fp32 DEST accumulation halves the DEST capacity: 4 tiles per acquire
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_x, cb_out);
    cb_wait_front(cb_x, Kt);
    cb_wait_front(cb_sc, 1);
    cb_wait_front(cb_eps, 1);
    // x^2
    reconfig_data_format(cb_x, cb_x);
    mul_init(cb_x, cb_x);
    pack_reconfig_data_format(cb_sq);
    cb_reserve_back(cb_sq, Kt);
    for (uint32_t k0 = 0; k0 < Kt; k0 += DST) {
        const uint32_t cnt = (Kt - k0 < DST) ? (Kt - k0) : DST;
        tile_regs_acquire();
        for (uint32_t j = 0; j < cnt; ++j) mul_tiles(cb_x, cb_x, k0 + j, k0 + j, j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) pack_tile(j, cb_sq);
        tile_regs_release();
    }
    cb_push_back(cb_sq, Kt);
    cb_wait_front(cb_sq, Kt);
    // row sum * scaler (REDUCE_ROW SUM wants the scaler as srcA)
    reconfig_data_format(cb_sc, cb_sq);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, cb_sum);
    pack_reconfig_data_format(cb_sum);
    cb_reserve_back(cb_sum, 1);
    tile_regs_acquire();
    for (uint32_t k = 0; k < Kt; ++k) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_sc, k, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_sum);
    tile_regs_release();
    reduce_uninit(cb_sq);
    cb_push_back(cb_sum, 1);
    cb_wait_front(cb_sum, 1);
    // rsqrt(sum + eps)
    reconfig_data_format(cb_sum, cb_eps);
    add_init(cb_sum, cb_eps);
    rsqrt_tile_init();
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, 1);
    tile_regs_acquire();
    add_tiles(cb_sum, cb_eps, 0, 0, 0);
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
