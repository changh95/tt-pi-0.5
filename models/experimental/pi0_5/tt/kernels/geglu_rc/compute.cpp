// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// geglu_rc compute: h = (r * u + cu) * gelu(r * g + cg) for Kb tiles of one tile-row, where r is the row rsqrt
// (bcast over columns) and cu / cg the folded adaRMS shift biases (bcast over rows). This is adaRMS + up|gate
// bias + GeGLU with the norm's scale folded into the up|gate weights.
#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

namespace {
constexpr uint32_t cb_u = 0, cb_g = 1, cb_r = 2, cb_cu = 3, cb_cg = 4, cb_t1 = 5, cb_u2 = 6, cb_g2 = 7, cb_out = 16;
constexpr uint32_t DST = 8;

// out = (x * r) + c   (r: column-broadcast tile, c: row-broadcast tiles), optionally gelu'd
template <bool GELU>
void scale_bias(uint32_t cb_x, uint32_t cb_c, uint32_t cb_o, uint32_t Kb) {
    // pass 1: t1 = x * r (bcast cols)
    reconfig_data_format(cb_x, cb_r);
    mul_bcast_cols_init(cb_x, cb_r);
    pack_reconfig_data_format(cb_t1);
    cb_reserve_back(cb_t1, Kb);
    for (uint32_t k0 = 0; k0 < Kb; k0 += DST) {
        const uint32_t cnt = (Kb - k0 < DST) ? (Kb - k0) : DST;
        tile_regs_acquire();
        for (uint32_t j = 0; j < cnt; ++j) mul_tiles_bcast_cols(cb_x, cb_r, k0 + j, 0, j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) pack_tile(j, cb_t1);
        tile_regs_release();
    }
    cb_push_back(cb_t1, Kb);
    cb_wait_front(cb_t1, Kb);
    // pass 2: o = t1 + c (bcast rows) [, gelu]
    reconfig_data_format(cb_t1, cb_c);
    add_bcast_rows_init(cb_t1, cb_c);
    if constexpr (GELU) gelu_tile_init();
    pack_reconfig_data_format(cb_o);
    cb_reserve_back(cb_o, Kb);
    for (uint32_t k0 = 0; k0 < Kb; k0 += DST) {
        const uint32_t cnt = (Kb - k0 < DST) ? (Kb - k0) : DST;
        tile_regs_acquire();
        for (uint32_t j = 0; j < cnt; ++j) {
            add_tiles_bcast_rows(cb_t1, cb_c, k0 + j, k0 + j, j);
            if constexpr (GELU) gelu_tile(j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) pack_tile(j, cb_o);
        tile_regs_release();
    }
    cb_push_back(cb_o, Kb);
    cb_pop_front(cb_t1, Kb);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Kb = get_compile_time_arg_val(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_u, cb_r, cb_out);
    cb_wait_front(cb_u, Kb);
    cb_wait_front(cb_g, Kb);
    cb_wait_front(cb_r, 1);
    cb_wait_front(cb_cu, Kb);
    cb_wait_front(cb_cg, Kb);
    scale_bias<false>(cb_u, cb_cu, cb_u2, Kb);
    scale_bias<true>(cb_g, cb_cg, cb_g2, Kb);
    cb_wait_front(cb_u2, Kb);
    cb_wait_front(cb_g2, Kb);
    reconfig_data_format(cb_u2, cb_g2);
    mul_init(cb_u2, cb_g2);
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, Kb);
    for (uint32_t k0 = 0; k0 < Kb; k0 += DST) {
        const uint32_t cnt = (Kb - k0 < DST) ? (Kb - k0) : DST;
        tile_regs_acquire();
        for (uint32_t j = 0; j < cnt; ++j) mul_tiles(cb_u2, cb_g2, k0 + j, k0 + j, j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < cnt; ++j) pack_tile(j, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, Kb);
}
