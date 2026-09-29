// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// geglu_rc reader: one core per (tile-row rt, column block blk of Kb tiles) of ug = [up | gate] (unnormalised input
// times the scale-folded weights): the row's rsqrt tile, the folded-bias tiles of the block (up and gate halves), and
// the up / gate tiles.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t ug_addr = get_arg_val<uint32_t>(i++);
    const uint32_t r_addr = get_arg_val<uint32_t>(i++);
    const uint32_t c_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rt = get_arg_val<uint32_t>(i++);
    const uint32_t blk = get_arg_val<uint32_t>(i++);
    const uint32_t Kb = get_arg_val<uint32_t>(i++);
    const uint32_t NUG_T = get_arg_val<uint32_t>(i++);  // tile columns of ug (2 * mlp / 32)
    const uint32_t HALF_T = NUG_T / 2;                  // tile columns of one half (mlp / 32)
    constexpr uint32_t cb_u = 0, cb_g = 1, cb_r = 2, cb_cu = 3, cb_cg = 4;
    constexpr auto a_ug = TensorAccessorArgs<0>();
    constexpr auto a_r = TensorAccessorArgs<a_ug.next_compile_time_args_offset()>();
    constexpr auto a_c = TensorAccessorArgs<a_r.next_compile_time_args_offset()>();
    const auto s_ug = TensorAccessor(a_ug, ug_addr);
    const auto s_r = TensorAccessor(a_r, r_addr);
    const auto s_c = TensorAccessor(a_c, c_addr);
    auto read_pages = [&](uint32_t cb, uint32_t n, auto&& page_of, const auto& acc) {
        cb_reserve_back(cb, n);
        uint32_t a = get_write_ptr(cb);
        const uint32_t tb = get_local_cb_interface(cb).fifo_page_size;
        for (uint32_t j = 0; j < n; ++j, a += tb) noc_async_read_page(page_of(j), acc, a);
    };
    read_pages(cb_r, 1, [&](uint32_t) { return rt; }, s_r);
    read_pages(cb_cu, Kb, [&](uint32_t t) { return blk * Kb + t; }, s_c);
    read_pages(cb_cg, Kb, [&](uint32_t t) { return HALF_T + blk * Kb + t; }, s_c);
    read_pages(cb_u, Kb, [&](uint32_t t) { return rt * NUG_T + blk * Kb + t; }, s_ug);
    read_pages(cb_g, Kb, [&](uint32_t t) { return rt * NUG_T + HALF_T + blk * Kb + t; }, s_ug);
    noc_async_read_barrier();
    cb_push_back(cb_r, 1);
    cb_push_back(cb_cu, Kb);
    cb_push_back(cb_cg, Kb);
    cb_push_back(cb_u, Kb);
    cb_push_back(cb_g, Kb);
}
