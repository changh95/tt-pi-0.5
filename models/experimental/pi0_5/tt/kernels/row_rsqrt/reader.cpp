// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// row_rsqrt reader: one core per tile-row of x [rows, D]: the Kt tiles of that row, a scaler tile (1/D) and an eps tile.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(i++);
    const uint32_t scaler_addr = get_arg_val<uint32_t>(i++);
    const uint32_t eps_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rt = get_arg_val<uint32_t>(i++);
    const uint32_t Kt = get_arg_val<uint32_t>(i++);
    constexpr uint32_t cb_x = 0, cb_sc = 1, cb_eps = 2;
    constexpr auto a_x = TensorAccessorArgs<0>();
    constexpr auto a_sc = TensorAccessorArgs<a_x.next_compile_time_args_offset()>();
    constexpr auto a_eps = TensorAccessorArgs<a_sc.next_compile_time_args_offset()>();
    const auto s_x = TensorAccessor(a_x, x_addr);
    const auto s_sc = TensorAccessor(a_sc, scaler_addr);
    const auto s_eps = TensorAccessor(a_eps, eps_addr);
    cb_reserve_back(cb_sc, 1);
    noc_async_read_page(0, s_sc, get_write_ptr(cb_sc));
    cb_reserve_back(cb_eps, 1);
    noc_async_read_page(0, s_eps, get_write_ptr(cb_eps));
    cb_reserve_back(cb_x, Kt);
    uint32_t a = get_write_ptr(cb_x);
    const uint32_t tb = get_local_cb_interface(cb_x).fifo_page_size;
    for (uint32_t k = 0; k < Kt; ++k, a += tb) noc_async_read_page(rt * Kt + k, s_x, a);
    noc_async_read_barrier();
    cb_push_back(cb_sc, 1);
    cb_push_back(cb_eps, 1);
    cb_push_back(cb_x, Kt);
}
