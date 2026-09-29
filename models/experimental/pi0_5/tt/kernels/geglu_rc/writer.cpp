// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// geglu_rc writer: the Kb output tiles of this (row tile, block) -> h[rt, blk*Kb + t].
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t h_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rt = get_arg_val<uint32_t>(i++);
    const uint32_t blk = get_arg_val<uint32_t>(i++);
    const uint32_t Kb = get_arg_val<uint32_t>(i++);
    const uint32_t NH_T = get_arg_val<uint32_t>(i++);  // tile columns of h (mlp / 32)
    constexpr uint32_t cb_out = 16;
    constexpr auto a_h = TensorAccessorArgs<0>();
    const auto s_h = TensorAccessor(a_h, h_addr);
    const uint32_t tb = get_local_cb_interface(cb_out).fifo_page_size;
    cb_wait_front(cb_out, Kb);
    uint32_t a = get_read_ptr(cb_out);
    for (uint32_t t = 0; t < Kb; ++t, a += tb) noc_async_write_page(rt * NH_T + blk * Kb + t, s_h, a, tb);
    noc_async_write_barrier();
    cb_pop_front(cb_out, Kb);
}
