// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// row_rsqrt writer: the result tile (rsqrt(mean(x^2) + eps) per row in column 0) -> r[rt].
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t r_addr = get_arg_val<uint32_t>(0);
    const uint32_t rt = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_out = 16;
    constexpr auto a_r = TensorAccessorArgs<0>();
    const auto s_r = TensorAccessor(a_r, r_addr);
    cb_wait_front(cb_out, 1);
    noc_async_write_page(rt, s_r, get_read_ptr(cb_out), get_local_cb_interface(cb_out).fifo_page_size);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}
