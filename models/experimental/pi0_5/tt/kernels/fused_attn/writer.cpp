// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Fused expert attention, writer: this core's DHt context tiles -> ctx[r, h*DHt + t] (head-concatenated layout).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t ctx_addr = get_arg_val<uint32_t>(i++);
    const uint32_t h = get_arg_val<uint32_t>(i++);
    const uint32_t r = get_arg_val<uint32_t>(i++);
    const uint32_t NCTX_T = get_arg_val<uint32_t>(i++);
    const uint32_t DHt = get_arg_val<uint32_t>(i++);
    const uint32_t b = get_arg_val<uint32_t>(i++);
    const uint32_t ctx_bstride = get_arg_val<uint32_t>(i++);  // tiles per request in ctx (St * NCTX_T)
    constexpr uint32_t cb_out = 16;
    constexpr auto a_ctx = TensorAccessorArgs<0>();
    const auto s_ctx = TensorAccessor(a_ctx, ctx_addr);
    const uint32_t tile_bytes = get_local_cb_interface(cb_out).fifo_page_size;
    cb_wait_front(cb_out, DHt);
    uint32_t a = get_read_ptr(cb_out);
    for (uint32_t t = 0; t < DHt; ++t, a += tile_bytes) noc_async_write_page(b * ctx_bstride + r * NCTX_T + h * DHt + t, s_ctx, a, tile_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, DHt);
}
