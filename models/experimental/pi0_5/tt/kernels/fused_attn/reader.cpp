// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Fused expert attention, reader (one core per (request b, head h, query tile-row r)): this core's q tiles, the KV head's
// k and v suffix tiles (all suffix rows), the RoPE tables (scale folded into the q tables, rotate-half sign folded
// into the sin tables), the key mask tile and the reduce scaler tile, then the K and V PREFIX rows of the caches.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t i = 0;
    const uint32_t xqkv_addr = get_arg_val<uint32_t>(i++);
    const uint32_t cosq_addr = get_arg_val<uint32_t>(i++);
    const uint32_t sinq_addr = get_arg_val<uint32_t>(i++);
    const uint32_t cosk_addr = get_arg_val<uint32_t>(i++);
    const uint32_t sink_addr = get_arg_val<uint32_t>(i++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(i++);
    const uint32_t scaler_addr = get_arg_val<uint32_t>(i++);
    const uint32_t kc_addr = get_arg_val<uint32_t>(i++);
    const uint32_t vc_addr = get_arg_val<uint32_t>(i++);
    const uint32_t h = get_arg_val<uint32_t>(i++);
    const uint32_t r = get_arg_val<uint32_t>(i++);
    const uint32_t Pt = get_arg_val<uint32_t>(i++);      // prefix key tile rows
    const uint32_t NQKV_T = get_arg_val<uint32_t>(i++);  // tile columns of xqkv
    const uint32_t DHt = get_arg_val<uint32_t>(i++);     // head_dim tiles
    const uint32_t NH = get_arg_val<uint32_t>(i++);      // q heads
    const uint32_t St = get_arg_val<uint32_t>(i++);      // suffix tile rows
    const uint32_t b = get_arg_val<uint32_t>(i++);       // request (batch index) of this core
    const uint32_t xqkv_bstride = get_arg_val<uint32_t>(i++);  // tiles per request in xqkv (St * NQKV_T)
    const uint32_t kv_bstride = get_arg_val<uint32_t>(i++);    // tiles per request in a cache (padded rows/32 * DHt)
    const uint32_t xb = b * xqkv_bstride;
    const uint32_t kb = b * kv_bstride;
#ifdef RC_PROLOGUE
    const uint32_t r_addr = get_arg_val<uint32_t>(i++);  // row rsqrt tiles [rows, 32] (tile b*St + s)
    const uint32_t c_addr = get_arg_val<uint32_t>(i++);  // folded bias row [1, 1, 32, (H+2)*dh]
#endif

    constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_cosq = 3, cb_sinq = 4, cb_cosk = 5, cb_sink = 6;
    constexpr uint32_t cb_mask = 7, cb_scaler = 8, cb_kpre = 9, cb_vpre = 11;
    constexpr auto a_xqkv = TensorAccessorArgs<0>();
    constexpr auto a_cosq = TensorAccessorArgs<a_xqkv.next_compile_time_args_offset()>();
    constexpr auto a_sinq = TensorAccessorArgs<a_cosq.next_compile_time_args_offset()>();
    constexpr auto a_cosk = TensorAccessorArgs<a_sinq.next_compile_time_args_offset()>();
    constexpr auto a_sink = TensorAccessorArgs<a_cosk.next_compile_time_args_offset()>();
    constexpr auto a_mask = TensorAccessorArgs<a_sink.next_compile_time_args_offset()>();
    constexpr auto a_scaler = TensorAccessorArgs<a_mask.next_compile_time_args_offset()>();
    constexpr auto a_kc = TensorAccessorArgs<a_scaler.next_compile_time_args_offset()>();
    constexpr auto a_vc = TensorAccessorArgs<a_kc.next_compile_time_args_offset()>();
#ifdef RC_PROLOGUE
    constexpr auto a_r = TensorAccessorArgs<a_vc.next_compile_time_args_offset()>();
    constexpr auto a_c = TensorAccessorArgs<a_r.next_compile_time_args_offset()>();
#endif
    const auto s_xqkv = TensorAccessor(a_xqkv, xqkv_addr);
    const auto s_cosq = TensorAccessor(a_cosq, cosq_addr);
    const auto s_sinq = TensorAccessor(a_sinq, sinq_addr);
    const auto s_cosk = TensorAccessor(a_cosk, cosk_addr);
    const auto s_sink = TensorAccessor(a_sink, sink_addr);
    const auto s_mask = TensorAccessor(a_mask, mask_addr);
    const auto s_scaler = TensorAccessor(a_scaler, scaler_addr);
    const auto s_kc = TensorAccessor(a_kc, kc_addr);
    const auto s_vc = TensorAccessor(a_vc, vc_addr);
#ifdef RC_PROLOGUE
    const auto s_r = TensorAccessor(a_r, r_addr);
    const auto s_c = TensorAccessor(a_c, c_addr);
    constexpr uint32_t cb_r = 19, cb_cq = 24, cb_ck = 25, cb_cv = 26;
#endif

    auto read_pages = [&](uint32_t cb, uint32_t n, auto&& page_of, const auto& acc) {
        cb_reserve_back(cb, n);
        uint32_t a = get_write_ptr(cb);
        const uint32_t page_bytes = get_local_cb_interface(cb).fifo_page_size;
        for (uint32_t j = 0; j < n; ++j, a += page_bytes) noc_async_read_page(page_of(j), acc, a);
    };
    // small operands
    read_pages(cb_q, DHt, [&](uint32_t t) { return xb + r * NQKV_T + h * DHt + t; }, s_xqkv);
    read_pages(cb_k, St * DHt, [&](uint32_t j) { return xb + (j / DHt) * NQKV_T + NH * DHt + (j % DHt); }, s_xqkv);
    read_pages(cb_v, St * DHt, [&](uint32_t j) { return xb + (j / DHt) * NQKV_T + (NH + 1) * DHt + (j % DHt); }, s_xqkv);
    read_pages(cb_cosq, DHt, [&](uint32_t t) { return r * DHt + t; }, s_cosq);
    read_pages(cb_sinq, DHt, [&](uint32_t t) { return r * DHt + t; }, s_sinq);
    read_pages(cb_mask, 1, [&](uint32_t) { return 0u; }, s_mask);
    read_pages(cb_scaler, 1, [&](uint32_t) { return 0u; }, s_scaler);
#ifdef RC_PROLOGUE
    read_pages(cb_r, St, [&](uint32_t s) { return b * St + s; }, s_r);
    read_pages(cb_cq, DHt, [&](uint32_t t) { return h * DHt + t; }, s_c);
    read_pages(cb_ck, DHt, [&](uint32_t t) { return NH * DHt + t; }, s_c);
    read_pages(cb_cv, DHt, [&](uint32_t t) { return (NH + 1) * DHt + t; }, s_c);
#endif
    noc_async_read_barrier();
    cb_push_back(cb_q, DHt);
    cb_push_back(cb_k, St * DHt);
    cb_push_back(cb_v, St * DHt);
    cb_push_back(cb_cosq, DHt);
    cb_push_back(cb_sinq, DHt);
    cb_push_back(cb_mask, 1);
    cb_push_back(cb_scaler, 1);
#ifdef RC_PROLOGUE
    cb_push_back(cb_r, St);
    cb_push_back(cb_cq, DHt);
    cb_push_back(cb_ck, DHt);
    cb_push_back(cb_cv, DHt);
#endif
    // RoPE tables for the k suffix rows, one row at a time (the CBs hold one row; compute pops after each row)
    for (uint32_t s = 0; s < St; ++s) {
        read_pages(cb_cosk, DHt, [&](uint32_t t) { return s * DHt + t; }, s_cosk);
        read_pages(cb_sink, DHt, [&](uint32_t t) { return s * DHt + t; }, s_sink);
        noc_async_read_barrier();
        cb_push_back(cb_cosk, DHt);
        cb_push_back(cb_sink, DHt);
    }
    // K prefix rows, then V prefix rows (tile (n, t) at n*DHt + t), pushed in chunks of CH tile rows so the
    // compute's QK^T / PV rounds start on the first chunk while the rest is still in flight.
    // Chunks are ALWAYS CH rows: a Metal circular buffer never splits a block across its end, so every push must
    // have the same size that the ring capacity is a multiple of. The last chunk may read a few tile rows past the
    // prefix (they exist in the tile-padded cache); the compute ignores them.
    constexpr uint32_t CH = 4;
    for (uint32_t n0 = 0; n0 < Pt; n0 += CH) {
        read_pages(cb_kpre, CH * DHt, [&](uint32_t j) { return kb + n0 * DHt + j; }, s_kc);
        noc_async_read_barrier();
        cb_push_back(cb_kpre, CH * DHt);
    }
    for (uint32_t n0 = 0; n0 < Pt; n0 += CH) {
        read_pages(cb_vpre, CH * DHt, [&](uint32_t j) { return kb + n0 * DHt + j; }, s_vc);
        noc_async_read_barrier();
        cb_push_back(cb_vpre, CH * DHt);
    }
}
