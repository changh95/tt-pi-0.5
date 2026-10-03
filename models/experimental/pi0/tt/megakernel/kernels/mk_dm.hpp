// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pi0.5 expert megakernel: data-movement helpers shared by the BRISC and NCRISC kernels.
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "mk_defs.hpp"
#ifdef MK_TRACE
#include "api/debug/dprint.h"
#define TRD(tag, a, b) DPRINT(tag " {} {}\n", (uint32_t)(a), (uint32_t)(b))
#else
#define TRD(tag, a, b) \
    do {               \
    } while (0)
#endif

namespace mk {

FORCE_INLINE uint32_t rt(uint32_t i) { return get_arg_val<uint32_t>(i); }
FORCE_INLINE uint32_t ct_arg(uint32_t i) { return get_common_arg_val<uint32_t>(i); }

// Byte address of a CB's buffer base (LocalCBInterface keeps it in 16 B units): identical on every core.
FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    const auto& iface = get_local_cb_interface(cb);
    return (iface.fifo_limit - iface.fifo_size) << cb_addr_shift;
}
FORCE_INLINE uint32_t cb_pages(uint32_t cb) { return get_local_cb_interface(cb).fifo_num_pages; }

// Free pages from the producer's point of view (non-blocking cb_reserve_back maths).
FORCE_INLINE uint32_t cb_free_pages(uint32_t cb) {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
    const uint16_t received = (uint16_t)get_cb_tiles_received_ptr(cb)[0];
    const uint16_t acked = (uint16_t)reg_read((uintptr_t)get_cb_tiles_acked_ptr(cb));
    return (uint16_t)(get_local_cb_interface(cb).fifo_num_pages - (uint16_t)(received - acked));
}

// Sync words live in CB_SYNC, SYNC_STRIDE bytes apart (every word owns a 16 B line: F0 device fact 6).
FORCE_INLINE uint32_t sync_addr(uint32_t w) { return cb_base(CB_SYNC) + w * SYNC_STRIDE; }
FORCE_INLINE volatile tt_l1_ptr uint32_t* sync_ptr(uint32_t w) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync_addr(w));
}
FORCE_INLINE uint32_t sync_read(uint32_t w) {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");  // the arch fence is not a compiler barrier (rule 27)
    return *sync_ptr(w);
}

// Debug words: (s, l, phase) of the last blocking wait entered, for post-mortem reads after a hang.
FORCE_INLINE void dbg_mark(uint32_t g, uint32_t phase) {
    TRD("B", g, phase);
    *sync_ptr(S_DBG) = g;
    *sync_ptr(S_DBG + 1) = phase;
}

// Multicast NoC address of the rectangle (x0, y0)-(x1, y1) given in NoC-0 virtual coordinates (start = min
// corner). NoC 1 addresses the rectangle end-first (deepseek_v3_b1 unified_kernels/mcast.hpp form).
FORCE_INLINE uint64_t mcast_addr(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t addr) {
    if (noc_index == 0) {
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(0, x0), DYNAMIC_NOC_Y(0, y0), DYNAMIC_NOC_X(0, x1), DYNAMIC_NOC_Y(0, y1), addr);
    }
    return NOC_MULTICAST_ADDR(
        DYNAMIC_NOC_X(1, x1), DYNAMIC_NOC_Y(1, y1), DYNAMIC_NOC_X(1, x0), DYNAMIC_NOC_Y(1, y0), addr);
}

}  // namespace mk
