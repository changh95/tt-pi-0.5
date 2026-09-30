// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefix engine: data-movement helpers shared by the BRISC and NCRISC kernels (sync words, CB re-pointing, the arena,
// accessors, the multicast primitives, the global barrier).
#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/waypoint.h"
#include "pe_common.hpp"
#ifdef MK_TRACE
#include "api/debug/dprint.h"
#define PTRD(tag, a, b) DPRINT(tag " {} {}\n", (uint32_t)(a), (uint32_t)(b))
#else
#define PTRD(tag, a, b) \
    do {                \
    } while (0)
#endif

namespace pe {

FORCE_INLINE uint32_t rta(uint32_t i) { return get_arg_val<uint32_t>(i); }
FORCE_INLINE uint32_t cra(uint32_t i) { return get_common_arg_val<uint32_t>(i); }

// free pages from the producer's view: capacity - (received - acked); `received` is this RISC's own counter
FORCE_INLINE uint32_t cb_free(uint32_t cb) {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
    const uint16_t received = (uint16_t)get_cb_tiles_received_ptr(cb)[0];
    const uint16_t acked = (uint16_t)reg_read((uintptr_t)get_cb_tiles_acked_ptr(cb));
    return (uint16_t)(get_local_cb_interface(cb).fifo_num_pages - (uint16_t)(received - acked));
}

FORCE_INLINE uint32_t ps_addr(uint32_t w) { return cbase(P_SYNC) + w * PSTRIDE; }
FORCE_INLINE volatile tt_l1_ptr uint32_t* ps_ptr(uint32_t w) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ps_addr(w));
}
FORCE_INLINE uint32_t ps_read(uint32_t w) {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
    return *ps_ptr(w);
}
// debug words of the last blocking step entered (post-mortem after a hang): NCRISC 10..13, BRISC 14..15
FORCE_INLINE void ps_dbg(uint32_t op, uint32_t phase, uint32_t a = 0, uint32_t b = 0) {
    if (noc_index == 0) {
        *ps_ptr(PS_DBG) = op;
        *ps_ptr(PS_DBG + 1) = phase;
        *ps_ptr(PS_DBG + 2) = a;
        *ps_ptr(PS_DBG + 3) = b;
    } else {
        *ps_ptr(PS_DBG + 4) = op;
        *ps_ptr(PS_DBG + 5) = phase;
    }
}

#define PWAIT_GE(word, value, tag)             \
    do {                                       \
        WAYPOINT(tag);                         \
        const uint32_t _v = (value);           \
        while (ps_read(word) < _v) {           \
        }                                      \
    } while (0)

// Multicast NoC address of (x0, y0)-(x1, y1) in NoC-0 virtual coordinates (min corner first); NoC 1 is end-first.
FORCE_INLINE uint64_t pmcast(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t addr) {
    if (noc_index == 0) {
        return NOC_MULTICAST_ADDR(DYNAMIC_NOC_X(0, x0), DYNAMIC_NOC_Y(0, y0), DYNAMIC_NOC_X(0, x1), DYNAMIC_NOC_Y(0, y1),
                                  addr);
    }
    return NOC_MULTICAST_ADDR(DYNAMIC_NOC_X(1, x1), DYNAMIC_NOC_Y(1, y1), DYNAMIC_NOC_X(1, x0), DYNAMIC_NOC_Y(1, y0), addr);
}

FORCE_INLINE void inc_word(uint32_t x, uint32_t y, uint32_t w, uint32_t n = 1) {
    noc_semaphore_inc(get_noc_addr(x, y, ps_addr(w)), n);
}

// data (optional) + flag multicast, linked on one path: receivers see the flag only after the data
FORCE_INLINE void mcast_flag(uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1, uint32_t ndst, uint32_t src,
                             uint32_t dst, uint32_t bytes, uint32_t src_word, uint32_t flag_word, uint32_t value) {
    if (bytes) {
        noc_async_write_multicast(src, pmcast(x0, y0, x1, y1, dst), bytes, ndst, true);
    }
    *ps_ptr(src_word) = value;
    noc_semaphore_set_multicast(ps_addr(src_word), pmcast(x0, y0, x1, y1, ps_addr(flag_word)), ndst);
    noc_async_writes_flushed();
}

// ---------------------------------------------------------------- core identity
struct Core {
    uint32_t x, y, lin, wfx, wfy, ifx, ify, hubx, huby, colx, rowy, rowx0, rowx1, gx0, gy0, gx1, gy1;
    uint32_t nocy[GRID_Y];
    bool is_hub() const { return x == HUB_X && y == HUB_Y; }
};

PE_OS void load_core(Core& c) {
    c.x = rta(PR_X);
    c.y = rta(PR_Y);
    c.lin = rta(PR_LIN);
    c.wfx = rta(PR_WFX);
    c.wfy = rta(PR_WFY);
    c.ifx = rta(PR_IFX);
    c.ify = rta(PR_IFY);
    c.hubx = rta(PR_HUBX);
    c.huby = rta(PR_HUBY);
    c.colx = rta(PR_COLX);
    c.rowy = rta(PR_ROWY);
    c.rowx0 = rta(PR_ROWX0);
    c.rowx1 = rta(PR_ROWX1);
    c.gx0 = rta(PR_GX0);
    c.gy0 = rta(PR_GY0);
    c.gx1 = rta(PR_GX1);
    c.gy1 = rta(PR_GY1);
    for (uint32_t i = 0; i < GRID_Y; ++i) {
        c.nocy[i] = rta(PR_NOCY0 + i);
    }
}

// roles in a matmul op
FORCE_INLINE bool mm_compute(const Op& o, const Core& c) { return c.y < o.nb; }
FORCE_INLINE bool mm_wfeeder(const Op& o, const Core& c) { return c.y == WF_Y; }
FORCE_INLINE bool mm_ifeeder(const Op& o, const Core& c) { return c.y == IF_Y && c.x < o.nb; }

}  // namespace pe
