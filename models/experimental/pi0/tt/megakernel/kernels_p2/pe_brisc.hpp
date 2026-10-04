// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefix engine, BRISC (NoC 1): every output of every op, the in0 band feeders, the boot and the global barriers.
//   matmul, compute core : output pairs -> DRAM (or the L1 K / V caches) in pair order
//   matmul, (b, IF_Y)    : the in0 feeder of band b (DRAM tiles -> multicast along the band's grid row)
//   norm / attention / embedding: the item outputs
//   barrier after op k   : writes acknowledged, the NCRISC's PS_NCDONE >= k + 1, arrive at the hub, wait for go;
//                          the hub (10, 9) waits for all 110 arrivals and multicasts go = k + 1
#pragma once

#include "pe_dm.hpp"

namespace pe {

constexpr auto bacc_dram = TensorAccessorArgs<PE_CT0 + PT_ACC>();
constexpr auto bacc_l1 = TensorAccessorArgs<bacc_dram.next_compile_time_args_offset()>();

FORCE_INLINE auto bdram(uint32_t arg, uint32_t page) { return TensorAccessor(bacc_dram, cra(arg), page); }
NOINL void wrp(uint32_t arg, uint32_t z, uint32_t page, uint32_t l1) { noc_async_write_page(page, bdram(arg, z), l1); }

struct BState {
    uint32_t ibase[NCOL] = {};  // in0 feeder: pages each receiver column was sent
};

// ---------------------------------------------------------------- matmul outputs
PE_OS void write_pair(const Op& o, uint32_t p, uint32_t r0) {
    const uint32_t ocb = out_cb(o, p), n = out_tiles(o);
    const uint32_t tb = ocb == P_O16 ? T16 : (ocb == P_O8 ? T8 : T32);
    cb_wait_front(ocb, n);
    const uint32_t l1 = get_read_ptr(ocb);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t r = o.epi == E_GEGLU ? i : i / 2, t = o.epi == E_GEGLU ? 0 : i % 2;
        const uint32_t row = r0 + r, col = pair_col(o, p, t), src = l1 + i * tb;
        switch (o.what) {
            case W_PATCH:
            case W_SO:
            case W_SFC2: noc_async_write_page(row * S_D + col, bdram(PA_X_S, T32), src); break;
            case W_SQKV: noc_async_write_page(row * S_NQKV + col, bdram(PA_QKV_S, T16), src); break;
            case W_SFC1: noc_async_write_page(row * S_I + col, bdram(PA_H_S, T16), src); break;
            case W_PROJ:
            case W_VO:
            case W_VDOWN: noc_async_write_page(row * V_D + col, bdram(PA_X_V, T32), src); break;
            case W_VGU: noc_async_write_page(row * V_I + p, bdram(PA_H_V, T8), src); break;
            case W_VQKV: {
                if (p < 32) {
                    const uint32_t h = p / 4, d = p % 4 + 4 * t;
                    noc_async_write_page((h * MT + row) * V_DH + d, bdram(PA_Q_V, T16), src);
                } else if (row < PTV) {  // pad query rows never become keys
                    const uint32_t d = (p - 32) % 4 + 4 * t;
                    const uint32_t arg = (p < 36 ? mk::C_K_ADDR : mk::C_V_ADDR) + o.layer;
                    noc_async_write_page(row * V_DH + d, TensorAccessor(bacc_l1, cra(arg), T8), src);
                }
                break;
            }
            default: break;
        }
    }
    noc_async_writes_flushed();
    cb_pop_front(ocb, n);
}

PE_OS void mm_write(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    const uint32_t rp = o.rpb, np = mm_pairs(o, c.x), p0 = mm_pair0(o, c.x);
    cb_point(P_O16, A + L.o16, 4 * rp, T16);
    cb_point(P_O8, A + L.o8, 4 * rp, T8);
    cb_point(P_O32, A + L.o32, 4 * rp, T32);
    for (uint32_t s = 0; s < np; ++s) {
        ps_dbg(o.op, 1, s);
        write_pair(o, p0 + s, c.y * rp);
    }
}

// in0 feeder of band b = c.x: rows [b * rpb, (b + 1) * rpb) of the op's in0 tensor
FORCE_INLINE uint32_t in0_arg(const Op& o) {
    switch (o.what) {
        case W_PATCH: return PA_IM2COL;
        case W_SQKV:
        case W_SFC1:
        case W_PROJ: return PA_XN_S;
        case W_SO: return PA_CTX_S;
        case W_SFC2: return PA_H_S;
        case W_VQKV:
        case W_VGU: return PA_XN_V;
        case W_VO: return PA_CTX_V;
        default: return PA_H_V;  // W_VDOWN
    }
}

// mode R in0, distributed: the band's compute core in column q reads K piece q (all rp rows) from DRAM into its own
// resident band and multicasts it along its row, then flags PS_IV0 + q. One reader per row (b, 9) capped the 8 bands
// at ~90 GB/s together (row-9 links; 27 GB/s for one feeder alone, 11 each for eight: FC2 in0 51 us of reads).
#ifdef PE_IN0_LAG
constexpr uint32_t IN0_LAG = PE_IN0_LAG;
#else
constexpr uint32_t IN0_LAG = 1;
#endif
PE_OS void in0_source(const Op& o, const Core& c, const Lay& L, uint32_t A, uint32_t k) {
    const uint32_t rp = o.rpb, kt = o.kt, pc = o.piece, q = c.x, r0 = c.y * rp;
    const auto src = bdram(in0_arg(o), T16);
    const uint32_t dst = A + L.in0, y = c.nocy[c.y];
    // every source reads its piece at once; only the multicasts are staggered (piece q after piece q - IN0_LAG
    // landed), so the row's pieces arrive in K order without serialising the DRAM reads (arm PE_IN0_READ_STAGGER:
    // the read waits too, as before)
#ifdef PE_IN0_READ_STAGGER
    if (q >= IN0_LAG) {
        PWAIT_GE(PS_IV0 + q - IN0_LAG, (k << 16) + 1, "PIVS");
    }
#endif
    for (uint32_t r = 0; r < rp; ++r) {
        for (uint32_t kk = 0; kk < pc; ++kk) {
            noc_async_read_page((r0 + r) * kt + q * pc + kk, src, dst + (r * kt + q * pc + kk) * T16);
        }
    }
    noc_async_read_barrier();
#ifndef PE_IN0_READ_STAGGER
    if (q >= IN0_LAG) {
        PWAIT_GE(PS_IV0 + q - IN0_LAG, (k << 16) + 1, "PIVS");
    }
#endif
    for (uint32_t r = 0; r < rp; ++r) {
        const uint32_t a = dst + (r * kt + q * pc) * T16;
        noc_async_write_multicast(a, pmcast(c.rowx0, y, c.rowx1, y, a), pc * T16, NCOL - 1, false);
    }
    noc_async_write_barrier();  // landed on every core of the row before the flag
    *ps_ptr(PS_IV0 + q) = (k << 16) + 1;
    mcast_flag(c.rowx0, y, c.rowx1, y, NCOL - 1, 0, 0, 0, PS_SRC_I, PS_IV0 + q, (k << 16) + 1);
}

PE_OS void feed_in0(const Op& o, const Core& c, const Lay& L, uint32_t A, uint32_t k, BState& st) {
    const uint32_t rp = o.rpb, r0 = c.x * rp, kt = o.kt, nr = NCOL;
    const uint32_t s0 = A + L.stage, dst = A + L.in0;
    const uint32_t x0 = c.rowx0, x1 = c.rowx1, y = c.rowy;
    const auto src = bdram(in0_arg(o), T8);
    const uint32_t pc = o.piece, nq = kt / pc, pt = rp * pc, pb = pt * T8;
    auto read_page = [&](uint32_t j) {
        const uint32_t buf = s0 + (j & 1) * pb;
        for (uint32_t r = 0; r < rp; ++r) {
            for (uint32_t kk = 0; kk < pc; ++kk) {
                noc_async_read_page((r0 + r) * kt + j * pc + kk, src, buf + (r * pc + kk) * T8);
            }
        }
    };
    read_page(0);
    noc_async_read_barrier();
    for (uint32_t j = 0; j < nq; ++j) {
        ps_dbg(o.op, 3, j);
        WAYPOINT("PISR");
        wait_credits(PS_I_RDY0, nr, st.ibase, j + 1);
        mcast_flag(
            x0, y, x1, y, nr, s0 + (j & 1) * pb, dst + (j % IN0_SLOTS) * pb, pb, PS_SRC_I, PS_I_VAL, (k << 16) + j + 1);
        if (j + 1 < nq) {
            read_page(j + 1);
            noc_async_read_barrier();
        }
    }
    for (uint32_t r = 0; r < nr; ++r) {
        st.ibase[r] += nq;
    }
}

// ---------------------------------------------------------------- norm / attention / embedding outputs
PE_OS void norm_write(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    cb_point(P_O16, A + L.o16, NORM_O16_PAGES, T16);
    const uint32_t out_a = o.what >= W_VRMS1 ? PA_XN_V : PA_XN_S, out_z = T16;
    const uint32_t ncg = o.nkind == N_LN ? NCG_S : NCG_VLM, w = o.nk / ncg;
    for (uint32_t it = c.lin; it < o.items; it += NNORM) {
        const uint32_t r = it / ncg, c0 = (it % ncg) * w;
        {  // this group's partial statistics -> slot (it % ncg) of P_R on the row's ncg item cores (item == core)
            cb_point(P_O32, A + L.o32, 2, T32);
            cb_wait_front(P_O32, 2);
            // NORM2: round 1 uses the second half of P_R and counts in PS_NR's high half (a peer still in round 0
            // reads the first half and waits on the low half)
            const bool hi = NORM2 && it >= NNORM;
            const uint32_t slot = (hi ? ncg : 0) + it % ncg;
            const uint32_t src = get_read_ptr(P_O32), row0 = it - it % ncg, dst = A + L.r + slot * 2 * T32;
            for (uint32_t g = 0; g < ncg; ++g) {
                const uint32_t pl = NORM2 ? (row0 + g) % NNORM : row0 + g;
                noc_async_write(src, get_noc_addr(cra(PA_NOCX0 + pl % NCOL), c.nocy[pl / NCOL], dst), 2 * T32);
            }
            noc_async_write_barrier();  // landed before the counts
            for (uint32_t g = 0; g < ncg; ++g) {
                const uint32_t pl = NORM2 ? (row0 + g) % NNORM : row0 + g;
                inc_word(cra(PA_NOCX0 + pl % NCOL), c.nocy[pl / NCOL], PS_NR, hi ? 0x10000u : 1u);
            }
            cb_pop_front(P_O32, 2);
        }
        for (uint32_t t0 = 0; t0 < w; t0 += 4) {  // 4 tiles per flush (the TRISC ring holds 8)
            const uint32_t n = w - t0 < 4 ? w - t0 : 4;
            cb_wait_front(P_O16, n);
            const uint32_t l1 = get_read_ptr(P_O16);
            for (uint32_t t = 0; t < n; ++t) {
                wrp(out_a, out_z, r * o.nk + c0 + t0 + t, l1 + t * T16);
            }
            noc_async_writes_flushed();
            cb_pop_front(P_O16, n);
        }
    }
}

PE_OS void attn_write(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    const bool v = o.what == W_VATTN;
    const uint32_t dh = v ? V_DH : S_DH;
    cb_point(P_O16, A + L.o16, 2 * dh, T16);
    const uint32_t out_a = v ? PA_CTX_V : PA_CTX_S, out_z = T16;
    for (uint32_t it = c.lin; it < o.items; it += NCORES) {
        uint32_t img = 0, sh = 0, r0 = 0, nr = 2;
        if (!v) {
            s_attn_item(it, img, sh, r0, nr);
        }
        for (uint32_t hh = 0; hh < nr; ++hh) {
            uint32_t base;
            if (v) {
                const uint32_t r = it / 4, h = 2 * (it % 4) + hh;
                base = r * V_D + h * V_DH;
            } else {
                base = (r0 + hh) * S_DCTX + sh * S_DH;
            }
            cb_wait_front(P_O16, dh);
            const uint32_t l1 = get_read_ptr(P_O16);
            for (uint32_t d = 0; d < dh; ++d) {
                wrp(out_a, out_z, base + d, l1 + d * T16);
            }
            noc_async_writes_flushed();
            cb_pop_front(P_O16, dh);
        }
    }
}

PE_OS void embed_write(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    cb_point(P_O32, A + L.o32, 16, T32);
    const uint32_t out_a = PA_X_V, out_z = T32;
    for (uint32_t it = c.lin; it < o.items; it += NCORES) {
        const uint32_t row = S_M + it / 4, cq = it % 4;
        for (uint32_t t0 = 0; t0 < 16; t0 += 4) {
            cb_wait_front(P_O32, 4);
            const uint32_t l1 = get_read_ptr(P_O32);
            for (uint32_t t = 0; t < 4; ++t) {
                wrp(out_a, out_z, row * V_D + cq * 16 + t0 + t, l1 + t * T32);
            }
            noc_async_writes_flushed();
            cb_pop_front(P_O32, 4);
        }
    }
}

// ---------------------------------------------------------------- barriers
PE_OS void boot(const Core& c) {
    for (uint32_t w = 0; w < PS_N; ++w) {
        *ps_ptr(w) = 0;
    }
    for (uint32_t w = PS_IV0; w < PS_WORDS; ++w) {
        *ps_ptr(w) = 0;
    }
    const uint32_t sem_arr = get_semaphore(2), sem_go = get_semaphore(3);
    noc_semaphore_inc(get_noc_addr(c.hubx, c.huby, sem_arr), 1);
    if (c.is_hub()) {
        WAYPOINT("PBOT");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_arr), NCORES);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go) = 1;
        noc_semaphore_set_multicast(sem_go, pmcast(c.gx0, c.gy0, c.gx1, c.gy1, sem_go), NCORES - 1);
        noc_async_writes_flushed();
    }
    WAYPOINT("PBGO");
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go), 1);
    if (c.is_hub()) {  // boot stamp: record 4095
        volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ps_addr(PS_TSTAMP));
        ts[0] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        ts[1] = 0xFFFFFFFFu;
        ts[2] = 0xFFFFFFFFu;
        ts[3] = 0x424f4f54u;  // "BOOT"
        noc_async_write_page(4095, bdram(PA_TIMES, 64), ps_addr(PS_TSTAMP), 16);
        noc_async_writes_flushed();
    }
}

PE_OS void op_end(const Core& c, uint32_t k, uint32_t op) {
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    ps_dbg(k, 8);
    PWAIT_GE(PS_NCDONE, k + 1, "PBNC");
    inc_word(c.hubx, c.huby, PS_ARR);
    if (c.is_hub()) {
        PWAIT_GE(PS_ARR, NCORES * (k + 1), "PHUB");
        *ps_ptr(PS_GO) = k + 1;
        mcast_flag(c.gx0, c.gy0, c.gx1, c.gy1, NCORES - 1, 0, 0, 0, PS_SRC_GO, PS_GO, k + 1);
        // time stamp of op k's end (wall clock) -> PA_TIMES record k % 4096 (the flush above freed the staging)
        volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ps_addr(PS_TSTAMP));
        ts[0] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
        ts[1] = k;
        ts[2] = op;
        ts[3] = 0x54494d45u;  // "TIME"
        noc_async_write_page(k % 4096, bdram(PA_TIMES, 64), ps_addr(PS_TSTAMP), 16);
        noc_async_writes_flushed();
    }
    PWAIT_GE(PS_GO, k + 1, "PBGW");
}

PE_OS void run_brisc() {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
    Core c;
    load_core(c);
    const uint32_t first = cra(PA_OPFIRST), stop = cra(PA_DBGSTOP);
    boot(c);
    const uint32_t A = arena_lo();
    uint32_t need = 0;
    BState st;
    uint32_t k = 0;
    const uint32_t reps = cra(PA_REPS);
    for (uint32_t rep = 0; rep < reps; ++rep) {
        for (uint32_t op = first; op < stop; ++op, ++k) {
            Op o;
            Lay L;
            take_oplay(o, L, k);
            trace_mark(k, 8);
            if (L.end > need) {
                need = L.end;
            }
            switch (o.kind) {
                case K_MM:
                    if (mm_compute(o, c)) {
                        if (o.mode == MM_R && c.x < mm_nq(o)) {
                            in0_source(o, c, L, A, k);
                        }
                        mm_write(o, c, L, A);
                    } else if (mm_ifeeder(o, c) && o.mode == MM_S) {
#ifdef PE_DBG_IN0_DIRECT
                        if (o.mode == MM_S) {
                            break;
                        }
#endif
                        feed_in0(o, c, L, A, k, st);
                    }
                    break;
                case K_NORM:
                    if (c.lin < o.items && (!NORM2 || c.lin < NNORM)) {
                        norm_write(o, c, L, A);
                    }
                    break;
                case K_ATTN:
                    if (c.lin < o.items) {
                        attn_write(o, c, L, A);
                    }
                    break;
                case K_EMBED:
                    if (c.lin < o.items) {
                        embed_write(o, c, L, A);
                    }
                    break;
                default: break;
            }
            noc_async_write_barrier();
            trace_mark(k, 9);
            op_end(c, k, op);
        }
    }
    // diagnostics (PA_DIAG page = this core): arena bounds as seen here, the largest op layout, ops executed
    volatile tt_l1_ptr uint32_t* dg = ps_ptr(PS_DIAG);
    dg[0] = 0x50455047u;  // "PEPG"
    dg[1] = A;
    dg[2] = arena_hi();
    dg[3] = ARENA_BYTES;
    dg[4] = need;
    dg[5] = k;
    dg[6] = first;
    dg[7] = stop;
    noc_async_write_page(c.lin, bdram(PA_DIAG, 64), ps_addr(PS_DIAG), 64);
    noc_async_write_barrier();
}

}  // namespace pe
