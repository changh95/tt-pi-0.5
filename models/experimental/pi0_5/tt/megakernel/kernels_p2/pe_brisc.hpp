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

struct BState {
    uint32_t ibase = 0;  // in0-feeder credits consumed so far (cumulative)
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
                    const uint32_t arg = (p < 36 ? PA_KC : PA_VC) + o.layer;
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

PE_OS void feed_in0(const Op& o, const Core& c, const Lay& L, uint32_t A, uint32_t k, BState& st) {
    const uint32_t rp = o.rpb, r0 = c.x * rp, kt = o.kt, nr = NCOL;
    const uint32_t s0 = A + L.stage, dst = A + L.in0;
    const uint32_t x0 = c.rowx0, x1 = c.rowx1, y = c.rowy;
    if (o.mode == MM_R) {
        const auto src = bdram(in0_arg(o), T16);
        const uint32_t tiles = rp * kt, nch = (tiles + IN0_CHUNK - 1) / IN0_CHUNK;
        auto read_chunk = [&](uint32_t ch) {
            const uint32_t t0 = ch * IN0_CHUNK, n = tiles - t0 < IN0_CHUNK ? tiles - t0 : IN0_CHUNK;
            const uint32_t buf = s0 + (ch & 1) * IN0_CHUNK * T16;
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t t = t0 + i;
                noc_async_read_page((r0 + t / kt) * kt + t % kt, src, buf + i * T16);
            }
            return n;
        };
        ps_dbg(o.op, 2, st.ibase + nr);
        PWAIT_GE(PS_I_RDY, st.ibase + nr, "PICR");
        uint32_t n = read_chunk(0);
        noc_async_read_barrier();
        for (uint32_t ch = 0; ch < nch; ++ch) {
            noc_async_write_multicast(s0 + (ch & 1) * IN0_CHUNK * T16, pmcast(x0, y, x1, y, dst + ch * IN0_CHUNK * T16),
                                      n * T16, nr, false);
            uint32_t n_next = 0;
            if (ch + 1 < nch) {
                n_next = read_chunk(ch + 1);
                noc_async_read_barrier();
            }
            noc_async_writes_flushed();
            n = n_next;
        }
        noc_async_write_barrier();  // every chunk landed before the flag
        mcast_flag(x0, y, x1, y, nr, 0, 0, 0, PS_SRC_I, PS_I_VAL, (k << 16) + 1);
        st.ibase += nr;
    } else {
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
            PWAIT_GE(PS_I_RDY, st.ibase + nr * (j + 1), "PISR");
            mcast_flag(x0, y, x1, y, nr, s0 + (j & 1) * pb, dst + (j % IN0_SLOTS) * pb, pb, PS_SRC_I, PS_I_VAL,
                       (k << 16) + j + 1);
            if (j + 1 < nq) {
                read_page(j + 1);
                noc_async_read_barrier();
            }
        }
        st.ibase += nr * nq;
    }
}

// ---------------------------------------------------------------- norm / attention / embedding outputs
PE_OS void norm_write(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    cb_point(P_O16, A + L.o16, 8, T16);
    const auto out = bdram(o.what >= W_VRMS1 ? PA_XN_V : PA_XN_S, T16);
    for (uint32_t r = c.lin; r < o.items; r += NCORES) {
        for (uint32_t t = 0; t < o.nk; ++t) {
            cb_wait_front(P_O16, 1);
            noc_async_write_page(r * o.nk + t, out, get_read_ptr(P_O16));
            noc_async_writes_flushed();
            cb_pop_front(P_O16, 1);
        }
    }
}

PE_OS void attn_write(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    const bool v = o.what == W_VATTN;
    const uint32_t dh = v ? V_DH : S_DH;
    cb_point(P_O16, A + L.o16, 2 * dh, T16);
    const auto out = bdram(v ? PA_CTX_V : PA_CTX_S, T16);
    for (uint32_t it = c.lin; it < o.items; it += NCORES) {
        for (uint32_t hh = 0; hh < 2; ++hh) {
            uint32_t base;
            if (v) {
                const uint32_t r = it / 4, h = 2 * (it % 4) + hh;
                base = r * V_D + h * V_DH;
            } else {
                const uint32_t img = it / 64, h = (it / 4) % S_HEADS, qb = it % 4;
                base = (img * S_IMG + qb * 2 + hh) * S_DCTX + h * S_DH;
            }
            cb_wait_front(P_O16, dh);
            const uint32_t l1 = get_read_ptr(P_O16);
            for (uint32_t d = 0; d < dh; ++d) {
                noc_async_write_page(base + d, out, l1 + d * T16);
            }
            noc_async_writes_flushed();
            cb_pop_front(P_O16, dh);
        }
    }
}

PE_OS void embed_write(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    cb_point(P_O32, A + L.o32, 16, T32);
    const auto out = bdram(PA_X_V, T32);
    for (uint32_t it = c.lin; it < o.items; it += NCORES) {
        const uint32_t row = S_M + it / 4, cq = it % 4;
        for (uint32_t t = 0; t < 16; ++t) {
            cb_wait_front(P_O32, 1);
            noc_async_write_page(row * V_D + cq * 16 + t, out, get_read_ptr(P_O32));
            noc_async_writes_flushed();
            cb_pop_front(P_O32, 1);
        }
    }
}

// ---------------------------------------------------------------- barriers
PE_OS void boot(const Core& c) {
    for (uint32_t w = 0; w < PS_N; ++w) {
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
}

PE_OS void op_end(const Core& c, uint32_t k) {
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    ps_dbg(k, 8);
    PWAIT_GE(PS_NCDONE, k + 1, "PBNC");
    inc_word(c.hubx, c.huby, PS_ARR);
    if (c.is_hub()) {
        PWAIT_GE(PS_ARR, NCORES * (k + 1), "PHUB");
        *ps_ptr(PS_GO) = k + 1;
        mcast_flag(c.gx0, c.gy0, c.gx1, c.gy1, NCORES - 1, 0, 0, 0, PS_SRC_GO, PS_GO, k + 1);
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
    for (uint32_t op = first; op < stop; ++op, ++k) {
        const Op o = describe(op);
        const Lay L = layout(o);
        if (L.end > need) {
            need = L.end;
        }
        switch (o.kind) {
            case K_MM:
                if (mm_compute(o, c)) {
                    mm_write(o, c, L, A);
                } else if (mm_ifeeder(o, c)) {
                    feed_in0(o, c, L, A, k, st);
                }
                break;
            case K_NORM:
                if (c.lin < o.items) {
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
        op_end(c, k);
    }
    // diagnostics (PA_DIAG page = this core): arena bounds as seen here, the largest op layout, ops executed
    volatile tt_l1_ptr uint32_t* dg = ps_ptr(PS_N);
    dg[0] = 0x50455047u;  // "PEPG"
    dg[1] = A;
    dg[2] = arena_hi();
    dg[3] = ARENA_BYTES;
    dg[4] = need;
    dg[5] = k;
    dg[6] = first;
    dg[7] = stop;
    noc_async_write_page(c.lin, bdram(PA_DIAG, 64), ps_addr(PS_N), 32);
    noc_async_write_barrier();
}

}  // namespace pe
