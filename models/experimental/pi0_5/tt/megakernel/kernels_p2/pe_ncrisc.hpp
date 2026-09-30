// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefix engine, NCRISC (NoC 0): every input of every op.
//   matmul, compute core : receiver side of the in0 ring (band feeder) and the weight ring (column feeder) + the side
//                          tiles of every output pair (bias, residual, RoPE tables, position table) from DRAM
//   matmul, (x, WF_Y)    : the weight feeder of column x (bank-striped arena pages -> multicast down the column)
//   norm / attention / embedding: the item inputs from DRAM / the L1 K / V caches
// After its part of op k the NCRISC flushes, publishes PS_NCDONE = k + 1 (its BRISC arrives at the barrier only then)
// and waits for the barrier's go before op k + 1.
#pragma once

#include "pe_dm.hpp"

namespace pe {

constexpr auto acc_dram = TensorAccessorArgs<PE_CT0 + PT_ACC>();
constexpr auto acc_l1 = TensorAccessorArgs<acc_dram.next_compile_time_args_offset()>();

FORCE_INLINE auto dram(uint32_t arg, uint32_t page) { return TensorAccessor(acc_dram, cra(arg), page); }

struct NState {
    uint32_t wbase[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // weight feeder: pages each receiver row was sent so far
};

FORCE_INLINE uint32_t w_arena_arg(const Op& o) {
    if (o.what == W_PATCH) {
        return PA_WPATCH;
    }
    if (o.what == W_PROJ) {
        return PA_WPROJ;
    }
    return (o.what >= W_VRMS1 ? PA_WV : PA_WS) + o.layer;
}

FORCE_INLINE uint32_t bias_page(const Op& o, uint32_t n) {
    switch (o.what) {
        case W_SQKV: return o.layer * SV_N + SV_BQKV + n;
        case W_SO: return o.layer * SV_N + SV_BO + n;
        case W_SFC1: return o.layer * SV_N + SV_BFC1 + n;
        case W_SFC2: return o.layer * SV_N + SV_BFC2 + n;
        default: return GV_BPROJ + n;  // W_PROJ
    }
}

// the side tiles of pair p (TRISC order: S16 then S32 of pe_trisc.hpp mm_r / mm_s)
PE_OS void read_side(const Op& o, uint32_t p, uint32_t r0) {
    const uint32_t rp = o.rpb;
    const uint32_t n16 = side16(o, p), n32 = side32(o);
    if (n16) {
        cb_reserve_back(P_S16, n16);
        uint32_t a = get_write_ptr(P_S16);
        if (o.epi == E_POS) {
            const auto t = dram(PA_POS, T16);
            for (uint32_t r = 0; r < rp; ++r) {
                for (uint32_t u = 0; u < 2; ++u, a += T16) {
                    noc_async_read_page(((r0 + r) % S_IMG) * S_D + pair_col(o, p, u), t, a);
                }
            }
        } else if (o.epi == E_ROPE) {
            const bool q = p < 32;
            const auto c = dram(q ? PA_COSQ : PA_COSK, T16);
            const auto s = dram(q ? PA_SINQ : PA_SINK, T16);
            const uint32_t d0 = p % 4, d1 = d0 + 4;
            for (uint32_t r = 0; r < rp; ++r) {
                const uint32_t row = (r0 + r) * V_DH;
                noc_async_read_page(row + d0, c, a);
                noc_async_read_page(row + d0, s, a + T16);
                noc_async_read_page(row + d1, c, a + 2 * T16);
                noc_async_read_page(row + d1, s, a + 3 * T16);
                a += 4 * T16;
            }
        } else {  // bias (row-broadcast tiles)
            const auto b = dram(o.what == W_PROJ ? PA_GVEC : PA_SVEC, T16);
            noc_async_read_page(bias_page(o, pair_col(o, p, 0)), b, a);
            noc_async_read_page(bias_page(o, pair_col(o, p, 1)), b, a + T16);
        }
    }
    if (n32) {
        cb_reserve_back(P_S32, n32);
        uint32_t a = get_write_ptr(P_S32);
        const bool v = o.what >= W_VRMS1;
        const auto x = dram(v ? PA_X_V : PA_X_S, T32);
        const uint32_t nt = v ? V_D : S_D;
        for (uint32_t r = 0; r < rp; ++r) {
            for (uint32_t u = 0; u < 2; ++u, a += T32) {
                noc_async_read_page((r0 + r) * nt + pair_col(o, p, u), x, a);
            }
        }
    }
    noc_async_read_barrier();
    if (n16) {
        cb_push_back(P_S16, n16);
    }
    if (n32) {
        cb_push_back(P_S32, n32);
    }
}

// compute core: in0 ring + weight ring receiver, side tiles
PE_OS void mm_receive(const Op& o, const Core& c, const Lay& L, uint32_t A, uint32_t k) {
    const uint32_t rp = o.rpb, r0 = c.y * rp, nq = mm_nq(o);
    const uint32_t np = mm_pairs(o, c.x), p0 = mm_pair0(o, c.x);
    const uint32_t wcb = o.wbf16 ? P_W16 : P_W8, pt = mm_page_tiles(o), wpages = np * nq;
    cb_point(wcb, A + L.w, L.w_slots * pt, mm_wtile(o));
    const uint32_t icb = o.mode == MM_S ? P_IN08 : P_IN0;
    const uint32_t ipages = mm_in0_pages(o), itiles = mm_in0_page_tiles(o);
    if (o.mode == MM_S) {
        cb_point(P_IN08, A + L.in0, IN0_SLOTS * itiles, T8);
    } else {
        cb_point(P_IN0, A + L.in0, itiles, T16);
    }
    cb_point(P_S16, A + L.s16, 24, T16);
    cb_point(P_S32, A + L.s32, 4 * rp, T32);
    const uint32_t spairs = (side16(o, 0) | side32(o) | side16(o, 32)) ? np : 0;
    const uint32_t fl = k << 16;
    uint32_t ic = 0, ip = 0, wc = 0, wp = 0, sp = 0;
    ps_dbg(k, 1, 0, 0);
    while (ip < ipages || wp < wpages || sp < spairs) {
        WAYPOINT("PRCV");
#ifdef PE_DBG_IN0_DIRECT
        if (o.mode == MM_S) {  // debug arm: every receiver reads its own in0 K block from DRAM (no band feeder)
            if (ip < ipages && cb_free(icb) >= itiles) {
                cb_reserve_back(icb, itiles);
                const uint32_t a = get_write_ptr(icb);
                const auto src = dram(PA_H_V, T8);
                for (uint32_t r = 0; r < rp; ++r) {
                    for (uint32_t kk = 0; kk < o.piece; ++kk) {
                        noc_async_read_page((r0 + r) * o.kt + ip * o.piece + kk, src, a + (r * o.piece + kk) * T8);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(icb, itiles);
                ++ip;
                ic = ip;
            }
        } else
#endif
        {
        if (ic < ipages && cb_free(icb) >= (ic - ip + 1) * itiles) {
            inc_word(c.ifx, c.ify, PS_I_RDY0 + c.x);
            ++ic;
        }
        if (ip < ic && ps_read(PS_I_VAL) >= fl + ip + 1) {
            cb_push_back(icb, itiles);
            ++ip;
        }
        }
        if (wc < wpages && cb_free(wcb) >= (wc - wp + 1) * pt) {
            inc_word(c.wfx, c.wfy, PS_W_RDY0 + c.y);
            ++wc;
        }
        if (wp < wc && ps_read(PS_W_VAL) >= fl + wp + 1) {
            cb_push_back(wcb, pt);
            ++wp;
        }
        if (sp < spairs) {
            const uint32_t n16 = side16(o, p0 + sp), n32 = side32(o);
            if (cb_free(P_S16) >= n16 && cb_free(P_S32) >= n32) {
                if (n16 | n32) {
                    read_side(o, p0 + sp, r0);
                }
                ++sp;
            }
        }
        ps_dbg(k, 2, (ip << 16) | wp, sp);
    }
}

// weight feeder of column x: this column's pages of the op, bank-striped in the layer arena
PE_OS void feed_w(const Op& o, const Core& c, const Lay& L, uint32_t A, uint32_t k, NState& st) {
    const uint32_t nq = mm_nq(o), np = mm_pairs(o, c.x), pages = np * nq, pb = mm_page_bytes(o), nr = o.nb;
    if (pages == 0) {
        return;
    }
    const uint32_t warena = cra(w_arena_arg(o)), off = mm_arena_off(o), g0 = mm_pair0(o, c.x) * nq;
    const uint32_t s0 = A + L.stage, dst = A + L.w;
    const uint32_t y0 = c.nocy[0], y1 = c.nocy[nr - 1];
    auto issue = [&](uint32_t j) {
        const uint32_t g = g0 + j;
#ifdef PE_DBG_NO_WREAD
        if (j >= 2) {
            return;  // timing arm: multicast the staging buffers' stale content (no DRAM read after the first two)
        }
#endif
        noc_async_read_set_trid(1 + (j & 1));
        noc_async_read(get_noc_addr_from_bank_id<true>(g % N_BANKS, warena + off + (g / N_BANKS) * pb), s0 + (j & 1) * pb,
                       pb);
    };
    issue(0);
    if (pages > 1) {
        issue(1);
    }
    for (uint32_t j = 0; j < pages; ++j) {
        WAYPOINT("PWRD");
        while (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + (j & 1))) {
        }
        invalidate_l1_cache();
        ps_dbg(k, 3, j, st.wbase[0] + j + 1);
        WAYPOINT("PWCR");
        wait_credits(PS_W_RDY0, nr, st.wbase, j + 1);
        mcast_flag(c.colx, y0, c.colx, y1, nr, s0 + (j & 1) * pb, dst + (j % L.w_slots) * pb, pb, PS_SRC_W, PS_W_VAL,
                   (k << 16) + j + 1);
        if (j + 2 < pages) {
            issue(j + 2);
        }
    }
    noc_async_read_set_trid(0);
    for (uint32_t r = 0; r < nr; ++r) {
        st.wbase[r] += pages;
    }
}

PE_OS void read_consts(uint32_t A, const Lay& L) {
    cb_point(P_CONST, A + L.cst, PC_N, T16);
    cb_reserve_back(P_CONST, PC_N);
    const auto t = dram(PA_CONST, T16);
    for (uint32_t i = 0; i < PC_N; ++i) {
        noc_async_read_page(i, t, get_write_ptr(P_CONST) + i * T16);
    }
    noc_async_read_barrier();
    cb_push_back(P_CONST, PC_N);
}

PE_OS void norm_read(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    const bool ln = o.nkind == N_LN;
    const uint32_t nk = o.nk;
    cb_point(P_S32, A + L.x32, nk, T32);
    cb_point(P_S16, A + L.s16, 2 * nk, T16);
    read_consts(A, L);
    uint32_t gw = 0, gb = 0, garg = PA_VVEC;
    switch (o.what) {
        case W_VRMS1: gw = o.layer * VV_N + VV_G1; break;
        case W_VRMS2: gw = o.layer * VV_N + VV_G2; break;
        case W_SLN1: garg = PA_SVEC; gw = o.layer * SV_N + SV_LN1W; gb = o.layer * SV_N + SV_LN1B; break;
        case W_SLN2: garg = PA_SVEC; gw = o.layer * SV_N + SV_LN2W; gb = o.layer * SV_N + SV_LN2B; break;
        default: garg = PA_GVEC; gw = GV_PLNW; gb = GV_PLNB; break;  // W_POSTLN
    }
    const auto x = dram(o.what >= W_VRMS1 ? PA_X_V : PA_X_S, T32);
    const auto g = dram(garg, T16);
    for (uint32_t r = c.lin; r < o.items; r += NCORES) {
        for (uint32_t t0 = 0; t0 < nk; t0 += 8) {  // x in chunks of 8 tiles (the TRISC's statistics start on the first)
            const uint32_t n = nk - t0 < 8 ? nk - t0 : 8;
            cb_reserve_back(P_S32, n);
            const uint32_t ax = get_write_ptr(P_S32);
            for (uint32_t t = 0; t < n; ++t) {
                noc_async_read_page(r * nk + t0 + t, x, ax + t * T32);
            }
            noc_async_read_barrier();
            cb_push_back(P_S32, n);
        }
        cb_reserve_back(P_S16, ln ? 2 * nk : nk);
        const uint32_t ag = get_write_ptr(P_S16);
        for (uint32_t t = 0; t < nk; ++t) {
            noc_async_read_page(gw + t, g, ag + t * T16);
            if (ln) {
                noc_async_read_page(gb + t, g, ag + (nk + t) * T16);
            }
        }
        noc_async_read_barrier();
        cb_push_back(P_S16, ln ? 2 * nk : nk);
    }
}

PE_OS void attn_read(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    const bool v = o.what == W_VATTN;
    const uint32_t dh = v ? V_DH : S_DH, nk = v ? PTV : S_NK;
    cb_point(P_Q, A + L.q, 2 * dh, T16);
    if (v) {
        cb_point(P_KV8, A + L.kv, 2 * PTV * V_DH, T8);
        cb_point(P_MSK, A + L.msk, PTV, T16);
    } else {
        cb_point(P_KV16, A + L.kv, 2 * S_NK * S_DH, T16);
    }
    read_consts(A, L);
    if (v) {
        cb_reserve_back(P_MSK, PTV);
        const auto m = dram(PA_VMASK, T16);
        for (uint32_t t = 0; t < PTV; ++t) {
            noc_async_read_page(t, m, get_write_ptr(P_MSK) + t * T16);
        }
        noc_async_read_barrier();
        cb_push_back(P_MSK, PTV);
    }
    for (uint32_t it = c.lin; it < o.items; it += NCORES) {
        cb_reserve_back(P_Q, 2 * dh);
        const uint32_t kvcb = v ? P_KV8 : P_KV16;
        cb_reserve_back(kvcb, 2 * nk * dh);
        const uint32_t aq = get_write_ptr(P_Q), akv = get_write_ptr(kvcb);
        if (v) {
            const uint32_t r = it / 4, hp = it % 4;
            const auto q = dram(PA_Q_V, T16);
            for (uint32_t hh = 0; hh < 2; ++hh) {
                const uint32_t h = 2 * hp + hh;
                for (uint32_t d = 0; d < V_DH; ++d) {
                    noc_async_read_page((h * MT + r) * V_DH + d, q, aq + (hh * V_DH + d) * T16);
                }
            }
            const auto kc = TensorAccessor(acc_l1, cra(PA_KC + o.layer), T8);
            const auto vc = TensorAccessor(acc_l1, cra(PA_VC + o.layer), T8);
            for (uint32_t t = 0; t < PTV * V_DH; ++t) {
                noc_async_read_page(t, kc, akv + t * T8);
                noc_async_read_page(t, vc, akv + (PTV * V_DH + t) * T8);
            }
        } else {
            const uint32_t img = it / 64, h = (it / 4) % S_HEADS, qb = it % 4;
            const auto qkv = dram(PA_QKV_S, T16);
            for (uint32_t rr = 0; rr < 2; ++rr) {
                const uint32_t r = img * S_IMG + qb * 2 + rr;
                for (uint32_t d = 0; d < S_DH; ++d) {
                    noc_async_read_page(r * S_NQKV + h * S_DH + d, qkv, aq + (rr * S_DH + d) * T16);
                }
            }
            for (uint32_t t = 0; t < S_NK; ++t) {
                const uint32_t r = img * S_IMG + t;
                for (uint32_t d = 0; d < S_DH; ++d) {
                    noc_async_read_page(r * S_NQKV + S_DCTX + h * S_DH + d, qkv, akv + (t * S_DH + d) * T16);
                    noc_async_read_page(r * S_NQKV + 2 * S_DCTX + h * S_DH + d, qkv,
                                        akv + ((S_NK + t) * S_DH + d) * T16);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(P_Q, 2 * dh);
        cb_push_back(kvcb, 2 * nk * dh);
    }
}

// language rows: 32 token rows x 16 tiles (one quarter of the 2048 columns) gathered row-major into the P_RM region,
// then TILIZED HERE by word copies into P_S16 (face layout: element (r, c) of a tile in face (r / 16) * 2 + c / 16 at
// (r % 16) * 16 + c % 16) -- the TRISC only scales. Pad row tiles have no NCRISC input (the TRISC packs zeros).
PE_OS void embed_read(const Op& o, const Core& c, const Lay& L, uint32_t A) {
    cb_point(P_S16, A + L.s16, 16, T16);
    read_consts(A, L);
    const uint32_t ids = A + L.tok;  // token ids
    const uint32_t rm = A + L.rm;    // row-major staging, 32 rows x 1024 B
    bool have_ids = false;
    const auto emb = dram(PA_EMB, V_D * 32 * 2);
    for (uint32_t it = c.lin; it < o.items; it += NCORES) {
        const uint32_t rt = it / 4, cq = it % 4;
        if (rt >= LT) {
            continue;
        }
        if (!have_ids) {
            const auto tk = dram(PA_TOK, NTOK * 4);
            noc_async_read_page(0, tk, ids);
            noc_async_read_barrier();
            invalidate_l1_cache();
            have_ids = true;
        }
        volatile tt_l1_ptr uint32_t* tok = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids);
        for (uint32_t i = 0; i < 32; ++i) {
            noc_async_read(emb.get_noc_addr(tok[rt * 32 + i], cq * 1024), rm + i * 1024, 1024);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        cb_reserve_back(P_S16, 16);
        const uint32_t dst = get_write_ptr(P_S16);
        for (uint32_t r = 0; r < 32; ++r) {
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rm + r * 1024);
            for (uint32_t t = 0; t < 16; ++t) {
                for (uint32_t h = 0; h < 2; ++h) {  // 16 bf16 = 8 words per face row
                    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        dst + t * T16 + (((r >> 4) * 2 + h) * 256 + (r & 15) * 16) * 2);
                    const uint32_t s0 = (t * 32 + h * 16) / 2;
                    for (uint32_t w = 0; w < 8; ++w) {
                        d[w] = src[s0 + w];
                    }
                }
            }
        }
        cb_push_back(P_S16, 16);
    }
}

// the TRISCs' copy of op k (pushed after op k's go: a TRISC can never run ahead of the barrier)
PE_OS void push_opd(const Op& o, const Lay& L, const Core& c, uint32_t A) {
    cb_reserve_back(P_OPD, 1);
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(P_OPD));
    for (uint32_t i = 0; i < OD_N; ++i) {
        d[i] = 0;
    }
    uint32_t kind = K_NOP;
    if (o.kind == K_MM) {
        kind = mm_compute(o, c) ? K_MM : K_NOP;
    } else if (o.kind != K_NOP && c.lin < o.items) {
        kind = o.kind;
    }
    d[OD_KIND] = kind;
    d[OD_WHAT] = o.what;
    d[OD_MODE] = o.mode;
    d[OD_RPB] = o.rpb;
    d[OD_KT] = o.kt;
    d[OD_PIECE] = o.piece;
    if (o.kind == K_MM) {
        d[OD_NP] = mm_pairs(o, c.x);
        d[OD_P0] = mm_pair0(o, c.x);
    }
    d[OD_EPI] = o.epi;
    d[OD_FID] = o.fid_hi;
    d[OD_WBF16] = o.wbf16;
    d[OD_NKIND] = o.nkind;
    d[OD_NK] = o.nk;
    d[OD_NITEMS] = c.lin < o.items ? (o.items - c.lin + NCORES - 1) / NCORES : 0;
    d[OD_IT0] = c.lin;
    d[OD_WSLOTS] = L.w_slots;
    const uint32_t off[OA_N] = {L.in0, L.w,  L.s16, L.s32, L.o16, L.o8, L.o32, L.part, L.x32, L.scr, L.r, L.q,
                                L.kv,  L.msk, L.ss, L.m,   L.op_, L.pm, L.pl,  L.d,    L.cst, L.rm,  L.tok};
    for (uint32_t i = 0; i < OA_N; ++i) {
        d[OD_A + i] = A + off[i];
    }
    cb_push_back(P_OPD, 1);
}

PE_OS void run_ncrisc() {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");
    Core c;
    load_core(c);
    const uint32_t first = cra(PA_OPFIRST), stop = cra(PA_DBGSTOP);
    // boot: the BRISC zeroed every sync word before the boot barrier (semaphores 2 / 3)
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3)), 1);
    const uint32_t A = arena_lo();
    NState st;
    uint32_t k = 0;
    const uint32_t reps = cra(PA_REPS);
    for (uint32_t rep = 0; rep < reps; ++rep)
    for (uint32_t op = first; op < stop; ++op, ++k) {
        const Op o = describe(op);
        const Lay L = layout(o);
        push_opd(o, L, c, A);
        switch (o.kind) {
            case K_MM:
                if (mm_compute(o, c)) {
                    mm_receive(o, c, L, A, k);
                } else if (mm_wfeeder(o, c)) {
                    feed_w(o, c, L, A, k, st);
                }
                break;
            case K_NORM:
                if (c.lin < o.items) {
                    norm_read(o, c, L, A);
                }
                break;
            case K_ATTN:
                if (c.lin < o.items) {
                    attn_read(o, c, L, A);
                }
                break;
            case K_EMBED:
                if (c.lin < o.items) {
                    embed_read(o, c, L, A);
                }
                break;
            default: break;
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        *ps_ptr(PS_NCDONE) = k + 1;
        ps_dbg(k, 9, 0, 0);
        PWAIT_GE(PS_GO, k + 1, "PNGO");
    }
}

}  // namespace pe
