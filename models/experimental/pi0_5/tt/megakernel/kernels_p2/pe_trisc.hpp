// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefix engine, TRISC: all arithmetic of SigLIP x2, the projector, the language embedding and the VLM prefill.
// Included by whole_trisc.cpp AFTER ../kernels/mk_trisc.cpp, whose LLK wrappers (mm_init_f / mm_f / load / fmul / ...
// pack_to, all noinline + noclone) it reuses. Every op starts with the op descriptor its NCRISC pushes into P_OPD after
// the op's barrier go (read with read_tile_value: UNPACK reads, mailboxes to MATH / PACK), so nothing here ever runs
// ahead of the op boundary, and no geometry code (describe / layout) is compiled into the three TRISC binaries.
#pragma once

#include "api/compute/tilize.h"
#include "pe_common.hpp"

namespace pe {

// ---------------------------------------------------------------- the block-matmul K loop (every prefix matmul)
// One output pair (rp x 2 tiles in DST 0 .. 2 rp - 1) accumulates nq weight pages of pc K tiles each; in0 tile of
// (row r, k) at q * pc + k + r * kt. The fidelity is a runtime switch on the MATH thread only (the unpacker and the
// packer code is shared: one copy of the loop per binary).
NOINL void mm_k(uint32_t in0, uint32_t w_cb, uint32_t rp, uint32_t kt, uint32_t pc, uint32_t nq, uint32_t fid_hi) {
    reconfig_data_format(w_cb, in0);  // matmul: SrcA <- in1, SrcB <- in0
    if (fid_hi) {
        MATH((llk_math_matmul_init<MathFidelity::HiFi2, MM_THROTTLE>(in0, w_cb, 0, 2, rp)));
    } else {
        MATH((llk_math_matmul_init<MathFidelity::LoFi, MM_THROTTLE>(in0, w_cb, 0, 2, rp)));
    }
    UNPACK((llk_unpack_AB_matmul_init(in0, w_cb, 0, 2, rp, kt)));
    const uint32_t pt = 2 * pc;
    for (uint32_t q = 0; q < nq; ++q) {
        cb_wait_front(w_cb, pt);
        for (uint32_t k = 0; k < pc; ++k) {
            UNPACK((llk_unpack_AB_matmul(in0, w_cb, q * pc + k, 2 * k, 2, rp, kt)));
            if (fid_hi) {
                MATH((llk_math_matmul<MathFidelity::HiFi2, MM_THROTTLE>(0, 2, rp)));
            } else {
                MATH((llk_math_matmul<MathFidelity::LoFi, MM_THROTTLE>(0, 2, rp)));
            }
        }
        cb_pop_front(w_cb, pt);
    }
}

struct TOp {
    uint32_t kind, what, mode, rpb, kt, piece, np, p0, epi, fid_hi, wbf16, nkind, nk, nitems, it0, w_slots;
    uint32_t a[OA_N];
};

PE_OS void read_opd(TOp& o) {
    cb_wait_front(P_OPD, 1);
    uint32_t* f = &o.kind;
    for (uint32_t i = 0; i < OD_A; ++i) {
        f[i] = read_tile_value(P_OPD, 0, i);
    }
    if (o.kind != K_NOP) {
        for (uint32_t i = 0; i < OA_N; ++i) {
            o.a[i] = read_tile_value(P_OPD, 0, OD_A + i);
        }
    }
    cb_pop_front(P_OPD, 1);
}

PE_OS void epilogue(const TOp& o, uint32_t p) {
    const uint32_t rp = o.rpb;
    if (o.epi == E_BIAS || o.epi == E_BIAS_GELU || o.epi == E_RES_BIAS) {
        load(P_S16, 0, 6);
        load(P_S16, 1, 7);
        for (uint32_t r = 0; r < rp; ++r) {
            fadd(2 * r, 6, 2 * r);
            fadd(2 * r + 1, 7, 2 * r + 1);
        }
        if (o.epi == E_BIAS_GELU) {
            gelu_tanh_tile_init();
            for (uint32_t i = 0; i < 2 * rp; ++i) {
                gelu_tanh_tile(i);
            }
        }
    } else if (o.epi == E_ROPE && p < 36) {
        // out0 = x0 cos0 + x1 sin0 ; out1 = x1 cos1 + x0 sin1 (signed sin tables: split-half rotate)
        for (uint32_t r = 0; r < rp; ++r) {
            const uint32_t a = 2 * r, b = 2 * r + 1;
            load(P_S16, 4 * r + 1, 6);  // sin0
            fmul(b, 6, 6);              // x1 sin0
            load(P_S16, 4 * r + 2, 7);  // cos1
            fmul(b, 7, b);              // x1 cos1
            load(P_S16, 4 * r + 3, 7);  // sin1
            fmul(a, 7, 7);              // x0 sin1
            fadd(b, 7, b);              // out1
            load(P_S16, 4 * r + 0, 7);  // cos0
            fmul(a, 7, a);              // x0 cos0
            fadd(a, 6, a);              // out0
        }
    } else if (o.epi == E_GEGLU) {
        gelu_tanh_tile_init();
        for (uint32_t r = 0; r < rp; ++r) {
            gelu_tanh_tile(2 * r + 1);
        }
        for (uint32_t r = 0; r < rp; ++r) {
            fmul(2 * r, 2 * r + 1, 2 * r);
        }
    }
}

// ================================================================ matmul, mode R (resident in0 band, N-outer)
PE_OS void mm_r(const TOp& o, uint32_t w_cb, uint32_t p0, uint32_t npairs) {
    const uint32_t rp = o.rpb, kt = o.kt;
    cb_wait_front(P_IN0, rp * kt);
    for (uint32_t s = 0; s < npairs; ++s) {
        const uint32_t p = p0 + s;
        const uint32_t n16 = side16(o, p), n32 = side32(o);
        if (n16) {
            cb_wait_front(P_S16, n16);
        }
        if (n32) {
            cb_wait_front(P_S32, n32);
        }
        tile_regs_acquire();
        if (o.epi == E_POS) {
            for (uint32_t i = 0; i < 2 * rp; ++i) {
                load(P_S16, i, i);
            }
        } else if (n32) {
            for (uint32_t i = 0; i < 2 * rp; ++i) {
                load(P_S32, i, i);
            }
        }
        mm_k(P_IN0, w_cb, rp, kt, o.piece, kt / o.piece, o.fid_hi);
        epilogue(o, p);
        tile_regs_commit();
        if (n16) {
            cb_pop_front(P_S16, n16);
        }
        if (n32) {
            cb_pop_front(P_S32, n32);
        }
        const uint32_t ocb = out_cb(o, p), nout = out_tiles(o);
        cb_reserve_back(ocb, nout);
        tile_regs_wait();
        if (o.epi == E_GEGLU) {
            for (uint32_t r = 0; r < rp; ++r) {
                pack_to(2 * r, ocb, r);
            }
        } else {
            for (uint32_t i = 0; i < nout; ++i) {
                pack_to(i, ocb, i);
            }
        }
        tile_regs_release();
        cb_push_back(ocb, nout);
    }
    cb_pop_front(P_IN0, rp * kt);
}

// ================================================================ matmul, mode S (streamed in0 K blocks, K-outer)
NOINL void mm_s(const TOp& o, uint32_t npairs) {
    const uint32_t rp = o.rpb, pc = o.piece, nq = o.kt / pc, pt = 2 * pc, nt = 2 * rp;
    for (uint32_t kb = 0; kb < nq; ++kb) {
        cb_wait_front(P_IN08, rp * pc);
        for (uint32_t s = 0; s < npairs; ++s) {
            const uint32_t src = kb == 0 ? P_S32 : P_PART;
            cb_wait_front(src, nt);
            tile_regs_acquire();
            for (uint32_t i = 0; i < nt; ++i) {
                load(src, i, i);
            }
            mm_k(P_IN08, P_W8, rp, pc, pc, 1, 0);
            tile_regs_commit();
            cb_pop_front(src, nt);
            const uint32_t dst = kb + 1 == nq ? P_O32 : P_PART;
            cb_reserve_back(dst, nt);
            tile_regs_wait();
            for (uint32_t i = 0; i < nt; ++i) {
                pack_to(i, dst, i);
            }
            tile_regs_release();
            cb_push_back(dst, nt);
        }
        cb_pop_front(P_IN08, rp * pc);
    }
}

PE_OS void run_mm(const TOp& o) {
    const uint32_t rp = o.rpb;
    if (o.mode == MM_S) {
        cb_point(P_IN08, o.a[OA_IN0], IN0_SLOTS * rp * o.piece, T8);
    } else {
        cb_point(P_IN0, o.a[OA_IN0], rp * o.kt, T16);
    }
    const uint32_t wcb = o.wbf16 ? P_W16 : P_W8;
    cb_point(wcb, o.a[OA_W], o.w_slots * 2 * o.piece, o.wbf16 ? T16 : T8);
    cb_point(P_S16, o.a[OA_S16], 24, T16);
    cb_point(P_S32, o.a[OA_S32], 4 * rp, T32);
    cb_point(P_O16, o.a[OA_O16], 4 * rp, T16);
    cb_point(P_O8, o.a[OA_O8], 4 * rp, T8);
    cb_point(P_O32, o.a[OA_O32], 4 * rp, T32);
    if (o.mode == MM_S) {
        cb_point(P_PART, o.a[OA_PART], 3 * 2 * rp, T32);
        mm_s(o, o.np);
    } else {
        mm_r(o, wcb, o.p0, o.np);
    }
}

// ================================================================ norms (one row tile per item)
PE_OS void row_reduce_to(uint32_t scale_bits, uint32_t eps_bits, bool rsq, uint32_t out_idx) {
    // SCR[0] (per-column partial sums) -> row sum in every column (HiFi4 @ ONES) * scale (+ eps, rsqrt) -> P_R[out_idx]
    cb_wait_front(P_SCR, 1);
    tile_regs_acquire();
    mm_init_f<MathFidelity::HiFi4>(P_SCR, P_CONST);
    mm_f<MathFidelity::HiFi4>(P_SCR, P_CONST, 0, PC_ONES, 0);
    fscale(0, scale_bits);
    if (rsq) {
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(0);
    }
    tile_regs_commit();
    cb_pop_front(P_SCR, 1);
    cb_reserve_back(P_R, 1);
    tile_regs_wait();
    pack_to(0, P_R, 0);
    tile_regs_release();
    cb_push_back(P_R, 1);
    (void)out_idx;
}

PE_OS void pack_scr0() {
    tile_regs_commit();
    cb_reserve_back(P_SCR, 1);
    tile_regs_wait();
    pack_to(0, P_SCR, 0);
    tile_regs_release();
    cb_push_back(P_SCR, 1);
}

// x in P_S32 (exact fp32 copies), gamma (and beta) in P_S16, out bf16 -> P_O16 (ring of 8)
PE_OS void norm_row(const TOp& o) {
    const uint32_t nk = o.nk;
    const bool ln = o.nkind == N_LN;
    cb_wait_front(P_S32, nk);
    cb_wait_front(P_S16, ln ? 2 * nk : nk);
    constexpr uint32_t ONE_OVER_2048 = 0x3A000000u;  // 1 / 2048
    constexpr uint32_t ONE_OVER_1152 = 0x3A638E39u;  // 1 / 1152 (fp32)
    const uint32_t inv_n = ln ? ONE_OVER_1152 : ONE_OVER_2048;
    const uint32_t eps = ln ? EPS_S : EPS_V;
    if (ln) {
        // mu (exact fp32 column partials, row sum by matmul)
        tile_regs_acquire();
        load(P_S32, 0, 0);
        for (uint32_t k = 1; k < nk; ++k) {
            load(P_S32, k, 1);
            fadd(0, 1, 0);
        }
        pack_scr0();
        row_reduce_to(inv_n, 0, false, 0);  // P_R[0] = mu
        cb_wait_front(P_R, 1);
        tile_regs_acquire();
        load(P_R, 0, 2);
        load(P_S32, 0, 0);
        fsub(0, 2, 0);
        fmul(0, 0, 0);
        for (uint32_t k = 1; k < nk; ++k) {
            load(P_S32, k, 1);
            fsub(1, 2, 1);
            fmul(1, 1, 1);
            fadd(0, 1, 0);
        }
        pack_scr0();
        row_reduce_to(inv_n, eps, true, 1);  // P_R[1] = rstd
        cb_wait_front(P_R, 2);
    } else {
        tile_regs_acquire();
        load(P_S32, 0, 0);
        fmul(0, 0, 0);
        for (uint32_t k = 1; k < nk; ++k) {
            load(P_S32, k, 1);
            fmul(1, 1, 1);
            fadd(0, 1, 0);
        }
        pack_scr0();
        row_reduce_to(inv_n, eps, true, 0);  // P_R[0] = r
        cb_wait_front(P_R, 1);
    }
    for (uint32_t k = 0; k < nk; ++k) {
        tile_regs_acquire();
        load(P_S32, k, 0);
        if (ln) {
            load(P_R, 0, 1);
            fsub(0, 1, 0);
            load(P_R, 1, 1);
            fmul(0, 1, 0);
        } else {
            load(P_R, 0, 1);
            fmul(0, 1, 0);
        }
        load(P_S16, k, 1);
        fmul(0, 1, 0);
        if (ln) {
            load(P_S16, nk + k, 1);
            fadd(0, 1, 0);
        }
        tile_regs_commit();
        cb_reserve_back(P_O16, 1);
        tile_regs_wait();
        pack_to(0, P_O16, 0);
        tile_regs_release();
        cb_push_back(P_O16, 1);
    }
    cb_pop_front(P_R, ln ? 2 : 1);
    cb_pop_front(P_S16, ln ? 2 * nk : nk);
    cb_pop_front(P_S32, nk);
}

PE_OS void run_norm(const TOp& o) {
    cb_point(P_S32, o.a[OA_X32], o.nk, T32);
    cb_point(P_S16, o.a[OA_S16], 2 * o.nk, T16);
    cb_point(P_O16, o.a[OA_O16], 8, T16);
    cb_point(P_SCR, o.a[OA_SCR], 2, T32);
    cb_point(P_R, o.a[OA_R], 2, T32);
    cb_point(P_CONST, o.a[OA_CST], PC_N, T16);
    cb_wait_front(P_CONST, PC_N);
    for (uint32_t i = 0; i < o.nitems; ++i) {
        norm_row(o);
    }
    cb_pop_front(P_CONST, PC_N);
}

// ================================================================ attention
// one key chunk of one q row tile: q tiles at q_off (dh), keys [t0, t0 + n) of the resident K / V (K at kv_k, V at
// kv_v, [t][d] layout), optional mask tiles -> part (O dh tiles, m full tile, l fp32 full tile) pushed to OP / PM / PL
PE_OS void flash_part(uint32_t kv_cb, uint32_t q_off, uint32_t dh, uint32_t t0, uint32_t n, uint32_t kv_v, bool mask,
                      uint32_t ss_addr) {
    // P_SS in full-capacity cycles of n tiles (a chunk may be shorter than the others: a ring of a fixed capacity
    // would straddle its end and pack_tile / unpack index past it). P_SS is TRISC-private (pack -> unpack).
    cb_point(P_SS, ss_addr, n, T16);
    cb_reserve_back(P_SS, n);
    tile_regs_acquire();
    mm_init_f<MathFidelity::HiFi2>(P_Q, kv_cb, 1);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t d = 0; d < dh; ++d) {
            mm_f<MathFidelity::HiFi2>(P_Q, kv_cb, q_off + d, (t0 + i) * dh + d, i);
        }
    }
    if (mask) {
        for (uint32_t i = 0; i < n; ++i) {
            load(P_MSK, t0 + i, 7);
            fadd(i, 7, i);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < n; ++i) {
        pack_to(i, P_SS, i);
    }
    tile_regs_release();
    cb_push_back(P_SS, n);
    cb_wait_front(P_SS, n);
    // m = rowmax (column vector)
    reconfig_data_format(P_SS, P_CONST);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(P_SS, P_CONST, P_M);
    cb_reserve_back(P_M, 1);
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(P_SS, P_CONST, i, PC_ONES, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, P_M, 0);
    tile_regs_release();
    reduce_uninit(P_SS);
    cb_push_back(P_M, 1);
    cb_wait_front(P_M, 1);
    // MF = m broadcast (full tile) -> the part's m
    reconfig_data_format(P_CONST, P_M);
    mul_bcast_cols_init(P_CONST, P_M);
    cb_reserve_back(P_PM, 1);
    tile_regs_acquire();
    mul_tiles_bcast_cols(P_CONST, P_M, PC_ONES, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, P_PM, 0);
    tile_regs_release();
    cb_push_back(P_PM, 1);
    // P = exp(S - m), in place
    reconfig_data_format(P_SS, P_M);
    sub_bcast_cols_init(P_SS, P_M);
    exp_tile_init<false>();
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        sub_tiles_bcast_cols(P_SS, P_M, i, 0, i);
        exp_tile<false>(i);
    }
    tile_regs_commit();
    cb_pop_front(P_SS, n);
    cb_reserve_back(P_SS, n);
    tile_regs_wait();
    for (uint32_t i = 0; i < n; ++i) {
        pack_to(i, P_SS, i);
    }
    tile_regs_release();
    cb_push_back(P_SS, n);
    cb_wait_front(P_SS, n);
    cb_pop_front(P_M, 1);
    // l = sum_i P_i @ ONES
    cb_reserve_back(P_PL, 1);
    tile_regs_acquire();
    mm_init_f<MathFidelity::HiFi4>(P_SS, P_CONST);
    for (uint32_t i = 0; i < n; ++i) {
        mm_f<MathFidelity::HiFi4>(P_SS, P_CONST, i, PC_ONES, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_to(0, P_PL, 0);
    tile_regs_release();
    cb_push_back(P_PL, 1);
    // O = sum_i P_i @ V_i
    cb_reserve_back(P_OP, dh);
    tile_regs_acquire();
    mm_init_f<MathFidelity::HiFi2>(P_SS, kv_cb);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t d = 0; d < dh; ++d) {
            mm_f<MathFidelity::HiFi2>(P_SS, kv_cb, i, kv_v + (t0 + i) * dh + d, d);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < dh; ++d) {
        pack_to(d, P_OP, d);
    }
    tile_regs_release();
    cb_push_back(P_OP, dh);
    cb_pop_front(P_SS, n);
}

// NP parts (in OP / PM / PL) -> ctx (dh tiles) -> P_O16
PE_OS void merge_parts(uint32_t np, uint32_t dh) {
    cb_wait_front(P_OP, np * dh);
    cb_wait_front(P_PM, np);
    cb_wait_front(P_PL, np);
    cb_reserve_back(P_D, np);
    tile_regs_acquire();
    load(P_PM, 0, 0);
    for (uint32_t i = 1; i < np; ++i) {
        load(P_PM, i, 1);
        fmax(0, 1, 0);
    }
    for (uint32_t i = 0; i < np; ++i) {
        load(P_PM, i, 1 + i);
        fsub(1 + i, 0, 1 + i);
        fexp(1 + i);
    }
    load(P_PL, 0, 0);
    fmul(0, 1, 0);
    for (uint32_t i = 1; i < np; ++i) {
        load(P_PL, i, 7);
        fmul(7, 1 + i, 7);
        fadd(0, 7, 0);
    }
    frecip(0);
    for (uint32_t i = 0; i < np; ++i) {
        fmul(1 + i, 0, 1 + i);
    }
    load(P_CONST, PC_IDENT, 0);
    for (uint32_t i = 0; i < np; ++i) {
        fmul(1 + i, 0, 1 + i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < np; ++i) {
        pack_to(1 + i, P_D, i);
    }
    tile_regs_release();
    cb_push_back(P_D, np);
    cb_wait_front(P_D, np);
    cb_reserve_back(P_O16, dh);
    tile_regs_acquire();
    mm_init_f<MathFidelity::HiFi4>(P_D, P_OP);
    for (uint32_t i = 0; i < np; ++i) {
        for (uint32_t d = 0; d < dh; ++d) {
            mm_f<MathFidelity::HiFi4>(P_D, P_OP, i, i * dh + d, d);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < dh; ++d) {
        pack_to(d, P_O16, d);
    }
    tile_regs_release();
    cb_push_back(P_O16, dh);
    cb_pop_front(P_D, np);
    cb_pop_front(P_OP, np * dh);
    cb_pop_front(P_PM, np);
    cb_pop_front(P_PL, np);
}

PE_OS void run_attn(const TOp& o) {
    const bool v = o.what == W_VATTN;
    const uint32_t dh = v ? V_DH : S_DH, nk = v ? PTV : S_NK, ch = v ? V_CH : S_CH, np = (nk + ch - 1) / ch;
    const uint32_t kv_cb = v ? P_KV8 : P_KV16;
    cb_point(P_Q, o.a[OA_Q], 2 * dh, T16);
    if (v) {
        cb_point(P_KV8, o.a[OA_KV], 2 * PTV * V_DH, T8);
        cb_point(P_MSK, o.a[OA_MSK], PTV, T16);
    } else {
        cb_point(P_KV16, o.a[OA_KV], 2 * S_NK * S_DH, T16);
    }
    cb_point(P_SS, o.a[OA_SS], 8, T16);
    cb_point(P_M, o.a[OA_M], 1, T16);
    cb_point(P_OP, o.a[OA_OP], 2 * np * dh, T16);
    cb_point(P_PM, o.a[OA_PM], 2 * np, T16);
    cb_point(P_PL, o.a[OA_PL], 2 * np, T32);
    cb_point(P_D, o.a[OA_D], np, T32);
    cb_point(P_O16, o.a[OA_O16], 2 * dh, T16);
    cb_point(P_CONST, o.a[OA_CST], PC_N, T16);
    cb_wait_front(P_CONST, PC_N);
    if (v) {
        cb_wait_front(P_MSK, PTV);
    }
    const uint32_t kv_v = nk * dh;  // V after K in the resident buffer
    for (uint32_t it = 0; it < o.nitems; ++it) {
        cb_wait_front(P_Q, 2 * dh);
        cb_wait_front(kv_cb, 2 * nk * dh);
        for (uint32_t h = 0; h < 2; ++h) {  // VLM: 2 heads of one q row tile; SigLIP: 2 q row tiles of one head
            for (uint32_t t0 = 0; t0 < nk; t0 += ch) {
                const uint32_t n = nk - t0 < ch ? nk - t0 : ch;
                flash_part(kv_cb, h * dh, dh, t0, n, kv_v, v, o.a[OA_SS]);
            }
            merge_parts(np, dh);
        }
        cb_pop_front(kv_cb, 2 * nk * dh);
        cb_pop_front(P_Q, 2 * dh);
    }
    if (v) {
        cb_pop_front(P_MSK, PTV);
    }
    cb_pop_front(P_CONST, PC_N);
}

// ================================================================ embedding (tilize + scale) / pad rows (zeros)
PE_OS void run_embed(const TOp& o) {
    cb_point(P_TOK, o.a[OA_TOK], 1, 64);
    cb_point(P_RM, o.a[OA_RM], 16, T16);
    cb_point(P_S16, o.a[OA_S16], 16, T16);
    cb_point(P_O32, o.a[OA_O32], 16, T32);
    cb_point(P_CONST, o.a[OA_CST], PC_N, T16);
    cb_wait_front(P_CONST, PC_N);
    for (uint32_t i = 0; i < o.nitems; ++i) {
        const uint32_t it = o.it0 + i * NCORES;
        const bool pad = it / 4 >= LT;
        if (!pad) {
            cb_wait_front(P_RM, 16);
            cb_reserve_back(P_S16, 16);
            tilize_init(P_RM, 16, P_S16);
            tilize_block(P_RM, 16, P_S16);
            tilize_uninit(P_RM, P_S16);
            cb_push_back(P_S16, 16);
            cb_pop_front(P_RM, 16);
            cb_wait_front(P_S16, 16);
        }
        for (uint32_t t = 0; t < 16; ++t) {
            tile_regs_acquire();
            if (!pad) {
                load(P_S16, t, 0);
                fscale(0, EMB_SCALE);
            } else {
                load(P_CONST, PC_ZERO, 0);
            }
            tile_regs_commit();
            cb_reserve_back(P_O32, 1);
            tile_regs_wait();
            pack_to(0, P_O32, 0);
            tile_regs_release();
            cb_push_back(P_O32, 1);
        }
        if (!pad) {
            cb_pop_front(P_S16, 16);
        }
    }
    cb_pop_front(P_CONST, PC_N);
}

// ================================================================ the op loop
PE_OS void run_trisc() {
    const uint32_t first = get_common_arg_val<uint32_t>(PA_OPFIRST);
    const uint32_t stop = get_common_arg_val<uint32_t>(PA_DBGSTOP);
    compute_kernel_hw_startup<SrcOrder::Reverse>(P_IN0, P_W8, P_O16);
    TOp o;
    const uint32_t reps = get_common_arg_val<uint32_t>(PA_REPS);
    for (uint32_t rep = 0; rep < reps; ++rep)
    for (uint32_t op = first; op < stop; ++op) {
        read_opd(o);
        switch (o.kind) {
            case K_MM: run_mm(o); break;
            case K_NORM: run_norm(o); break;
            case K_ATTN: run_attn(o); break;
            case K_EMBED: run_embed(o); break;
            default: break;
        }
    }
}

}  // namespace pe
