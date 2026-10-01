// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Prefix engine, TRISC: all arithmetic of SigLIP x2, the projector, the language embedding and the VLM prefill.
// Included by whole_trisc.cpp AFTER ../kernels/mk_trisc.cpp, whose LLK wrappers (mm_init_f / mm_f / load / fmul / ...
// pack_to, all noinline + noclone) it reuses. Every op starts with the op descriptor its NCRISC pushes into P_OPD after
// the op's barrier go (read with read_tile_value: UNPACK reads, mailboxes to MATH / PACK), so nothing here ever runs
// ahead of the op boundary, and no geometry code (describe / layout) is compiled into the three TRISC binaries.
#pragma once

#include "pe_common.hpp"

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// GELU (tanh form) without the fp32-accurate tanh (the stock gelu_tanh costs ~2,200 cycles per tile here, 40 us of the
// SigLIP fc1: arm PE_DBG_NO_GELU, 2026-09-30; x sigmoid(2u) by exp_21f + reciprocal still cost 32 us). Outputs bf16 /
// bfp8.
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void pe_calculate_gelu_fast() {
    // gelu(x) = relu(x) - h(|x|), h(t) = t q(t) on [0, 4.25] (degree-9 least-squares-minimax fit of the tanh form;
    // max abs error 1.3e-5 in fp64, 4.0e-5 in fp32 Horner, 2026-09-30), h = 0 beyond (|x| clamped: q(4.25) ~ 0)
#ifdef PE_DBG_GELU_EMPTY  // timing arm: the SFPU call without its arithmetic
    return;
#endif
    // two rows per step, the two Horner chains interleaved (a lone chain is SFPMAD-latency bound)
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat t0 = sfpi::dst_reg[0];
        sfpi::vFloat t1 = sfpi::dst_reg[1];
        t0 = sfpi::setsgn(t0, 0);
        t1 = sfpi::setsgn(t1, 0);
        v_if(t0 > 4.25f) { t0 = 4.25f; }
        v_endif;
        v_if(t1 > 4.25f) { t1 = 4.25f; }
        v_endif;
        sfpi::vFloat q0 = t0 * -1.432145018e-05f + 3.306026920e-04f;
        sfpi::vFloat q1 = t1 * -1.432145018e-05f + 3.306026920e-04f;
        q0 = q0 * t0 + -3.156597493e-03f;
        q1 = q1 * t1 + -3.156597493e-03f;
        q0 = q0 * t0 + 1.561119035e-02f;
        q1 = q1 * t1 + 1.561119035e-02f;
        q0 = q0 * t0 + -3.921917826e-02f;
        q1 = q1 * t1 + -3.921917826e-02f;
        q0 = q0 * t0 + 3.103299625e-02f;
        q1 = q1 * t1 + 3.103299625e-02f;
        q0 = q0 * t0 + 4.830191657e-02f;
        q1 = q1 * t1 + 4.830191657e-02f;
        q0 = q0 * t0 + 5.249063484e-03f;
        q1 = q1 * t1 + 5.249063484e-03f;
        q0 = q0 * t0 + -3.992681503e-01f;
        q1 = q1 * t1 + -3.992681503e-01f;
        q0 = q0 * t0 + 4.999349117e-01f;
        q1 = q1 * t1 + 4.999349117e-01f;
        q0 = t0 * q0;
        q1 = t1 * q1;
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = (x0 + sfpi::setsgn(x0, 0)) * 0.5f - q0;
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::dst_reg[1] = (x1 + sfpi::setsgn(x1, 0)) * 0.5f - q1;
        sfpi::dst_reg += 2;
    }
}
inline void pe_gelu_fast_init() {}
// exp for the softmax: exp_21f's range reduction with a degree-4 2^f (max rel err 2.7e-6; exp_21f's quadratic is
// ~1.7e-3 and failed seed 707 of the base gate in two builds while the fp32-accurate exp passed, 2026-09-30).
// p(0) = 1.0000026 >= 1 and p(1) = 1.9999948 < 2, as setexp needs.
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void pe_calculate_exp21() {
#pragma GCC unroll 1
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat xlog2 = sfpi::dst_reg[0] * 1.4426950216293334961f + 127.f;
        xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);
        sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
        sfpi::vInt ep = sfpi::exexp(z, sfpi::ExponentMode::Biased);
        sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest) * 1.1920928955078125e-07f;
        sfpi::vFloat q = f * 1.353495196e-02f + 5.200919136e-02f;
        q = q * f + 2.414447218e-01f;
        q = q * f + 6.930032969e-01f;
        q = q * f + 1.000002623e+00f;
        sfpi::dst_reg[0] = sfpi::setexp(q, ep);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif

namespace pe {

// softmax exp: the fp32-accurate library exp by default. Faster SFPU exps (exp_21f; degree-4 2^f, max rel err
// 2.7e-6: arm PE_EXP_FAST) failed seed 707 of the amended base gate in three builds (0.978 / 0.980 / 0.975 vs the
// shipped 0.992) while every build with the library exp passed 22 / 22 (2026-09-30, p2/results/seeds_*.json).
#ifndef PE_EXP_FAST
ALWI void pe_exp_init() { exp_tile_init<false>(); }
ALWI void pe_exp(uint32_t idst) { exp_tile<false>(idst); }
#else
ALWI void pe_exp_init() { MATH(llk_math_eltwise_unary_sfpu_init<SfpuType::exponential>(sfpu::pe_gelu_fast_init)); }
ALWI void pe_exp(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, pe_calculate_exp21, (DST_ACCUM_MODE), idst, VectorMode::RC));
}
#endif
#ifdef PE_GELU_STOCK  // precision arm: the fp32-accurate gelu_tanh
ALWI void pe_gelu_init() { gelu_tanh_tile_init(); }
ALWI void pe_gelu(uint32_t idst) { gelu_tanh_tile(idst); }
#else
ALWI void pe_gelu_init() { MATH(llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_tanh>(sfpu::pe_gelu_fast_init)); }
ALWI void pe_gelu(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, pe_calculate_gelu_fast, (DST_ACCUM_MODE), idst,
                         VectorMode::RC));
}
#endif

// ---------------------------------------------------------------- the block-matmul K loop (every prefix matmul)
// One output pair (rp x 2 tiles in DST 0 .. 2 rp - 1) accumulates nq weight pages of pc K tiles each; in0 tile of
// (row r, k) at q * pc + k + r * kt. The fidelity is a runtime switch on the MATH thread only (the unpacker and the
// packer code is shared: one copy of the loop per binary). Arm PE_MM_HIFI4: the "high" fidelity is HiFi4.
#ifdef PE_MM_HIFI4
constexpr MathFidelity MM_FID_HI = MathFidelity::HiFi4;
#else
constexpr MathFidelity MM_FID_HI = MathFidelity::HiFi2;
#endif
NOINL void mm_k(uint32_t in0, uint32_t w_cb, uint32_t rp, uint32_t kt, uint32_t pc, uint32_t nq, uint32_t fid_hi,
                bool wait_in0) {
    reconfig_data_format(w_cb, in0);  // matmul: SrcA <- in1, SrcB <- in0
    if (fid_hi) {
        MATH((llk_math_matmul_init<MM_FID_HI, MM_THROTTLE>(in0, w_cb, 0, 2, rp)));
    } else {
        MATH((llk_math_matmul_init<MathFidelity::LoFi, MM_THROTTLE>(in0, w_cb, 0, 2, rp)));
    }
    UNPACK((llk_unpack_AB_matmul_init(in0, w_cb, 0, 2, rp, kt)));
    const uint32_t pt = 2 * pc;
    for (uint32_t q = 0; q < nq; ++q) {
        if (wait_in0) {
            cb_wait_front(in0, rp * pc * (q + 1));  // mode R, first pair: the band arrives piece by piece
        }
        cb_wait_front(w_cb, pt);
#ifndef PE_DBG_NO_MATH
        for (uint32_t k = 0; k < pc; ++k) {
            UNPACK((llk_unpack_AB_matmul(in0, w_cb, q * pc + k, 2 * k, 2, rp, kt)));
            if (fid_hi) {
                MATH((llk_math_matmul<MM_FID_HI, MM_THROTTLE>(0, 2, rp)));
            } else {
                MATH((llk_math_matmul<MathFidelity::LoFi, MM_THROTTLE>(0, 2, rp)));
            }
        }
#endif
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
#ifndef PE_DBG_NO_GELU  // timing arm: the FC1 epilogue without its GELU
        if (o.epi == E_BIAS_GELU) {
#else
        if (false) {
#endif
            pe_gelu_init();
            for (uint32_t i = 0; i < 2 * rp; ++i) {
                pe_gelu(i);
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
        pe_gelu_init();
        for (uint32_t r = 0; r < rp; ++r) {
            pe_gelu(2 * r + 1);
        }
        for (uint32_t r = 0; r < rp; ++r) {
            fmul(2 * r, 2 * r + 1, 2 * r);
        }
    }
}

// ================================================================ matmul, mode R (resident in0 band, N-outer)
PE_OS void mm_r(const TOp& o, uint32_t w_cb, uint32_t p0, uint32_t npairs) {
    const uint32_t rp = o.rpb, kt = o.kt;
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
        mm_k(P_IN0, w_cb, rp, kt, o.piece, kt / o.piece, o.fid_hi, s == 0);
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
            mm_k(P_IN08, P_W8, rp, pc, pc, 1, o.fid_hi, false);
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
// x (fp32, P_X32: an FPU operand, default unpack) arrives in chunks; statistics by FPU (sum of squares by an
// accumulating ELWMUL, row sums by a HiFi4 matmul with ONES), E[x^2] - mu^2 for the LN variance (SigLIP inputs:
// mu^2 / var <= 0.46 on real activations, 2026-09-30), the apply batched 3 tiles per DST acquire with the inits hoisted.
ALWI void fpu_mul_init(uint32_t a, uint32_t b) {
    reconfig_data_format(a, b);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MathFidelity::HiFi4>(a, b, 0)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(a, b)));
}
ALWI void fpu_mul(uint32_t a, uint32_t b, uint32_t ta, uint32_t tb, uint32_t dst) {
    UNPACK((llk_unpack_AB(a, b, ta, tb)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, DST_ACCUM_MODE, MathFidelity::HiFi4,
                                  EltwiseBinaryReuseDestType::NONE>(a, b, dst, true)));
}
ALWI void fpu_sub_init(uint32_t a, uint32_t b) {
    reconfig_data_format(a, b);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::NONE, MathFidelity::HiFi4>(a, b, 0)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(a, b)));
}
ALWI void fpu_sub(uint32_t a, uint32_t b, uint32_t ta, uint32_t tb, uint32_t dst) {
    UNPACK((llk_unpack_AB(a, b, ta, tb)));
    MATH((llk_math_eltwise_binary<EltwiseBinaryType::ELWSUB, BroadcastType::NONE, DST_ACCUM_MODE, MathFidelity::HiFi4,
                                  EltwiseBinaryReuseDestType::NONE>(a, b, dst, true)));
}

PE_OS void norm_row(const TOp& o) {
    const uint32_t nk = o.nk;
    const bool ln = o.nkind == N_LN;
    const uint32_t ncg = ln ? NCG_S : NCG_V, w = nk / ncg;
    constexpr uint32_t ONE_OVER_2048 = 0x3A000000u;  // 1 / 2048
    constexpr uint32_t ONE_OVER_1152 = 0x3A638E39u;  // 1 / 1152 (fp32)
    constexpr uint32_t ONE_OVER_1152_32 = 0x37E38E39u;  // 1 / (1152 * 32) (fp32: the same mantissa, exponent - 5)
    const uint32_t inv_n = ln ? ONE_OVER_1152 : ONE_OVER_2048;
    const uint32_t eps = ln ? EPS_S : EPS_V;
    // pass 1 over this group's w tiles: per-element sum of x^2 (DST 0) and, for LN, row sums of x (DST 1) -> P_O32;
    // the BRISC sends the pair to slot (it % ncg) of the P_R of every core of the row (itself included)
    tile_regs_acquire();
    mul_acc_init(P_X32, P_X32);
    for (uint32_t k = 0; k < w; ++k) {
        cb_wait_front(P_X32, k + 1);
        mul_acc(P_X32, P_X32, k, k, 0);
    }
    if (ln) {
        mm_init_f<MathFidelity::HiFi4>(P_X32, P_CONST);
        for (uint32_t k = 0; k < w; ++k) {
            mm_f<MathFidelity::HiFi4>(P_X32, P_CONST, k, PC_ONES, 1);
        }
    }
    tile_regs_commit();
    cb_reserve_back(P_O32, 2);
    tile_regs_wait();
    pack_to(0, P_O32, 0);
    pack_to(1, P_O32, 1);
    tile_regs_release();
    cb_push_back(P_O32, 2);
    // pass 2 over the ncg partials: r (RMS) or (rstd, mu) (LN) as full tiles -> P_SCR [0] (and [1])
    cb_wait_front(P_R, 2 * ncg);
    tile_regs_acquire();
    mm_init_f<MathFidelity::HiFi4>(P_R, P_CONST);
    for (uint32_t g = 0; g < ncg; ++g) {
        mm_f<MathFidelity::HiFi4>(P_R, P_CONST, 2 * g, PC_ONES, 0);  // row sums of x^2
        if (ln) {  // the partial row sums are already row sums in every column: this sums them x 32
            mm_f<MathFidelity::HiFi4>(P_R, P_CONST, 2 * g + 1, PC_ONES, 1);
        }
    }
    fscale(0, inv_n);  // E[x^2]
    if (ln) {
        fscale(1, ONE_OVER_1152_32);  // mu
        fmul(1, 1, 2);
        fsub(0, 2, 0);  // var = E[x^2] - mu^2
    }
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    cb_pop_front(P_R, 2 * ncg);
    cb_reserve_back(P_SCR, 2);
    tile_regs_wait();
    pack_to(0, P_SCR, 0);
    pack_to(1, P_SCR, 1);
    tile_regs_release();
    cb_push_back(P_SCR, 2);
    cb_wait_front(P_SCR, 2);
    // apply to this item's column group [c0, c0 + w), 3 tiles per acquire: RMS x * r, LN (x - mu) * rstd (gamma /
    // beta / the RMS (1 + w) are folded into the next matmul's weights and bias on the host, pe_host.fold_norms)
    for (uint32_t k0 = 0; k0 < w; k0 += 3) {
        const uint32_t n = w - k0 < 3 ? w - k0 : 3;
        tile_regs_acquire();
        if (ln) {
            fpu_sub_init(P_X32, P_SCR);
            for (uint32_t j = 0; j < n; ++j) {
                fpu_sub(P_X32, P_SCR, k0 + j, 1, j);  // x - mu
            }
            reconfig_data_format_srca(P_SCR);
            copy_init(P_SCR);
            copy_tile(P_SCR, 0, 6);  // rstd
            mul_binary_tile_init();
            for (uint32_t j = 0; j < n; ++j) {
                mul_binary_tile(j, 6, j);
            }
        } else {
            fpu_mul_init(P_X32, P_SCR);
            for (uint32_t j = 0; j < n; ++j) {
                fpu_mul(P_X32, P_SCR, k0 + j, 0, j);  // x * r
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {  // one tile per push: any ring capacity works
            cb_reserve_back(P_O16, 1);
            pack_to(j, P_O16, 0);
            cb_push_back(P_O16, 1);
        }
        tile_regs_release();
    }
    cb_pop_front(P_SCR, 2);
    cb_pop_front(P_X32, w);
}

PE_OS void run_norm(const TOp& o) {
    cb_point(P_X32, o.a[OA_X32], o.nk, T32);
    cb_point(P_O16, o.a[OA_O16], 8, T16);  // == the BRISC writer (norm_write)
    cb_point(P_SCR, o.a[OA_SCR], 2, T32);
    cb_point(P_R, o.a[OA_R], 2 * (o.nkind == N_LN ? NCG_S : NCG_V), T32);
    cb_point(P_O32, o.a[OA_O32], 2, T32);
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
                      uint32_t ss_addr, bool single) {
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
    if (!single) {
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
    }
    // P = exp(S - m), in place
    reconfig_data_format(P_SS, P_M);
    sub_bcast_cols_init(P_SS, P_M);
    pe_exp_init();
    tile_regs_acquire();
    for (uint32_t i = 0; i < n; ++i) {
        sub_tiles_bcast_cols(P_SS, P_M, i, 0, i);
        pe_exp(i);
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
    // l = sum_i P_i @ ONES (fp32, P_PL) and O = sum_i P_i @ V_i (single: normalised by l in DST, the output is ctx)
    const uint32_t ocb = single ? P_O16 : P_OP;
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
    cb_reserve_back(ocb, dh);
    tile_regs_acquire();
    mm_init_f<MathFidelity::HiFi2>(P_SS, kv_cb);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t d = 0; d < dh; ++d) {
            mm_f<MathFidelity::HiFi2>(P_SS, kv_cb, i, kv_v + (t0 + i) * dh + d, d);
        }
    }
    if (single) {  // every key in this chunk
        cb_wait_front(P_PL, 1);
        load(P_PL, 0, dh);
        cb_pop_front(P_PL, 1);
        frecip(dh);
        for (uint32_t d = 0; d < dh; ++d) {
            fmul(d, dh, d);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t d = 0; d < dh; ++d) {
        pack_to(d, ocb, d);
    }
    tile_regs_release();
    cb_push_back(ocb, dh);
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
    const uint32_t qn = v ? 2 * V_DH : 3 * S_DH;
    cb_point(P_Q, o.a[OA_Q], qn, T16);
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
    if (v) {
        cb_wait_front(kv_cb, 2 * nk * dh);  // resident for the whole op (multicast once)
    }
    for (uint32_t it = 0; it < o.nitems; ++it) {
        uint32_t nr = 2;
        if (!v) {  // SigLIP item rows: groups {3, 3, 2}
            nr = ((o.it0 + it * NCORES) % (S_HEADS * 3)) % 3 < 2 ? 3 : 2;
        }
        cb_wait_front(P_Q, qn);
        if (!v) {
            cb_wait_front(kv_cb, 2 * nk * dh);
        }
        for (uint32_t h = 0; h < nr; ++h) {  // VLM: 2 heads of one q row tile; SigLIP: 2-3 q row tiles of one head
#ifdef PE_DBG_ATTN_NOCOMPUTE
            cb_reserve_back(P_O16, dh);  // timing arm: inputs in, zeros out
            tile_regs_acquire();
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t d = 0; d < dh; ++d) {
                pack_to(0, P_O16, d);
            }
            tile_regs_release();
            cb_push_back(P_O16, dh);
            continue;
#endif
            if (!v) {
                flash_part(kv_cb, h * dh, dh, 0, nk, kv_v, false, o.a[OA_SS], true);
                continue;
            }
            for (uint32_t t0 = 0; t0 < nk; t0 += ch) {
                const uint32_t n = nk - t0 < ch ? nk - t0 : ch;
                flash_part(kv_cb, h * dh, dh, t0, n, kv_v, v, o.a[OA_SS], false);
            }
            merge_parts(np, dh);
        }
        if (!v) {
            cb_pop_front(kv_cb, 2 * nk * dh);
        }
        cb_pop_front(P_Q, qn);
    }
    if (v) {
        cb_pop_front(kv_cb, 2 * nk * dh);
        cb_pop_front(P_MSK, PTV);
    }
    cb_pop_front(P_CONST, PC_N);
}

// ================================================================ embedding (tilize + scale) / pad rows (zeros)
PE_OS void run_embed(const TOp& o) {
    cb_point(P_S16, o.a[OA_S16], 16, T16);
    cb_point(P_O32, o.a[OA_O32], 16, T32);
    cb_point(P_CONST, o.a[OA_CST], PC_N, T16);
    cb_wait_front(P_CONST, PC_N);
    for (uint32_t i = 0; i < o.nitems; ++i) {
        const uint32_t it = o.it0 + i * NCORES;
        const bool pad = it / 4 >= LT;
        if (!pad) {
            cb_wait_front(P_S16, 16);  // tiles tilized by the NCRISC
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
