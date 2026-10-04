// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pi0.5 expert megakernel, BRISC (NoC 1): every exchange, the KV chunk prefetch, prologue loads and the output.
//
// One generation g = s * 18 + l per (step, layer). Every core walks the same per-generation sequence below and does the
// parts its role bits select (mk_defs.hpp R_*); the order is the dataflow order of the layer, so no core ever waits on
// something that needs a later step of its own sequence (tests/megakernel/test_cpu_mk_protocol.py simulates it).
//
//   1 x round (H0 -> pair cores + owners)   5 part send (units -> merger)     9  x_mid round (H0 -> MLP cores)
//   2 pair send (-> Q leader / KL)          6 merger: parts in, ctx -> H1     10 h send (-> row leader)
//   3 Q leader multicast, KL multicast      7 ctx round (H1 -> owners)        11 row multicast (row leader -> row)
//   4 Q / K_s|V_s receive (units)           8 x_mid send (owners -> H0)       12 partials (-> owners)
//                                                                              13 reduce, x -> H0
//
// Sync words (mk_defs.hpp S_*) are cumulative counters; a flag carries g + 1 after generation g. Multicast rounds use
// explicit per-receiver ready credits; point-to-point deposits rely on the layer's dependency chain (every deposit of
// generation g + 1 transitively depends on every consumer of the same buffer finishing generation g).
#include "mk_dm.hpp"

using namespace mk;

namespace {

constexpr uint32_t RT = get_compile_time_arg_val(CT_RT);
constexpr uint32_t PT = get_compile_time_arg_val(CT_PT);
constexpr uint32_t CHT = get_compile_time_arg_val(CT_CHT);
constexpr uint32_t NCH = get_compile_time_arg_val(CT_NCH);
// row loop (NCH x RT > the grid's rows, c = 4 at 64 action rows): a unit is (head, chunk) and runs both query rows
constexpr bool ROWLOOP = NCH * RT > GRID_ROWS;
constexpr uint32_t NU = ROWLOOP ? NCH : NCH * RT;  // units per head column
constexpr uint32_t NRU = ROWLOOP ? 1 : RT;         // unit rows per key chunk (the last chunk's units: 8 x NRU)
constexpr uint32_t T16 = 2048, T8 = 1088, T32 = 4096;
constexpr uint32_t IN0_PAGES = 64 * RT;
constexpr uint32_t X_TILES = 32 * RT;

constexpr auto acc_cache = TensorAccessorArgs<CT_ACC0>();
constexpr auto acc_mask = TensorAccessorArgs<acc_cache.next_compile_time_args_offset()>();
constexpr auto acc_tab = TensorAccessorArgs<acc_mask.next_compile_time_args_offset()>();
constexpr auto acc_noise = TensorAccessorArgs<acc_tab.next_compile_time_args_offset()>();
constexpr auto acc_out = TensorAccessorArgs<acc_noise.next_compile_time_args_offset()>();
constexpr auto acc_const = TensorAccessorArgs<acc_out.next_compile_time_args_offset()>();
constexpr auto acc_dbg = TensorAccessorArgs<acc_const.next_compile_time_args_offset()>();

#define WAIT_GE(word, value, tag)      \
    do {                               \
        WAYPOINT(tag);                 \
        const uint32_t _v = (value);   \
        while (sync_read(word) < _v) { \
        }                              \
    } while (0)

FORCE_INLINE void inc_at(uint32_t x, uint32_t y, uint32_t word, uint32_t n = 1) {
    noc_semaphore_inc(get_noc_addr(x, y, sync_addr(word)), n);
}
FORCE_INLINE void write_to(uint32_t src, uint32_t x, uint32_t y, uint32_t dst, uint32_t bytes) {
    noc_async_write(src, get_noc_addr(x, y, dst), bytes);
}

struct Rect {
    uint32_t x0, y0, x1, y1, n;  // NoC coordinates (min corner, max corner), destination count
};

// Multicast `bytes` at local `src` to `dst` on every core of `r`, then its flag: the local source word `src_word`
// set to `value`, multicast into the receivers' `flag_word`. Data then flag on one NoC path (linked).
FORCE_INLINE void mcast_round(
    const Rect& r, uint32_t src, uint32_t dst, uint32_t bytes, uint32_t src_word, uint32_t flag_word, uint32_t value) {
    if (bytes) {
        noc_async_write_multicast(src, mcast_addr(r.x0, r.y0, r.x1, r.y1, dst), bytes, r.n, true /* linked */);
    }
    *sync_ptr(src_word) = value;
    noc_semaphore_set_multicast(sync_addr(src_word), mcast_addr(r.x0, r.y0, r.x1, r.y1, sync_addr(flag_word)), r.n);
    noc_async_writes_flushed();
}

// ---------------------------------------------------------------- state from the runtime args
struct Me {
    uint32_t bits, kind, j, head, dx, dy, dt, ur, kc, npre, kt0, mx, my, qx, qy, n, rlx, rly, kg, ng;
    Rect col, row;
    uint32_t ox[4], oy[4];
    uint32_t h0x, h0y, h1x, h1y, klx, kly;
    Rect rx, rm, ro, rk, ra;
    uint32_t n_xrdy, n_cores, ngen;
    bool has(uint32_t b) const { return (bits & b) != 0; }
};

Me me;
bool kv_pending = false;

void load_args() {
    me.bits = rt(A_ROLE);
    me.kind = rt(A_PAIR_KIND);
    me.j = rt(A_PAIR_J);
    me.head = rt(A_HEAD);
    me.dx = rt(A_PAIR_DX);
    me.dy = rt(A_PAIR_DY);
    me.dt = rt(A_PAIR_DT);
    me.ur = rt(A_UNIT_R);
    me.kc = rt(A_UNIT_KC);
    me.npre = rt(A_UNIT_NPRE);
    me.kt0 = rt(A_UNIT_KT0);
    me.mx = rt(A_MERGER_X);
    me.my = rt(A_MERGER_Y);
    me.qx = rt(A_QLEAD_X);
    me.qy = rt(A_QLEAD_Y);
    me.col = {rt(A_COL_X0), rt(A_COL_Y0), rt(A_COL_X1), rt(A_COL_Y1), NU - 1};
    me.n = rt(A_OWNER_N);
    me.rlx = rt(A_ROWL_X);
    me.rly = rt(A_ROWL_Y);
    me.row = {rt(A_ROW_X0), rt(A_ROW_Y0), rt(A_ROW_X1), rt(A_ROW_Y1), 7};
    me.kg = rt(A_MLP_KG);
    me.ng = rt(A_MLP_NG);
    for (uint32_t i = 0; i < 4; ++i) {
        me.ox[i] = rt(A_OWNX0 + i);
        me.oy[i] = rt(A_OWNY0 + i);
    }
    me.h0x = ct_arg(C_H0_X);
    me.h0y = ct_arg(C_H0_Y);
    me.h1x = ct_arg(C_H1_X);
    me.h1y = ct_arg(C_H1_Y);
    me.klx = ct_arg(C_KL_X);
    me.kly = ct_arg(C_KL_Y);
    me.rx = {ct_arg(C_RX_X0), ct_arg(C_RX_Y0), ct_arg(C_RX_X1), ct_arg(C_RX_Y1), ct_arg(C_N_RX)};
    me.rm = {ct_arg(C_RM_X0), ct_arg(C_RM_Y0), ct_arg(C_RM_X1), ct_arg(C_RM_Y1), 64};
    me.ro = {ct_arg(C_RO_X0), ct_arg(C_RO_Y0), ct_arg(C_RO_X1), ct_arg(C_RO_Y1), 32};
    me.rk = {ct_arg(C_RK_X0), ct_arg(C_RK_Y0), ct_arg(C_RK_X1), ct_arg(C_RK_Y1), 8 * NRU};
    me.n_cores = ct_arg(C_N_CORES);
    me.ra = {ct_arg(C_RA_X0), ct_arg(C_RA_Y0), ct_arg(C_RA_X1), ct_arg(C_RA_Y1), me.n_cores - 1};
    me.n_xrdy = ct_arg(C_N_XRDY);
    me.ngen = ct_arg(C_DEBUG);
}

// ---------------------------------------------------------------- prologue loads
template <typename Acc>
FORCE_INLINE void read_tiles_into(uint32_t cb, uint32_t n, const Acc& acc, uint32_t first_page, uint32_t stride = 1) {
    cb_reserve_back(cb, n);
    uint32_t a = get_write_ptr(cb);
    const uint32_t pb = get_local_cb_interface(cb).fifo_page_size;
    for (uint32_t i = 0; i < n; ++i, a += pb) {
        noc_async_read_page(first_page + i * stride, acc, a);
    }
    noc_async_read_barrier();
    cb_push_back(cb, n);
}

#if MK_KV_MCAST
// K / V chunk row multicast (production, MULTICONFIG WP4): the 8 head units of one (chunk, row) read the same chunk
// (one KV head), so the head-0 unit reads it and multicasts it along the row into the other 7 units' CB_KV
// (single-chunk ring: the same address on every unit) after their per-receiver ready credits, then the flag
// S_KVM_FLAG = prefetch count. 8x fewer cache reads: -1.8 ms per call at c3 L224 S64, bit-identical (r_kvd2 / 3)
uint32_t kvm_n = 0;
#endif

// KV chunk of generation g's layer: prefix K tiles then V tiles of this unit's chunk (issued, pushed later).
void prefetch_kv(uint32_t g) {
    if (!me.has(R_UNIT) || me.npre == 0 || g >= me.ngen) {
        return;
    }
    const uint32_t l = g % N_LAYERS;
    cb_reserve_back(CB_KV, 2 * CHT * DH_T);
    const uint32_t dst = get_write_ptr(CB_KV);
#if MK_KV_MCAST
    ++kvm_n;
    if (me.head != 0) {
        inc_at(rt(A_KVM_LX), rt(A_KVM_LY), S_KVM_RDY);
        kv_pending = true;
        return;
    }
    WAIT_GE(S_KVM_RDY, 7 * kvm_n, "KVMR");
#endif
    const auto k = TensorAccessor(acc_cache, ct_arg(C_K_ADDR + l));
    const auto v = TensorAccessor(acc_cache, ct_arg(C_V_ADDR + l));
    for (uint32_t i = 0; i < me.npre; ++i) {
        for (uint32_t d = 0; d < DH_T; ++d) {
            noc_async_read_page((me.kt0 + i) * DH_T + d, k, dst + (i * DH_T + d) * T8);
        }
    }
    for (uint32_t i = 0; i < me.npre; ++i) {
        for (uint32_t d = 0; d < DH_T; ++d) {
            noc_async_read_page((me.kt0 + i) * DH_T + d, v, dst + ((CHT + i) * DH_T + d) * T8);
        }
    }
#if MK_KV_MCAST
    noc_async_read_barrier();
    const Rect r = {rt(A_KVM_X0), rt(A_KVM_Y0), rt(A_KVM_X1), rt(A_KVM_Y1), 7};
    mcast_round(r, dst, dst, 2 * CHT * DH_T * T8, S_SRC0 + 4, S_KVM_FLAG, kvm_n);
#endif
    kv_pending = true;
}

void land_kv() {
    if (kv_pending) {
        WAYPOINT("KVRB");
#if MK_KV_MCAST
        if (me.head != 0) {
            WAIT_GE(S_KVM_FLAG, kvm_n, "KVMF");
        }
#endif
        noc_async_read_barrier();
        cb_push_back(CB_KV, 2 * CHT * DH_T);
        kv_pending = false;
    }
}

// a unit's part (O, m, l of one query row) -> the merger's slot kc - 1
FORCE_INLINE void send_part() {
    cb_wait_front(CB_OP, DH_T);
    cb_wait_front(CB_MF, 1);
    cb_wait_front(CB_L, 1);
    const uint32_t slot = me.kc - 1;
    write_to(get_read_ptr(CB_OP), me.mx, me.my, cb_base(CB_PART) + slot * DH_T * T16, DH_T * T16);
    write_to(get_read_ptr(CB_MF), me.mx, me.my, cb_base(CB_PM) + slot * T16, T16);
    write_to(get_read_ptr(CB_L), me.mx, me.my, cb_base(CB_PL) + slot * T32, T32);
    noc_async_write_barrier();
    inc_at(me.mx, me.my, S_PART_ARR);
    cb_pop_front(CB_OP, DH_T);
    cb_pop_front(CB_MF, 1);
    cb_pop_front(CB_L, 1);
}

// ---------------------------------------------------------------- the compute cores' generation
void compute_gen(uint32_t g) {
    const uint32_t l = g % N_LAYERS;
    const uint32_t in0 = cb_base(CB_IN0);
    // 1. x round
    if (me.has(R_PAIR) || me.has(R_OWNER)) {
        dbg_mark(g, 1);
        WAYPOINT("XRSV");
        cb_reserve_back(CB_IN0, IN0_PAGES);
        inc_at(me.h0x, me.h0y, S_X_RDY);
        if (me.has(R_PAIR) || l == 0) {
            WAIT_GE(S_X_FLAG, g + 1, "XFLG");
            cb_push_back(CB_IN0, IN0_PAGES);
        }
        if (me.has(R_PAIR)) {
            WAIT_GE(S_R_FLAG, g + 1, "RFLG");
            cb_push_back(CB_RTOK, 1);
        }
    }
    // 2. pair send: tiles (r, j), (r, j + 4) of this pair -> the Q leader's CB_Q / KL's CB_KSV
    if (me.has(R_PAIR)) {
        dbg_mark(g, 2);
        WAYPOINT("PQKV");
        cb_wait_front(CB_QKVO, 2 * RT);
        const uint32_t src = get_read_ptr(CB_QKVO);
        const uint32_t dst = (me.kind == PK_Q ? cb_base(CB_Q) : cb_base(CB_KSV)) + me.dt * T16;
        for (uint32_t r = 0; r < RT; ++r) {
            write_to(src + (2 * r) * T16, me.dx, me.dy, dst + (r * DH_T) * T16, T16);
            write_to(src + (2 * r + 1) * T16, me.dx, me.dy, dst + (r * DH_T + 4) * T16, T16);
        }
        noc_async_write_barrier();
        inc_at(me.dx, me.dy, me.kind == PK_Q ? S_Q_ARR : S_KV_ARR);
        cb_pop_front(CB_QKVO, 2 * RT);
    }
    // 3. Q leader: gather -> column multicast -> tell KL
    if (me.has(R_QLEAD)) {
        dbg_mark(g, 3);
        WAIT_GE(S_Q_ARR, 4 * (g + 1), "QARR");
        WAIT_GE(S_Q_RDY, (NU - 1) * (g + 1), "QRDY");
        WAYPOINT("QRSV");
        cb_reserve_back(CB_Q, DH_T * RT);
        mcast_round(me.col, cb_base(CB_Q), cb_base(CB_Q), DH_T * RT * T16, S_SRC0 + 0, S_Q_FLAG, g + 1);
        noc_async_write_barrier();  // multicast delivered before KL may multicast into an intersecting rectangle
        inc_at(me.klx, me.kly, S_KV_QDONE);
        cb_push_back(CB_Q, DH_T * RT);
    }
    // 4. unit receives
    if (me.has(R_UNIT)) {
        dbg_mark(g, 4);
        land_kv();
        if (!me.has(R_QLEAD)) {
            WAYPOINT("URSV");
            cb_reserve_back(CB_Q, DH_T * RT);
            inc_at(me.qx, me.qy, S_Q_RDY);
            WAIT_GE(S_Q_FLAG, g + 1, "QFLG");
            cb_push_back(CB_Q, DH_T * RT);
        }
        if (me.has(R_LASTCH)) {
            WAYPOINT("KRSV");
            cb_reserve_back(CB_KSV, 2 * DH_T * RT);
            inc_at(me.klx, me.kly, S_KV_RDY);
            WAIT_GE(S_KV_FLAG, g + 1, "KFLG");
            cb_push_back(CB_KSV, 2 * DH_T * RT);
        }
    }
    // 5. part send (units kc >= 1 -> merger slot kc - 1)
    if (me.has(R_UNIT) && !me.has(R_MERGER)) {
        dbg_mark(g, 5);
        WAYPOINT("PART");
        if constexpr (ROWLOOP) {
            for (uint32_t r = 0; r < RT; ++r) {
                if (r > 0) {  // the merger holds one row's parts: wait until it freed them
                    WAIT_GE(S_PRT_FLAG, g * (RT - 1) + r, "PRTF");
                }
                send_part();
            }
        } else {
            send_part();
        }
        if (!me.has(R_MLP)) {
            prefetch_kv(g + 1);
        }
    }
    // 6. merger: the NCH - 1 remote parts in, the merged ctx out to H1
    if (me.has(R_MERGER)) {
        dbg_mark(g, 6);
        if constexpr (ROWLOOP) {  // one row's parts at a time: row r's slots are freed by the TRISC's merge of r - 1
            for (uint32_t r = 0; r < RT; ++r) {
                WAYPOINT("MRSV");
                cb_reserve_back(CB_PART, DH_T * (NCH - 1));
                cb_reserve_back(CB_PM, NCH - 1);
                cb_reserve_back(CB_PL, NCH - 1);
                if (r > 0) {  // tell the column's units
                    mcast_round(me.col, 0, 0, 0, S_SRC0 + 5, S_PRT_FLAG, g * (RT - 1) + r);
                }
                WAIT_GE(S_PART_ARR, (NCH - 1) * (g * RT + r + 1), "MARR");
                cb_push_back(CB_PART, DH_T * (NCH - 1));
                cb_push_back(CB_PM, NCH - 1);
                cb_push_back(CB_PL, NCH - 1);
                WAYPOINT("CTXO");
                cb_wait_front(CB_CTXO, DH_T);
                write_to(get_read_ptr(CB_CTXO), me.h1x, me.h1y, in0 + (r * 64 + me.head * DH_T) * T16, DH_T * T16);
                noc_async_write_barrier();
                inc_at(me.h1x, me.h1y, S_CTX_ARR);
                cb_pop_front(CB_CTXO, DH_T);
            }
        } else {
            WAYPOINT("MRSV");
            cb_reserve_back(CB_PART, DH_T * (NCH - 1));
            cb_reserve_back(CB_PM, NCH - 1);
            cb_reserve_back(CB_PL, NCH - 1);
            WAIT_GE(S_PART_ARR, (NCH - 1) * (g + 1), "MARR");
            cb_push_back(CB_PART, DH_T * (NCH - 1));
            cb_push_back(CB_PM, NCH - 1);
            cb_push_back(CB_PL, NCH - 1);
            WAYPOINT("CTXO");
            cb_wait_front(CB_CTXO, DH_T);
            write_to(get_read_ptr(CB_CTXO), me.h1x, me.h1y, in0 + (me.ur * 64 + me.head * DH_T) * T16, DH_T * T16);
            noc_async_write_barrier();
            inc_at(me.h1x, me.h1y, S_CTX_ARR);
            cb_pop_front(CB_CTXO, DH_T);
        }
    }
    // 7. owners: ctx round
    if (me.has(R_OWNER)) {
        dbg_mark(g, 7);
        WAYPOINT("CRSV");
        cb_reserve_back(CB_IN0, IN0_PAGES);
        inc_at(me.h1x, me.h1y, S_CTX_RDY);
        WAIT_GE(S_CTX_FLAG, g + 1, "CFLG");
        cb_push_back(CB_IN0, IN0_PAGES);
        // 8. x_mid -> H0
        dbg_mark(g, 8);
        WAYPOINT("XMSD");
        cb_wait_front(CB_XS, RT);
        for (uint32_t r = 0; r < RT; ++r) {
            write_to(get_read_ptr(CB_XS) + r * T16, me.h0x, me.h0y, in0 + (r * 32 + me.n) * T16, T16);
        }
        noc_async_write_barrier();
        inc_at(me.h0x, me.h0y, S_XM_ARR);
        cb_pop_front(CB_XS, RT);
    }
    if (me.has(R_MLP)) {
        // 9. x_mid round
        dbg_mark(g, 9);
        WAYPOINT("MRSX");
        cb_reserve_back(CB_IN0, IN0_PAGES);
        inc_at(me.h0x, me.h0y, S_XM_RDY);
        WAIT_GE(S_XM_FLAG, g + 1, "MFLG");
        cb_push_back(CB_IN0, IN0_PAGES);
        WAIT_GE(S_RP_FLAG, g + 1, "PFLG");
        cb_push_back(CB_RTOK, 1);
        prefetch_kv(g + 1);
        // 10. h -> row leader
        dbg_mark(g, 10);
        WAYPOINT("HSND");
        cb_wait_front(CB_H, 2 * RT);
        for (uint32_t r = 0; r < RT; ++r) {
            write_to(
                get_read_ptr(CB_H) + 2 * r * T16, me.rlx, me.rly, cb_base(CB_HG) + (r * 16 + 2 * me.ng) * T16, 2 * T16);
        }
        noc_async_write_barrier();
        inc_at(me.rlx, me.rly, S_H_ARR);
        cb_pop_front(CB_H, 2 * RT);
        // 11. row multicast
        dbg_mark(g, 11);
        if (me.has(R_ROWLEAD)) {
            WAIT_GE(S_H_ARR, 8 * (g + 1), "HARR");
            WAIT_GE(S_H_RDY, 7 * (g + 1), "HRDY");
            WAYPOINT("HRSV");
            cb_reserve_back(CB_HG, 16 * RT);
            mcast_round(me.row, cb_base(CB_HG), cb_base(CB_HG), 16 * RT * T16, S_SRC0 + 1, S_H_FLAG, g + 1);
            cb_push_back(CB_HG, 16 * RT);
        } else {
            WAYPOINT("HRCV");
            cb_reserve_back(CB_HG, 16 * RT);
            inc_at(me.rlx, me.rly, S_H_RDY);
            WAIT_GE(S_H_FLAG, g + 1, "HFLG");
            cb_push_back(CB_HG, 16 * RT);
        }
        // 12. down partials -> the 4 owners of this column, slot kg
        dbg_mark(g, 12);
        WAYPOINT("DPSD");
        cb_wait_front(CB_DP, 4 * RT);
        for (uint32_t i = 0; i < 4; ++i) {
            write_to(
                get_read_ptr(CB_DP) + i * RT * T32, me.ox[i], me.oy[i], cb_base(CB_RED) + me.kg * RT * T32, RT * T32);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < 4; ++i) {
            inc_at(me.ox[i], me.oy[i], S_RED_ARR);
        }
        cb_pop_front(CB_DP, 4 * RT);
    }
    // 13. owners: reduce in, x -> H0
    if (me.has(R_OWNER)) {
        dbg_mark(g, 13);
        WAYPOINT("RRSV");
        cb_reserve_back(CB_RED, 8 * RT);
        WAIT_GE(S_RED_ARR, 8 * (g + 1), "RARR");
        cb_push_back(CB_RED, 8 * RT);
        WAYPOINT("XSND");
        cb_wait_front(CB_XS, RT);
        for (uint32_t r = 0; r < RT; ++r) {
            write_to(get_read_ptr(CB_XS) + r * T16, me.h0x, me.h0y, in0 + (r * 32 + me.n) * T16, T16);
        }
        noc_async_write_barrier();
        inc_at(me.h0x, me.h0y, S_X_ARR);
        cb_pop_front(CB_XS, RT);
    }
}

// ---------------------------------------------------------------- phase stamps (MK_STAMPS, test-only diagnostic)
// The hubs record the wall clock (1350 MHz cycles, low word) at their per-generation phase boundaries into their idle
// CB_KV (no hub is an attention unit) and write them to the debug output tensor at the end (instead of the residual
// dump): H0 pages 0 / 1 = x_mid gathered / x gathered, H1 page 2 = ctx gathered, KL page 3 = suffix K / V gathered.
#ifdef MK_STAMPS
FORCE_INLINE void stamp(uint32_t slot, uint32_t g) {
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(CB_KV) + slot * 2048)[g] = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
}
void stamps_out(uint32_t slot0, uint32_t n) {
    const auto dbg = TensorAccessor(acc_dbg, ct_arg(C_DBGOUT));
    for (uint32_t i = 0; i < n; ++i) {
        noc_async_write_page(slot0 + i, dbg, cb_base(CB_KV) + i * 2048);
    }
    noc_async_write_barrier();
}
#define STAMP(slot, g) stamp(slot, g)
#else
#define STAMP(slot, g)
#endif

// ---------------------------------------------------------------- hubs
void h0_gather(uint32_t word, uint32_t target) {
    WAYPOINT("HGRS");
    cb_reserve_back(CB_IN0, IN0_PAGES);
    WAIT_GE(word, target, "HGAR");
    cb_push_back(CB_IN0, IN0_PAGES);
}

void run_h0() {
    const uint32_t in0 = cb_base(CB_IN0);
    const uint32_t rout = cb_base(CB_ROUT);
    for (uint32_t g = 0; g < me.ngen; ++g) {
        const uint32_t l = g % N_LAYERS;
        dbg_mark(g, 20);
        if (l == 0) {
            if (g > 0) {
                h0_gather(S_X_ARR, 32 * g);  // x_final of the previous step -> TRISC tail + in-proj
                STAMP(1, g - 1);
            }
            // TRISC in-proj: x_new row 0 in CB_PART, row 1 in CB_HG (every CB has ONE producer RISC and ONE consumer
            // RISC: the TRISC packer / unpacker keep local copies of the CB counters)
            WAYPOINT("H0IP");
            cb_wait_front(CB_PART, DH_T * (NCH - 1));
            if (RT == 2) {
                cb_wait_front(CB_HG, 16 * RT);
            }
            const uint32_t rows[2] = {cb_base(CB_PART), cb_base(CB_HG)};
            cb_reserve_back(CB_IN0, IN0_PAGES);
            for (uint32_t r = 0; r < RT; ++r) {  // own copy for the TRISC's r_in
                write_to(rows[r], me.h0x, me.h0y, in0 + r * 32 * T16, 32 * T16);
            }
            noc_async_write_barrier();
            cb_push_back(CB_IN0, IN0_PAGES);
            WAIT_GE(S_X_RDY, me.n_xrdy * (g + 1), "H0XR");
            for (uint32_t r = 0; r < RT; ++r) {
                noc_async_write_multicast(
                    rows[r],
                    mcast_addr(me.rx.x0, me.rx.y0, me.rx.x1, me.rx.y1, in0 + r * 32 * T16),
                    32 * T16,
                    me.rx.n,
                    true);
            }
            mcast_round(me.rx, 0, 0, 0, S_SRC0 + 0, S_X_FLAG, g + 1);
            noc_async_write_barrier();
            cb_pop_front(CB_PART, DH_T * (NCH - 1));
            if (RT == 2) {
                cb_pop_front(CB_HG, 16 * RT);
            }
            WAYPOINT("H0RI");
            cb_wait_front(CB_ROUT, RT);
            mcast_round(me.rx, rout, in0 + X_TILES * T16, RT * T16, S_SRC0 + 1, S_R_FLAG, g + 1);
            noc_async_write_barrier();
            cb_pop_front(CB_ROUT, RT);
        } else {
            h0_gather(S_X_ARR, 32 * g);
            STAMP(1, g - 1);
            WAIT_GE(S_X_RDY, me.n_xrdy * (g + 1), "H0XR");
            mcast_round(me.rx, in0, in0, X_TILES * T16, S_SRC0 + 0, S_X_FLAG, g + 1);
            WAYPOINT("H0RI");
            cb_wait_front(CB_ROUT, RT);
            mcast_round(me.rx, rout, in0 + X_TILES * T16, RT * T16, S_SRC0 + 1, S_R_FLAG, g + 1);
            noc_async_write_barrier();
            cb_pop_front(CB_ROUT, RT);
        }
        dbg_mark(g, 21);
        h0_gather(S_XM_ARR, 32 * (g + 1));
        STAMP(0, g);
        WAIT_GE(S_XM_RDY, 64 * (g + 1), "H0MR");
        mcast_round(me.rm, in0, in0, X_TILES * T16, S_SRC0 + 2, S_XM_FLAG, g + 1);
        WAYPOINT("H0RP");
        cb_wait_front(CB_ROUT, RT);
        mcast_round(me.rm, rout, in0 + X_TILES * T16, RT * T16, S_SRC0 + 3, S_RP_FLAG, g + 1);
        noc_async_write_barrier();
        cb_pop_front(CB_ROUT, RT);
    }
    // the residual after the last generation: debug dump of x, then the TRISC's x_t (the output)
    dbg_mark(me.ngen, 22);
    h0_gather(S_X_ARR, 32 * me.ngen);
#ifdef MK_STAMPS
    STAMP(1, me.ngen - 1);
    stamps_out(0, 2);
#else
    {
        const auto dbg = TensorAccessor(acc_dbg, ct_arg(C_DBGOUT));
        for (uint32_t t = 0; t < X_TILES; ++t) {
            noc_async_write_page(t, dbg, in0 + t * T16);
        }
        noc_async_write_barrier();
    }
#endif
    WAYPOINT("H0OT");
    cb_wait_front(CB_ROUT, RT);
    const auto out = TensorAccessor(acc_out, ct_arg(C_OUT));
    for (uint32_t r = 0; r < RT; ++r) {
        noc_async_write_page(r, out, rout + r * T16);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_ROUT, RT);
}

void run_h1() {
    const uint32_t in0 = cb_base(CB_IN0);
    for (uint32_t g = 0; g < me.ngen; ++g) {
        dbg_mark(g, 30);
        WAIT_GE(S_CTX_ARR, NH * RT * (g + 1), "H1AR");
        STAMP(0, g);
        WAIT_GE(S_CTX_RDY, 32 * (g + 1), "H1RD");
        mcast_round(me.ro, in0, in0, IN0_PAGES * T16, S_SRC0 + 0, S_CTX_FLAG, g + 1);
        noc_async_write_barrier();
    }
#ifdef MK_STAMPS
    stamps_out(2, 1);
#endif
}

void run_kl() {
    const uint32_t ksv = cb_base(CB_KSV);
    for (uint32_t g = 0; g < me.ngen; ++g) {
        dbg_mark(g, 40);
        WAIT_GE(S_KV_ARR, 8 * (g + 1), "KLAR");
        STAMP(0, g);
        WAIT_GE(S_KV_QDONE, NH * (g + 1), "KLQD");
        WAIT_GE(S_KV_RDY, 8 * NRU * (g + 1), "KLRD");
        mcast_round(me.rk, ksv, ksv, 2 * DH_T * RT * T16, S_SRC0 + 0, S_KV_FLAG, g + 1);
        noc_async_write_barrier();
    }
#ifdef MK_STAMPS
    stamps_out(3, 1);
#endif
}

}  // namespace

void kernel_main() {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");  // runtime args are written by the dispatcher: fence before the first read
    load_args();
    // zero every sync word, then the boot barrier (tt-metal semaphores: re-initialised by every enqueue)
    for (uint32_t w = 0; w < N_SYNC; ++w) {
        *sync_ptr(w) = 0;
    }
    const uint32_t sem_arr = get_semaphore(0), sem_go = get_semaphore(1);
    noc_semaphore_inc(get_noc_addr(me.h0x, me.h0y, sem_arr), 1);
    if (me.has(R_H0)) {
        WAYPOINT("BOOT");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_arr), me.n_cores);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go) = 1;
        noc_semaphore_set_multicast(sem_go, mcast_addr(me.ra.x0, me.ra.y0, me.ra.x1, me.ra.y1, sem_go), me.ra.n);
        noc_async_writes_flushed();
    }
    WAYPOINT("BTGO");
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go), 1);

    // prologue loads (persistent CBs, never popped: CONST, TAB, MASK)
    const bool computes = (me.bits & (R_PAIR | R_UNIT | R_OWNER | R_MLP | R_H0)) != 0;
    if (computes) {
        read_tiles_into(CB_CONST, 4, TensorAccessor(acc_const, ct_arg(C_CONSTS)), 0);
    }
    if (me.has(R_PAIR) && me.kind != PK_V) {
        const auto c = TensorAccessor(acc_tab, ct_arg(me.kind == PK_Q ? C_COSQ : C_COSK));
        const auto s = TensorAccessor(acc_tab, ct_arg(me.kind == PK_Q ? C_SINQ : C_SINK));
        cb_reserve_back(CB_TAB, 4 * RT);
        const uint32_t a = get_write_ptr(CB_TAB);
        for (uint32_t r = 0; r < RT; ++r) {
            noc_async_read_page(r * DH_T + me.j, c, a + (4 * r + 0) * T16);
            noc_async_read_page(r * DH_T + me.j + 4, c, a + (4 * r + 1) * T16);
            noc_async_read_page(r * DH_T + me.j, s, a + (4 * r + 2) * T16);
            noc_async_read_page(r * DH_T + me.j + 4, s, a + (4 * r + 3) * T16);
        }
        noc_async_read_barrier();
        cb_push_back(CB_TAB, 4 * RT);
    }
    if (me.has(R_UNIT)) {
        read_tiles_into(CB_MASK, CHT, TensorAccessor(acc_mask, ct_arg(C_MASK)), me.kt0);
        prefetch_kv(0);
    }
    if (me.has(R_H0)) {
        read_tiles_into(CB_Q, RT, TensorAccessor(acc_noise, ct_arg(C_NOISE)), 0);
        run_h0();
    } else if (me.has(R_H1)) {
        run_h1();
    } else if (me.has(R_KL)) {
        run_kl();
    } else if (me.bits & (R_PAIR | R_UNIT | R_OWNER | R_MLP)) {
        for (uint32_t g = 0; g < me.ngen; ++g) {
            compute_gen(g);
        }
    }
    dbg_mark(0xFFFF, 99);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
