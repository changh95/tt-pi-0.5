// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pi0.5 expert megakernel, NCRISC (NoC 0): the weight streams, nothing else.
//
// Direct mode: every streaming core reads its own contiguous byte ranges of ONE DRAM bank, per denoising step s from
// that step's two arena tensors (w8 = bfp8 pages of 8 tiles; w16 = bf16 pages of 8 tiles, holding the w16 stream and
// the per-(step, layer) constant stream WC at their own offsets). The three rings (CB_W8, CB_W16, CB_WC) are filled
// independently in stream order: the TRISC consumes each ring in its own order, so no interleave table is needed.
// Reads carry a transaction id per ring slot so each landed page is pushed in order as soon as it is complete.
#include "mk_dm.hpp"

using namespace mk;

namespace {

struct Ring {
    uint32_t cb = 0, page_bytes = 0, base = 0, end = 0, wr = 0;
    uint32_t trid0 = 0, depth = 0;  // transaction ids trid0 .. trid0 + depth - 1
    uint32_t pages = 0;             // pages per step
    uint32_t off = 0;               // byte offset of the stream inside its bank (same in every step tensor)
    uint32_t issued = 0;            // pages issued in the current step
    uint32_t in_flight = 0, issue_slot = 0, done_slot = 0;
    uint32_t bank = 0;

    void init(uint32_t cb_, uint32_t page_bytes_, uint32_t trid0_, uint32_t depth_, uint32_t pages_, uint32_t off_,
              uint32_t bank_) {
        cb = cb_;
        page_bytes = page_bytes_;
        base = cb_base(cb_);
        end = base + cb_pages(cb_) * get_local_cb_interface(cb_).fifo_page_size;
        wr = base;
        trid0 = trid0_;
        depth = depth_;
        pages = pages_;
        off = off_;
        bank = bank_;
    }

    // One issue attempt for step tensor `addr`; true when a read was issued.
    FORCE_INLINE bool try_issue(uint32_t addr) {
        if (issued >= pages || in_flight >= depth) {
            return false;
        }
        if (cb_free_pages(cb) < PAGE_TILES * (in_flight + 1)) {
            return false;
        }
        const uint32_t trid = trid0 + issue_slot;
        noc_async_read_set_trid(trid);
        const uint64_t src = get_noc_addr_from_bank_id<true>(bank, addr + off + issued * page_bytes);
        noc_async_read(src, wr, page_bytes);
        wr += page_bytes;
        if (wr >= end) {
            wr = base;
        }
        issue_slot = (issue_slot + 1 == depth) ? 0 : issue_slot + 1;
        ++issued;
        ++in_flight;
        return true;
    }

    // Push the oldest page if it landed; true when one was pushed.
    FORCE_INLINE bool try_complete() {
        if (in_flight == 0) {
            return false;
        }
        if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid0 + done_slot)) {
            return false;
        }
        invalidate_l1_cache();
        asm volatile("" ::: "memory");
        cb_push_back(cb, PAGE_TILES);
        done_slot = (done_slot + 1 == depth) ? 0 : done_slot + 1;
        --in_flight;
        return true;
    }

    bool step_done() const { return issued >= pages && in_flight == 0; }
};

}  // namespace

void kernel_main() {
    invalidate_l1_cache();
    asm volatile("" ::: "memory");  // runtime args are written by the dispatcher: fence before the first read
    const uint32_t bank = rt(A_BANK);
    const uint32_t n8 = rt(A_W8_PAGES), n16 = rt(A_W16_PAGES), nc = rt(A_WC_PAGES);
    if (n8 + n16 + nc == 0) {
        return;
    }
    Ring w8, w16, wc;
    w8.init(CB_W8, PAGE_TILES * 1088, 1, 4, n8, rt(A_W8_OFF), bank);
    w16.init(CB_W16, PAGE_TILES * 2048, 5, 4, n16, rt(A_W16_OFF), bank);
    wc.init(CB_WC, PAGE_TILES * 2048, 9, 2, nc, rt(A_WC_OFF), bank);
    // debug stop (C_DEBUG < N_GEN): stream only what the TRISC will consume. Compute cores consume per layer; H0
    // consumes its whole per-step stream for every step it starts.
    const uint32_t ngen = ct_arg(C_DEBUG);
    const bool per_step = (rt(A_ROLE) & R_H0) != 0;
    for (uint32_t s = 0; s < N_STEPS; ++s) {
        if (ngen <= s * N_LAYERS) {
            break;
        }
        const uint32_t layers = (ngen - s * N_LAYERS < N_LAYERS) ? ngen - s * N_LAYERS : N_LAYERS;
        const uint32_t a8 = ct_arg(C_W8_ADDR + s), a16 = ct_arg(C_W16_ADDR + s);
        w8.pages = per_step ? n8 : n8 / N_LAYERS * layers;
        w16.pages = per_step ? n16 : n16 / N_LAYERS * layers;
        wc.pages = per_step ? nc : nc / N_LAYERS * layers;
        w8.issued = w16.issued = wc.issued = 0;
        while (!(w8.step_done() && w16.step_done() && wc.step_done())) {
            WAYPOINT("NWST");
            // constants first: they gate the start of every layer
            wc.try_complete();
            wc.try_issue(a16);
            w8.try_complete();
            w8.try_issue(a8);
            w16.try_complete();
            w16.try_issue(a16);
        }
    }
    noc_async_read_set_trid(0);
}
