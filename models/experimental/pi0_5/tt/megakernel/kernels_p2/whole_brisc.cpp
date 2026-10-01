// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pi0.5 whole-model megakernel (phase 2), BRISC: ONE kernel per RISC for the whole sample_actions.
// The phase-1 expert kernel (../kernels/mk_brisc.cpp, byte-identical to the verified phase-1 build) is compiled in as
// mk_expert_kernel_main(); the prefix engine (pe_brisc.hpp) runs first. PE_WHOLE=0 builds the prefix-only test program
// (the expert code is compiled but never entered).
#define kernel_main mk_expert_kernel_main
#include "../kernels/mk_brisc.cpp"
#undef kernel_main
#include "pe_brisc.hpp"

void kernel_main() {
    pe::run_brisc();
#if PE_WHOLE
    mk_expert_kernel_main();
#endif
}
