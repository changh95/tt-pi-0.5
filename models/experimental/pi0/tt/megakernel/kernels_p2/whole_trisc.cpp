// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// pi0.5 megakernel, TRISC: the prefix engine (the VISION and PREFIX programs).
// The phase-1 expert kernel (../kernels/mk_trisc.cpp) is compiled in as mk_expert_kernel_main(); the prefix engine
// (pe_trisc.hpp) runs the op range [PA_OPFIRST, PA_DBGSTOP) first. The VISION and PREFIX programs of a call
// (pe_program.py) are this binary with PA_EXPERT = 0, the expert loop runs as its own program (../kernels/mk_*.cpp).
// The host compiles these programs with PE_NO_EXPERT (pe_program.CODEGEN_DEFINES): the expert loop's call is
// compiled out, so its code is dead and dropped; the shared helpers it defines stay. The prefix engine's hot bodies
// are flattened (PE_ATT_FLAT) so their codegen does not depend on what else the translation unit contains.
#define kernel_main mk_expert_kernel_main
#include "../kernels/mk_trisc.cpp"
#undef kernel_main
#include "pe_trisc.hpp"

void kernel_main() {
    pe::run_trisc();
#ifndef PE_NO_EXPERT
    if (get_common_arg_val<uint32_t>(pe::PA_EXPERT)) {
        mk_expert_kernel_main();
    }
#endif
}
