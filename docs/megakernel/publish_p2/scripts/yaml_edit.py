p='tt-model.yaml'; s=open(p).read()
def rep(a,b):
    global s
    assert s.count(a)==1, a
    s=s.replace(a,b)
rep("""# Served path: sample_actions_fused, one Metal trace. Inside it the SigLIP + VLM prefix is traced stock TT-NN ops and the
# whole 10-step x 18-layer action-expert loop (+ action in / out + Euler) is ONE persistent generic_op on 110 cores
# (phase-1 megakernel, code/models/experimental/pi0_5/tt/megakernel/, PI05_MEGAKERNEL=expert).""",
"""# Served path: sample_actions_fused, one Metal trace whose replay holds exactly ONE device op: the whole model (SigLIP on
# both cameras, projector, language embedding, VLM prefill -> K/V, the 10-step x 18-layer action-expert loop, action in /
# out, Euler) is ONE persistent generic_op on 110 cores (phase-2 whole-model megakernel,
# code/models/experimental/pi0_5/tt/megakernel/pe_program.py + kernels_p2/, PI05_MEGAKERNEL=whole).""")
rep("""The port's generic_op kernels (tt/kernels/*, tt/megakernel/kernels/*) use the compute API of that tree""",
"""The port's generic_op kernels (tt/kernels/*, tt/megakernel/kernels/*, tt/megakernel/kernels_p2/*) use the compute API of that tree""")
rep("""  # C++ kernels (code/models/experimental/pi0_5/tt/kernels/*/*.cpp, tt/megakernel/kernels/mk_*.{cpp,hpp}) include only
  # tt_metal `api/compute/*` / `api/dataflow/*` / `api/debug/*` headers and their own sibling headers, which are in the
  # image (verify: below).""",
"""  # C++ kernels (code/models/experimental/pi0_5/tt/kernels/*/*.cpp, tt/megakernel/kernels/mk_*.{cpp,hpp},
  # tt/megakernel/kernels_p2/{pe_*.hpp,whole_*.cpp}) include only tt_metal/hw/inc headers (`api/compute/*`,
  # `api/dataflow/*`, `api/debug/*`, `internal/circular_buffer_interface.h`) and their own port files (kernels_p2 includes
  # ../kernels/mk_*.cpp), all of which are in the image (verify: below).""")
rep("""    # The phase-1 expert megakernel (also the default when unset); "off" = the previous all-stock-op expert path.
    PI05_MEGAKERNEL: "expert\"""",
"""    # The whole-model megakernel (also the default when unset). Comparators only: "expert" = phase 1 (traced stock-op
    # prefix + the expert loop as one generic_op, 892 ops); "off" = stock TT-NN ops plus the 3 custom programs (fused
    # attention, row_rsqrt, geglu_rc; 2,551 ops).
    PI05_MEGAKERNEL: "whole\"""")
rep('PI05_SOURCE_COMMIT: "f7f173bd61241f09b286e56a9eb4038f76ee0782"', 'PI05_SOURCE_COMMIT: "821e8c528dfffa0d1d6e73ad6abf181a749b39d9"')
rep("""  - "import models.experimental.pi0_5.tt.megakernel.program as p, models.experimental.pi0_5.tt.megakernel.geometry, models.experimental.pi0_5.tt.megakernel.host_model\"""",
"""  - "import models.experimental.pi0_5.tt.megakernel.program as p, models.experimental.pi0_5.tt.megakernel.geometry, models.experimental.pi0_5.tt.megakernel.host_model"
  - "import models.experimental.pi0_5.tt.megakernel.pe_program as pp, models.experimental.pi0_5.tt.megakernel.pe_geometry, models.experimental.pi0_5.tt.megakernel.pe_host, models.experimental.pi0_5.tt.megakernel.pe_size_check; assert pp.WholeMegakernel\"""")
rep("""  - "from models.experimental.pi0_5.common.fused_config import FusedConfig as F; assert F.from_env({}).megakernel == F.from_env({'PI05_MEGAKERNEL': 'expert'}).megakernel == 'expert' and F.from_env({'PI05_MEGAKERNEL': 'off'}).megakernel == 'off'\"""",
"""  - "from models.experimental.pi0_5.common.fused_config import FusedConfig as F; assert F.from_env({}).megakernel == F.from_env({'PI05_MEGAKERNEL': ''}).megakernel == F.from_env({'PI05_MEGAKERNEL': 'whole'}).megakernel == 'whole' and F.from_env({'PI05_MEGAKERNEL': 'expert'}).megakernel == 'expert' and F.from_env({'PI05_MEGAKERNEL': 'off'}).megakernel == 'off'\"""")
rep("""  # The megakernel sources ship complete: the five files, the three entry points, and the kernel digest the device gates ran on.""",
"""  # The expert-loop megakernel sources ship complete (phase 2 includes them): the five files, the three entry points, the digest.""")
old_inc = [l for l in s.split('\n') if "assert len(ks) == 14, ks" in l]; assert len(old_inc)==1
new_inc = old_inc[0].replace("assert len(ks) == 14, ks", "assert len(ks) == 23, ks").replace("assert len(pairs) >= 40, len(pairs)", "assert len(pairs) >= 69, len(pairs)")
assert new_inc != old_inc[0] and "len(pairs) >= 69" in new_inc
rep(old_inc[0], """  # The whole-model megakernel ships complete: the nine kernels_p2 files, its three entry points, and kernel_digest2 = the digest every phase-2 device gate ran on.
  - "import os; from pathlib import Path; import models.experimental.pi0_5.tt.megakernel.pe_program as pp; d = Path('/opt/tt-metal/models/experimental/pi0_5/tt/megakernel/kernels_p2'); fs = sorted(x.name for x in d.iterdir()); assert fs == ['pe_brisc.hpp', 'pe_common.hpp', 'pe_defs.hpp', 'pe_dm.hpp', 'pe_ncrisc.hpp', 'pe_trisc.hpp', 'whole_brisc.cpp', 'whole_ncrisc.cpp', 'whole_trisc.cpp'], fs; assert all(os.path.isfile(k) for k in pp.KERNELS2.values()), pp.KERNELS2; assert len(pp.KERNEL_SOURCES2) == 14, pp.KERNEL_SOURCES2; assert pp.kernel_digest2() == '4aa02cdf21ed0c94', pp.kernel_digest2()"
""" + new_inc)
open(p,'w').write(s)
