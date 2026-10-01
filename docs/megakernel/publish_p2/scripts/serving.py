"""Staging SERVING.md = HF cf08fb95 SERVING.md with the phase-2 (whole-model megakernel) facts."""
S = "/home/deepgadget/experiments/tt-models/models/pi05-base-p150-fused"
s = open("/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/pub2/hf_head/SERVING.md").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep("""                 + tt/megakernel/: the phase-1 expert megakernel (host program + kernels/mk_{brisc,ncrisc,trisc}.cpp,
                 mk_defs.hpp, mk_dm.hpp): the whole 10-step x 18-layer expert loop as ONE persistent generic_op""",
"""                 + tt/megakernel/: the whole-model megakernel (pe_geometry / pe_host / pe_program.py, kernels_p2/
                 whole_{brisc,ncrisc,trisc}.cpp + pe_*.hpp, which include phase 1's kernels/mk_*): SigLIP x2, projector,
                 embedding, VLM prefill -> K/V, the 10-step x 18-layer expert loop and Euler as ONE persistent generic_op""")
rep("@ [`f7f173b`](https://github.com/changh95/tt-pi-0.5/commit/f7f173bd61241f09b286e56a9eb4038f76ee0782) (`main`, merge of PR #2).",
    "@ [`821e8c5`](https://github.com/changh95/tt-pi-0.5/commit/821e8c528dfffa0d1d6e73ad6abf181a749b39d9) (`main`, merge of PR #3).")
rep("""tree the fused code was validated on. The port's `generic_op` kernels (tt/kernels/* and the megakernel's
tt/megakernel/kernels/*) include only tt_metal `api/compute/*` / `api/dataflow/*` / `api/debug/*` headers and
their own sibling headers. `verify:` lines resolve every include of every port C++ source inside the image and
check the megakernel's five source files and kernel digest (`328761c8a1ce3fd9`).
and they use that tree's compute API: they do not build on the older fork `changh95/pi05` @ `4c9fbfcceb9`
that the previous image used.""",
"""tree the fused code was validated on. The port's `generic_op` kernels (tt/kernels/*, tt/megakernel/kernels/* and
tt/megakernel/kernels_p2/*) include only tt_metal/hw/inc headers (`api/compute/*`, `api/dataflow/*`, `api/debug/*`,
`internal/circular_buffer_interface.h`) and their own port files. `verify:` lines resolve every include of all 23 port
C++ sources inside the image, and check the whole-model megakernel's nine source files and kernel digest
(`4aa02cdf21ed0c94`) and the expert-loop sources it includes (`328761c8a1ce3fd9`). The kernels use that tree's
compute API: they do not build on the older fork `changh95/pi05` @ `4c9fbfcceb9`.""")
rep("export PI05_MEGAKERNEL=expert   # the default; `off` = the previous expert path (stock TT-NN ops plus 3 custom programs; comparator)",
    "export PI05_MEGAKERNEL=whole    # the default; comparators: `expert` = phase 1, `off` = stock TT-NN ops plus 3 custom programs")
rep("""  (`pi05_libero`: PCC 0.999884 mean, 0.999778 min over 8 observations) and the LIBERO closed loop (99/100);
  both use the same code with different weights and shape (see `demo/`). For `pi05_base` at the served
  shape the expert is checked against an fp32 expert-loop oracle fed the device's own prefix K/V (22 random
  inputs: mean PCC 0.99974, closer than the previous path on 22/22); the whole call vs the fixed fp32 torch
  reference is 0.804-0.9994 on the same inputs, the spread coming from the bf16 / bf8 prefix.""",
"""  (`pi05_libero`: PCC 0.999976 mean, 0.999955 min over 8 observations) and the LIBERO closed loop (99/100);
  both use the same code with different weights and shape (see `demo/`). For `pi05_base` at the served
  shape the whole call is checked against the fp32 torch reference of the whole model on the same inputs
  (32 random inputs with 1-224 real tokens: mean PCC 0.99851, min 0.98882, closer than the `off` path on 32/32),
  and the VLM K / V caches per layer against the reference's own (closer than the TT-NN caches on 288/288).""")
rep("""* **Megakernel limits.** Single p150a, batch 1, bf8 K/V caches, 10 steps; the device must be opened with""",
    """* **Megakernel limits.** Single p150a, batch 1, 2 cameras, bf8 K/V caches, 10 steps; the device must be opened with""")
open(f"{S}/SERVING.md", "w").write(s)
print("ok")
