# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-role timing of single prefix-engine ops (timing arm PE_DBG_TRACE=0: wall-clock marks of the first executed op).

For each op of layer 0 run ALONE (run(op, op + 1)): per core the BRISC / NCRISC start (go seen) and end (role work
done, writes acked) in us after the earliest start, summarised by role (compute cores, in0 feeders, weight feeders,
item cores). Real layer-0 weights, zero activations (timing only).

    PI05_PE_DEFINES=PE_DBG_TRACE=0 python .../pe_trace.py --ops 1,2,3,4,5,6,7,193,194,195,196,198,199 --out t.json
"""
import argparse
import json
import os

import torch
import ttnn

from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.tests.megakernel.pe_bringup import AICLK_MHZ, Rig, stamp
from models.experimental.pi0_5.tt.megakernel import pe_geometry as P
from models.experimental.pi0_5.tt.megakernel import pe_host as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="base")
    ap.add_argument("--ops", default="1,2,3,4,5,6,7,193,194,195,196,198,199")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    assert "PE_DBG_TRACE=0" in os.environ.get("PI05_PE_DEFINES", ""), "needs PI05_PE_DEFINES=PE_DBG_TRACE=0"
    torch.set_num_threads(16)
    wl = PI0WeightLoader(os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
    cat = wl.categorized_weights
    pp = H.prefix_params(cat, n_sig=1, n_vlm=1)
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, worker_l1_size=1395712)
    res = {}
    try:
        emb = ttnn.from_torch(cat["vlm_language"]["lm_head.weight"][:256], dtype=ttnn.bfloat16,
                              layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rig = Rig(dev, a.shape, pp, emb)
        names = {v: k for k, v in P.PD.items() if k.startswith("W_")}
        for op in [int(x) for x in a.ops.split(",")]:
            o = P.describe(op, rig.ps)
            rig.run(op, op + 1)
            rig.run(op, op + 1)  # the second run is the one read (warm caches, no first-launch effects)
            d = rig.diag()
            t = d[:, 8:12].double()
            t0 = float(t[:, [0, 2]].min())
            us = (t - t0) / AICLK_MHZ
            rows = {}
            for lin in range(P.NCORES):
                x, y = lin % 11, lin // 11
                if o.kind == P.K_MM:
                    role = "compute" if y < o.nb else ("wfeed" if y == P.WF_Y else ("ifeed" if y == P.IF_Y and x < 8
                                                                                   else "idle"))
                else:
                    role = "item" if lin < o.items else "idle"
                rows.setdefault(role, []).append([round(float(v), 2) for v in us[lin]])
            summ = {}
            for role, v in rows.items():
                v = torch.tensor(v)
                summ[role] = {"n": len(v), "brisc_end_max": round(float(v[:, 1].max()), 2),
                              "brisc_end_med": round(float(v[:, 1].median()), 2),
                              "ncrisc_end_max": round(float(v[:, 3].max()), 2),
                              "ncrisc_end_med": round(float(v[:, 3].median()), 2),
                              "start_max": round(float(v[:, [0, 2]].max()), 2)}
            res[f"{op}_{names.get(o.what, o.what)}"] = {"summary": summ, "per_core": rows}
            stamp(f"op {op} {names.get(o.what, o.what)}: {summ}")
    finally:
        ttnn.close_device(dev)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
