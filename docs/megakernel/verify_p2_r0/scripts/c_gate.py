"""verify-p2-r0 CPU analysis: (1) amended per-seed gate: per seed, PCC(arm, fp32 whole-model reference) for whole /
expert / off; margin = whole - off (the binding comparator = the pre-megakernel shipped path) and whole - expert;
(2) amended P2-3 layer gate: per input, layer, K / V: PCC and rel-L2 of the device caches (valid prefix rows only)
vs the fp32 reference's own VLM cache; whole must be at least as close as the ttnn caches (off) on every one;
(3) LIBERO: the same per-record whole-model comparison.  python c_gate.py ROUND (r1)"""
import json, statistics, sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vc2 import OUT, SPEC, pcc, rel  # noqa: E402

rnd = sys.argv[1] if len(sys.argv) > 1 else "r1"
res = {}
for shape in ("base", "libero"):
    R = torch.load(f"{OUT}/ref_{shape}.pt", weights_only=False)
    A = {arm: torch.load(f"{OUT}/A_{arm}_{shape}_{rnd}.pt", weights_only=False) for arm in ("whole", "expert", "off")}
    tags = A["whole"]["tags"]
    assert tags == A["off"]["tags"] == A["expert"]["tags"]
    rows = []
    for i, t in enumerate(tags):
        ref = R["refs"][t]
        r = {"tag": t, **{arm: pcc(A[arm]["outs"][i], ref) for arm in A}}
        r["margin_vs_off"], r["margin_vs_expert"] = r["whole"] - r["off"], r["whole"] - r["expert"]
        r["whole_eq_off_bitwise"] = bool(torch.equal(A["whole"]["outs"][i], A["off"]["outs"][i]))
        r["whole_vs_expert_pcc"] = pcc(A["whole"]["outs"][i], A["expert"]["outs"][i])
        rows.append(r)
    mo = [r["margin_vs_off"] for r in rows]
    me = [r["margin_vs_expert"] for r in rows]
    s = {"n": len(rows), "pass_vs_off": sum(m >= 0 for m in mo), "pass_vs_expert": sum(m >= 0 for m in me),
         "mean": {arm: statistics.mean(r[arm] for r in rows) for arm in A},
         "min": {arm: min(r[arm] for r in rows) for arm in A},
         "margin_vs_off": {"min": min(mo), "median": statistics.median(mo), "max": max(mo),
                           "sorted": sorted(round(m, 5) for m in mo)},
         "margin_vs_expert": {"min": min(me), "median": statistics.median(me), "max": max(me)},
         "argmin_vs_off": rows[mo.index(min(mo))]["tag"], "argmin_vs_expert": rows[me.index(min(me))]["tag"],
         "any_whole_eq_off": any(r["whole_eq_off_bitwise"] for r in rows),
         "max_whole_vs_expert_pcc": max(r["whole_vs_expert_pcc"] for r in rows)}
    if shape == "base":
        s["seed_707"] = next(r for r in rows if r["tag"] == 707)
    res[f"seed_gate_{shape}"] = {"summary": s, "rows": rows}
    # P2-3 layer gate
    kvrows, worst = [], []
    for t, kvw in A["whole"]["kvs"].items():
        kvo, kvr = A["off"]["kvs"][t], R["kv"][t]
        valid = R["valid"][t].reshape(-1)
        P = valid.shape[0]
        same = bool(all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(A["expert"]["kvs"][t], kvo)))
        for l in range(18):
            for j, nm in enumerate("KV"):
                ref = kvr[l][j][0, 0, :P][valid]
                w = kvw[l][j][0, 0, :P][valid]
                o = kvo[l][j][0, 0, :P][valid]
                kvrows.append({"tag": str(t), "layer": l, "t": nm, "pcc_whole": pcc(w, ref), "pcc_ttnn": pcc(o, ref),
                               "rel_whole": rel(w, ref), "rel_ttnn": rel(o, ref), "pcc_whole_vs_ttnn": pcc(w, o),
                               "expert_kv_eq_off": same})
    n_ok = sum(r["pcc_whole"] >= r["pcc_ttnn"] for r in kvrows)
    n_ok_rel = sum(r["rel_whole"] <= r["rel_ttnn"] for r in kvrows)
    res[f"p23_{shape}"] = {"n": len(kvrows), "n_inputs": len(A["whole"]["kvs"]), "pcc_closer": n_ok, "rel_closer": n_ok_rel,
                           "whole_min_pcc": min(r["pcc_whole"] for r in kvrows), "ttnn_min_pcc": min(r["pcc_ttnn"] for r in kvrows),
                           "min_pcc_margin": min(r["pcc_whole"] - r["pcc_ttnn"] for r in kvrows),
                           "whole_vs_ttnn_min_pcc": min(r["pcc_whole_vs_ttnn"] for r in kvrows),
                           "expert_kv_eq_off_all": all(r["expert_kv_eq_off"] for r in kvrows),
                           "pass": n_ok == len(kvrows) and n_ok_rel == len(kvrows), "rows": kvrows}
    print(shape, json.dumps(s, default=str)[:1500])
    print(shape, "P23", {k: v for k, v in res[f"p23_{shape}"].items() if k != "rows"})
json.dump(res, open(f"{OUT}/G_gates_{rnd}.json", "w"), indent=1, default=str)
