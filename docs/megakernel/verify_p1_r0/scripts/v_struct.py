"""Parse the profiled runs: per traced replay session, op count, ops after the LAST UpdateKVCache / fill-cache op
(the VLM's last cache write), their identity, and the megakernel kernel duration."""
import collections
import csv
import gzip
import json
import statistics
import sys

OUT = __file__.rsplit("/", 1)[0] + "/out"
res = {}
for arm in ("expert", "off"):
    for shape in ("base", "libero"):
        p = f"{OUT}/ops_{arm}_{shape}.csv.gz"
        try:
            rows = list(csv.DictReader(gzip.open(p, "rt")))
        except FileNotFoundError:
            continue
        sess = collections.defaultdict(list)
        for r in rows:
            s = r.get("METAL TRACE REPLAY SESSION ID", "")
            if s:
                sess[s].append(r)
        per = []
        for s, ops in sorted(sess.items(), key=lambda kv: int(kv[0])):
            codes = [o["OP CODE"] for o in ops]
            kv_idx = [i for i, c in enumerate(codes) if "KVCache" in c or "FillCache" in c or "UpdateCache" in c]
            last = kv_idx[-1] if kv_idx else -1
            after = ops[last + 1:]
            missing = sum(1 for o in ops if not o.get("DEVICE KERNEL DURATION [ns]"))
            per.append({"session": s, "ops": len(ops), "n_kv_ops": len(kv_idx), "last_kv_index": last,
                        "ops_after_last_kv": [(o["OP CODE"], o.get("COMPUTE KERNEL SOURCE", "")[-60:],
                                               o.get("CORE COUNT"), o.get("DEVICE KERNEL DURATION [ns]"),
                                               o.get("PROGRAM HASH")) for o in after],
                        "n_generic": sum(1 for c in codes if "Generic" in c),
                        "missing_device_duration": missing,
                        "span_ns_sum_kernel": sum(float(o.get("DEVICE KERNEL DURATION [ns]") or 0) for o in ops)})
        mk = [float(p_["ops_after_last_kv"][0][3]) for p_ in per
              if len(p_["ops_after_last_kv"]) == 1 and "Generic" in p_["ops_after_last_kv"][0][0]]
        res[f"{arm}_{shape}"] = {"n_sessions": len(per), "op_counts": sorted({p_["ops"] for p_ in per}),
                                 "ops_after_last_kv_counts": sorted({len(p_["ops_after_last_kv"]) for p_ in per}),
                                 "first_session": per[0] if per else None,
                                 "mk_kernel_ms_median": statistics.median(mk) / 1e6 if mk else None,
                                 "mk_kernel_ms_all": [round(x / 1e6, 4) for x in mk],
                                 "missing_durations": sum(p_["missing_device_duration"] for p_ in per),
                                 "op_code_histogram_first": collections.Counter(o["OP CODE"] for o in sess[sorted(sess, key=int)[0]]).most_common(40) if per else None}
json.dump(res, open(f"{OUT}/S_verifier.json", "w"), indent=1)
for k, v in res.items():
    print(k, json.dumps({kk: vv for kk, vv in v.items() if kk not in ("op_code_histogram_first",)})[:1500])
