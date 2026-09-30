"""verify-p1-r1: parse the tracy op CSVs per traced replay session (METAL TRACE REPLAY SESSION ID).
Gate: after the LAST UpdateKVCache op of the replay (the VLM's last cache write) exactly ONE op, a GenericOp whose
kernels are the megakernel's; and the prefix (ops up to and incl. the last cache write) is the SAME op-code sequence
in both arms (no expert work hidden before the cache write). A/B: the off arm's post-cache ops are the stock graph."""
import collections
import csv
import gzip
import json
import statistics

OUT = __file__.rsplit("/", 1)[0] + "/out"


def sessions(path):
    rows = list(csv.DictReader(gzip.open(path, "rt")))
    s = collections.defaultdict(list)
    for r in rows:
        k = r.get("METAL TRACE REPLAY SESSION ID", "")
        if k:
            s[int(k)].append(r)
    return rows, [s[k] for k in sorted(s)]


res, prefix = {}, {}
for arm in ("default", "off"):
    for shape in ("base", "libero"):
        try:
            rows, per = sessions(f"{OUT}/ops_{arm}_{shape}.csv.gz")
        except FileNotFoundError:
            continue
        info = []
        for ops in per:
            codes = [o["OP CODE"] for o in ops]
            kv = [i for i, c in enumerate(codes) if "UpdateKVCache" in c or "FillCache" in c or "PagedUpdate" in c]
            after = ops[kv[-1] + 1:]
            info.append({"n_ops": len(ops), "n_kv": len(kv), "last_kv": kv[-1], "n_after": len(after),
                         "after_codes": collections.Counter(o["OP CODE"] for o in after).most_common(8),
                         "after0": {k: after[0].get(k) for k in ("OP CODE", "CORE COUNT", "DEVICE KERNEL DURATION [ns]",
                                                                  "COMPUTE KERNEL SOURCE", "DATA MOVEMENT KERNEL SOURCE",
                                                                  "PROGRAM HASH")} if after else None,
                         "missing_dur": sum(1 for o in ops if not o.get("DEVICE KERNEL DURATION [ns]")),
                         "prefix_codes": codes[: kv[-1] + 1]})
        prefix[(arm, shape)] = info[0]["prefix_codes"]
        mk = [float(i["after0"]["DEVICE KERNEL DURATION [ns]"]) / 1e6 for i in info
              if i["n_after"] == 1 and i["after0"]["DEVICE KERNEL DURATION [ns]"]]
        res[f"{arm}_{shape}"] = {
            "n_rows_csv": len(rows), "n_sessions": len(info),
            "op_counts": sorted({i["n_ops"] for i in info}), "n_kv_ops": sorted({i["n_kv"] for i in info}),
            "n_after_last_kv": sorted({i["n_after"] for i in info}),
            "after_codes_first": info[0]["after_codes"], "after0_first": info[0]["after0"],
            "after0_program_hashes": sorted({str(i["after0"]["PROGRAM HASH"]) for i in info if i["after0"]}),
            "missing_durations": sum(i["missing_dur"] for i in info),
            "mk_kernel_ms": {"median": statistics.median(mk), "min": min(mk), "max": max(mk), "n": len(mk)} if mk and arm == "default" else None,
            "post_cache_device_ms_first": sum(float(o.get("DEVICE KERNEL DURATION [ns]") or 0) for o in per[0][info[0]["last_kv"] + 1:]) / 1e6}
for shape in ("base", "libero"):
    if ("default", shape) in prefix and ("off", shape) in prefix:
        res[f"prefix_codes_equal_{shape}"] = prefix[("default", shape)] == prefix[("off", shape)]
        res[f"prefix_len_ops_{shape}"] = len(prefix[("default", shape)])
json.dump(res, open(f"{OUT}/S_struct.json", "w"), indent=1)
print(json.dumps(res, indent=1)[:6000])
