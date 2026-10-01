"""verify-p2-r0 structural gate from the tracy op CSVs, per traced replay session (METAL TRACE REPLAY SESSION ID).
whole: every replay = exactly ONE op, a GenericOp with whole_* kernels. expert / off: op counts and codes (the arms
differ), the prefix op-code sequence (ops up to the last cache write) of expert == off."""
import collections, csv, gzip, json, statistics
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
for arm in ("whole", "expert", "off"):
    for shape in ("base", "libero"):
        try:
            rows, per = sessions(f"{OUT}/ops_{arm}_{shape}.csv.gz")
        except FileNotFoundError:
            continue
        codes = [[o["OP CODE"] for o in p] for p in per]
        d = {"n_rows_csv": len(rows), "n_sessions": len(per), "ops_per_session": sorted({len(p) for p in per}),
             "non_session_rows": len(rows) - sum(len(p) for p in per),
             "missing_durations": sum(1 for p in per for o in p if not o.get("DEVICE KERNEL DURATION [ns]")),
             "op_code_counts_first": collections.Counter(codes[0]).most_common(12) if codes else None}
        if arm == "whole":
            dur = [float(p[0]["DEVICE KERNEL DURATION [ns]"]) / 1e6 for p in per if len(p) == 1 and p[0].get("DEVICE KERNEL DURATION [ns]")]
            d.update({"op_codes": sorted({c for cs in codes for c in cs}),
                      "kernel_sources": sorted({(o.get("COMPUTE KERNEL SOURCE", ""), o.get("DATA MOVEMENT KERNEL SOURCE", "")) for p in per for o in p}),
                      "core_counts": sorted({o.get("CORE COUNT") for p in per for o in p}),
                      "program_hashes": sorted({o.get("PROGRAM HASH") for p in per for o in p}),
                      "device_ms": {"median": statistics.median(dur), "min": min(dur), "max": max(dur), "n": len(dur)} if dur else None})
            d["pass"] = (d["ops_per_session"] == [1] and len(per) >= 20 and d["op_codes"] == ["GenericOpDeviceOperation"] or
                         d["ops_per_session"] == [1] and len(per) >= 20 and all("Generic" in c for c in d["op_codes"])) and \
                all("whole_trisc" in c[0] and "whole_brisc" in c[1] and "whole_ncrisc" in c[1] for c in d["kernel_sources"])
        else:
            kv = [i for i, c in enumerate(codes[0]) if "UpdateKVCache" in c or "FillCache" in c or "PagedUpdate" in c]
            prefix[(arm, shape)] = codes[0][: kv[-1] + 1]
            after = per[0][kv[-1] + 1:]
            d.update({"n_kv": len(kv), "n_after_last_kv": len(after),
                      "after0_sources": (after[0].get("COMPUTE KERNEL SOURCE"), after[0].get("DATA MOVEMENT KERNEL SOURCE")) if after else None,
                      "session_device_ms_median": statistics.median(sum(float(o.get("DEVICE KERNEL DURATION [ns]") or 0) for o in p) / 1e6 for p in per)})
        res[f"{arm}_{shape}"] = d
for shape in ("base", "libero"):
    if ("expert", shape) in prefix and ("off", shape) in prefix:
        res[f"prefix_codes_equal_expert_off_{shape}"] = prefix[("expert", shape)] == prefix[("off", shape)]
json.dump(res, open(f"{OUT}/S_struct.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str)[:8000])
