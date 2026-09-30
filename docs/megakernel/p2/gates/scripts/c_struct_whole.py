"""mk2: structural gate of phase 2 from the tracy op CSVs, per traced replay session (METAL TRACE REPLAY SESSION ID):
the replay holds EXACTLY ONE device op, a GenericOp whose kernels are the whole-model megakernel (whole_*.cpp)."""
import collections, csv, gzip, json, statistics, sys
OUT = __file__.rsplit("/", 1)[0] + "/out"
res = {}
for shape in ("base", "libero"):
    try:
        rows = list(csv.DictReader(gzip.open(f"{OUT}/ops_whole_{shape}.csv.gz", "rt")))
    except FileNotFoundError:
        continue
    s = collections.defaultdict(list)
    for r in rows:
        k = r.get("METAL TRACE REPLAY SESSION ID", "")
        if k:
            s[int(k)].append(r)
    per = [s[k] for k in sorted(s)]
    dur = [float(p[0]["DEVICE KERNEL DURATION [ns]"]) / 1e6 for p in per if len(p) == 1 and p[0].get("DEVICE KERNEL DURATION [ns]")]
    res[shape] = {"n_rows_csv": len(rows), "n_sessions": len(per), "ops_per_session": sorted({len(p) for p in per}),
                  "op_codes": sorted({o["OP CODE"] for p in per for o in p}),
                  "kernel_sources": sorted({(o.get("COMPUTE KERNEL SOURCE", ""), o.get("DATA MOVEMENT KERNEL SOURCE", "")) for p in per for o in p}),
                  "core_counts": sorted({o.get("CORE COUNT") for p in per for o in p}),
                  "program_hashes": sorted({o.get("PROGRAM HASH") for p in per for o in p}),
                  "non_session_rows": len(rows) - sum(len(p) for p in per),
                  "whole_kernel_ms": {"median": statistics.median(dur), "min": min(dur), "max": max(dur), "n": len(dur)} if dur else None}
    res[shape]["pass"] = res[shape]["ops_per_session"] == [1] and len(per) >= 20 and all("whole_trisc" in c[0] for c in res[shape]["kernel_sources"])
json.dump(res, open(f"{OUT}/S_struct_whole.json", "w"), indent=1)
print(json.dumps(res, indent=1))
