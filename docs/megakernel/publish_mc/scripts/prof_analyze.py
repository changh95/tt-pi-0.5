"""Programs per trace replay session in cpp_device_perf_report.csv (device profiler of prof_ops.py)."""
import csv, collections, json, statistics as st, sys
rows = list(csv.DictReader(open(sys.argv[1]))); rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
sess = collections.defaultdict(list); non = []
for r in rows:
    s = r["METAL TRACE REPLAY SESSION ID"]
    (sess[(r["METAL TRACE ID"], int(s))] if s else non).append(r)
per = collections.Counter(len(v) for v in sess.values())
by_trace = collections.defaultdict(list)
for (t, s), v in sess.items():
    by_trace[t].append((len(v), sorted({r["OP NAME"] for r in v}), sorted({r["CORE COUNT"] for r in v}),
                        sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in v) / 1e6))
first = min(int(v[0]["GLOBAL CALL COUNT"]) for v in sess.values())
res = {"n_rows": len(rows), "n_replay_sessions": len(sess), "programs_per_session": sorted(per.items()),
       "per_trace": {t: {"sessions": len(v), "programs": sorted({x[0] for x in v}), "op_codes": v[0][1], "cores": v[0][2],
                         "device_ms_median": st.median(x[3] for x in v)} for t, v in sorted(by_trace.items(), key=lambda kv: int(kv[0]))},
       "non_session_rows_total": len(non), "non_session_rows_after_first_replay": len([r for r in non if int(r["GLOBAL CALL COUNT"]) > first])}
json.dump(res, open(sys.argv[2], "w"), indent=1); print(json.dumps(res, indent=1))
