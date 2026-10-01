"""final-critic: programs per trace replay in the pulled shipped image (cpp_device_perf_report.csv of the served run)."""
import csv, collections, json, statistics as st
rows = list(csv.DictReader(open("prof/.logs/cpp_device_perf_report.csv")))
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
sess = collections.defaultdict(list); non = []
for r in rows:
    s = r["METAL TRACE REPLAY SESSION ID"]
    (sess[(r["METAL TRACE ID"], int(s))] if s else non).append(r)
first_sess_gcc = min(int(v[0]["GLOBAL CALL COUNT"]) for v in sess.values()) if sess else None
after = [r for r in non if first_sess_gcc is not None and int(r["GLOBAL CALL COUNT"]) > first_sess_gcc]
dur = [float(v[0]["DEVICE KERNEL DURATION [ns]"]) / 1e6 for v in sess.values() if len(v) == 1]
res = {"n_rows": len(rows), "n_replay_sessions": len(sess), "trace_ids": sorted({k[0] for k in sess}),
       "programs_per_session": sorted(collections.Counter(len(v) for v in sess.values()).items()),
       "session_core_counts": sorted({r["CORE COUNT"] for v in sess.values() for r in v}),
       "session_global_call_counts": sorted({r["GLOBAL CALL COUNT"] for v in sess.values() for r in v}),
       "non_session_rows_total": len(non), "non_session_rows_after_first_replay": len(after),
       "device_kernel_ms_per_replay": {"median": st.median(dur), "min": min(dur), "max": max(dur), "n": len(dur)} if dur else None}
json.dump(res, open("FC_prof_pulled.json", "w"), indent=1); print(json.dumps(res))
