"""integrate-p2: request-path structural check from ops3_whole_base.csv.gz (d_prof3: first call = compile + capture,
then 9 requests alternating n 1 / 128 / 224). Rows in CSV order (GLOBAL CALL COUNT): replay sessions, their op
counts / codes, and the non-session rows AFTER the first replay session (a request-path device op outside the trace)."""
import collections, csv, gzip, json
OUT = __file__.rsplit("/", 1)[0] + "/out"
rows = list(csv.DictReader(gzip.open(f"{OUT}/ops3_whole_base.csv.gz", "rt")))
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
sid = [r.get("METAL TRACE REPLAY SESSION ID", "") for r in rows]
first = next(i for i, s in enumerate(sid) if s)
sess = collections.defaultdict(list)
for r, s in zip(rows, sid):
    if s:
        sess[int(s)].append(r["OP CODE"])
after = [r["OP CODE"] for r, s in zip(rows[first:], sid[first:]) if not s]
res = {"n_rows": len(rows), "n_sessions": len(sess), "ops_per_session": sorted({len(v) for v in sess.values()}),
       "session_op_codes": sorted({c for v in sess.values() for c in v}),
       "non_session_rows_before_first_session": first, "non_session_rows_after_first_session": len(after),
       "after_codes": collections.Counter(after).most_common(10)}
json.dump(res, open(f"{OUT}/S_prof3.json", "w"), indent=1)
print(json.dumps(res))
