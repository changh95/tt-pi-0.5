"""integrate-p2 speed gate: whole (default, env unset) vs expert vs off, same session, arms alternated across processes,
30 calls each, two rounds. se of the median = 1.4826 * MAD * 1.2533 / sqrt(n); pass = diff > 2 x sqrt(se_a^2 + se_b^2)."""
import json, math, statistics
OUT = __file__.rsplit("/", 1)[0] + "/out"
def st(x):
    m = statistics.median(x); mad = statistics.median([abs(v - m) for v in x])
    return m, 1.4826 * mad * 1.2533 / math.sqrt(len(x))
res = {}
for s in ("base", "libero"):
    for r in ("r1", "r2"):
        a = {k: json.load(open(f"{OUT}/A_{k}_{s}_{r}.json")) for k in ("whole", "expert", "off")}
        row = {}
        for k, d in a.items():
            m, se = st(d["call_ms"]["all"]); rm, rse = st(d["replay_ms"]["all"])
            row[k] = {"call_median": round(m, 3), "call_se": round(se, 3), "replay_median": round(rm, 3), "n": len(d["call_ms"]["all"]),
                      "aiclk": [d["smi_bench_before"].get("aiclk"), d["smi_bench_after"].get("aiclk")], "env": d["env"].get("PI05_MEGAKERNEL"),
                      "stamp": d["stamp"]["backend"], "replays_in_10s": d["replays_in_10s_time_time"]}
        for b in ("expert", "off"):
            diff = row[b]["call_median"] - row["whole"]["call_median"]
            bar = 2 * math.sqrt(row[b]["call_se"] ** 2 + row["whole"]["call_se"] ** 2)
            row[f"whole_faster_than_{b}"] = {"diff_ms": round(diff, 3), "2se": round(bar, 3), "pass": diff > bar}
        res[f"{s}_{r}"] = row
json.dump(res, open(f"{OUT}/G_speed.json", "w"), indent=1)
for k, v in res.items():
    print(k, {a: (v[a]["call_median"], v[a]["replay_median"], v[a]["call_se"], v[a]["env"], v[a]["aiclk"]) for a in ("whole", "expert", "off")}, v["whole_faster_than_expert"], v["whole_faster_than_off"])
