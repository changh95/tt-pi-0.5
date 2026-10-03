"""DRAFT (ASD-STE100 prose): write the card: block of draft_ste/tt-model.yaml from measured files only (prose in ste_text.py).
usage: card_mc.py --bench BENCH_JSON --image TAG --check IMG_CHECK_JSON [--perf PERF_JSON] [--served SERVED_JSON]
Every number below is read from a file named in SRC (asserted where the file and a fixed value must agree)."""
import argparse, json, os, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--bench", required=True)  # image bench, default profile (c2 H50 N10), 100 warm requests
ap.add_argument("--image", required=True)
ap.add_argument("--check", required=True)  # img_check.json of the image
ap.add_argument("--profiles", required=True)  # dir with bench-<profile>-*.json of the validate run (30 requests each)
ap.add_argument("--source-commit", required=True)
ap.add_argument("--yaml", default="/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc/draft_ste/tt-model.yaml")
a = ap.parse_args()

MK = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel"
W = "/tmp/claude-1000/-home-deepgadget-experiments-gr00t/12fd0cff-5a02-45cf-ac15-0681e5ee81aa/scratchpad/wp6"
LM = "/home/deepgadget/experiments/gr00t/libero_eval/pi05/tt_mc_matrix"
SRC = {}

bench = json.load(open(a.bench)); SRC["bench"] = a.bench
assert bench["n"] == 100 and bench["identical_actions_all"]
st = bench["stats"]
inf, p90, tot, wall = st["inference"]["median"], st["inference"]["p90"], st["total"]["median"], st["client_wall"]["median"]
prev = json.load(open(f"{MK}/publish_p2/results/bench-c2-mkp2-20261001-131455.json"))["stats"]
assert round(prev["inference"]["median"], 2) == 55.84
prev_inf, prev_tot = prev["inference"]["median"], prev["total"]["median"]
fr = bench["first_response"]

chk = json.load(open(a.check)); SRC["check"] = a.check
g = chk["golden"]; assert g["all_inputs_equal"] and g["n"] == 8
bit = chk["bitid"]
assert chk["refuse_all_ok"]

T = json.load(open(f"{W}/table.json"))
assert T["sets_evaluated"] == T["sets_expected"] == 320 and not T["missing_control"]
pc = T["pass_counts"]
assert pc["A2"] == "314/320" and pc["A4"] == "286/288"
a2 = sorted([c for c in T["failing_cells"] if c["gate"] == "A2"], key=lambda c: c["N"])
a4 = [c for c in T["failing_cells"] if c["gate"] == "A4"]
n1 = T["n1_a4"]
GI = json.load(open(f"{W}/gpu_inc_c2_L224_H64.json"))
gpu_bf16 = {int(N): [s["pcc"] for s in v["seeds"] if s["seed"] == 5][0] for N, v in GI["arms"]["bf16"]["per_N"].items()}
mk_s5 = {c["N"]: [s["pcc"] for s in c["seeds"] if s["seed"] == 5][0] for c in a2}
# per-seed A2 of the failing cell (c2 L224 S64 H64) at every N, recomputed from wp6/out/outs/c2_S64_N*.pt vs the matrix
# refs with the same comparand as wp6_an.py (min asserted == an/c2_S64_N*.gates.json a2.L224_H64.min)
CS = "/home/deepgadget/experiments/tt-models/ports/tt-pi-0.5/docs/megakernel/publish_mc/results/a2_cell_seeds.json"
cs = {int(k): v for k, v in json.load(open(CS)).items()}; SRC["a2_cell_seeds"] = CS
for c in a2:
    assert abs(cs[c["N"]][5] - [x["pcc"] for x in c["seeds"] if x["seed"] == 5][0]) < 1e-12
others = min(min(cs[N][:5]) for N in range(5, 11)); others_all = min(min(cs[N][:5]) for N in range(1, 11))

P = json.load(open(f"{LM}/paired_vs_gpu.json"))
LS = {k: json.load(open(f"{LM}/{k}_summary.json")) for k in P}
for k in P:
    assert LS[k]["episodes"] == 100 and not LS[k]["errors"] and LS[k]["success"] == P[k]["tt_success"]

f = lambda x, d=2: f"{x:.{d}f}"
import glob
PROF = [("c2-h50-n10", None, 2, 50, 10), ("c1-h50-n10", "c1-h50-n10", 1, 50, 10), ("c3-h50-n10", "c3-h50-n10", 3, 50, 10),
        ("c4-h50-n10", "c4-h50-n10", 4, 50, 10), ("c2-h10-n10", "c2-h10-n10", 2, 10, 10), ("c2-h10-n5", "c2-h10-n5", 2, 10, 5)]
prof_rows = []
for label, pname, cams, H, N in PROF:
    if pname is None:
        b = bench
    else:
        fs = glob.glob(f"{a.profiles}/bench-{pname}-*.json"); assert len(fs) == 1, fs
        b = json.load(open(fs[0])); SRC[pname] = fs[0]
        assert b["identical_actions_all"] and b["n"] == 30
    fr_ = b["first_response"]
    assert (fr_["images_used"], fr_["action_horizon"], fr_["denoising_steps"]) == (cams, H, N), (label, fr_)
    tag = "c2-default-c2" if pname is None else pname
    fi = glob.glob(f"{a.profiles}/info-{tag}-*.json"); assert len(fi) == 1, fi
    info = json.load(open(fi[0])); SRC[f"info {tag}"] = fi[0]
    ops = info["megakernel"]["program"]["device_ops_per_call"]
    assert info["megakernel"]["backend"] == "mc" and ops == (4 if cams >= 3 else 3), (tag, ops)
    prof_rows.append(f"| `{label}`{' (default)' if pname is None else ''} | {cams} | {H} | {N} | {ops} | {fr_['prompt_bucket']} | **{f(b['stats']['inference']['median'])}** | "
                     f"{f(b['stats']['total']['median'])} | {b['n']} |")
prof_md = "\n".join(prof_rows)

# ---------------------------------------------------------------- tables
libero_rows = "\n".join(
    f"| {k[1]} | {k.split('_n')[1]} | **{P[k]['tt_success']} / 100** | {P[k]['gpu_success']} / 100 | "
    f"{len(P[k]['tt_only'])} / {len(P[k]['gpu_only'])} | {f(LS[k]['policy_latency']['mean_ms'], 1)} ms |"
    for k in ["c2_n10", "c2_n5", "c2_n1", "c1_n10", "c1_n5", "c1_n1"])

bit_rows = "\n".join(
    f"| {r['cameras']} | {r['preset'][1]} ({r['n_tokens']} real tokens) | {r['preset'][2]} (H = {r['H']}) | {r['N']} | "
    f"{r['device_ops_per_call']} | {'DRAM' if r['kv_dram'] else 'L1'} |"
    for r in (bit[c] for c in ["c1", "c2", "c3", "c4"]))

RT = "/home/deepgadget/experiments/tt-metal-pr/.val/mc_impl/wp6/perf/out/RELEASE_TABLE.md"
rt = open(RT).read(); SRC["perf"] = RT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ste_text

a2_tab = ("| steps | " + " | ".join(str(N) for N in sorted(mk_s5)) + " |\n|---|" + "---|" * len(mk_s5) + "\n"
          "| openpi GPU bf16 vs fp32 | " + " | ".join(f(gpu_bf16[N], 3) for N in sorted(mk_s5)) + " |\n"
          "| this megakernel vs fp32 | " + " | ".join(f(mk_s5[N], 3) for N in sorted(mk_s5)) + " |")
V = dict(inf=inf, p90=p90, tot=tot, wall=wall, prev_inf=prev_inf, prev_tot=prev_tot, fr=fr, g=g, pc=pc, T=T, a4=a4, n1=n1,
         digest=chk["kernel_digest"], prof_md=prof_md, perf_md=ste_text.perf_md(rt, f), bench_md=ste_text.bench_md(rt), libero_rows=libero_rows,
         a2_tab=a2_tab, others=others, others_all=others_all, src=a.source_commit)
desc = ste_text.description(V, f)
quick = ste_text.quickstart(V, f)

def block(text):
    return "\n".join(("    " + l) if l.strip() else "" for l in text.rstrip("\n").split("\n"))

y = open(a.yaml).read()
i = y.index("\ncard:\n")
y = y[:i] + "\ncard:\n  description: |\n" + block(desc) + "\n  quickstart: |\n" + block(quick) + "\n"
open(a.yaml, "w").write(y)
import yaml
d = yaml.safe_load(open(a.yaml))
assert d["card"]["quickstart"].startswith("### Run with tt-cli")
print("card written; sources:", json.dumps(SRC))
print("bitid table (for the report):\n" + bit_rows)
