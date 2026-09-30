"""verify-p1-r1 amended base gate (user 2026-09-30): per seed, PCC(megakernel, oracle) >= PCC(shipped, oracle), where
oracle = my fp32 expert loop (oracle.py) around the torch reference, fed the device's OWN prefix K/V (bf8 caches read
back) + the device noise, over >= 18 seeds. Each arm is scored against the oracle built from ITS OWN K/V.
Controls: prefix K/V and noise bit-identical across arms (the gate isolates the expert); negative control = the oracle
fed ANOTHER seed's K/V must be clearly farther (the comparison is sensitive to the prefix it is fed)."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from oracle import expert_loop  # noqa: E402
from vc import BASE_WEIGHTS, OUT, base_config, now, pcc  # noqa: E402

from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model

torch.set_grad_enabled(False)
R = torch.load(f"{OUT}/ref_base.pt", weights_only=False)
D = {a: torch.load(f"{OUT}/A_{a}_base.pt", weights_only=False) for a in ("off", "default")}
assert D["off"]["tags"] == D["default"]["tags"]
ref = PI0Model(base_config(), PI0WeightLoader(BASE_WEIGHTS))
tags = D["off"]["tags"]
rows = []
for i, seed in enumerate(tags):
    row = {"seed": seed, "n_lang": int(R["valid"][seed].sum()) - 512}
    row["kv_identical_across_arms"] = all(torch.equal(D["off"]["kvs"][i][l][j], D["default"]["kvs"][i][l][j])
                                          for l in range(18) for j in range(2))
    row["noise_identical_across_arms"] = bool(torch.equal(D["off"]["noises"][i], D["default"]["noises"][i]))
    orc = {}
    for a in ("off", "default"):
        kv = [(k.float(), v.float()) for k, v in D[a]["kvs"][i]]
        orc[a] = expert_loop(ref, kv, R["valid"][seed], D[a]["noises"][i], 50)
        row[f"{a}_vs_oracle"] = pcc(D[a]["outs"][i], orc[a])
        row[f"{a}_vs_fullref"] = pcc(D[a]["outs"][i], R["refs"][seed])
    row["oracle_vs_fullref"] = pcc(orc["off"], R["refs"][seed])
    row["oracles_equal_across_arms"] = bool(torch.equal(orc["off"], orc["default"]))
    row["mk_vs_off"] = pcc(D["default"]["outs"][i], D["off"]["outs"][i])
    row["pass"] = row["default_vs_oracle"] >= row["off_vs_oracle"]
    if i < 3:  # negative control: another seed's K/V (same validity, same noise)
        j = (i + 7) % len(tags)
        kvw = [(k.float(), v.float()) for k, v in D["default"]["kvs"][j]]
        wrong = expert_loop(ref, kvw, R["valid"][seed], D["default"]["noises"][i], 50)
        row["neg_ctrl_mk_vs_oracle_wrong_kv"] = pcc(D["default"]["outs"][i], wrong)
    rows.append(row)
    print(now(), json.dumps(row), flush=True)
n = len(rows)
summ = {"n": n, "n_pass": sum(r["pass"] for r in rows), "all_pass": all(r["pass"] for r in rows),
        "mean_mk_vs_oracle": sum(r["default_vs_oracle"] for r in rows) / n, "min_mk_vs_oracle": min(r["default_vs_oracle"] for r in rows),
        "mean_off_vs_oracle": sum(r["off_vs_oracle"] for r in rows) / n, "min_off_vs_oracle": min(r["off_vs_oracle"] for r in rows),
        "mean_mk_vs_fullref": sum(r["default_vs_fullref"] for r in rows) / n,
        "mean_off_vs_fullref": sum(r["off_vs_fullref"] for r in rows) / n,
        "n_mk_closer_fullref": sum(r["default_vs_fullref"] >= r["off_vs_fullref"] for r in rows),
        "kv_identical_all": all(r["kv_identical_across_arms"] for r in rows),
        "noise_identical_all": all(r["noise_identical_across_arms"] for r in rows),
        "min_margin": min(r["default_vs_oracle"] - r["off_vs_oracle"] for r in rows),
        "oracle_ctrl": R["ctrl"]}
json.dump({"rows": rows, "summary": summ}, open(f"{OUT}/O_ip1.json", "w"), indent=1)
print("SUMMARY", json.dumps(summ))
