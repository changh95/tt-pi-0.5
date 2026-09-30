"""CPU: O1-method oracle per seed, re-derived with the torch reference's expert loop fed each ARM's own device prefix
K/V + device noise. Gate (user amendment 2026-09-30): per seed, PCC(mk, oracle_mk) >= PCC(shipped, oracle_shipped)."""
import json
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from vcommon import BASE_WEIGHTS, OUT, base_config, now, oracle, pcc  # noqa: E402

from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.reference.torch_pi0_model import PI0Model

torch.set_grad_enabled(False)
R = torch.load(f"{OUT}/ref_base.pt", weights_only=False)
D = {a: torch.load(f"{OUT}/A_{a}_base.pt", weights_only=False) for a in ("off", "expert")}
ref = PI0Model(base_config(), PI0WeightLoader(BASE_WEIGHTS))
rows = []
for i, seed in enumerate(D["off"]["tags"]):
    row = {"seed": seed}
    kv_same = all(torch.equal(D["off"]["kvs"][i][l][j], D["expert"]["kvs"][i][l][j]) for l in range(18) for j in range(2))
    row["prefix_kv_bit_identical_across_arms"] = kv_same
    row["noise_identical"] = bool(torch.equal(D["off"]["noises"][i], D["expert"]["noises"][i]))
    for a in ("off", "expert"):
        kv = [(k.reshape(1, 1, -1, 256), v.reshape(1, 1, -1, 256)) for k, v in D[a]["kvs"][i]]
        o = oracle(ref, kv, R["valid"][seed], D[a]["noises"][i], 50)
        if a == "off":
            row["oracle_vs_ref"] = pcc(o, R["refs"][seed])
        row[f"{a}_vs_oracle"] = pcc(D[a]["outs"][i], o)
        row[f"{a}_vs_ref"] = pcc(D[a]["outs"][i], R["refs"][seed])
    row["mk_vs_off"] = pcc(D["expert"]["outs"][i], D["off"]["outs"][i])
    row["mk_not_worse"] = row["expert_vs_oracle"] >= row["off_vs_oracle"]
    rows.append(row)
    print(now(), json.dumps(row), flush=True)
summ = {"n": len(rows), "mk_not_worse_all": all(r["mk_not_worse"] for r in rows),
        "n_mk_not_worse": sum(r["mk_not_worse"] for r in rows),
        "mean_off_vs_oracle": sum(r["off_vs_oracle"] for r in rows) / len(rows),
        "mean_mk_vs_oracle": sum(r["expert_vs_oracle"] for r in rows) / len(rows),
        "min_off_vs_oracle": min(r["off_vs_oracle"] for r in rows),
        "min_mk_vs_oracle": min(r["expert_vs_oracle"] for r in rows),
        "mean_off_vs_ref": sum(r["off_vs_ref"] for r in rows) / len(rows),
        "mean_mk_vs_ref": sum(r["expert_vs_ref"] for r in rows) / len(rows),
        "kv_identical_all": all(r["prefix_kv_bit_identical_across_arms"] for r in rows),
        "ref_ctrl": R["ctrl"]}
json.dump({"rows": rows, "summary": summ}, open(f"{OUT}/O_verifier.json", "w"), indent=1)
print("SUMMARY", json.dumps(summ))
