"""verify-p2-r0 LIBERO prompt-length edge inputs (n_lang 32 = no pad key, 1): images + noise of golden records 0 / 5,
random prompt ids of length n (seeded), right-padded to 32."""
import torch
from models.experimental.pi0_5.tests.pcc.golden_openpi import load_records, record_inputs

EDGE = [(0, 32, 901), (5, 32, 902), (0, 1, 903), (5, 1, 904)]


def edge_obs():
    recs = load_records()
    out = []
    for ri, n, seed in EDGE:
        im, tk, mk, nz = record_inputs(recs[ri], 32)
        g = torch.Generator().manual_seed(seed)
        tk = torch.zeros(1, 32, dtype=torch.long)
        tk[0, :n] = torch.randint(1, 256000, (n,), generator=g)
        mk = torch.zeros(1, 32, dtype=torch.bool)
        mk[0, :n] = True
        out.append((f"r{ri}_n{n}", (im, tk, nz, mk)))
    return out
