"""CPU emulation: which quantisation dominates the prefix K/V error vs fp32 (host_vlm with quantisers)."""
import os, sys, json, time
import torch, torch.nn.functional as F
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.common.fused_host import im2col_patches, prefix_valid_mask
from models.experimental.pi0_5.tt.megakernel import geometry as G, pe_geometry as P, pe_host as H

torch.set_num_threads(32)
def bf16(t): return t.to(torch.bfloat16).float()
def bfp8(t):
    s = t.shape
    b = t.reshape(-1, 16)
    m = b.abs().amax(1, keepdim=True).clamp_min(1e-38)
    e = torch.floor(torch.log2(m))
    step = torch.pow(2.0, e - 6)
    q = torch.clamp(torch.round(b / step), -127, 127) * step
    return q.reshape(s)
ident = lambda t: t

wl = PI0WeightLoader(os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base"))
cat = wl.categorized_weights
pp = H.prefix_params(cat)
ps = P.pshape_for(G.SHAPES["base"])
g = torch.Generator().manual_seed(0)
pix = torch.rand(2, 3, 224, 224, generator=g) * 2 - 1
n_lang = 150
tokens = torch.zeros(1, ps.ntok, dtype=torch.int64)
tokens[0, :n_lang] = torch.randint(1, 257152, (n_lang,), generator=g)
lmask = torch.zeros(1, ps.ntok, dtype=torch.bool); lmask[0, :n_lang] = True
valid = prefix_valid_mask(2, lmask)[0]
im = im2col_patches(pix, 14, pad_to=608).reshape(512, 608).to(torch.bfloat16).float()
t0 = time.time()
img = H.host_siglip(pp, im) @ pp.proj_w + pp.proj_b
ew = cat["vlm_language"].get("model.embed_tokens.weight")
ew = cat["vlm_language"]["lm_head.weight"] if ew is None else ew
Pr = ps.ptv * 32
x0 = torch.zeros(ps.mt * 32, 2048); x0[:512] = img; x0[512:Pr] = ew[tokens[0]].to(torch.bfloat16).float() * (2048 ** 0.5)
print('siglip', time.time() - t0, flush=True)
tab = H.rope_tables(ps)
rows = ps.mt * 32
bias = torch.where(valid.reshape(-1).bool(), 0.0, -1.0e9)[None, :]
def rope(t, c, s):
    h = t.shape[-1] // 2
    return t * c + torch.cat([t[..., h:], t[..., :h]], -1) * s
wq = {}
def run(qa, qh, qw, qkv_out):
    x = x0.clone(); kv = []
    for li, lp in enumerate(pp.vlm):
        W = wq.setdefault((qw.__name__, li), (qw(lp.wqkv), qw(lp.wo), qw(lp.wug), qw(lp.wd)))
        xn = qa(H._rms(x, lp.g1, pp.eps_v))
        qkv = xn @ W[0]
        q = qkv[:, :2048].reshape(rows, 8, 256).transpose(0, 1)
        k = qkv[:, 2048:2304]; v = qkv[:, 2304:2560]
        q = rope(q, tab["cosq"], tab["sinq"]); k = rope(k, tab["cosk"], tab["sink"])
        kp, vp = qkv_out(k[:Pr]), qkv_out(v[:Pr])
        kv.append((kp, vp))
        if li == len(pp.vlm) - 1: break
        a = torch.softmax(qa(q) @ kp.T + bias, dim=-1) @ vp
        x = x + qa(a.transpose(0, 1).reshape(rows, 2048)) @ W[1]
        xn = qa(H._rms(x, lp.g2, pp.eps_v))
        ug = xn @ W[2]
        x = x + qh(ug[:, :16384] * F.gelu(ug[:, 16384:], approximate="tanh")) @ W[3]
    return kv
ref = run(ident, ident, ident, ident)
vr = valid
def rel(a, b): return float((a - b).norm() / b.norm())
res = {}
for name, cfg in [("device", (bf16, bfp8, bfp8, bfp8)), ("h_bf16", (bf16, bf16, bfp8, bfp8)),
                  ("w_fp32", (bf16, bfp8, ident, bfp8)), ("act_fp32", (ident, bfp8, bfp8, bfp8)),
                  ("kv_fp32", (bf16, bfp8, bfp8, ident)), ("only_kvq", (ident, ident, ident, bfp8))]:
    t0 = time.time()
    kv = run(*cfg)
    r = [(round(rel(kv[l][0][vr], ref[l][0][vr]), 4), round(rel(kv[l][1][vr], ref[l][1][vr]), 4)) for l in range(18)]
    res[name] = r
    print(name, 'L0', r[0], 'L8', r[8], 'L17', r[17], '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
