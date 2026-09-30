"""CPU emulation: SigLIP output error vs fp32 by quantisation choice (with the kernel's folded parameters)."""
import os, sys, time
import torch, torch.nn.functional as F
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader
from models.experimental.pi0_5.common.fused_host import im2col_patches
from models.experimental.pi0_5.tt.megakernel import pe_host as H
torch.set_num_threads(32)
def bf16(t): return t.to(torch.bfloat16).float()
def bfp8(t):
    s = t.shape; b = t.reshape(-1, 16)
    m = b.abs().amax(1, keepdim=True).clamp_min(1e-38); e = torch.floor(torch.log2(m)); step = torch.pow(2.0, e - 6)
    return (torch.clamp(torch.round(b / step), -127, 127) * step).reshape(s)
ident = lambda t: t
cat = PI0WeightLoader(os.environ.get("PI05_WEIGHTS_DIR", "lerobot/pi05_base")).categorized_weights
pp = H.prefix_params(cat)
g = torch.Generator().manual_seed(0)
pix = torch.rand(2, 3, 224, 224, generator=g) * 2 - 1
im = im2col_patches(pix, 14, pad_to=608).reshape(512, 608).to(torch.bfloat16).float()
def run(qa, qw, qh, wsel, qs=lambda t: t, qp=lambda t: t):
    x = im @ pp.patch_w + pp.pos_b.repeat(2, 1)
    for s in pp.sig:
        W = [qw(w) if sel else bf16(w) if sel == 0 else w for w, sel in zip((s.wqkv, s.wo, s.wfc1, s.wfc2), wsel)]
        xn = qa(H._ln(x, s.ln1w, s.ln1b, pp.eps_s))
        qkv = xn @ W[0] + s.bqkv
        q, k, v = qkv.split(1536, dim=1)
        ctx = torch.zeros(512, 1536)
        for img in range(2):
            rs = slice(img * 256, (img + 1) * 256)
            qh_, kh, vh = (qa(u[rs]).reshape(256, 16, 96).transpose(0, 1) for u in (q, k, v))
            sc = qs(qh_ @ kh.transpose(1, 2)); m = sc.amax(-1, keepdim=True); p = qp(torch.exp(sc - m))
            ctx[rs] = ((p @ vh) / p.sum(-1, keepdim=True)).transpose(0, 1).reshape(256, 1536)
        x = x + qa(ctx) @ W[1] + s.bo
        xn = qa(H._ln(x, s.ln2w, s.ln2b, pp.eps_s))
        x = x + qh(F.gelu(xn @ W[2] + s.bfc1, approximate="tanh")) @ W[3] + s.bfc2
    return H._ln(x, pp.post_w, pp.post_b, pp.eps_s) @ pp.proj_w + pp.proj_b
ref = run(ident, ident, ident, (None, None, None, None))
rel = lambda a: float((a - ref).norm() / ref.norm())
for name, cfg in [("device(bfp8 W, bf16 act/h)", (bf16, bfp8, bf16, (1, 1, 1, 1))),
                  ("+ bf16 scores + bf16 P", (bf16, bfp8, bf16, (1, 1, 1, 1), bf16, bf16)),
                  ("+ bf16 scores", (bf16, bfp8, bf16, (1, 1, 1, 1), bf16)),
                  ("+ bf16 P", (bf16, bfp8, bf16, (1, 1, 1, 1), ident, bf16))]:
    t0 = time.time()
    print('%-28s rel %.4f  (%.0fs)' % (name, rel(run(*cfg)), time.time() - t0), flush=True)
