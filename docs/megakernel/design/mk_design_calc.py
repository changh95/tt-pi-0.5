BF16, BFP8, FP32 = 2048, 1088, 4096
# ---- weights per expert layer (tiles) ----
qkv = 32*80; o = 64*32; ug = 32*256; dn = 128*32
mix = qkv*BFP8 + o*BF16 + ug*BFP8 + dn*BF16
allb8 = (qkv+o+ug+dn)*BFP8
print("expert layer bytes mix %d allbfp8 %d" % (mix, allb8))
print("per step mix MB %.1f, per request GB %.3f" % (mix*18/1e6, mix*180/1e9))
s_b8 = (qkv+ug)*BFP8/414e3; s_b16 = (o+dn)*BF16/464e3   # us (bytes / (GB/s*1e3))
print("stream per layer us: bfp8 part %.1f bf16 part %.1f total %.1f ; allbfp8 %.1f" % (s_b8, s_b16, s_b8+s_b16, allb8/414e3))
# ---- per-core shares ----
print("qkv share/core (80 cores, 32 K tiles bfp8) B", 32*BFP8)
print("o share/core (32 cores, 64 tiles bf16) B", 64*BF16)
print("ug share/core (64 cores, 4N x 32K bfp8) B", 128*BFP8)
print("down share/core (64 cores, 16K x 4N bf16) B", 64*BF16)
# ---- L1 heaviest compute core, phase 1 base ----
cbs = {
 "c0 cb_in0 arena max(x bf16 64t, ctx bfp8 128t)": max(64*BF16, 128*BFP8),
 "c1 cb_w weight ring 16 x 16 KiB": 16*16384,
 "c2 cb_acc fp32 8t": 8*FP32, "c3 cb_out bf16 8t": 8*BF16,
 "c4 cb_q landing 16t bf16": 16*BF16, "c5 cb_qr 8t bf16": 8*BF16, "c6 cb_rtmp 8t bf16": 8*BF16,
 "c7 cb_rope cos/sin 16t bf16": 16*BF16,
 "c8 cb_kv 2 x (5 key x 8 dh x 2) bfp8": 2*5*8*2*BFP8,
 "c9 cb_ksuf 32t bf16": 32*BF16, "c10 cb_mask 5t bf16": 5*BF16,
 "c11 cb_s 5t fp32": 5*FP32, "c12 cb_oacc 8t fp32": 8*FP32, "c13 cb_ml 4t fp32": 4*FP32,
 "c14 cb_part 4 x (8t bf16 + 2t fp32)": 4*(8*BF16+2*FP32), "c15 cb_ctx 8t bfp8": 8*BFP8,
 "c16 cb_r 2t bf16": 2*BF16, "c17 cb_bias 6t bf16": 6*BF16, "c18 cb_ug 4t fp32": 4*FP32,
 "c19 cb_h 4t bf16": 4*BF16, "c20 cb_hg 32t bf16": 32*BF16, "c21 cb_dpart 8t fp32": 8*FP32,
 "c22 cb_red 7x2t fp32": 14*FP32, "c23 cb_x fp32 2t": 2*FP32, "c24 cb_xs bf16 2t": 2*BF16,
 "c25 cb_const 4t": 4*BF16, "c26 cb_fence": 4096, "c27 cb_sync": 2048,
}
tot = sum(cbs.values())
for k,v in cbs.items(): print("  %-48s %8d" % (k, v))
print("heaviest core total", tot, "KiB %.1f" % (tot/1024))
alloc = 1395712 - 32768
print("allocatable under 64 KiB cut", alloc)
kv_l1 = 18*2*25*8*BFP8   # current L1-interleaved caches base, cache_len 800
print("co-tenant KV caches total %d per core over 110 banks %d" % (kv_l1, kv_l1//110))
print("headroom", alloc - tot - kv_l1//110)
# ---- chain per layer ----
base = dict(x_round=6.5, qkv=64*0.029, qkv_dist=3.5, attn=5*1.72+1.0, merge=1.5+3*3.5, ctx_round=6.5,
            o_proj=128*0.029+0.5, xmid_round=6.5, ug=256*0.029, geglu=4*1.5, h_exch=4.0, down=128*0.029, reduce=5.0)
lib  = dict(x_round=4.5, qkv=32*0.029, qkv_dist=2.5, attn=3*1.66+0.5, merge=1.5+3*3.5, ctx_round=4.5,
            o_proj=64*0.029+0.5, xmid_round=4.5, ug=128*0.029, geglu=2*1.5, h_exch=3.0, down=64*0.029, reduce=4.0)
for name, d in (("base", base), ("libero", lib)):
    c = sum(d.values()); print(name, {k: round(v,2) for k,v in d.items()}, "chain %.1f" % c)
    lay = max(c, s_b8+s_b16); step = 18*lay + 8.0; req = 10*step/1000
    print("  layer %.1f us  step %.1f us  request %.2f ms  x1.36 %.2f  x1.68 %.2f" % (lay, step, req, req*1.36, req*1.68))
print("---- phase 2 ----")
F = lambda m,k,n: 2*m*k*n
m=736
vf = dict(qkv=F(m,2048,2560), o=F(m,2048,2048), gu=F(m,2048,32768), dn=F(m,16384,2048))
print({k: v/1e9 for k,v in vf.items()}, "layer GF", sum(vf.values())/1e9)
gelu_tiles = 23*512/100
lo = dict(qkv=32.93, o=26.23, gu=vf['gu']/168e6, gelu=gelu_tiles*0.58, mul=gelu_tiles*0.1, dn=200.0, attn=139.6+20, norms=2*15.3)
lo['exch'] = 0.1*(lo['qkv']+lo['o']+lo['gu']+lo['dn'])
hi = dict(qkv=vf['qkv']/133e6, o=vf['o']/133e6, gu=vf['gu']/94e6, gelu=gelu_tiles*1.5, mul=gelu_tiles*0.1, dn=vf['dn']/117e6, attn=159.6, norms=30.6, exch=5*38.0)
for nm,d in (("VLM lower",lo),("VLM upper",hi)):
    print(nm, {k: round(v,1) for k,v in d.items()}, "sum us %.0f" % sum(d.values()))
vlo, vhi = sum(lo.values()), sum(hi.values()); vpt = vlo*1.36
print("VLM point (lower x1.36) %.0f us ; stack 17 layers: lower %.1f point %.1f upper %.1f ms" % (vpt, 17*vlo/1e3+0.1, 17*vpt/1e3+0.1, 17*vhi/1e3+0.1))
ml=544; sc=ml/736
print("LIBERO VLM (FLOP-scaled %.3f): lower %.1f point %.1f upper %.1f ms" % (sc, 17*vlo*sc/1e3+0.1, 17*vpt*sc/1e3+0.1, 17*vhi*sc/1e3+0.1))
s = 512
sf = dict(qkv=F(s,1152,4608), o=F(s,1536,1152), fc1=F(s,1152,4320), fc2=F(s,4320,1152))
print({k: v/1e9 for k,v in sf.items()})
gt = 16*135/100
slo = dict(qkv=39.47, o=17.52, fc1=sf['fc1']/140.6e6, gelu=gt*0.58, fc2=sf['fc2']/97.1e6, attn=25.14+5, ln=2*15.3)
slo['exch']=0.1*(slo['qkv']+slo['o']+slo['fc1']+slo['fc2'])
sl = sum(slo.values()); extras = 51.21+8.50+6.29+5.95+22.70
print("SigLIP lower", {k: round(v,1) for k,v in slo.items()}, "layer %.0f us tower lower %.2f point %.2f ms" % (sl, (27*sl+extras)/1e3, 1.36*(27*sl+extras)/1e3))
exp_pt = 18.91; exp_lo = 13.90; exp_hi = 23.36
p1 = lambda e: 84.21 - 31.43 + e
print("phase1 whole call base: bottom-up %.1f point %.1f first-build %.1f" % (p1(exp_lo), p1(exp_pt), p1(exp_hi)))
p1l = lambda e: 76.77 - (27.845+0.163) + e
print("phase1 whole call libero: %.1f %.1f %.1f" % (p1l(10.05), p1l(13.66), p1l(16.88)))
sig_pt = 1.36*(27*sl+extras)/1e3; sig_lo=(27*sl+extras)/1e3; sig_hi=10.77
pp = lambda sg, v, e: sg + 0.115 + v + 0.05 + e + 1.40
print("phase2 base: low %.1f point %.1f high %.1f" % (pp(sig_lo, 17*vlo/1e3+0.1, exp_lo), pp(sig_pt, 17*vpt/1e3+0.1, exp_pt), pp(sig_hi, 17*vhi/1e3+0.1, exp_hi)))
ppl = lambda sg, v, e: sg + 0.101 + v + 0.05 + e + 1.13
print("phase2 libero: low %.1f point %.1f high %.1f" % (ppl(sig_lo, 17*vlo*sc/1e3+0.1, 10.05), ppl(sig_pt, 17*vpt*sc/1e3+0.1, 13.66), ppl(sig_hi, 17*vhi*sc/1e3+0.1, 16.88)))
