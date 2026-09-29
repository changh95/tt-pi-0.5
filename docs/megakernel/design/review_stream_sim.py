# Fluid model of DESIGN.md sec 4.3/4.4: one in-order weight stream per core into two rings (w8, w16),
# consumed only in the compute phases of the sec-4.3 chain. Aggregate DRAM cap shared equally by
# non-blocked cores (optionally a per-core cap). Reports steady-state layer time.
import sys
QKV, WO, UG, WD = 34816, 131072, 139264, 131072
def run(phases, cap_gbs, core_cap_gbs, w8=87040, w16=131072, order="design", layers=40, dt=0.01):
    # core classes: (count, items per layer)
    if order == "design":
        t12 = [("w8",QKV,"C1"),("w8",UG,"C4"),("w16",WD,"C5")]
        t3  = [("w8",QKV,"C1"),("w16",WO,"C3"),("w8",UG,"C4"),("w16",WD,"C5")]
    else:  # reorder: Wd of the SAME layer ahead of ug where the w16 ring is free (T1/T2)
        t12 = [("w8",QKV,"C1"),("w16",WD,"C5"),("w8",UG,"C4")]
        t3  = [("w8",QKV,"C1"),("w16",WO,"C3"),("w8",UG,"C4"),("w16",WD,"C5")]
    kv = [("w8",QKV,"C1")]
    classes = [(32,t12),(32,t3),(16,kv)]
    ringcap = {"w8":w8,"w16":w16}
    st = []
    for n,items in classes:
        st.append(dict(n=n, items=items, cur=0, got=0.0, layer=0,
                       ring={"w8":0.0,"w16":0.0}, avail={}))  # avail[(layer,phase)] bytes landed
    t = 0.0; layer_start=[]
    cap = cap_gbs*1e3*dt  # bytes per dt (GB/s * 1e3 = B/us)
    ccap = core_cap_gbs*1e3*dt if core_cap_gbs else 1e18
    for L in range(layers):
        layer_start.append(t)
        for (pname, dur) in phases:
            need = {}
            for c in st:
                need[id(c)] = sum(b for (r,b,p) in c["items"] if p==pname)
            consumed = {id(c):0.0 for c in st}
            el = 0.0
            while True:
                # stream: active cores = those whose current item ring has space and layer lookahead ok
                act = []
                for c in st:
                    r,b,p = c["items"][c["cur"]]
                    if c["layer"] <= L+1 and c["ring"][r] < ringcap[r] - 1e-9:
                        act.append(c)
                ncores = sum(c["n"] for c in act)
                if ncores:
                    per = min(cap/ncores, ccap)
                    for c in act:
                        r,b,p = c["items"][c["cur"]]
                        amt = min(per, ringcap[r]-c["ring"][r], b-c["got"])
                        c["ring"][r]+=amt; c["got"]+=amt
                        k=(c["layer"],p); c["avail"][k]=c["avail"].get(k,0.0)+amt
                        if c["got"] >= b-1e-6:
                            c["got"]=0.0; c["cur"]+=1
                            if c["cur"]==len(c["items"]): c["cur"]=0; c["layer"]+=1
                # consume
                done = True
                for c in st:
                    nd = need[id(c)]
                    if nd==0: continue
                    rate = nd/dur*dt
                    k=(L,pname); a=c["avail"].get(k,0.0)
                    take=min(rate, a, nd-consumed[id(c)])
                    if take>0:
                        consumed[id(c)]+=take; c["avail"][k]=a-take
                        r=[it[0] for it in c["items"] if it[2]==pname][0]
                        c["ring"][r]-=take
                    if consumed[id(c)] < nd-1e-6: done=False
                el+=dt; t+=dt
                if done and el>=dur-1e-9: break
    per = [layer_start[i+1]-layer_start[i] for i in range(len(layer_start)-1)]
    return sum(per[-10:])/10
base = [("R1",6.5),("C1",1.86),("R2",3.5),("C2",9.6),("R3",12.0),("R4",6.5),("C3",4.21),("R5",6.5),
        ("C4",7.42+6.0),("R6",4.0),("C5",3.71),("R7",5.0)]
lib = [("R1",4.5),("C1",0.93),("R2",2.5),("C2",5.48),("R3",12.0),("R4",4.5),("C3",2.36),("R5",4.5),
       ("C4",3.71+3.0),("R6",3.0),("C5",1.86),("R7",4.0)]
if __name__=="__main__":
 for name,ph in (("base",base),("libero",lib)):
    chain=sum(d for _,d in ph)
    for cc in (None, 15.0):
        for order in ("design","reorder"):
            print(name, "chain %.1f"%chain, "percore cap", cc, order, "layer us %.1f"%run(ph, 440.0, cc, order=order))
