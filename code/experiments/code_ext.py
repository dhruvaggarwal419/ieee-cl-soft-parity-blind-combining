# Letter 4 extension: FER vs Eb/N0 for blind frame combining on a second LDPC code (n=512) and the n=256 code
import numpy as np, torch, json, sys, time
sys.path.insert(0,'.')
from lib import *
quick=len(sys.argv)>1
dev='cuda' if torch.cuda.is_available() else 'cpu'
codes={}
try: H=np.load('H.npy')
except Exception: H,_=make_H(256)
codes[256]=H
H5,c5=make_H(512); codes[512]=H5
AL=0.85; res={}; t0=time.time()
def wts(r): r=r.clamp(0.02,0.98); return 2*torch.sqrt(r)/(1-r)
for n,H in codes.items():
    R=(n-gf2_rank(H))/n; Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H])); dec=Dec(Hidx,n,dev)
    tab=make_table(H,1.0); g=torch.Generator(device=dev); g.manual_seed(11)
    rho_t=lambda eb:1/(1+1/(2*R*10**(eb/10)))
    for c,SP in ([(3.0,8)] if quick else [(c,sp) for sp in (8,16) for c in (1.5,2.0,2.5,3.0,3.5,4.5)]):
        NF=2000 if quick else 20000; BS=2000 if quick else 5000
        err={k:0 for k in ['oracle','syndrome','m2m4','egc','select']}
        for b in range(NF//BS):
            e1=c+SP*(torch.rand(BS,device=dev,generator=g)-.5); e2=c+SP*(torch.rand(BS,device=dev,generator=g)-.5)
            r1=rho_t(e1); r2=rho_t(e2)
            y1=torch.sqrt(r1)[:,None]+torch.sqrt(1-r1)[:,None]*torch.randn(BS,n,device=dev,generator=g)
            y2=torch.sqrt(r2)[:,None]+torch.sqrt(1-r2)[:,None]*torch.randn(BS,n,device=dev,generator=g)
            def est_syn(y):
                s=syndrome_stat(y,Hidx.to(dev),1.0).cpu().numpy(); return torch.tensor(invert(s,tab),dtype=torch.float32,device=dev)
            L={'oracle':wts(r1)[:,None]*y1+wts(r2)[:,None]*y2}
            s1,s2=est_syn(y1),est_syn(y2); L['syndrome']=wts(s1)[:,None]*y1+wts(s2)[:,None]*y2
            q1,q2=m2m4(y1),m2m4(y2); L['m2m4']=wts(q1)[:,None]*y1+wts(q2)[:,None]*y2
            L['egc']=y1+y2
            pick=(s1>=s2)[:,None]; L['select']=torch.where(pick,y1,y2)
            for k,v in L.items(): err[k]+=int((dec.run(v,50,'ms',AL)<0).any(1).sum())
        res[f'n{n}_c{c}_sp{SP}']={k:v/NF for k,v in err.items()}; json.dump(res,open('code_ext.json','w'),indent=1)
        print(n,c,SP,res[f'n{n}_c{c}_sp{SP}'],time.time()-t0,flush=True)
print('done',flush=True)
