import numpy as np, torch, json, sys
from lib import *
dev='cuda'; n=256; H=np.load('../results/H.npy'); R=(n-gf2_rank(H))/n
Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H])); dec=Dec(Hidx,n,dev); AL=0.85
g=torch.Generator(device=dev); g.manual_seed(5)
def rho_t(eb): return 1/(1+1/(2*R*10**(eb/10)))
def wts(r): r=r.clamp(0.02,0.98); return 2*torch.sqrt(r)/(1-r)
res={}
for scen,c in [('B',3.0),('B',2.0),('A',2.0)]:
    NF=100000; BS=20000; err={k:0 for k in ['oracle','n03','n06','n12','n25','egc']}
    for b in range(NF//BS):
        if scen=='A': e1=torch.full((BS,),c,device=dev); e2=e1-6
        else: e1=c+8*(torch.rand(BS,device=dev,generator=g)-.5); e2=c+8*(torch.rand(BS,device=dev,generator=g)-.5)
        r1=rho_t(e1); r2=rho_t(e2)
        y1=torch.sqrt(r1)[:,None]+torch.sqrt(1-r1)[:,None]*torch.randn(BS,n,device=dev,generator=g)
        y2=torch.sqrt(r2)[:,None]+torch.sqrt(1-r2)[:,None]*torch.randn(BS,n,device=dev,generator=g)
        for k,sg in [('oracle',0),('n03',0.03),('n06',0.06),('n12',0.12),('n25',0.25)]:
            a1=r1*torch.exp(sg*torch.randn(BS,device=dev,generator=g)); a2=r2*torch.exp(sg*torch.randn(BS,device=dev,generator=g))
            L=wts(a1)[:,None]*y1+wts(a2)[:,None]*y2; err[k]+=int((dec.run(L,50,'ms',AL)<0).any(1).sum())
        err['egc']+=int((dec.run(y1+y2,50,'ms',AL)<0).any(1).sum())
    res[f'{scen}{c}']=err; print(scen,c,err,flush=True)
json.dump(res,open('../results/exp9.json','w'))
