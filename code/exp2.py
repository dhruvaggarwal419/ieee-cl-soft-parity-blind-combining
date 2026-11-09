import numpy as np, torch, json, time, sys
from lib import *
dev='cuda'; print(torch.cuda.mem_get_info())
n=256; H=np.load('../results/H.npy'); rank=gf2_rank(H); R=(n-rank)/n
Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H])); Hd=Hidx.to(dev)
dec=Dec(Hidx,n,dev)
A_SOFT=5.0
tab={a:make_table(H,a) for a in [A_SOFT,np.inf]}
def rho_of(eb): return 1/(1+1/(2*R*10**(eb/10)))
def llr(y,r): r=r.clamp(0.02,0.98); return 2*torch.sqrt(r)[:,None]*y/(1-r)[:,None]
NF=int(sys.argv[1]) if len(sys.argv)>1 else 100000; BS=20000
EBS=[1.5,2.0,2.5,3.0,3.5,4.0]; DESIGN=2.5
g=torch.Generator(device=dev); g.manual_seed(123)
def alpha_tune():
    best=None
    for al in [0.6,0.7,0.75,0.8,0.85,0.9,1.0]:
        e=0
        for eb in [2.0,3.0]:
            rho=rho_of(eb); y=np.sqrt(rho)+np.sqrt(1-rho)*torch.randn(20000,n,device=dev,generator=g)
            L=llr(y,torch.full((20000,),rho,device=dev))
            e+=(dec.run(L,50,'ms',al)<0).any(1).float().mean().item()
        print('alpha',al,e); 
        if best is None or e<best[0]: best=(e,al)
    return best[1]
AL=alpha_tune(); print('alpha*',AL)
out={'R':R,'alpha':AL,'a_soft':A_SOFT,'nf':NF,'points':[]}
methods=['oracle','soft','hard','m2m4','fixed','nms']
for eb in EBS:
    rho=rho_of(eb); err={m:0 for m in methods}; est={m:[] for m in ['soft','hard','m2m4']}
    t0=time.time()
    for b in range(NF//BS):
        y=np.sqrt(rho)+np.sqrt(1-rho)*torch.randn(BS,n,device=dev,generator=g)
        rh={}
        rh['oracle']=torch.full((BS,),rho,device=dev)
        S=syndrome_stat(y,Hd,A_SOFT).double().cpu().numpy(); rh['soft']=torch.tensor(invert(S,tab[A_SOFT]),device=dev).float()
        S=syndrome_stat(y,Hd,np.inf).double().cpu().numpy(); rh['hard']=torch.tensor(invert(S,tab[np.inf]),device=dev).float()
        rh['m2m4']=m2m4(y); rh['fixed']=torch.full((BS,),rho_of(DESIGN),device=dev)
        for k in est: est[k].append(rh[k].cpu().numpy())
        for m in ['oracle','soft','hard','m2m4','fixed']:
            err[m]+=int((dec.run(llr(y,rh[m]),50,'bp')<0).any(1).sum())
        err['nms']+=int((dec.run(y,50,'ms',AL)<0).any(1).sum())
    pt={'eb':eb,'rho':rho,'n':NF,'err':err}
    for k in est:
        e=np.concatenate(est[k]); pt['est_'+k]={'mean_rel':float(e.mean()/rho-1),'relstd':float(e.std()/rho),'rmse_rel':float(np.sqrt(((e-rho)**2).mean())/rho),'frac_clipped':float(((e<=0.0201)|(e>=0.9799)).mean())}
    out['points'].append(pt); print(eb,err,{k:pt['est_'+k] for k in est},time.time()-t0,flush=True)
    json.dump(out,open('../results/exp2.json','w'),indent=1)
print('DONE')
