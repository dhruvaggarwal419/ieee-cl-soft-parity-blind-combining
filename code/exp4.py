import numpy as np, torch, json
from lib import *
dev='cpu'; n=256; H=np.load('../results/H.npy'); rank=gf2_rank(H); R=(n-rank)/n
Hd=torch.tensor(np.stack([np.nonzero(r)[0] for r in H])).to(dev)
def rho_of(eb): return 1/(1+1/(2*R*10**(eb/10)))
out={}
# theory curves of relative std
ebs=np.arange(0,5.01,0.5); th={}
for a in [2.0,5.0,12.0,np.inf]:
    v=[]
    for eb in ebs:
        r=rho_of(eb); mean,var=stat_mean_var(H,r,a); sl=stat_slope(H,r,a); v.append(float(np.sqrt(var)/abs(sl)/r))
    th[str(a)]=v
th['m2m4']=[float(np.sqrt(m4_var(rho_of(eb),n))/rho_of(eb)) for eb in ebs]
out['eb']=ebs.tolist(); out['theory']=th
# sensitivity to a at 3 dB
r=rho_of(3.0); out['a_sweep']={str(a):float(np.sqrt(stat_mean_var(H,r,a)[1])/abs(stat_slope(H,r,a))/r) for a in [1,2,3,4,5,6,8,12,24,np.inf]}
# Chebyshev confidence sets
A=5.0; G=RHO_GRID[::4]
mu=np.zeros(len(G)); sd=np.zeros(len(G))
for i,r in enumerate(G):
    m,v=stat_mean_var(H,r,A); mu[i]=m; sd[i]=np.sqrt(v)
Gt=torch.tensor(G,device=dev).float(); mut=torch.tensor(mu,device=dev).float(); sdt=torch.tensor(sd,device=dev).float()
g=torch.Generator(device=dev); g.manual_seed(7); cov=[]
for eb in [2.0,3.0,4.0]:
    rho=rho_of(eb); N=40000; S=[]
    for _ in range(2):
        y=np.sqrt(rho)+np.sqrt(1-rho)*torch.randn(N//2,n,device=dev,generator=g); S.append(syndrome_stat(y,Hd,A))
    S=torch.cat(S)
    for k in [2,3]:
        inside=((S[:,None]-mut[None,:]).abs()<=k*sdt[None,:])
        lo=torch.where(inside,Gt[None,:],torch.full_like(inside,9.,dtype=torch.float)).min(1).values
        hi=torch.where(inside,Gt[None,:],torch.full_like(inside,-9.,dtype=torch.float)).max(1).values
        empty=(hi<0)
        ok=(~empty)&(lo<=rho)&(hi>=rho)
        w=((hi-lo)/rho)[~empty]
        # width in dB of Eb/N0 is monotone map of rho: report relative width in rho
        cov.append(dict(eb=eb,k=k,coverage=float(ok.float().mean()),guarantee=1-1/k**2,median_rel_width=float(w.median()),empty=float(empty.float().mean())))
        print(cov[-1])
out['cheb']=cov
json.dump(out,open('../results/exp4.json','w'),indent=1)
print(out['a_sweep'])
