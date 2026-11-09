import numpy as np, torch, json, sys
from lib import *
n=256
H,c=make_H(n,3,6,seed=1); rank=gf2_rank(H); R=(n-rank)/n
print('4cycle cost',c,'rank',rank,'R',R)
np.save('../results/H.npy',H)
out={'cost':c,'rank':rank,'R':R}
def rho_of(eb): 
    s2=1/(2*R*10**(eb/10)); return 1/(1+s2)
Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H]))
avals=[0.5,1,2,3,4,6,8,12,np.inf]
res={}
for eb in [1.5,2.0,2.5,3.0,3.5,4.0]:
    rho=rho_of(eb); row={}
    for a in avals:
        mean,var=stat_mean_var(H,rho,a); sl=stat_slope(H,rho,a)
        row[str(a)]=float(np.sqrt(var)/abs(sl)/rho)   # relative std of rho-hat
    row['m2m4']=float(np.sqrt(m4_var(rho,n))/rho)
    res[eb]=row; print(eb,round(rho,4),{k:round(v,3) for k,v in row.items()})
out['delta_relstd']=res
# MC validation
dev='cuda'; g=torch.Generator(device=dev); g.manual_seed(0)
val=[]
for eb in [2.0,3.0]:
    rho=rho_of(eb)
    for a in [2.0,4.0,np.inf]:
        N=200000; S=[]
        for _ in range(10):
            y=np.sqrt(rho)+np.sqrt(1-rho)*torch.randn(N//10,n,device=dev,generator=g)
            S.append(syndrome_stat(y,Hidx.to(dev),a).cpu())
        S=torch.cat(S).double().numpy()
        mean,var=stat_mean_var(H,rho,a)
        val.append(dict(eb=eb,a=str(a),mean_th=float(mean),mean_mc=float(S.mean()),se=float(S.std()/np.sqrt(N)),var_th=float(var),var_mc=float(S.var())))
        print(val[-1])
out['validation']=val
json.dump(out,open('../results/exp1.json','w'),indent=1,default=str)
