import numpy as np, json
from lib import *
H=np.load('../results/H.npy'); n=256; R=(n-gf2_rank(H))/n
agrid=np.concatenate([np.arange(0.25,3,0.25),np.arange(3,12.1,0.5)])
rows=[]
for eb in [-5,-4,-3,-2,-1,0,1,2,3,4,5,6]:
    rho=1/(1+1/(2*R*10**(eb/10)))
    sd=[np.sqrt(stat_mean_var(H,rho,a)[1])/abs(stat_slope(H,rho,a))/rho for a in agrid]
    i=int(np.argmin(sd)); am=2*np.sqrt(rho)/(1-rho)
    j=int(np.argmin(abs(agrid-am)))
    sh_=np.sqrt(stat_mean_var(H,rho,np.inf)[1])/abs(stat_slope(H,rho,np.inf))/rho
    row=dict(eb=eb,rho=rho,a_opt=float(agrid[i]),a_match=float(am),sd_opt=float(sd[i]),sd_match=float(sd[j]),sd_a5=float(np.sqrt(stat_mean_var(H,rho,5.0)[1])/abs(stat_slope(H,rho,5.0))/rho),sd_hard=float(sh_),sd_m2m4=float(np.sqrt(m4_var(rho,n))/rho))
    rows.append(row); print({k:round(v,3) for k,v in row.items()})
json.dump(rows,open('../results/exp6.json','w'),indent=1)
m,q=moments(1/(1+1/(2*R*10**0.3)),5.0); rho=1/(1+1/(2*R*10**0.3)); mean,var=stat_mean_var(H,rho,5.0)
print(dict(R=R,rho=rho,m=m,q=q,m6=m**6,mean=mean,sd=var**.5,M=H.shape[0],overl=np.unique((H@H.T)[~np.eye(128,dtype=bool)],return_counts=True)))
tab=make_table(H,5.0); print('inv',invert(np.array([0.20]),tab)); print('indep var',(q**6-m**12)/128,'exact',var)
