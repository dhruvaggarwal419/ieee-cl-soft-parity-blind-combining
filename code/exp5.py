import numpy as np, json
from lib import *
H=np.load('../results/H.npy'); n=256; R=(n-gf2_rank(H))/n
rho=1/(1+1/(2*R*10**(0.3))); a=5.0
m,q=moments(rho,a); mean,var=stat_mean_var(H,rho,a)
print(dict(R=R,rho=rho,m=m,q=q,m6=m**6,mean=mean,sd=var**.5,rel=np.sqrt(var)/abs(stat_slope(H,rho,a))/rho, M=H.shape[0], rowdeg=set(H.sum(1)), coldeg=set(H.sum(0)), overlaps=np.unique((H@H.T)[~np.eye(H.shape[0],dtype=bool)],return_counts=True)))
tab=make_table(H,a); S=0.20; print('inv',S, invert(np.array([S]),tab), 'sigma2', (1-invert(np.array([S]),tab))/invert(np.array([S]),tab))
for eb in [3.0]:
    pass
# indep-approx variance (ignoring overlaps) for comparison
d=6; Mn=H.shape[0]; v0=(q**d-m**(2*d))/Mn; print('indep var',v0,'exact',var)
