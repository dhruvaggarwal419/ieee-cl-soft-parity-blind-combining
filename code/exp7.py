import numpy as np, json
from lib import *
H=np.load('../results/H.npy'); n=256; R=(n-gf2_rank(H))/n; M=H.shape[0]; d=6
out=[]
for eb in [2.0,3.0]:
    rho=1/(1+1/(2*R*10**(eb/10)))
    for a in [2.0,4.0,np.inf]:
        m,q=moments(rho,a); mean,var=stat_mean_var(H,rho,a)
        out.append(dict(eb=eb,a=str(a),var_exact=var,var_indep=(q**d-m**(2*d))/M,ratio=var/((q**d-m**(2*d))/M)))
        print(out[-1])
json.dump(out,open('../results/exp7.json','w'),indent=1)
rows=json.load(open('../results/exp6.json')); print('max ratio match/opt',max(r['sd_match']/r['sd_opt'] for r in rows),'a=5 ratio',max(r['sd_a5']/r['sd_opt'] for r in rows if r['eb']>=0))
