# Same frames, two metrics: relative RMSE of rho_hat (old paper) and RMSE in dB of SNR (agent report).
import numpy as np, torch, json
from lib import *
torch.manual_seed(5); np.random.seed(5)
H=np.load('H.npy'); n=256; R=(n-gf2_rank(H))/n
Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H]))
tab5=make_table(H,5.0); tabi=make_table(H,np.inf)
def snr_db(r): return 10*np.log10(r/(1-r))
out={}
for eb in [-4,-2,0,2,3,4,5,6]:
    rho=1/(1+1/(2*R*10**(eb/10))); NF=60000; est={'soft5':[],'hard':[],'m2m4':[],'m2m4_sf':[]}
    for _ in range(NF//20000):
        y=np.sqrt(rho)+np.sqrt(1-rho)*torch.randn(20000,n)
        S=syndrome_stat(y,Hidx,5.0).double().numpy(); est['soft5'].append(invert(S,tab5))
        S=syndrome_stat(y,Hidx,np.inf).double().numpy(); est['hard'].append(invert(S,tabi))
        est['m2m4'].append(m2m4(y).double().numpy())
        m2=(y**2).mean(1); m4=(y**4).mean(1); sf=torch.sqrt(torch.clamp((3*m2**2-m4)/2,min=0.0))/m2; est['m2m4_sf'].append(torch.clamp(sf,0.02,0.98).double().numpy())
    row={}
    for k,v in est.items():
        v=np.concatenate(v); row[k]={'rel_rmse':float(np.sqrt(np.mean((v-rho)**2))/rho),'rel_bias':float(np.mean(v-rho)/rho),
          'rmse_dB':float(np.sqrt(np.mean((snr_db(np.clip(v,0.02,0.98))-snr_db(rho))**2))),'frac_at_floor':float(np.mean(v<=0.0201))}
    out[eb]=row; print(eb,round(rho,3),{k:(round(x['rel_rmse'],3),round(x['rmse_dB'],2),round(x['frac_at_floor'],2)) for k,x in row.items()},flush=True)
json.dump(out,open('m2m4_metric_check.json','w'),indent=1)
