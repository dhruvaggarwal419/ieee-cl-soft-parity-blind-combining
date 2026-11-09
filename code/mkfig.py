import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from data import *
plt.rcParams.update({'font.family':'serif','font.size':7,'axes.linewidth':0.6,'pdf.fonttype':42})
e6=json.load(open(R+'exp6.json'))
fig,ax=plt.subplots(1,3,figsize=(7.16,1.95))
a=ax[0]; eb6=[r['eb'] for r in e6]
a.plot(eb6,[r['sd_match'] for r in e6],'-',color='tab:green',lw=1,label='soft, formula (8)')
a.plot(eb6,[r['sd_hard'] for r in e6],'-',color='tab:orange',lw=1,label='hard, formula')
a.plot(eb6,[r['sd_m2m4'] for r in e6],'-',color='tab:red',lw=1,label='M2M4, formula')
est=b['est']; eb=[r['eb'] for r in est]
for k,c,m,l in [('soft','tab:green','o','soft, simulation'),('hard','tab:orange','s','hard, simulation'),('m2m4','tab:red','^','M2M4, simulation'),('dd','tab:blue','D','decode then ML, sim.')]:
    a.plot(eb,[r[k]['rmse'] for r in est],m,color=c,ms=3,mfc='none' if k!='soft' else c,ls='none',label=l)
a.set_yscale('log'); a.set_xlabel('$E_b/N_0$ of the frame (dB)'); a.set_ylabel('relative RMSE of $\\hat\\rho$'); a.set_xlim(-5.3,6.3); a.set_ylim(0.015,3); a.grid(alpha=0.3,lw=0.4)
a.legend(fontsize=5,loc='upper right',frameon=False,handlelength=1.2,labelspacing=0.25); a.set_title('(a) single-frame estimate',fontsize=7)
styles={'oracle':('k','-',None,'oracle'),'blind':('tab:purple','-','o','blind max (this letter)'),'soft':('tab:green','--','v','per-frame soft'),'hard':('tab:orange','--','s','hard syndrome'),'m2m4':('tab:red','--','^','M2M4'),'dd':('tab:blue','--','D','decode then ML'),'egc':('0.5',':','x','equal weights')}
def panel(a,scen,cs,title,xl,leg=True):
    for k,(c,ls,m,l) in styles.items():
        xs=[];f=[];lo=[];hi=[]
        for cc in cs:
            try: kk,n=get(scen,cc,k)
            except KeyError: continue
            if kk==0: continue
            p,h=wil(kk,n); xs.append(cc); f.append(p); lo.append(max(p-h,1e-9)); hi.append(p+h)
        f=np.array(f); a.errorbar(xs,f,yerr=[f-np.array(lo),np.array(hi)-f],color=c,ls=ls,marker=m,ms=3,lw=0.9,capsize=1.2,elinewidth=0.5,mfc='none' if k not in('blind',) else c,label=l)
    a.set_yscale('log'); a.set_xlabel(xl); a.set_ylabel('frame error rate'); a.grid(alpha=0.3,lw=0.4,which='both'); a.set_title(title,fontsize=7)
    if leg: a.legend(fontsize=5,frameon=False,loc='lower left',handlelength=1.4,labelspacing=0.25)
panel(ax[1],'A',[1.0,1.5,2.0,2.5,3.0,3.5],'(b) scenario A, 6 dB gap','$E_b/N_0$ of the strong frame (dB)')
ax[1].set_ylim(5e-5,1)
panel(ax[2],'B',[1.5,2.0,2.5,3.0,3.5,4.0],'(c) scenario B, random SNRs','mean $E_b/N_0$ of the two frames (dB)',leg=False)
plt.tight_layout(pad=0.3,w_pad=0.6); plt.savefig('../paper/fig_results.pdf'); print('ok')
