# Letter 2 extension B: the paper's real claim is ROBUSTNESS WITHOUT RETUNING.
# Tune (margin, threshold) once on the base array config, then evaluate on every other
# config with no retuning. This tests transfer under array correlation and model mismatch.
import numpy as np, json, time, sys
sys.path.insert(0,'.')
import sim
quick=len(sys.argv)>1
def make_e2(pside,nside):
    def e2(rng,Mant,N,s0,B,guard=3):
        n=np.arange(Mant); L=1+nside
        th=rng.uniform(-1,1,(B,L)); g=np.empty((B,L),complex)
        g[:,0]=np.exp(1j*rng.uniform(0,2*np.pi,B))
        if nside: g[:,1:]=np.sqrt(pside/2)*(rng.standard_normal((B,nside))+1j*rng.standard_normal((B,nside)))
        u=-1+2*np.arange(N)/N; sig=np.zeros((B,N),complex)
        for l in range(L):
            d=th[:,l][:,None]-u[None,:]
            x=np.exp(1j*np.pi*d[:,:,None]*n[None,None,:]).sum(2)/np.sqrt(Mant)
            sig+=g[:,l][:,None]*x
        sig*=np.sqrt(s0/Mant)
        cn=lambda sh:(rng.standard_normal(sh)+1j*rng.standard_normal(sh))/np.sqrt(2)
        y=np.abs(sig+cn((B,N)))**2
        w=y.argmax(1); r=np.arange(B); yw=y[r,w]; ssel=np.abs(sig[r,w])**2
        idx=np.arange(N)[None,:]; dist=np.abs(idx-w[:,None]); dist=np.minimum(dist,N-dist)
        Mg=np.where(dist>guard,y,-np.inf).max(1); M2=np.sort(y,1)[:,-2]
        y2=np.abs(sig[r,w]+cn(B))**2; H=sim.H
        est={'raw':yw-1,'debias':yw-1-H(N-2*guard-1),'confirm':y2-1,'gap_noguard':yw-M2-1,'gap':yw-Mg-1}
        return est,ssel,None
    return e2
cfg=[('base',64,64,0.05,2),('noside',64,64,0.0,0),('side20',64,64,0.2,2),('side50',64,64,0.5,2),
     ('paths6',64,64,0.05,6),('M32',32,64,0.05,2),('N128',64,128,0.05,2),('N256',64,256,0.05,2),('N32',32,32,0.05,2)]
reps,B=(2,100) if quick else (10,500)
sd=[-5,0,5,10,15,20]
ci=lambda a:(float(np.mean(a)),float(2.093*np.std(a,ddof=1)/np.sqrt(len(a))) if len(a)>1 else 0.0)
t=time.time()
# 1. tune on base only
sim.e2_batch=make_e2(0.05,2)
cal=sim.run_mix(2,64,64,reps,B,sd,1000)
K={}
for m in sim.METH:
    o=cal[m]['out'].mean(0); g=cal[m]['gp'].mean(0); feas=np.where(o.max(0)<=0.1)[0]
    K[m]=int(feas[np.argmax(g[:,feas].mean(0))]) if len(feas) else None
print('tuned on base:',{m:(None if K[m] is None else list(sim.MARG[K[m]])) for m in sim.METH},flush=True)
R={'tuned_on':'base','knobs':{m:(None if K[m] is None else list(map(float,sim.MARG[K[m]]))) for m in sim.METH}}
for name,Ma,N,ps,ns in cfg:
    sim.e2_batch=make_e2(ps,ns)
    ev=sim.run_mix(2,N,Ma,reps,B,sd,2000+N+int(ps*100)+ns)
    row={}
    for m in sim.METH:
        if K[m] is None: row[m]=None; continue
        k=K[m]; row[m]={'gp':ci(ev[m]['gp'][:,:,k].mean(1)),'worst_out':ci(ev[m]['out'][:,:,k].max(1))}
    R[name]={'Mant':Ma,'N':N,'pside':ps,'nside':ns,'transfer':row}
    json.dump(R,open('beam_ext2.json','w'),indent=1); print(name,round(time.time()-t,1),flush=True)
print('done',flush=True)
