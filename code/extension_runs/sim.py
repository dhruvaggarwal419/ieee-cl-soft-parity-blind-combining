import numpy as np, json, sys, time
rng_master = np.random.SeedSequence(2026)
METH = ['raw','debias','confirm','gap_noguard','gap']
def H(n): return float(np.sum(1.0/np.arange(1,n+1)))

def e1_batch(rng,N,s,B):
    y = rng.exponential(1.0,(B,N)); y[:,0]*= (1+s)
    w = y.argmax(1); r=np.arange(B)
    yw = y[r,w]; ys=np.sort(y,1); M=ys[:,-2]
    ssel = np.where(w==0,s,0.0)
    y2 = rng.exponential(1.0,B)*(1+ssel)
    est = {'raw':yw-1,'debias':yw-1-H(N-1),'confirm':y2-1,'gap_noguard':yw-M-1}
    est['gap']=est['gap_noguard']
    return est,ssel,(w==0)

def e2_batch(rng,Mant,N,s0,B,guard=3):
    n=np.arange(Mant)
    th=rng.uniform(-1,1,(B,3))
    g=np.empty((B,3),complex)
    g[:,0]=np.exp(1j*rng.uniform(0,2*np.pi,B))
    g[:,1:]=np.sqrt(0.05/2)*(rng.standard_normal((B,2))+1j*rng.standard_normal((B,2)))
    u=-1+2*np.arange(N)/N
    sig=np.zeros((B,N),complex)
    for l in range(3):
        d=th[:,l][:,None]-u[None,:]            # B,N
        # a_j^H a(theta) = (1/sqrt(M)) sum exp(j pi n d)
        x=np.exp(1j*np.pi*d[:,:,None]*n[None,None,:]).sum(2)/np.sqrt(Mant)
        sig+=g[:,l][:,None]*x
    sig*=np.sqrt(s0/Mant)
    cn=lambda sh:(rng.standard_normal(sh)+1j*rng.standard_normal(sh))/np.sqrt(2)
    y=np.abs(sig+cn((B,N)))**2
    w=y.argmax(1); r=np.arange(B); yw=y[r,w]
    ssel=np.abs(sig[r,w])**2
    idx=np.arange(N)[None,:]; dist=np.abs(idx-w[:,None]); dist=np.minimum(dist,N-dist)
    Mg=np.where(dist>guard,y,-np.inf).max(1)
    ys=np.sort(y,1); M2=ys[:,-2]
    y2=np.abs(sig[r,w]+cn(B))**2
    est={'raw':yw-1,'debias':yw-1-H(N-2*guard-1),'confirm':y2-1,'gap_noguard':yw-M2-1,'gap':yw-Mg-1}
    return est,ssel,None

def outage_e1(x,ssel):
    with np.errstate(divide='ignore',invalid='ignore'):
        out=np.where(ssel>0,1-np.exp(-x/np.maximum(ssel,1e-12)),(x>0).astype(float))
    return out, np.log2(1+x)*(1-out)
def outage_e2(x,ssel):
    ok=(x<=ssel); return (~ok).astype(float), np.log2(1+x)*ok

MARG=np.array([(m,t) for m in np.geomspace(0.5,16,10) for t in [0,1,2,3,4,5,6,8,10,13,16]])
def run_mix(kind,N,Mant,reps,B,sdbs,seed):
    ss=np.random.SeedSequence(seed).spawn(reps)
    S=len(sdbs); K=len(MARG)
    res={m:{'out':np.zeros((reps,S,K)),'gp':np.zeros((reps,S,K))} for m in METH}
    for r,sq in enumerate(ss):
        rng=np.random.default_rng(sq)
        for i,sdb in enumerate(sdbs):
            s=10**(sdb/10)
            est,ssel,_=(e1_batch(rng,N,s,B) if kind==1 else e2_batch(rng,Mant,N,s,B))
            for m in METH:
                e=est[m]; x0=np.maximum(e,0)
                for k,(mg,th) in enumerate(MARG):
                    oo,gg=(outage_e1 if kind==1 else outage_e2)(np.where(e>=th,x0/mg,0.0),ssel)
                    res[m]['out'][r,i,k]=oo.mean(); res[m]['gp'][r,i,k]=gg.mean()
    return res
def ci(a):
    a=np.asarray(a); n=len(a); return float(a.mean()), float(2.093*a.std(ddof=1)/np.sqrt(n)) if n>1 else 0.0

def tune_and_eval(kind,N,Mant,sdbs,reps,B,target=0.1):
    cal=run_mix(kind,N,Mant,reps,B,sdbs,1000+N)
    ev=run_mix(kind,N,Mant,reps,B,sdbs,2000+N)
    out={}
    for m in METH:
        o=cal[m]['out'].mean(0); g=cal[m]['gp'].mean(0)
        feas=np.where(o<=target)[0]
        k=feas[np.argmax(g[feas])] if len(feas) else int(np.argmin(o))
        out[m]={'margin':float(MARG[k][0]),'thr':float(MARG[k][1]),'feasible':bool(len(feas)),'out':ci(ev[m]['out'][:,k]),'gp':ci(ev[m]['gp'][:,k])}
    return out

def bias_curve(kind,N,Mant,sdbs,reps,B):
    ss=np.random.SeedSequence(7+N).spawn(reps); res={m:{s:[] for s in sdbs} for m in METH}
    for sq in ss:
        rng=np.random.default_rng(sq)
        for sdb in sdbs:
            s=10**(sdb/10)
            est,ssel,_=(e1_batch(rng,N,s,B) if kind==1 else e2_batch(rng,Mant,N,s,B))
            for m in METH: res[m][sdb].append(np.mean(est[m]-ssel))
    return {m:{str(s):ci(v) for s,v in d.items()} for m,d in res.items()}

if __name__=='__main__':
    quick=len(sys.argv)>1
    reps,B=(4,300) if quick else (20,3000)
    reps2,B2=(4,150) if quick else (20,800)
    sd=[-5,0,5,10,15,20]; t=time.time(); Z={}
    for kind,N,Ma in [(1,16,None),(1,64,None),(1,256,None),(1,1024,None),(2,64,64)]:
        for tag,sd0 in [('cal',1000),('ev',2000)]:
            rr=reps if kind==1 else reps2; BB=B if kind==1 else B2
            res=run_mix(kind,N,Ma,rr,BB,sd,sd0+N+kind)
            for m in METH:
                Z[f'{kind}_{N}_{tag}_{m}_out']=res[m]['out']; Z[f'{kind}_{N}_{tag}_{m}_gp']=res[m]['gp']
        print(kind,N,time.time()-t,flush=True)
    Z['MARG']=MARG; Z['sd']=np.array(sd)
    np.savez_compressed('arrays_quick.npz' if quick else 'arrays.npz',**Z)
    curve=[-10,-5,0,5,10,15,20]; R={}
    for N in [16,64,256]: R[f'E1_bias_N{N}']=bias_curve(1,N,None,curve,reps,B)
    R['E2_bias_N64']=bias_curve(2,64,64,curve,reps2,B2)
    json.dump(R,open('bias.json','w'),indent=1)
    print('done',time.time()-t)
