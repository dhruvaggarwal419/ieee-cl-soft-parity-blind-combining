import numpy as np, torch, json, time, sys
from lib import *
dev='cuda'
n=256; H=np.load('../results/H.npy'); rank=gf2_rank(H); R=(n-rank)/n
Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H])); Hd=Hidx.to(dev); dec=Dec(Hidx,n,dev)
AL=json.load(open('../results/exp2.json'))['alpha']; A_SOFT=5.0
AG=[0.5,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0]
tab={a:make_table(H,a) for a in AG+[np.inf]}
AGt=torch.tensor(AG,device=dev)
def soft_adapt(y):
    S1=syndrome_stat(y,Hd,3.0).double().cpu().numpy(); r1=invert(S1,tab[3.0])
    am=2*np.sqrt(r1)/(1-r1); idx=np.abs(np.log(am[:,None])-np.log(np.array(AG)[None,:])).argmin(1)
    out=np.zeros(len(r1))
    for k,a in enumerate(AG):
        sel=np.nonzero(idx==k)[0]
        if len(sel)==0: continue
        Sk=syndrome_stat(y[torch.tensor(sel,device=dev)],Hd,a).double().cpu().numpy(); out[sel]=invert(Sk,tab[a])
    return torch.tensor(out,device=dev).float()
def rho_of(eb): return 1/(1+1/(2*R*10**(np.asarray(eb)/10)))
g=torch.Generator(device=dev); g.manual_seed(77)
RG=torch.linspace(0.02,0.98,961,device=dev)
def estimates(y):
    B=y.shape[0]; r={}
    r['soft']=soft_adapt(y)
    S=syndrome_stat(y,Hd,5.0).double().cpu().numpy(); r['soft5']=torch.tensor(invert(S,tab[5.0]),device=dev).float()
    S=syndrome_stat(y,Hd,np.inf).double().cpu().numpy(); r['hard']=torch.tensor(invert(S,tab[np.inf]),device=dev).float()
    r['m2m4']=m2m4(y)
    xh=torch.where(dec.run(y,20,'ms',AL)<0,-1.0,1.0)
    z=xh*y; zb=z.mean(1,keepdim=True); Sx=((z-zb)**2).sum(1,keepdim=True)
    G=RG[None,:]; ll=-n/2*torch.log(1-G)-(Sx+n*(zb-torch.sqrt(G))**2)/(2*(1-G))
    r['dd']=RG[ll.argmax(1)]
    return r
def wts(r): r=r.clamp(0.02,0.98); return 2*torch.sqrt(r)/(1-r)
NF=int(sys.argv[1]); BS=20000
out={'R':R,'alpha':AL,'nf':NF}
out['est']=[]
# ---- combining
methods=['oracle','soft','soft5','hard','m2m4','dd','egc']
def scenario(name,gen,centers):
    res=[]
    for c in centers:
        err={m:0 for m in methods}; t0=time.time()
        for b in range(NF//BS):
            eb1,eb2=gen(c,BS)  # tensors of dB
            r1=rho_of_t(eb1); r2=rho_of_t(eb2)
            y1=torch.sqrt(r1)[:,None]+torch.sqrt(1-r1)[:,None]*torch.randn(BS,n,device=dev,generator=g)
            y2=torch.sqrt(r2)[:,None]+torch.sqrt(1-r2)[:,None]*torch.randn(BS,n,device=dev,generator=g)
            e1=estimates(y1); e2=estimates(y2)
            e1['oracle']=r1; e2['oracle']=r2
            for m in methods:
                if m=='egc': L=y1+y2
                else: L=wts(e1[m])[:,None]*y1+wts(e2[m])[:,None]*y2
                err[m]+=int((dec.run(L,50,'ms',AL)<0).any(1).sum())
        res.append({'c':c,'err':err}); print(name,c,err,time.time()-t0,flush=True)
        out[name]=res; json.dump(out,open('../results/exp3c.json','w'),indent=1)
def rho_of_t(eb): return 1/(1+1/(2*R*10**(eb/10)))
def genA(c,B): return torch.full((B,),float(c),device=dev), torch.full((B,),float(c)-6.0,device=dev)
def genB(c,B): return c+8*(torch.rand(B,device=dev,generator=g)-0.5), c+8*(torch.rand(B,device=dev,generator=g)-0.5)
scenario('B',genB,[1.5,2.0,2.5,3.0,3.5,4.0])
print('DONE')
