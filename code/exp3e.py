import numpy as np, torch, json, sys, time
from lib import *
dev='cuda'; n=256; H=np.load('../results/H.npy'); R=(n-gf2_rank(H))/n
Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H])); Hd=Hidx.to(dev); dec=Dec(Hidx,n,dev)
AL=json.load(open('../results/exp2.json'))['alpha']
g=torch.Generator(device=dev); g.manual_seed(31)
RG=torch.tensor(np.geomspace(0.1,10,31),device=dev).float()
def rho_t(eb): return 1/(1+1/(2*R*10**(eb/10)))
def wts(r): r=r.clamp(0.02,0.98); return 2*torch.sqrt(r)/(1-r)
def blind_ratio(y1,y2,a=5.0):
    best=None; bi=None
    for k,r in enumerate(RG):
        z=y1+r*y2; z=z/z.pow(2).mean(1,keepdim=True).sqrt()
        S=syndrome_stat(z,Hd,a)
        if best is None: best=S.clone(); bi=torch.zeros_like(S,dtype=torch.long)
        else:
            m=S>best; best=torch.where(m,S,best); bi=torch.where(m,torch.full_like(bi,k),bi)
    return RG[bi]
NF=int(sys.argv[1]); BS=20000; out={'nf':NF}
def run(name,gen,centers):
    res=[]
    for c in centers:
        err={m:0 for m in ['oracle','blind','egc']}; lr=[]; t0=time.time()
        for b in range(NF//BS):
            e1,e2=gen(c,BS); r1=rho_t(e1); r2=rho_t(e2)
            y1=torch.sqrt(r1)[:,None]+torch.sqrt(1-r1)[:,None]*torch.randn(BS,n,device=dev,generator=g)
            y2=torch.sqrt(r2)[:,None]+torch.sqrt(1-r2)[:,None]*torch.randn(BS,n,device=dev,generator=g)
            ro=wts(r2)/wts(r1); rb=blind_ratio(y1,y2)
            lr.append((torch.log(rb)-torch.log(ro)).cpu().numpy())
            for m,L in [('oracle',y1+ro[:,None]*y2),('blind',y1+rb[:,None]*y2),('egc',y1+y2)]:
                err[m]+=int((dec.run(L,50,'ms',AL)<0).any(1).sum())
        lr=np.concatenate(lr); res.append({'c':c,'err':err,'logratio_err_mean':float(lr.mean()),'logratio_err_std':float(lr.std())}); print(name,c,err,res[-1]['logratio_err_std'],time.time()-t0,flush=True)
        out[name]=res; json.dump(out,open('../results/exp3e.json','w'),indent=1)
def genA(c,B): return torch.full((B,),float(c),device=dev), torch.full((B,),float(c)-6.0,device=dev)
def genB(c,B): return c+8*(torch.rand(B,device=dev,generator=g)-0.5), c+8*(torch.rand(B,device=dev,generator=g)-0.5)
run('A',genA,[1.0,1.5,2.0,2.5,3.0,3.5]); run('B',genB,[1.5,2.0,2.5,3.0,3.5,4.0]); print('DONE')
