# Letter 4 extension B: the paper's actual blind ratio search, on two codes,
# with grid sensitivity (range and density) and both fixed-gap and random-spread scenarios.
import numpy as np, torch, json, sys, time
sys.path.insert(0,'.')
from lib import *
quick=len(sys.argv)>1
dev='cuda' if torch.cuda.is_available() else 'cpu'
AL=0.85; A_SOFT=5.0
codes={}
try: codes[256]=np.load('H.npy')
except Exception: codes[256]=make_H(256)[0]
codes[512]=make_H(512)[0]
GRIDS={'g31_0.1_10':np.geomspace(0.1,10,31),'g61_0.1_10':np.geomspace(0.1,10,61),
       'g31_0.02_50':np.geomspace(0.02,50,31),'g121_0.02_50':np.geomspace(0.02,50,121)}
out={}; t0=time.time()
def wts(r): r=r.clamp(0.02,0.98); return 2*torch.sqrt(r)/(1-r)
for n,H in codes.items():
    R=(n-gf2_rank(H))/n; Hidx=torch.tensor(np.stack([np.nonzero(r)[0] for r in H])); Hd=Hidx.to(dev)
    dec=Dec(Hidx,n,dev); g=torch.Generator(device=dev); g.manual_seed(77)
    rho_t=lambda eb: 1/(1+1/(2*R*10**(eb/10)))
    def blind(y1,y2,RG):
        best=None; bi=None
        for k in range(len(RG)):
            z=y1+RG[k]*y2; z=z/z.pow(2).mean(1,keepdim=True).sqrt()
            S=syndrome_stat(z,Hd,A_SOFT)
            if best is None: best=S.clone(); bi=torch.zeros_like(S,dtype=torch.long)
            else:
                m=S>best; best=torch.where(m,S,best); bi=torch.where(m,torch.full_like(bi,k),bi)
        return RG[bi]
    scen=[('A6',lambda c,B:(torch.full((B,),float(c),device=dev),torch.full((B,),float(c)-6.,device=dev))),
          ('A12',lambda c,B:(torch.full((B,),float(c),device=dev),torch.full((B,),float(c)-12.,device=dev))),
          ('A0',lambda c,B:(torch.full((B,),float(c),device=dev),torch.full((B,),float(c),device=dev))),
          ('B8',lambda c,B:(c+8*(torch.rand(B,device=dev,generator=g)-.5),c+8*(torch.rand(B,device=dev,generator=g)-.5)))]
    for sname,gen in scen:
        for c in ([2.5] if quick else [1.5,2.0,2.5,3.0,3.5]):
            NF=2000 if quick else 20000; BS=2000 if quick else 5000
            err={k:0 for k in ['oracle','egc']}; err.update({f'blind_{k}':0 for k in GRIDS})
            lr={k:[] for k in GRIDS}
            for b in range(NF//BS):
                e1,e2=gen(c,BS); r1=rho_t(e1); r2=rho_t(e2)
                y1=torch.sqrt(r1)[:,None]+torch.sqrt(1-r1)[:,None]*torch.randn(BS,n,device=dev,generator=g)
                y2=torch.sqrt(r2)[:,None]+torch.sqrt(1-r2)[:,None]*torch.randn(BS,n,device=dev,generator=g)
                ro=wts(r2)/wts(r1)
                err['oracle']+=int((dec.run(y1+ro[:,None]*y2,50,'ms',AL)<0).any(1).sum())
                err['egc']+=int((dec.run(y1+y2,50,'ms',AL)<0).any(1).sum())
                for gname,gv in GRIDS.items():
                    RG=torch.tensor(gv,device=dev).float(); rb=blind(y1,y2,RG)
                    lr[gname].append((torch.log(rb)-torch.log(ro)).cpu().numpy())
                    err[f'blind_{gname}']+=int((dec.run(y1+rb[:,None]*y2,50,'ms',AL)<0).any(1).sum())
            rec={k:v/NF for k,v in err.items()}
            for gname in GRIDS:
                a=np.concatenate(lr[gname]); rec[f'lr_bias_{gname}']=float(a.mean()); rec[f'lr_std_{gname}']=float(a.std())
            out[f'n{n}_{sname}_c{c}']=rec; json.dump(out,open('code_ext2.json','w'),indent=1)
            print(n,sname,c,{k:round(v,4) for k,v in rec.items() if not k.startswith('lr_')},round(time.time()-t0,1),flush=True)
print('done',flush=True)
