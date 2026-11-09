import json, numpy as np
R='../results/'
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return c,h
b=json.load(open(R+'exp3b.json')); c=json.load(open(R+'exp3c.json')); e=json.load(open(R+'exp3e.json'))
def pts(lst): return {p['c']:p for p in lst}
Ab,Bc,Ae,Be=pts(b['A']),pts(c['B']),pts(e['A']),pts(e['B'])
nb,nc,ne=b['nf'],c['nf'],e['nf']
def get(scen,cc,m):
    """return (k,n) for method m in scenario"""
    if scen=='A':
        if m=='blind': return Ae[cc]['err']['blind'],ne
        if m in('oracle','egc'):
            k=Ae[cc]['err'][m]+(Ab[cc]['err'][m] if cc in Ab else 0); n=ne+(nb if cc in Ab else 0); return k,n
        return Ab[cc]['err'][m],nb
    else:
        if m=='blind': return Be[cc]['err']['blind'],ne
        if m in('oracle','egc'):
            k=Be[cc]['err'][m]+(Bc[cc]['err'][m] if cc in Bc else 0); n=ne+(nc if cc in Bc else 0); return k,n
        return Bc[cc]['err'][m],nc
def fer(scen,cc,m):
    k,n=get(scen,cc,m); return k/n
def gap(scen,m,level=0.02):
    cs=sorted((Ae if scen=='A' else Be).keys()); 
    def cross(mm):
        xs=[];ys=[]
        for cc in cs:
            try: f=fer(scen,cc,mm)
            except KeyError: continue
            if f>0: xs.append(cc); ys.append(np.log10(f))
        xs=np.array(xs); ys=np.array(ys)
        for i in range(len(xs)-1):
            if ys[i]>=np.log10(level)>=ys[i+1]: 
                return xs[i]+(np.log10(level)-ys[i])*(xs[i+1]-xs[i])/(ys[i+1]-ys[i])
        return None
    a=cross(m); o=cross('oracle'); return None if a is None or o is None else a-o
