"""(G) the normalisation inside the blind ratio search, stated exactly and tested.

Algorithm actually implemented in the paper's code (exp3e.py):
  for each candidate ratio r on the grid:
      z_r   = y1 + r * y2                      (y1,y2 are the two received frames)
      zn_r  = z_r / sqrt( (1/n) sum_i z_r[i]^2 )          <-- OBSERVED sample power
      S(r)  = (1/M) sum_j prod_{i in check j} tanh( a * zn_r[i] / 2 ),  a fixed = 5
  r_hat = argmax_r S(r)
The same fixed a and the same form of normalisation is used for every candidate,
but the divisor is re-estimated from the data for every candidate, so the map
r -> S(r) is NOT a monotone reparametrisation of r -> S_unnormalised(r):
the normalisation changes the maximiser.  This script measures by how much.
"""
import numpy as np, torch, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4lib as L, r4codes as CB, r4est as E

QUICK = '--quick' in sys.argv
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
A_SOFT = 5.0; ALPHA = 0.85
H = CB.build_all(which=['reg_n256_R12'])['reg_n256_R12']
c = E.Ctx(H, dev); dec = E.Dec(c); R = c.R
g = torch.Generator(device=dev); g.manual_seed(2718)
NF = 20000 if QUICK else 100000
BS = 10000 if dev == 'cuda' else 2500
RG = torch.tensor(np.geomspace(0.1, 10, 31), device=dev).float()
def rho_eb(eb): return 1 / (1 + 1 / (2 * R * 10 ** (eb / 10)))
def wts(r): r = r.clamp(0.02, 0.98); return 2 * torch.sqrt(r) / (1 - r)
def fer(Lin): return int((dec.run(Lin, 50, ALPHA) < 0).any(1).sum())

NORMS = ['rms', 'det', 'genie', 'none', 'rms_scaled0.8', 'rms_scaled1.25']
out = {'algorithm': __doc__, 'grid': 'geomspace(0.1,10,31)', 'a': A_SOFT, 'nf': NF, 'rows': []}
EB_TOT = 2.5
def pair(gap):
    t = 10 ** (gap / 10.0); gt = 2 * R * 10 ** (EB_TOT / 10.0)
    g1 = gt * t / (1 + t); g2 = gt / (1 + t)
    return g1 / (1 + g1), g2 / (1 + g2)
for gap in ([6.0] if QUICK else [0.0, 3.0, 6.0, 12.0, 20.0]):
    RHO1, RHO2 = pair(gap)
    err = {k: 0 for k in NORMS}; err['oracle'] = 0; err['egc'] = 0
    lr = {k: [] for k in NORMS}; idx = {k: [] for k in NORMS}
    nb = max(1, NF // BS)
    for b in range(nb):
        r1 = torch.full((BS,), RHO1, device=dev); r2 = torch.full((BS,), RHO2, device=dev)
        y1 = torch.sqrt(r1)[:, None] + torch.sqrt(1 - r1)[:, None] * torch.randn(BS, c.n, device=dev, generator=g)
        y2 = torch.sqrt(r2)[:, None] + torch.sqrt(1 - r2)[:, None] * torch.randn(BS, c.n, device=dev, generator=g)
        ro = wts(r2) / wts(r1)
        err['oracle'] += fer(y1 + ro[:, None] * y2); err['egc'] += fer(y1 + y2)
        S = {k: [] for k in NORMS}
        for k in range(len(RG)):
            r = RG[k]; z = y1 + r * y2
            rms = z.pow(2).mean(1, keepdim=True).sqrt()
            gen = torch.sqrt((1 + r ** 2 * 1.0)).expand_as(rms)   # placeholder, replaced below
            gen = torch.sqrt(1.0 + r ** 2).expand_as(rms)          # E z^2 = 1 + r^2 (both frames unit power)
            S['rms'].append(c.soft_stat(z / rms, A_SOFT))
            S['det'].append(c.soft_stat(z / torch.sqrt(1 + r ** 2), A_SOFT))
            S['genie'].append(c.soft_stat(z / gen, A_SOFT))
            S['none'].append(c.soft_stat(z, A_SOFT))
            S['rms_scaled0.8'].append(c.soft_stat(0.8 * z / rms, A_SOFT))
            S['rms_scaled1.25'].append(c.soft_stat(1.25 * z / rms, A_SOFT))
        for k in NORMS:
            M = torch.stack(S[k], 1); i = M.argmax(1); rb = RG[i]
            err[k] += fer(y1 + rb[:, None] * y2)
            lr[k].append((torch.log(rb) - torch.log(ro)).cpu().numpy())
            idx[k].append(i.cpu().numpy())
    N = nb * BS
    row = dict(gap_db=gap, eb_total_db=EB_TOT, rho1=RHO1, rho2=RHO2, N=N, fer={k: v / N for k, v in err.items()})
    row['logratio'] = {k: dict(bias=float(np.concatenate(v).mean()), std=float(np.concatenate(v).std()))
                       for k, v in lr.items()}
    I = {k: np.concatenate(v) for k, v in idx.items()}
    row['agree_with_rms'] = {k: float((I[k] == I['rms']).mean()) for k in NORMS}
    row['mean_abs_index_shift_vs_rms'] = {k: float(np.abs(I[k] - I['rms']).mean()) for k in NORMS}
    out['rows'].append(row)
    print(gap, {k: round(v, 5) for k, v in row['fer'].items()}, flush=True)
    print('  agree', {k: round(v, 3) for k, v in row['agree_with_rms'].items()}, flush=True)
    json.dump(out, open('r4_G_norm.json', 'w'), indent=1)
print('DONE g', flush=True)
