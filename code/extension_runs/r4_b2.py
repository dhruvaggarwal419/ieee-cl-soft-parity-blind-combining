"""(B follow-up) exact mu4 where it is feasible, measured cost scaling,
Monte-Carlo kurtosis everywhere, and the fourth-moment confidence sets."""
import numpy as np, torch, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4lib as L, r4codes as CB, r4est as E
from scipy.special import erfc

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
TCAP = int(float(sys.argv[1])) if len(sys.argv) > 1 else 3_000_000      # connected triples
QCAP = int(float(sys.argv[2])) if len(sys.argv) > 2 else 600_000_000    # candidate quads
codes = CB.build_all()
g = torch.Generator(device=dev); g.manual_seed(808)
out = {'t_cap': TCAP, 'q_cap': QCAP, 'cost': [], 'conf_sets': [], 'mu4_validation': []}

# ---------------------------------------------------- cost + exact mu4 + MC kurtosis
for name, H in codes.items():
    Hc = L.to_csr(H); M = H.shape[0]
    OO = (Hc @ Hc.T).tocoo(); ei, ej = L.dep_edges(Hc, OO)
    indptr, indices = L._adj_csr(M, ei, ej)
    deg = (indptr[1:] - indptr[:-1]).astype(np.int64)
    ntc = int((deg * (deg - 1) // 2).sum())       # connected-triple candidates (with repeats)
    r = dict(code=name, n=H.shape[1], M=M, dc_mean=float(Hc.sum(1).mean()), dc_max=int(Hc.sum(1).max()),
             dv_mean=float(Hc.sum(0).mean()), dv_max=int(Hc.sum(0).max()),
             dep_deg_mean=float(deg.mean()), dep_deg_max=int(deg.max()),
             n_edges=int(len(ei)), n_triple_candidates=ntc)
    T = Q = None
    if ntc <= TCAP:
        t0 = time.time(); T = L.conn_triples(M, indptr, indices, cap=10 ** 12); r['t_triples_s'] = time.time() - t0
        r['n_triples'] = int(len(T))
        nqc = int(sum((indptr[T[:, s] + 1] - indptr[T[:, s]]).sum() for s in range(3)))
        r['n_quad_candidates'] = nqc
        if nqc <= QCAP:
            t0 = time.time(); Q = L.conn_quads(M, indptr, indices, T, cap=10 ** 12)
            r['t_quads_s'] = time.time() - t0; r['n_quads'] = int(len(Q))
    if T is not None and Q is not None:
        rl = [np.nonzero(Hc[i].toarray().ravel())[0] for i in range(M)]
        K = max(len(x) for x in rl)
        rowlists = -np.ones((M, K), dtype=np.int64)
        for i, x in enumerate(rl): rowlists[i, :len(x)] = x
        pre = dict(OO=OO, ei=ei, ej=ej, T=T, Q=Q, rowlists=rowlists, K=K)
        t0 = time.time(); res = L.mu4_exact(Hc, 0.6, 5.0, pre=pre); r['t_eval_s'] = time.time() - t0
        r['kurt_exact_rho0p6'] = res['kurt']; r['var_rho0p6'] = res['var']; r['mu4_rho0p6'] = res['mu4']
        r['feasible'] = True
    else:
        r['kurt_exact_rho0p6'] = None; r['feasible'] = False
        r['why'] = 'triple candidates %d > cap' % ntc if ntc > TCAP else 'quad candidates > cap'
    del T, Q
    c = E.Ctx(H, dev, rank=False)
    B = int(max(512, min(20000, 2.5e8 // (c.M * c.K))))
    acc = []
    for _ in range(max(1, 2000000 // B)):
        y = np.sqrt(0.6) + np.sqrt(0.4) * torch.randn(B, c.n, device=dev, generator=g)
        acc.append(c.soft_stat(y, 5.0).double().cpu().numpy())
    S = np.concatenate(acc); m = S.mean(); sd = S.std(ddof=1)
    r['N_mc'] = len(S); r['kurt_mc_rho0p6'] = float((((S - m) ** 4).mean()) / sd ** 4)
    r['kurt_mc_stderr'] = float(np.sqrt(24.0 / len(S))); r['sd_mc'] = float(sd)
    out['cost'].append(r); print(r, flush=True)
    json.dump(out, open('r4_B2_mu4.json', 'w'), indent=1)
    del c, acc, S

# ---------------------------------------------------- confidence sets on the paper's code
H0 = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'H.npy'))
Hc0 = L.to_csr(H0); R0 = (H0.shape[1] - L.packed_gf2_rank(H0)) / H0.shape[1]
c0 = E.Ctx(H0, dev)
pre = None
for ebdb in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
    rho = 1 / (1 + 1 / (2 * R0 * 10 ** (ebdb / 10)))
    res, pre = L.mu4_exact(Hc0, rho, 5.0, pre=pre, return_pre=True)
    kap = res['kurt']
    row = dict(eb_db=ebdb, rho=rho, sd=float(res['var'] ** .5), mu4=res['mu4'], kurt=kap)
    for k in [1.5, 2, 2.5, 3, 4]:
        row['cheb_k%s' % k] = 1 / k ** 2
        row['m4_k%s' % k] = ((kap - 1) / ((kap - 1) + (k ** 2 - 1) ** 2)) if k > 1 else 1.0
        row['gauss_k%s' % k] = float(erfc(k / np.sqrt(2)))
    out['conf_sets'].append(row); print('CS', row, flush=True)
    json.dump(out, open('r4_B2_mu4.json', 'w'), indent=1)

# ---------------------------------------------------- Monte-Carlo validation of mu4
for ebdb in [2.0, 3.0, 4.0]:
    rho = 1 / (1 + 1 / (2 * R0 * 10 ** (ebdb / 10)))
    acc = []
    Bc = 50000
    for _ in range(4000000 // Bc):
        y = np.sqrt(rho) + np.sqrt(1 - rho) * torch.randn(Bc, c0.n, device=dev, generator=g)
        acc.append(c0.soft_stat(y, 5.0).double().cpu().numpy())
    S = np.concatenate(acc)
    res, pre = L.mu4_exact(Hc0, rho, 5.0, pre=pre, return_pre=True)
    sd = S.std(ddof=1); m = S.mean()
    row = dict(eb_db=ebdb, N=len(S), mean_exact=res['mean'], mean_emp=float(m),
               sd_exact=float(res['var'] ** .5), sd_emp=float(sd),
               kurt_exact=res['kurt'], kurt_emp=float((((S - m) ** 4).mean()) / sd ** 4),
               kurt_emp_stderr=float(np.sqrt(24.0 / len(S))))
    for k in [2, 3, 4]:
        row['emp_fail_k%d' % k] = float((np.abs(S - res['mean']) >= k * res['var'] ** .5).mean())
    out['mu4_validation'].append(row); print('MC', row, flush=True)
    json.dump(out, open('r4_B2_mu4.json', 'w'), indent=1)
print('DONE b2', flush=True)
