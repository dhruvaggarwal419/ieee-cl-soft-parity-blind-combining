"""Exact mu4 for the one code that exceeded the caps in r4_b2 (5G BG1, n=4224)."""
import numpy as np, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4lib as L, r4codes as CB
H = CB.build_all(which=['nr_bg1_R12_n4224'])['nr_bg1_R12_n4224']
Hc = L.to_csr(H); M = H.shape[0]
t0 = time.time()
OO = (Hc @ Hc.T).tocoo(); ei, ej = L.dep_edges(Hc, OO)
indptr, indices = L._adj_csr(M, ei, ej)
deg = (indptr[1:] - indptr[:-1]).astype(np.int64)
print('M', M, 'dep deg mean', deg.mean(), 'max', deg.max(), flush=True)
T = L.conn_triples(M, indptr, indices, cap=10**12); print('triples', len(T), '%.1fs' % (time.time() - t0), flush=True)
nqc = int(sum((indptr[T[:, s] + 1] - indptr[T[:, s]]).sum() for s in range(3)))
print('quad candidates', nqc, flush=True)
t1 = time.time()
Q = L.conn_quads(M, indptr, indices, T, cap=10**12)
print('quads', len(Q), '%.1fs' % (time.time() - t1), flush=True)
rl = [np.nonzero(Hc[i].toarray().ravel())[0] for i in range(M)]
K = max(len(x) for x in rl)
rowlists = -np.ones((M, K), dtype=np.int64)
for i, x in enumerate(rl): rowlists[i, :len(x)] = x
pre = dict(OO=OO, ei=ei, ej=ej, T=T, Q=Q, rowlists=rowlists, K=K)
t2 = time.time(); res = L.mu4_exact(Hc, 0.6, 5.0, pre=pre, chunk=60000); te = time.time() - t2
out = dict(code='nr_bg1_R12_n4224', M=M, dep_deg_mean=float(deg.mean()), dep_deg_max=int(deg.max()),
           n_triples=int(len(T)), n_quad_candidates=nqc, n_quads=int(len(Q)),
           t_enumerate_s=time.time() - t0 - te, t_eval_s=te,
           kurt_exact_rho0p6=res['kurt'], var_rho0p6=res['var'], mu4_rho0p6=res['mu4'])
json.dump(out, open('r4_B3_mu4_bg1.json', 'w'), indent=1)
print(out, flush=True); print('DONE b3', flush=True)
