"""(A) estimator operating range  +  (B) variance / mu4 complexity."""
import numpy as np, scipy.sparse as sp, json, time, sys, os
from scipy.special import erfc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4lib as L, r4codes as C
import torch

QUICK = '--quick' in sys.argv
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = {}
t00 = time.time()

# ======================================================= (A) operating range
print('=== A ===', flush=True)
A = {}
# --- A1: monotonicity of m_1(rho,a) and of mu_a on a fine, wide grid
rg = np.unique(np.concatenate([
    np.geomspace(1e-6, 1e-2, 400),
    np.linspace(1e-2, 1 - 1e-2, 4000),
    1 - np.geomspace(1e-6, 1e-2, 400)]))
mono = []
for a in [0.25, 0.5, 1.0, 2.0, 5.0, 12.0, 30.0, 100.0, np.inf]:
    m1 = np.array([L.t_moments(r, a, 1)[1] for r in rg])
    dif = np.diff(m1)
    # saturation: once m1 is within 1e-13 of its ceiling the steps are below
    # double precision and 'decreasing' steps are pure round-off.
    ceil = np.tanh(a / 2) if np.isfinite(a) else 1.0
    unsat = m1 < ceil * (1 - 1e-13)
    du = unsat[:-1] & unsat[1:]
    row = dict(a=('inf' if not np.isfinite(a) else a),
               ceiling=float(ceil),
               m1_min=float(m1.min()), m1_max=float(m1.max()),
               rho_where_saturated=(float(rg[np.argmax(~unsat)]) if (~unsat).any() else None),
               n_steps=int(len(dif)), n_unsaturated_steps=int(du.sum()),
               n_decreasing_unsat=int((dif[du] < 0).sum()),
               n_decreasing_all=int((dif < 0).sum()),
               min_fwd_diff_unsat=float(dif[du].min()) if du.any() else None,
               worst_decrease_all=float(dif.min()))
    mono.append(row); print('A1', row, flush=True)
A['monotonicity_m1'] = mono
A['grid_note'] = ('m_1(rho,a) checked on %d points spanning rho in [%.1e, 1-%.1e]; '
                  'mu_a(rho)=mean_j m_1^{d_j} is increasing iff m_1 is.' % (len(rg), rg[0], 1 - rg[-1]))

# --- A2: analytic-derivative identity (the monotonicity proof)
#   d m1/d rho = E[h(Y)]/(2 sqrt(rho)) + (a^2/4) E[sech^2(aY/2) tanh(aY/2)],
#   h(y) = (a/2) sech^2(a y/2).  Both terms are > 0 for rho in (0,1), a > 0.
from scipy.integrate import quad
def _E(f, rho, lim=12.0):
    s_, v_ = np.sqrt(rho), np.sqrt(1 - rho)
    g_ = lambda z: f(s_ + v_ * z) * np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    return quad(g_, -lim, lim, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
def m1_quad(rho, a):
    return _E(lambda y: np.tanh(a * y / 2), rho)
def dm1_analytic(rho, a):
    h = lambda y: (a / 2) / np.cosh(a * y / 2) ** 2
    st = lambda y: np.tanh(a * y / 2) / np.cosh(a * y / 2) ** 2
    # positivity-preserving form for the odd integrand:
    #   E[st] = int_0^inf st(y) [phi(y-s) - phi(y+s)] dy / v
    s_, v_ = np.sqrt(rho), np.sqrt(1 - rho)
    ph = lambda u: np.exp(-u * u / 2) / np.sqrt(2 * np.pi)
    E_st = quad(lambda y: st(y) * (ph((y - s_) / v_) - ph((y + s_) / v_)) / v_,
                0, s_ + 14 * v_, limit=400, epsabs=1e-16, epsrel=1e-13)[0]
    E_h = _E(h, rho)
    return E_h / (2 * s_) + (a ** 2 / 4) * E_st, E_h, E_st
chk = []
for a in [0.5, 2.0, 5.0, 12.0]:
    for r in [1e-4, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999]:
        an, Eh, Est = dm1_analytic(r, a)
        d_ = 1e-3 * min(r, 1 - r)
        f = lambda x: m1_quad(x, a)
        nu = (-f(r + 2 * d_) + 8 * f(r + d_) - 8 * f(r - d_) + f(r - 2 * d_)) / (12 * d_)
        chk.append(dict(a=a, rho=r, analytic=an, numeric=float(nu),
                        rel_err=float(abs(an - nu) / max(abs(an), 1e-300)),
                        E_h=Eh, E_sech2tanh=Est,
                        both_terms_positive=bool(Eh > 0 and Est > 0)))
A['derivative_identity'] = chk
print('A2 max rel err', max(c['rel_err'] for c in chk),
      'min E_sech2tanh', min(c['E_sech2tanh'] for c in chk),
      'all positive', all(c['both_terms_positive'] for c in chk), flush=True)

# --- A3: achievable range of mu_a and conditioning, per check degree
H0 = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'H.npy'))
rng_tab = []
for dc in [3, 4, 6, 10, 18, 19]:
    for a in [1.0, 2.0, 5.0, 12.0, np.inf]:
        mu0 = L.t_moments(1e-9, a, 1)[1] ** dc
        mu1 = (np.tanh(a / 2) ** dc) if np.isfinite(a) else 1.0
        sl = lambda r: dc * L.t_moments(r, a, 1)[1] ** (dc - 1) * L.dm1_drho(r, a)
        rng_tab.append(dict(dc=dc, a=('inf' if not np.isfinite(a) else a),
                            mu_at_rho0=float(mu0), mu_at_rho1=float(mu1),
                            slope_rho_0p02=float(sl(0.02)), slope_rho_0p5=float(sl(0.5)),
                            slope_rho_0p98=float(sl(0.98)), slope_rho_0p999=float(sl(0.999))))
A['range_and_slope'] = rng_tab

# --- A4: estimator conditioning on the real (256,126) code
Hc0 = L.to_csr(H0); R0 = (H0.shape[1] - L.packed_gf2_rank(H0)) / H0.shape[1]
A['code256_rate'] = R0
cond = []
for a in [1.0, 2.0, 5.0, np.inf]:
    for r in [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 0.995]:
        m, v = L.stat_mean_var_sparse(Hc0, r, a)
        d = np.asarray(Hc0.sum(1)).ravel()
        sl = float((d * L.t_moments(r, a, 1)[1] ** (d - 1)).mean() * L.dm1_drho(r, a))
        cond.append(dict(a=('inf' if not np.isfinite(a) else a), rho=r, mu=m, sd=float(v ** .5),
                         slope=sl, sd_rho=(float(v ** .5 / abs(sl)) if sl != 0 else None),
                         rel_sd_rho=(float(v ** .5 / abs(sl) / r) if sl != 0 else None)))
A['conditioning_n256'] = cond

# --- A5: boundary rule + how often it fires
#  rule:  rho_lo/hi are the ends of the inversion table.
#         S <= mu(rho_lo)  -> rho_hat = rho_lo  (left censored)
#         S >= mu(rho_hi)  -> rho_hat = rho_hi  (right censored)
#  mu(rho_hi) < 1 for finite a, so right censoring is a real event.
Hidx = torch.tensor(np.stack([np.nonzero(r)[0] for r in H0])).to(dev)
g = torch.Generator(device=dev); g.manual_seed(1234)
NMC = 20000 if QUICK else 400000
bnd = []
for a in [1.0, 2.0, 5.0, 12.0]:
    for lo, hi in [(0.02, 0.98), (0.002, 0.998)]:
        grid = np.linspace(lo, hi, 2001)
        tab = L.make_table(Hc0, a, grid)
        for ebdb in [-4, -2, 0, 2, 4, 6, 8, 10]:
            rho = 1 / (1 + 1 / (2 * R0 * 10 ** (ebdb / 10)))
            nlo = nhi = 0; tot = 0; acc = []
            B = 20000
            for _ in range(NMC // B):
                y = np.sqrt(rho) + np.sqrt(1 - rho) * torch.randn(B, H0.shape[1], device=dev, generator=g)
                S = torch.tanh(a * y / 2)[:, Hidx].prod(-1).mean(-1).cpu().numpy()
                nlo += int((S <= tab[0]).sum()); nhi += int((S >= tab[-1]).sum()); tot += B
                acc.append(S)
            S = np.concatenate(acc)
            rh = L.invert_table(S, tab, grid)
            bnd.append(dict(a=a, lo=lo, hi=hi, eb_db=ebdb, rho=rho,
                            frac_below=nlo / tot, frac_above=nhi / tot,
                            mu_hi=float(tab[-1]), mu_lo=float(tab[0]),
                            rmse_rho=float(np.sqrt(((rh - rho) ** 2).mean())),
                            bias_rho=float(rh.mean() - rho)))
            print('A5', bnd[-1], flush=True)
A['boundary'] = bnd
OUT['A'] = A
json.dump(OUT, open('r4_A_range.json', 'w'), indent=1)

# ======================================================= (B) complexity
print('=== B ===', flush=True)
B = {}
codes = C.build_all()
props = {}
for k, H in codes.items():
    props[k] = L.code_props(H)
    print('code', k, props[k], flush=True)
B['code_props'] = props

rows = []
for k, H in codes.items():
    Hc = L.to_csr(H); M, n = H.shape
    dv = np.asarray(Hc.sum(0)).ravel().astype(np.int64)
    dc = np.asarray(Hc.sum(1)).ravel().astype(np.int64)
    t0 = time.time(); OO = (Hc @ Hc.T).tocoo(); t_prod = time.time() - t0
    P = OO.nnz
    pred = int(M + (dv * (dv - 1)).sum())      # exact when girth >= 6
    rho = 0.6
    t0 = time.time()
    for _ in range(5):
        L.stat_mean_var_sparse(Hc, rho, 5.0, OO=OO)
    t_sparse = (time.time() - t0) / 5
    t0 = time.time(); _ = (Hc @ Hc.T).tocoo(); L.stat_mean_var_sparse(Hc, rho, 5.0); t_sparse_full = time.time() - t0
    r = dict(code=k, n=n, M=M, dv_mean=float(dv.mean()), dc_mean=float(dc.mean()),
             nnz_HHt=P, pred_M_plus_sum_dv2=pred, dense_pairs=M * M,
             t_sparse_s=t_sparse, t_sparse_incl_product_s=t_sparse_full,
             mem_sparse_MB=P * 12 / 1e6, mem_dense_MB=M * M * 8 / 1e6)
    if M <= 2200:
        t0 = time.time(); vd = L.stat_var_dense(H, rho, 5.0); r['t_dense_s'] = time.time() - t0
        _, vs = L.stat_mean_var_sparse(Hc, rho, 5.0, OO=OO)
        r['dense_vs_sparse_rel_diff'] = float(abs(vd - vs) / abs(vs))
        r['speedup'] = r['t_dense_s'] / r['t_sparse_incl_product_s']
    rows.append(r); print('B-var', {kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in r.items()}, flush=True)
B['variance_cost'] = rows
json.dump(dict(OUT, B=B), open('r4_B_complexity.json', 'w'), indent=1)

# --- B2: exact mu4
mu4rows = []
for k, H in codes.items():
    Hc = L.to_csr(H); M = H.shape[0]
    if M > (600 if QUICK else 2200):
        mu4rows.append(dict(code=k, M=M, skipped='M too large for this pass')); continue
    try:
        t0 = time.time(); r, pre = L.mu4_exact(Hc, 0.6, 5.0, return_pre=True); t_first = time.time() - t0
        t0 = time.time(); r2 = L.mu4_exact(Hc, 0.7, 5.0, pre=pre); t_warm = time.time() - t0
        row = dict(code=k, n=H.shape[1], M=M, t_setup_plus_eval_s=t_first, t_warm_eval_s=t_warm,
                   n_edges=r['n_edges'], n_triples=r['n_triples'], n_quads=r['n_quads'],
                   kurt_rho0p6=r['kurt'], kurt_rho0p7=r2['kurt'], exact=r['exact'],
                   kurt_partial_rho0p6=r['kurt_partial'],
                   triples_capped=r['triples_capped'], quads_capped=r['quads_capped'],
                   var_rho0p6=r['var'], mu4_rho0p6=r['mu4'])
    except MemoryError as e:
        row = dict(code=k, M=M, error='MemoryError')
    mu4rows.append(row); print('B-mu4', row, flush=True)
B['mu4_cost'] = mu4rows

# --- B3: mu4 on the paper's code at several SNR + confidence sets
cs = []
pre = None
for ebdb in [1.0, 2.0, 3.0, 4.0, 5.0]:
    rho = 1 / (1 + 1 / (2 * R0 * 10 ** (ebdb / 10)))
    r, pre = L.mu4_exact(Hc0, rho, 5.0, pre=pre, return_pre=True)
    kap = r['kurt']
    row = dict(eb_db=ebdb, rho=rho, sd=float(r['var'] ** .5), mu4=r['mu4'], kurt=kap)
    for kk in [2, 3, 4]:
        row['cheb_k%d' % kk] = 1 / kk ** 2
        row['m4_k%d' % kk] = (kap - 1) / ((kap - 1) + (kk ** 2 - 1) ** 2)
        row['gauss_k%d' % kk] = float(erfc(kk / np.sqrt(2)))
    cs.append(row); print('B-cs', row, flush=True)
B['conf_sets'] = cs

# --- B4: Monte-Carlo validation of mu4 / kurtosis
mc = []
NV = 200000 if QUICK else 4000000
for ebdb in [2.0, 3.0, 4.0]:
    rho = 1 / (1 + 1 / (2 * R0 * 10 ** (ebdb / 10)))
    acc = []
    Bc = 50000
    for _ in range(NV // Bc):
        y = np.sqrt(rho) + np.sqrt(1 - rho) * torch.randn(Bc, H0.shape[1], device=dev, generator=g)
        acc.append(torch.tanh(5.0 * y / 2)[:, Hidx].prod(-1).mean(-1).double().cpu().numpy())
    S = np.concatenate(acc)
    r, pre = L.mu4_exact(Hc0, rho, 5.0, pre=pre, return_pre=True)
    sd = S.std(ddof=1); m = S.mean()
    k_emp = float((((S - m) ** 4).mean()) / sd ** 4)
    emp = {}
    for kk in [2, 3, 4]:
        emp['emp_fail_k%d' % kk] = float((np.abs(S - r['mean']) >= kk * r['var'] ** .5).mean())
    mc.append(dict(eb_db=ebdb, N=len(S), sd_exact=float(r['var'] ** .5), sd_emp=float(sd),
                   mean_exact=r['mean'], mean_emp=float(m),
                   kurt_exact=r['kurt'], kurt_emp=k_emp,
                   kurt_emp_stderr=float(np.sqrt(24.0 / len(S))), **emp))
    print('B-mc', mc[-1], flush=True)
B['mu4_validation'] = mc
OUT['B'] = B
json.dump(dict(A=OUT['A']), open('r4_A_range.json', 'w'), indent=1)
json.dump(dict(B=B), open('r4_B_complexity.json', 'w'), indent=1)
print('DONE ab %.1f s' % (time.time() - t00), flush=True)
