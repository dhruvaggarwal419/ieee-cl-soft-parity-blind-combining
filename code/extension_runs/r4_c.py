"""(C) generality across codes: estimator RMSE and exact-vs-independent variance."""
import numpy as np, torch, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4lib as L, r4codes as CB, r4est as E

QUICK = '--quick' in sys.argv
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
SNRS = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
NMC = 20000 if QUICK else 200000
codes = CB.build_all()
out = {'snr_db_def': '10 log10(rho/(1-rho)) = 10 log10(2 Es/N0)', 'nmc': NMC, 'codes': {}}
t00 = time.time()
g = torch.Generator(device=dev); g.manual_seed(2024)

for name, H in codes.items():
    t0 = time.time()
    c = E.Ctx(H, dev)
    B = int(max(256, min(20000, 3e7 // (c.M * c.K))))
    rec = dict(n=c.n, M=c.M, rank=c.rank, rate=c.R, props=L.code_props(H), batch=B)
    # variance structure
    vr = []
    for sdb in SNRS:
        rho = float(10 ** (sdb / 10) / (1 + 10 ** (sdb / 10)))
        for a in [2.0, 5.0, np.inf]:
            mean, ve = L.stat_mean_var_sparse(c.Hc, rho, a)
            vi = L.stat_var_indep(c.Hc, rho, a)
            vr.append(dict(snr_db=sdb, a=('inf' if not np.isfinite(a) else a),
                           mean=mean, var_exact=ve, var_indep=vi, ratio=ve / vi))
    rec['variance'] = vr
    # estimator accuracy + empirical variance check
    est5 = E.SoftEst(c, 5.0)
    rows = []
    for sdb in SNRS:
        s = 10 ** (sdb / 10); rho = s / (1 + s)
        am = 2 * np.sqrt(rho) / (1 - rho)
        estm = E.SoftEst(c, am)
        acc = {k: [] for k in ['a5', 'am', 'S5', 'Sm']}
        for _ in range(max(1, NMC // B)):
            y = np.sqrt(rho) + np.sqrt(1 - rho) * torch.randn(B, c.n, device=dev, generator=g)
            r5, S5 = est5(y); rm, Sm = estm(y)
            acc['a5'].append(r5); acc['am'].append(rm); acc['S5'].append(S5); acc['Sm'].append(Sm)
        A = {k: np.concatenate(v) for k, v in acc.items()}
        me, ve = L.stat_mean_var_sparse(c.Hc, rho, 5.0)
        row = dict(snr_db=sdb, rho=rho, a_match=am, N=len(A['a5']),
                   S5_mean_exact=me, S5_mean_emp=float(A['S5'].mean()),
                   S5_sd_exact=float(ve ** .5), S5_sd_emp=float(A['S5'].std(ddof=1)),
                   sd_ratio_emp_over_exact=float(A['S5'].std(ddof=1) / ve ** .5))
        for k, lab in [('a5', 'a5'), ('am', 'amatch')]:
            rh = np.clip(A[k], 1e-6, 1 - 1e-6)
            row['rmse_rho_' + lab] = float(np.sqrt(((rh - rho) ** 2).mean()))
            row['bias_rho_' + lab] = float(rh.mean() - rho)
            row['rmse_db_' + lab] = float(np.sqrt(((E.db_from_rho(rh) - sdb) ** 2).mean()))
            row['bias_db_' + lab] = float(E.db_from_rho(rh).mean() - sdb)
        rows.append(row); print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    rec['estimator'] = rows
    # 5G puncturing: which checks avoid the first 2Z (punctured) columns
    if name.startswith('nr_'):
        Z = 52 if 'bg2' in name else 96
        pun = np.arange(2 * Z)
        touch = (H[:, pun].sum(1) > 0)
        rec['punctured_cols'] = int(2 * Z)
        rec['frac_checks_touching_punctured'] = float(touch.mean())
        rec['n_checks_usable_if_punctured'] = int((~touch).sum())
    rec['secs'] = time.time() - t0
    out['codes'][name] = rec
    json.dump(out, open('r4_C_codes.json', 'w'), indent=1)
    print('== done', name, '%.1fs' % rec['secs'], flush=True)
print('DONE c %.1f' % (time.time() - t00), flush=True)
