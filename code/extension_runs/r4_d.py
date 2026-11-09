"""(D) fair single-frame SNR-estimator comparison:
proposed soft statistic vs Lechner-Pacher hard syndrome vs Fang soft LLR
vs M2M4 vs decision-directed, against three Cramer-Rao bounds."""
import numpy as np, torch, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4lib as L, r4codes as CB, r4est as E

QUICK = '--quick' in sys.argv
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
SNRS = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0]
NMC = 20000 if QUICK else 200000
CODES = ['reg_n256_R12', 'reg_n1024_R12'] + ([] if QUICK else ['nr_bg1_R12_n4224'])
codes = CB.build_all()
out = {'snr_db_def': '10 log10(rho/(1-rho))', 'nmc': NMC, 'codes': {}}
g = torch.Generator(device=dev); g.manual_seed(4242)
t00 = time.time()

for name in CODES:
    H = codes[name]; c = E.Ctx(H, dev); dec = E.Dec(c)
    B = int(max(512, min(10000, 2.5e8 // (c.M * c.K))))
    NM = NMC if c.n <= 1200 else max(50000, NMC // 4)
    FN = (4000 if QUICK else (20000 if c.n <= 1200 else 8000))
    print('== code', name, c.n, c.M, 'batch', B, flush=True)
    soft5 = E.SoftEst(c, 5.0)
    tp = E.TwoPassSoft(c, a0=1.0)
    softinf = E.SoftEst(c, np.inf)
    lp = E.HardLP(c)
    t_tab = {}
    t0 = time.time(); fang5 = E.FangEst(c, 5.0, grid=np.linspace(0.02,0.98,61), nmc=FN, bs=min(B, 4000))
    t_tab['fang5_table_s'] = time.time() - t0
    rows = []
    for sdb in SNRS:
        s = 10 ** (sdb / 10); rho = s / (1 + s)
        am = float(2 * np.sqrt(rho) / (1 - rho))
        softm = E.SoftEst(c, am)
        t0 = time.time(); fangm = E.FangEst(c, am, grid=np.linspace(0.02,0.98,61), nmc=FN, bs=min(B, 4000))
        t_fm = time.time() - t0
        acc = {k: [] for k in ['soft5', 'softm', 'soft2p', 'softinf', 'lp', 'fang5', 'fangm',
                               'm2m4', 'dd', 'genie']}
        tm = {k: 0.0 for k in acc}
        nb = max(1, NM // B)
        for _ in range(nb):
            y = np.sqrt(rho) + np.sqrt(1 - rho) * torch.randn(B, c.n, device=dev, generator=g)
            def timed(k, f):
                torch.cuda.synchronize() if dev == 'cuda' else None
                t = time.time(); v = f(); torch.cuda.synchronize() if dev == 'cuda' else None
                tm[k] += time.time() - t; acc[k].append(v)
            timed('soft5', lambda: soft5(y)[0])
            timed('softm', lambda: softm(y)[0])
            timed('softinf', lambda: softinf(y)[0])
            timed('lp', lambda: lp(y)[0])
            timed('fang5', lambda: fang5(y)[0])
            timed('fangm', lambda: fangm(y)[0])
            timed('m2m4', lambda: E.m2m4(y))
            timed('soft2p', lambda: tp(y))
            q = torch.tensor(E.m2m4(y), device=dev, dtype=torch.float32).clamp(0.02, 0.98)
            a0 = 2 * torch.sqrt(q) / (1 - q)
            timed('dd', lambda: E.dd_est(y, dec, a0, iters=50))
            A = (y * 1.0).mean(1); m2 = (y ** 2).mean(1)
            acc['genie'].append(torch.clamp(A ** 2 / m2, 1e-6, 1 - 1e-6).double().cpu().numpy())
        row = dict(snr_db=sdb, rho=rho, a_match=am, fangm_table_s=t_fm, N=nb * B)
        for k, v in acc.items():
            if not v: continue
            rh = np.clip(np.concatenate(v), 1e-6, 1 - 1e-6)
            db = E.db_from_rho(rh)
            row[k] = dict(bias_rho=float(rh.mean() - rho), std_rho=float(rh.std(ddof=1)),
                          rmse_rho=float(np.sqrt(((rh - rho) ** 2).mean())),
                          bias_db=float(db.mean() - sdb), std_db=float(db.std(ddof=1)),
                          rmse_db=float(np.sqrt(((db - sdb) ** 2).mean())),
                          us_per_frame=1e6 * tm.get(k, 0.0) / (nb * B),
                          dec_iters=(50 if k == 'dd' else 0))
        row['crb_da_rho'] = E.crb_da(rho, c.n)
        row['crb_nda_rho'] = float(E.crb_nda(rho, c.n))
        row['crb_syndrome_rho'] = float(E.crb_syndrome(c, rho))
        for kk, vv in [('crb_da', row['crb_da_rho']), ('crb_nda', row['crb_nda_rho']),
                       ('crb_syndrome', row['crb_syndrome_rho'])]:
            row[kk + '_db'] = float(np.sqrt(E.db_var_from_rho_var(rho, vv)))
        # does LP coincide with the a->inf soft inversion?
        d1 = np.concatenate(acc['lp']); d2 = np.concatenate(acc['softinf'])
        row['lp_vs_softinf_max_abs_diff_rho'] = float(np.abs(d1 - d2).max())
        row['lp_vs_softinf_corr'] = float(np.corrcoef(d1, d2)[0, 1])
        rows.append(row)
        print(name, sdb, {k: (round(row[k]['rmse_db'], 4) if isinstance(row[k], dict) else None)
                          for k in ['soft5', 'softm', 'soft2p', 'softinf', 'lp', 'fang5', 'fangm', 'm2m4', 'dd', 'genie']},
              'crb_nda_db', round(row['crb_nda_db'], 4), 'crb_syn_db', round(row['crb_syndrome_db'], 4), flush=True)
    out['codes'][name] = dict(n=c.n, M=c.M, rate=c.R, rows=rows, table_build=t_tab)
    json.dump(out, open('r4_D_baselines.json', 'w'), indent=1)
print('DONE d %.1f' % (time.time() - t00), flush=True)
