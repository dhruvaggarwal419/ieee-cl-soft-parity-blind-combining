"""(F) selection-aware analysis of the blind ratio search."""
import numpy as np, torch, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4lib as L, r4codes as CB, r4est as E

QUICK = '--quick' in sys.argv
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
A_SOFT = 5.0; ALPHA = 0.85
codes = CB.build_all(which=['reg_n256_R12'])
H = codes['reg_n256_R12']; c = E.Ctx(H, dev); dec = E.Dec(c)
R = c.R
g = torch.Generator(device=dev); g.manual_seed(31337)
NF = 20000 if QUICK else 100000
BS = 10000 if dev == 'cuda' else 2500
out = {'code': 'reg_n256_R12', 'n': c.n, 'rate': R, 'a': A_SOFT, 'nf': NF, 'alpha': ALPHA}
def rho_eb(eb): return 1 / (1 + 1 / (2 * R * 10 ** (eb / 10)))
def wts(r): r = r.clamp(0.02, 0.98); return 2 * torch.sqrt(r) / (1 - r)

GRIDS = {G: np.geomspace(0.1, 10, G) for G in [5, 11, 31, 61, 121, 241]}
GRIDS[481] = np.geomspace(0.1, 10, 481)   # stand-in for continuous optimisation

def all_S(y1, y2, RG):
    """(B,G) statistic for every candidate ratio, sample-rms normalisation."""
    o = []
    for k in range(len(RG)):
        z = y1 + RG[k] * y2
        z = z / z.pow(2).mean(1, keepdim=True).sqrt()
        o.append(c.soft_stat(z, A_SOFT))
    return torch.stack(o, 1)

def fer(Lin):
    return int((dec.run(Lin, 50, ALPHA) < 0).any(1).sum())

# ---------------------------------------------------------------- F1+F4 sweep
# Design: the TOTAL combined SNR gamma1+gamma2 is held fixed and only the
# imbalance gap = 10log10(gamma1/gamma2) is varied, so the oracle FER is flat
# and every change is caused by the imbalance alone.
rows = []
EB_TOT = 2.5     # dB; gamma_tot = 2 R 10^(EB_TOT/10)
def pair(gap):
    t = 10 ** (gap / 10.0); gt = 2 * R * 10 ** (EB_TOT / 10.0)
    g1 = gt * t / (1 + t); g2 = gt / (1 + t)
    return g1 / (1 + g1), g2 / (1 + g2)
for gap in ([6.0] if QUICK else [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]):
    RHO1, RHO2 = pair(gap)
    err = {}
    logr = {}
    corr_acc = []
    nb = max(1, NF // BS)
    for b in range(nb):
        r1 = torch.full((BS,), RHO1, device=dev); r2 = torch.full((BS,), RHO2, device=dev)
        y1 = torch.sqrt(r1)[:, None] + torch.sqrt(1 - r1)[:, None] * torch.randn(BS, c.n, device=dev, generator=g)
        y2 = torch.sqrt(r2)[:, None] + torch.sqrt(1 - r2)[:, None] * torch.randn(BS, c.n, device=dev, generator=g)
        ro = wts(r2) / wts(r1)
        err['oracle'] = err.get('oracle', 0) + fer(y1 + ro[:, None] * y2)
        err['egc'] = err.get('egc', 0) + fer(y1 + y2)
        err['best1'] = err.get('best1', 0) + fer(y1)          # keep the stronger frame only
        for G, gv in GRIDS.items():
            RG = torch.tensor(gv, device=dev).float()
            S = all_S(y1, y2, RG)
            i = S.argmax(1)
            rb = RG[i]
            key = 'G%d' % G
            err[key] = err.get(key, 0) + fer(y1 + rb[:, None] * y2)
            logr.setdefault(key, []).append((torch.log(rb) - torch.log(ro)).cpu().numpy())
            if G == 31:
                Sn = S.cpu().numpy()
                corr_acc.append(Sn)
                # parabolic peak interpolation in log r
                lg = np.log(gv); ii = i.cpu().numpy()
                ic = np.clip(ii, 1, G - 2)
                y0 = Sn[np.arange(len(ii)), ic - 1]; y1_ = Sn[np.arange(len(ii)), ic]; y2_ = Sn[np.arange(len(ii)), ic + 1]
                den = (y0 - 2 * y1_ + y2_)
                dl = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2_) / np.where(np.abs(den) > 1e-12, den, 1), 0.0)
                dl = np.clip(dl, -1, 1)
                step = lg[1] - lg[0]
                rp = np.exp(lg[ic] + dl * step)
                rp_t = torch.tensor(rp, device=dev, dtype=torch.float32)
                err['G31_parab'] = err.get('G31_parab', 0) + fer(y1 + rp_t[:, None] * y2)
                logr.setdefault('G31_parab', []).append((torch.log(rp_t) - torch.log(ro)).cpu().numpy())
                # 3-point smoothing before argmax
                Ss = np.copy(Sn); Ss[:, 1:-1] = (Sn[:, :-2] + 2 * Sn[:, 1:-1] + Sn[:, 2:]) / 4
                js = Ss.argmax(1); rs = torch.tensor(gv[js], device=dev, dtype=torch.float32)
                err['G31_smooth'] = err.get('G31_smooth', 0) + fer(y1 + rs[:, None] * y2)
                logr.setdefault('G31_smooth', []).append((torch.log(rs) - torch.log(ro)).cpu().numpy())
                # shrinkage of log r_hat
                for beta in [0.5, 0.7, 0.85]:
                    rsh = torch.exp(beta * torch.log(rb))
                    k = 'G31_shrink%.2f' % beta
                    err[k] = err.get(k, 0) + fer(y1 + rsh[:, None] * y2)
                    logr.setdefault(k, []).append((torch.log(rsh) - torch.log(ro)).cpu().numpy())
    N = nb * BS
    row = dict(gap_db=gap, eb_total_db=EB_TOT, rho1=RHO1, rho2=RHO2, N=N, fer={k: v / N for k, v in err.items()})
    row['logratio'] = {k: dict(bias=float(np.concatenate(v).mean()), std=float(np.concatenate(v).std()))
                       for k, v in logr.items()}
    Sn = np.concatenate(corr_acc, 0)
    cm = np.corrcoef(Sn.T)
    row['grid31_neighbour_corr'] = float(np.mean([cm[i, i + 1] for i in range(30)]))
    row['grid31_lag2_corr'] = float(np.mean([cm[i, i + 2] for i in range(29)]))
    row['grid31_corr_matrix_first_row'] = [float(x) for x in cm[15]]
    rows.append(row)
    print('gap', gap, {k: round(v, 5) for k, v in row['fer'].items()}, flush=True)
    print('   logr', {k: (round(v['bias'], 3), round(v['std'], 3)) for k, v in row['logratio'].items()}, flush=True)
    out['sweep'] = rows
    json.dump(out, open('r4_F_selection.json', 'w'), indent=1)
print('DONE f', flush=True)
