"""(E) beyond BPSK over AWGN."""
import numpy as np, torch, json, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r4lib as L, r4codes as CB, r4est as E

QUICK = '--quick' in sys.argv
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
NMC = 20000 if QUICK else 200000
codes = CB.build_all(which=['reg_n256_R12', 'reg_n1024_R12'])
H = codes['reg_n256_R12']; c = E.Ctx(H, dev)
soft5 = E.SoftEst(c, 5.0)
g = torch.Generator(device=dev); g.manual_seed(777)
out = {'note': 'all on the (n=256,R=1/2) PEG code unless stated', 'nmc': NMC}
SNRS = [-2.0, 0.0, 2.0, 4.0, 6.0]
def rho_of(db): s = 10 ** (db / 10); return s / (1 + s)
def randn(B, n): return torch.randn(B, n, device=dev, generator=g)
BS = 20000 if dev == 'cuda' else 4000

def mc(fn, N=None):
    """fn(B) -> dict of (B,) numpy arrays; concatenated."""
    N = N or NMC; acc = None
    for _ in range(max(1, N // BS)):
        d = fn(BS)
        if acc is None: acc = {k: [v] for k, v in d.items()}
        else:
            for k, v in d.items(): acc[k].append(v)
    return {k: np.concatenate(v) for k, v in acc.items()}

# ---------------------------------------------------------- E1 block fading
print('=== E1 block fading ===', flush=True)
r1 = []
for sdb in SNRS:
    rho = rho_of(sdb)
    def f(B):
        h = torch.sqrt(-2 * torch.log(torch.rand(B, 1, device=dev, generator=g)))  # Rayleigh, E|h|^2=2
        h = h / np.sqrt(2.0)                                                       # E|h|^2 = 1
        r = h * np.sqrt(rho) + np.sqrt(1 - rho) * randn(B, c.n)
        rho_eff = (h[:, 0] ** 2 * rho) / (h[:, 0] ** 2 * rho + (1 - rho))
        y_s = r / r.pow(2).mean(1, keepdim=True).sqrt()                  # sample-power norm
        y_g = r / torch.sqrt(h ** 2 * rho + (1 - rho))                   # genie norm
        return dict(rho_eff=rho_eff.double().cpu().numpy(),
                    rs=soft5(y_s)[0], rg=soft5(y_g)[0])
    A = mc(f)
    for lab in ['rs', 'rg']:
        e = E.db_from_rho(np.clip(A[lab], 1e-6, 1 - 1e-6)) - E.db_from_rho(A['rho_eff'])
        r1.append(dict(snr_db=sdb, norm=('sample' if lab == 'rs' else 'genie'),
                       bias_db=float(e.mean()), rmse_db=float(np.sqrt((e ** 2).mean())),
                       median_abs_db=float(np.median(np.abs(e)))))
        print(r1[-1], flush=True)
out['block_fading'] = r1

# ---------------------------------------------------------- E2 fast fading
print('=== E2 fast fading ===', flush=True)
def fade_moments(rho, a, cmax=2, nh=64):
    """m_c = E_h E_n [ tanh(a Y/2)^c ], Y = h sqrt(rho) + sqrt(1-rho) Z,
    |h| Rayleigh with E|h|^2 = 1 (so h^2 ~ Exp(1))."""
    xs, ws = np.polynomial.laguerre.laggauss(nh)   # for exp(-x) weight: h^2 = x
    ws = ws / ws.sum()
    hh = np.sqrt(xs)
    y = hh[:, None] * np.sqrt(rho) + np.sqrt(1 - rho) * L.GH_X[None, :]
    t = np.tanh(a * y / 2)
    W = ws[:, None] * L.GH_W[None, :]
    return np.array([1.0] + [float((W * t ** k).sum()) for k in range(1, cmax + 1)])
r2 = []
for sdb in SNRS:
    rho = rho_of(sdb); a = 5.0
    mmf = fade_moments(rho, a, 2)
    mean_f, var_f = L.stat_mean_var_sparse(c.Hc, rho, a, mm=mmf)
    def f(B):
        h2 = -torch.log(torch.rand(B, c.n, device=dev, generator=g))   # Exp(1)
        y = torch.sqrt(h2) * np.sqrt(rho) + np.sqrt(1 - rho) * randn(B, c.n)
        return dict(S=c.soft_stat(y, a).double().cpu().numpy())
    A = mc(f)
    # matched inversion table under fading
    grid = np.linspace(0.02, 0.98, 400)
    d = np.asarray(c.Hc.sum(1)).ravel()
    tabf = np.array([float((fade_moments(r, a, 1)[1] ** d).mean()) for r in grid])
    tabA = L.make_table(c.Hc, a, grid)
    rf = L.invert_table(A['S'], tabf, grid); ra = L.invert_table(A['S'], tabA, grid)
    ef = E.db_from_rho(np.clip(rf, 1e-6, 1 - 1e-6)) - sdb
    ea = E.db_from_rho(np.clip(ra, 1e-6, 1 - 1e-6)) - sdb
    r2.append(dict(snr_db=sdb, mean_exact_fading=mean_f, mean_emp=float(A['S'].mean()),
                   sd_exact_fading=float(var_f ** .5), sd_emp=float(A['S'].std(ddof=1)),
                   sd_ratio=float(A['S'].std(ddof=1) / var_f ** .5),
                   bias_db_matched_table=float(ef.mean()), rmse_db_matched_table=float(np.sqrt((ef ** 2).mean())),
                   bias_db_awgn_table=float(ea.mean()), rmse_db_awgn_table=float(np.sqrt((ea ** 2).mean()))))
    print(r2[-1], flush=True)
out['fast_fading'] = r2

# ---------------------------------------------------------- E3 scale / a mismatch
print('=== E3 normalisation / a mismatch ===', flush=True)
r3 = []
for sdb in [0.0, 4.0]:
    rho = rho_of(sdb)
    for ddb in [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]:
        sc = 10 ** (ddb / 20)    # amplitude-scale error from a wrong power estimate
        def f(B):
            y = np.sqrt(rho) + np.sqrt(1 - rho) * randn(B, c.n)
            return dict(r=soft5(y * sc)[0])
        A = mc(f, N=max(BS, NMC // 4))
        e = E.db_from_rho(np.clip(A['r'], 1e-6, 1 - 1e-6)) - sdb
        r3.append(dict(snr_db=sdb, power_err_db=ddb, amp_scale=sc,
                       bias_db=float(e.mean()), rmse_db=float(np.sqrt((e ** 2).mean()))))
        print(r3[-1], flush=True)
out['scale_mismatch'] = r3

# ---------------------------------------------------------- E4 QPSK / 16-QAM
print('=== E4 modulation ===', flush=True)
def qam16_llr(rx, N0, pts, bits):
    """exact bit LLRs, rx:(B,Ns) complex-as-2-channel -> (B,Ns,4)"""
    d2 = (rx[..., None, 0] - pts[None, None, :, 0]) ** 2 + (rx[..., None, 1] - pts[None, None, :, 1]) ** 2
    lg = -d2 / N0
    mx = lg.max(-1, keepdim=True).values
    ex = torch.exp(lg - mx)
    out = []
    for b in range(4):
        m1 = bits[:, b] == 1; m0 = ~m1
        out.append(torch.log(ex[..., m0].sum(-1) + 1e-300) - torch.log(ex[..., m1].sum(-1) + 1e-300))
    return torch.stack(out, -1)   # LLR = log P(b=0)/P(b=1)

r4 = []
# QPSK: Gray-mapped QPSK on AWGN is two parallel BPSK channels -> identical statistic
for sdb in SNRS:
    rho = rho_of(sdb)
    def f(B):
        # map n bits onto n/2 QPSK symbols, I and Q independent, same Es/N0 per bit
        y = np.sqrt(rho) + np.sqrt(1 - rho) * randn(B, c.n)   # equals the per-rail output
        return dict(S=c.soft_stat(y, 5.0).double().cpu().numpy())
    A = mc(f, N=max(BS, NMC // 4))
    me, ve = L.stat_mean_var_sparse(c.Hc, rho, 5.0)
    r4.append(dict(mod='QPSK(gray)', snr_db=sdb, mean_exact_bpsk=me, mean_emp=float(A['S'].mean()),
                   sd_exact_bpsk=float(ve ** .5), sd_emp=float(A['S'].std(ddof=1)),
                   note='Gray QPSK over AWGN = 2 parallel BPSK rails; statistic identical'))
    print(r4[-1], flush=True)

# 16-QAM, Gray, exact LLRs, random bit interleaver.
# x-axis here is Es/N0 in dB (Es = 1 per 16-QAM symbol, 4 bits/symbol).
lev = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)
gray = np.array([0, 1, 3, 2])          # Gray labels of the 4 amplitude levels
pts = []; bts = []
for i_ in range(4):
    for q_ in range(4):
        pts.append([lev[i_], lev[q_]])
        bts.append([(gray[i_] >> 1) & 1, gray[i_] & 1, (gray[q_] >> 1) & 1, gray[q_] & 1])
pts_t = torch.tensor(np.array(pts), dtype=torch.float32, device=dev)
bts_t = torch.tensor(np.array(bts), dtype=torch.bool, device=dev)
rngp = np.random.default_rng(5); perm = rngp.permutation(c.n)
perm_t = torch.tensor(perm, device=dev)
Ns = c.n // 4
# how often does one check contain two bits of the SAME 16-QAM symbol?
symb = perm // 4
same = 0
for row in H:
    v = np.nonzero(row)[0]; sy = symb[v]
    same += int(len(np.unique(sy)) < len(sy))
qam_same_symbol_checks = same / H.shape[0]
ESN0 = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]
qrows = []
for sdb in ESN0:
    N0 = 1.0 / (10 ** (sdb / 10))
    def f(B):
        tx = torch.empty(B, Ns, 2, device=dev)
        tx[:, :, 0] = pts_t[0, 0]; tx[:, :, 1] = pts_t[0, 1]     # all-zero bit word
        rx = tx + np.sqrt(N0 / 2) * torch.randn(B, Ns, 2, device=dev, generator=g)
        Lb = qam16_llr(rx, N0, pts_t, bts_t).reshape(B, Ns * 4)[:, perm_t]
        t = torch.tanh(Lb / 2)
        return dict(S=c.prod_over_checks(t).mean(-1).double().cpu().numpy(),
                    t1=t.double().mean(0).cpu().numpy(), t2=(t ** 2).double().mean(0).cpu().numpy())
    acc = {'S': [], 't1': [], 't2': []}
    nb = max(2, (NMC // 4) // BS)
    for _ in range(nb):
        d = f(BS)
        for k in acc: acc[k].append(d[k])
    S = np.concatenate(acc['S']); m1v = np.mean(acc['t1'], 0); m2v = np.mean(acc['t2'], 0)
    mp, vp = L.stat_mean_var_pervar(c.Hc, np.clip(m1v, 1e-9, None), np.clip(m2v, 1e-9, None))
    qrows.append(dict(mod='16QAM(gray,exactLLR)', esn0_db=sdb, N=len(S),
                      S_mean_emp=float(S.mean()), S_sd_emp=float(S.std(ddof=1)),
                      S_mean_pred_indep=mp, S_sd_pred_indep=float(vp ** .5),
                      sd_emp_over_pred=float(S.std(ddof=1) / vp ** .5),
                      checks_with_two_bits_same_symbol=qam_same_symbol_checks))
    print(qrows[-1], flush=True)
# monotone? and estimator RMSE with an MC-built table
mt = np.array([r['S_mean_emp'] for r in qrows])
qam_mono = bool(np.all(np.diff(mt) > 0))
r4 += qrows
out['qam16_monotone_mean'] = qam_mono
out['qam16_checks_two_bits_same_symbol'] = qam_same_symbol_checks
out['modulation'] = r4

# ---------------------------------------------------------- E5 misalignment
# NOTE: with the all-zero codeword the received vector is i.i.d., so a cyclic
# shift changes nothing.  Real random codewords are required here.
print('=== E5 misalignment (random codewords) ===', flush=True)
NB = L.gf2_nullspace(H)
NBt = torch.tensor(NB.astype(np.float32), device=dev)
def codewords(B):
    u = (torch.rand(B, NB.shape[0], device=dev, generator=g) < 0.5).float()
    cw = torch.remainder(u @ NBt, 2.0)
    return 1.0 - 2.0 * cw
r5 = []
for sdb in [0.0, 2.0, 4.0, 6.0]:
    rho = rho_of(sdb)
    for sh in [0, 1, 2, 4]:
        def f(B):
            x = codewords(B)
            y = np.sqrt(rho) * x + np.sqrt(1 - rho) * randn(B, c.n)
            ys = torch.roll(y, shifts=sh, dims=1)
            return dict(S=c.soft_stat(ys, 5.0).double().cpu().numpy(), r=soft5(ys)[0])
        A = mc(f, N=max(BS, NMC // 8))
        r5.append(dict(snr_db=sdb, shift=sh, S_mean=float(A['S'].mean()), S_sd=float(A['S'].std(ddof=1)),
                       rho_hat_mean=float(A['r'].mean()),
                       db_hat_mean=float(E.db_from_rho(np.clip(A['r'], 1e-6, 1 - 1e-6)).mean())))
        print(r5[-1], flush=True)
al = []
for sdb in [-2.0, 0.0, 2.0, 4.0]:
    rho = rho_of(sdb)
    def f(B):
        x = codewords(B)
        y = np.sqrt(rho) * x + np.sqrt(1 - rho) * randn(B, c.n)
        S = torch.stack([c.soft_stat(torch.roll(y, s, 1), 5.0) for s in range(8)], 1)
        return dict(ok=(S.argmax(1) == 0).double().cpu().numpy())
    A = mc(f, N=max(BS, NMC // 8))
    al.append(dict(snr_db=sdb, p_correct_shift=float(A['ok'].mean()), n_shifts=8, chance=1 / 8))
    print(al[-1], flush=True)
out['misalignment'] = r5; out['alignment_detection'] = al

# ---------------------------------------------------------- E6 non-Gaussian
print('=== E6 non-Gaussian interference ===', flush=True)
hardE = E.SoftEst(c, np.inf)
r6 = []
for sdb in [0.0, 2.0, 4.0]:
    rho = rho_of(sdb); nv = 1 - rho
    scen = [('gauss', None)]
    scen += [('mix_e0.01_K100', (0.01, 100.0)), ('mix_e0.05_K20', (0.05, 20.0)),
             ('mix_e0.10_K10', (0.10, 10.0))]
    for sir in [10.0, 6.0, 3.0]:
        scen.append(('cci_sir%gdB' % sir, ('cci', sir)))
    for nm, par in scen:
        def f(B):
            if par is None:
                w = np.sqrt(nv) * randn(B, c.n)
            elif par[0] == 'cci':
                sir = par[1]; Pi = rho / 10 ** (sir / 10)
                w = np.sqrt(max(nv - Pi, 1e-6)) * randn(B, c.n) + np.sqrt(Pi) * torch.sign(randn(B, c.n))
            else:
                eps, K = par
                s2 = nv / (1 - eps + eps * K)
                imp = (torch.rand(B, c.n, device=dev, generator=g) < eps).float()
                w = torch.sqrt(torch.tensor(s2, device=dev)) * torch.sqrt(1 + (K - 1) * imp) * randn(B, c.n)
            y = np.sqrt(rho) + w
            y = y / y.pow(2).mean(1, keepdim=True).sqrt()
            return dict(soft=soft5(y)[0], hard=hardE(y)[0], m2m4=E.m2m4(y))
        A = mc(f, N=max(BS, NMC // 4))
        row = dict(snr_db=sdb, noise=nm)
        for k in ['soft', 'hard', 'm2m4']:
            e = E.db_from_rho(np.clip(A[k], 1e-6, 1 - 1e-6)) - sdb
            row['bias_db_' + k] = float(e.mean()); row['rmse_db_' + k] = float(np.sqrt((e ** 2).mean()))
        r6.append(row); print(row, flush=True)
out['non_gaussian'] = r6
json.dump(out, open('r4_E_channels.json', 'w'), indent=1)
print('DONE e', flush=True)
