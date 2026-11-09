import numpy as np, torch
# ---------- code ----------
def make_H(n=256, dv=3, dc=6, seed=1):
    rng = np.random.default_rng(seed)
    M = n*dv//dc; per = n//dc  # rows per stripe
    base = np.zeros((per, n), dtype=np.int8)
    for i in range(per): base[i, i*dc:(i+1)*dc] = 1
    perms = [np.arange(n)] + [rng.permutation(n) for _ in range(dv-1)]
    def build(perms):
        return np.vstack([base[:, p] for p in perms])
    def cost(H):
        O = H.astype(np.int32) @ H.T.astype(np.int32); np.fill_diagonal(O, 0)
        return int((np.maximum(O-1, 0)).sum())
    H = build(perms); c = cost(H)
    for it in range(20000):
        if c == 0: break
        s = rng.integers(1, dv); p = perms[s].copy()
        i, j = rng.integers(0, n, 2); p[i], p[j] = p[j], p[i]
        P2 = list(perms); P2[s] = p; H2 = build(P2); c2 = cost(H2)
        if c2 <= c: perms, H, c = P2, H2, c2
    return H, c
def gf2_rank(H):
    A = H.copy().astype(np.uint8); r = 0
    for col in range(A.shape[1]):
        piv = np.nonzero(A[r:, col])[0]
        if len(piv) == 0: continue
        p = piv[0]+r; A[[r, p]] = A[[p, r]]
        rows = np.nonzero(A[:, col])[0]; rows = rows[rows != r]
        A[rows] ^= A[r]; r += 1
        if r == A.shape[0]: break
    return r
# ---------- exact statistics ----------
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(240); GH_W = GH_W/GH_W.sum()
def moments(rho, a):
    """m = E tanh(a y/2), q = E tanh^2, y = sqrt(rho)+sqrt(1-rho) n"""
    s, v = np.sqrt(rho), np.sqrt(1-rho)
    if not np.isfinite(a):
        from scipy.special import erf
        return float(erf(s/(v*np.sqrt(2)))), 1.0
    y = s + v*GH_X
    t = np.tanh(a*y/2)
    return (GH_W*t).sum(), (GH_W*t*t).sum()
def stat_mean_var(H, rho, a):
    """exact mean and variance of S = mean over checks of prod tanh(a y_i/2)"""
    m, q = moments(rho, a)
    d = H.sum(1).astype(int); O = (H.astype(np.int32) @ H.T.astype(np.int32))
    Mn = len(d); D = d[:, None]+d[None, :]
    E2 = q**O * m**(D-2*O)   # includes diagonal (O=d)
    mean = (m**d).mean()
    var = (E2 - m**D).sum()/Mn**2
    return mean, var
def stat_slope(H, rho, a, eps=1e-5):
    f = lambda r: (moments(r, a)[0]**H.sum(1)).mean()
    return (f(rho+eps)-f(rho-eps))/(2*eps)
def m4_var(rho, n):
    s, v = np.sqrt(rho), np.sqrt(1-rho); y = s+v*GH_X
    e4 = (GH_W*y**4).sum(); e8 = (GH_W*y**8).sum()
    return (e8-e4**2)/n/(4*rho)**2
# ---------- estimators ----------
RHO_GRID = np.linspace(0.02, 0.98, 1921)
def make_table(H, a):
    return np.array([(moments(r, a)[0]**H.sum(1)).mean() for r in RHO_GRID])
def invert(Sbar, table):
    """monotone table -> rho"""
    tb = np.maximum.accumulate(table)
    return np.interp(Sbar, tb, RHO_GRID)
def syndrome_stat(y, Hidx, a):
    # y: (B,n) torch; Hidx: (M,dc) long
    t = torch.sign(y) if not np.isfinite(a) else torch.tanh(a*y/2)
    return t[:, Hidx].prod(-1).mean(-1)
def m2m4(y):
    m4 = (y**4).mean(1)
    return torch.sqrt(torch.clamp((3-m4)/2, 0.02**2, 0.98**2))
# ---------- decoders ----------
class Dec:
    def __init__(self, Hidx, n, dev):
        self.Hidx = Hidx.to(dev); self.n = n; self.M, self.dc = Hidx.shape; self.dev = dev
        self.flat = self.Hidx.reshape(-1)
    def run(self, L, iters=50, kind='bp', alpha=0.8):
        B = L.shape[0]; Hf = self.flat
        R = torch.zeros(B, self.M*self.dc, device=L.device)
        for _ in range(iters):
            tot = L.clone(); tot.index_add_(1, Hf, R)
            Q = (tot[:, Hf]-R).reshape(B, self.M, self.dc)
            if kind == 'bp':
                t = torch.tanh(torch.clamp(Q, -30, 30)/2)
                sg = torch.where(t < 0, -1.0, 1.0); lg = torch.log(torch.clamp(t.abs(), min=1e-12))
                ps = sg.prod(-1, keepdim=True)*sg; pl = torch.exp(lg.sum(-1, keepdim=True)-lg)
                Rn = 2*torch.atanh(torch.clamp(ps*pl, -1+1e-7, 1-1e-7))
            else:
                a = Q.abs(); sg = torch.where(Q < 0, -1.0, 1.0)
                v, i = torch.topk(a, 2, dim=-1, largest=False)
                mn = v[..., :1].expand_as(a).clone()
                mask = torch.zeros_like(a, dtype=torch.bool).scatter_(-1, i[..., :1], True)
                mn = torch.where(mask, v[..., 1:2].expand_as(a), mn)
                Rn = alpha*sg.prod(-1, keepdim=True)*sg*mn
            R = Rn.reshape(B, -1)
        tot = L.clone(); tot.index_add_(1, Hf, R)
        return tot
