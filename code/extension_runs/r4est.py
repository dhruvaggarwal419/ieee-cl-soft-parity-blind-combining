"""Code context, decoder and the competing SNR estimators (torch, GPU)."""
import numpy as np, torch, time
import r4lib as L
from scipy.special import erfc

def Qf(x):
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))

# one common inversion support for EVERY table-based estimator, so that the
# comparison in (D) is not decided by different clipping ranges.
GRID = np.linspace(0.02, 0.98, 2001)

class Ctx:
    """padded check-index representation, works for irregular check degrees."""
    def __init__(self, H, dev='cuda', rank=True):
        H = np.asarray(H, dtype=np.int8)
        self.H = H; self.M, self.n = H.shape; self.dev = dev
        self.Hc = L.to_csr(H)
        rl = [np.nonzero(r)[0] for r in H]
        self.K = max(len(r) for r in rl)
        pad = np.zeros((self.M, self.K), dtype=np.int64)
        msk = np.zeros((self.M, self.K), dtype=bool)
        for i, r in enumerate(rl):
            pad[i, :len(r)] = r; msk[i, :len(r)] = True
        self.Hidx = torch.tensor(pad, device=dev)
        self.Hmsk = torch.tensor(msk, device=dev)
        self.deg = np.array([len(r) for r in rl])
        self.rank = L.packed_gf2_rank(H) if rank else None
        self.R = (self.n - self.rank) / self.n if rank else None
        self.flat = self.Hidx.reshape(-1)
        self.mkf = self.Hmsk.reshape(-1).float()

    def prod_over_checks(self, t):
        """t:(B,n) -> (B,M) product of t over each check."""
        g = t[:, self.Hidx]
        g = torch.where(self.Hmsk, g, torch.ones_like(g))
        return g.prod(-1)

    def tvec(self, y, a):
        if not np.isfinite(a):
            s = torch.sign(y); return torch.where(s == 0, torch.ones_like(s), s)
        return torch.tanh(a * y / 2)

    def soft_stat(self, y, a):
        return self.prod_over_checks(self.tvec(y, a)).mean(-1)

    def prods(self, y, a):
        return self.prod_over_checks(self.tvec(y, a))

    def syndrome_weight(self, y):
        p = self.prod_over_checks(torch.sign(y) + (y == 0).float())
        return (p < 0).float().sum(-1)

class Dec:
    """normalised min-sum, irregular-degree safe."""
    def __init__(self, ctx):
        self.c = ctx
    def run(self, Lch, iters=50, alpha=0.85):
        c = self.c; B = Lch.shape[0]
        R = torch.zeros(B, c.M * c.K, device=Lch.device)
        BIG = torch.tensor(1e9, device=Lch.device)
        for _ in range(iters):
            tot = Lch.clone(); tot.index_add_(1, c.flat, R * c.mkf)
            Q = (tot[:, c.flat] - R).reshape(B, c.M, c.K)
            Q = torch.where(c.Hmsk, Q, BIG.expand_as(Q))
            av = Q.abs(); sg = torch.where(Q < 0, -1.0, 1.0)
            v, i = torch.topk(av, 2, dim=-1, largest=False)
            mn = v[..., :1].expand_as(av).clone()
            sel = torch.zeros_like(av, dtype=torch.bool).scatter_(-1, i[..., :1], True)
            mn = torch.where(sel, v[..., 1:2].expand_as(av), mn)
            Rn = alpha * sg.prod(-1, keepdim=True) * sg * mn
            R = torch.where(c.Hmsk, Rn, torch.zeros_like(Rn)).reshape(B, -1)
        tot = Lch.clone(); tot.index_add_(1, c.flat, R * c.mkf)
        return tot

# ------------------------------------------------------------------ estimators
RGRID = None
def rho_grid(lo=0.02, hi=0.98, N=2001):
    return np.linspace(lo, hi, N)

class SoftEst:
    """proposed statistic S_a, inverted through the exact mean mu_a(rho)."""
    def __init__(self, ctx, a, grid=None):
        self.c = ctx; self.a = a
        self.grid = rho_grid() if grid is None else grid
        self.tab = L.make_table(ctx.Hc, a, self.grid)
    def __call__(self, y):
        S = self.c.soft_stat(y, self.a).double().cpu().numpy()
        return L.invert_table(S, self.tab, self.grid), S

class HardLP:
    """Lechner-Pacher (IEEE CL 2013): ML estimate of the channel parameter from
    the HARD-decision syndrome, under the independent-syndrome-bit model
        P(s_j=1) = (1 - (1-2p)^{d_j})/2,   p = Q(sqrt(rho/(1-rho))).
    The log-likelihood only depends on the syndrome weight inside each
    check-degree class, so we evaluate it on a rho grid and take the argmax."""
    def __init__(self, ctx, grid=None):
        self.c = ctx
        self.grid = rho_grid() if grid is None else grid
        self.degs = np.unique(ctx.deg)
        self.Md = np.array([(ctx.deg == d).sum() for d in self.degs], dtype=np.float64)
        p = Qf(np.sqrt(self.grid / (1 - self.grid)))
        P = (1 - (1 - 2 * p)[:, None] ** self.degs[None, :]) / 2     # (G, nd)
        P = np.clip(P, 1e-300, 1 - 1e-300)
        # log1p keeps the high-SNR regime (P -> 0) resolvable instead of
        # saturating to a tie, which would silently bias the argmax downwards.
        self.logP = np.log(P); self.log1P = np.log1p(-P)
        self.cls = torch.tensor(np.stack([(ctx.deg == d).astype(np.float64) for d in self.degs]),
                                device=ctx.dev)      # (nd, M)
        self.lp_t = torch.tensor(self.logP, device=ctx.dev)
        self.l1_t = torch.tensor(self.log1P, device=ctx.dev)
        self.Md_t = torch.tensor(self.Md, device=ctx.dev)
        self.grid_t = torch.tensor(self.grid, device=ctx.dev)
    def __call__(self, y):
        c = self.c
        pr = c.prod_over_checks(torch.sign(y) + (y == 0).float())
        s = (pr < 0).double()                                  # (B,M)
        w = s @ self.cls.T                                     # (B,nd)
        ll = w @ self.lp_t.T + (self.Md_t - w) @ self.l1_t.T   # (B,G)
        i = ll.argmax(-1)
        return self.grid_t[i].cpu().numpy(), (w.sum(-1) / c.M).cpu().numpy()

class FangEst:
    """Fang (IEEE CL 2009): mean intrinsic LLR of the syndrome,
        F = (1/M) sum_j 2 atanh( prod_{i in j} tanh(L_i/2) ),  L_i = a y_i.
    No closed form for E[F], so the inversion curve is built by Monte Carlo."""
    def __init__(self, ctx, a, grid=None, nmc=40000, bs=20000, seed=99):
        self.c = ctx; self.a = a
        self.cgrid = (np.linspace(0.02, 0.98, 61) if grid is None else np.asarray(grid))
        self.grid = GRID
        g = torch.Generator(device=ctx.dev); g.manual_seed(seed)
        tab = []
        for r in self.cgrid:
            acc = 0.0; nb = 0
            for _ in range(max(1, nmc // bs)):
                y = np.sqrt(r) + np.sqrt(1 - r) * torch.randn(bs, ctx.n, device=ctx.dev, generator=g)
                acc += float(self.stat(y).double().mean()); nb += 1
            tab.append(acc / nb)
        self.tab = np.interp(self.grid, self.cgrid, np.array(tab))
    def stat(self, y):
        p = self.c.prods(y, self.a).clamp(-1 + 1e-7, 1 - 1e-7)
        return (2 * torch.atanh(p)).mean(-1)
    def __call__(self, y):
        F = self.stat(y).double().cpu().numpy()
        return L.invert_table(F, self.tab, self.grid), F

def m2m4(y, lo=0.02, hi=0.98):
    """Scale-free M2M4 for REAL BPSK in REAL AWGN.
    m2 = S + N, m4 = S^2 + 6SN + 3N^2  =>  m4 = 3 m2^2 - 2 S^2,
    so S = sqrt((3 m2^2 - m4)/2) and rho = S/m2."""
    m2 = (y ** 2).mean(1); m4 = (y ** 4).mean(1)
    S = torch.sqrt(torch.clamp((3 * m2 ** 2 - m4) / 2, min=0.0))
    rho = torch.clamp(S / torch.clamp(m2, min=1e-12), lo, hi)
    return rho.double().cpu().numpy()

def dd_est(y, dec, a0, iters=50):
    """decision-directed: decode, then match the first two moments to the decision."""
    xh = torch.sign(dec.run(a0[:, None] * y if a0.dim() else a0 * y, iters))
    xh = torch.where(xh == 0, torch.ones_like(xh), xh)
    A = (y * xh).mean(1)
    m2 = (y ** 2).mean(1)
    rho = torch.clamp(A ** 2 / torch.clamp(m2, min=1e-12), 0.02, 0.98)
    return rho.double().cpu().numpy()

# ------------------------------------------------------------------ CRBs
def crb_da(rho, n):
    """known codeword: y_i ~ N(sqrt(rho) x_i, 1-rho)."""
    I1 = (1 / (4 * rho)) / (1 - rho) + 1 / (2 * (1 - rho) ** 2)
    return 1.0 / (n * I1)

def crb_nda(rho, n, lim=14.0, N=40001):
    """unknown equiprobable BPSK symbols, code ignored."""
    s, v = np.sqrt(rho), np.sqrt(1 - rho)
    y = np.linspace(-(s + lim * v), s + lim * v, N)
    def pdf(r):
        s_, v_ = np.sqrt(r), np.sqrt(1 - r)
        return 0.5 * (np.exp(-(y - s_) ** 2 / (2 * v_ ** 2)) + np.exp(-(y + s_) ** 2 / (2 * v_ ** 2))) / (v_ * np.sqrt(2 * np.pi))
    h = 1e-5 * min(rho, 1 - rho)
    p = pdf(rho); dp = (pdf(rho + h) - pdf(rho - h)) / (2 * h)
    I1 = np.trapz(dp ** 2 / np.maximum(p, 1e-300), y)
    return 1.0 / (n * I1)

def crb_syndrome(ctx, rho):
    """Fisher information carried by the hard syndrome bits alone
    (independent-bit model) -> the bound that applies to Lechner-Pacher."""
    degs, cnt = np.unique(ctx.deg, return_counts=True)
    def P(r):
        p = Qf(np.sqrt(r / (1 - r)))
        return (1 - (1 - 2 * p) ** degs) / 2
    h = 1e-6 * min(rho, 1 - rho)
    p0 = P(rho); dp = (P(rho + h) - P(rho - h)) / (2 * h)
    I = (cnt * dp ** 2 / np.maximum(p0 * (1 - p0), 1e-300)).sum()
    return 1.0 / I

def db_from_rho(r):
    r = np.clip(r, 1e-9, 1 - 1e-9)
    return 10 * np.log10(r / (1 - r))

def db_var_from_rho_var(rho, vr):
    """delta method: d/drho of 10 log10(rho/(1-rho)) = 10/(ln10 rho (1-rho))."""
    g = 10.0 / (np.log(10) * rho * (1 - rho))
    return vr * g ** 2

class TwoPassSoft:
    """practical proposed estimator: pass 1 with a0, then a tuned to the pass-1
    estimate (quantised to a small bank of precomputed inversion tables)."""
    def __init__(self, ctx, a0=1.0, grid=None, bank=None):
        self.c = ctx
        self.grid = rho_grid() if grid is None else grid
        self.bank = np.geomspace(0.3, 40.0, 25) if bank is None else np.asarray(bank)
        self.tabs = np.stack([L.make_table(ctx.Hc, a, self.grid) for a in self.bank])
        self.first = SoftEst(ctx, a0, self.grid)
        self.lb = np.log(self.bank)
    def __call__(self, y):
        r1 = np.clip(self.first(y)[0], 0.01, 0.99)
        a2 = 2 * np.sqrt(r1) / (1 - r1)
        j = np.abs(np.log(a2)[:, None] - self.lb[None, :]).argmin(1)
        aq = torch.tensor(self.bank[j], device=self.c.dev, dtype=torch.float32)
        S = self.c.prod_over_checks(torch.tanh(aq[:, None] * y / 2)).mean(-1).double().cpu().numpy()
        out = np.empty(len(S))
        for b in np.unique(j):
            m = j == b
            out[m] = L.invert_table(S[m], self.tabs[b], self.grid)
        return out
