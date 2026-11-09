"""Shared library for the Letter-4 reviewer experiments (r4_*).

Everything here is exact / measured.  No results are hard-coded.
"""
import numpy as np, scipy.sparse as sp, json, time, os

# ---------------------------------------------------------------- moments
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(240); GH_W = GH_W / GH_W.sum()

def t_moments(rho, a, cmax=4):
    """m_c = E[ tanh(a Y/2)^c ],  Y = sqrt(rho) + sqrt(1-rho) Z,  Z~N(0,1).
    Returns array [m_0=1, m_1, ..., m_cmax]."""
    rho = float(np.clip(rho, 1e-12, 1 - 1e-12))
    s, v = np.sqrt(rho), np.sqrt(1 - rho)
    if not np.isfinite(a):
        from scipy.special import erf
        m1 = float(erf(s / (v * np.sqrt(2))))
        return np.array([1.0] + [1.0 if c % 2 == 0 else m1 for c in range(1, cmax + 1)])
    y = s + v * GH_X
    t = np.tanh(a * y / 2)
    return np.array([1.0] + [float((GH_W * t ** c).sum()) for c in range(1, cmax + 1)])

def dm1_drho(rho, a, eps=1e-6):
    r = float(np.clip(rho, 2 * eps, 1 - 2 * eps))
    return (t_moments(r + eps, a, 1)[1] - t_moments(r - eps, a, 1)[1]) / (2 * eps)

# ---------------------------------------------------------------- codes
def to_csr(H):
    return sp.csr_matrix(np.asarray(H, dtype=np.int8))

def packed_gf2_rank(H):
    """GF(2) rank with uint64 bit packing."""
    H = np.asarray(H, dtype=np.uint8)
    M, n = H.shape
    W = (n + 63) // 64
    pad = np.zeros((M, W * 64), dtype=np.uint8); pad[:, :n] = H
    A = np.packbits(pad, axis=1).view(np.uint64).byteswap()   # (M, W)
    A = np.ascontiguousarray(A)
    r = 0
    for col in range(n):
        w, b = col // 64, np.uint64(63 - col % 64)
        bit = np.uint64(1) << b
        cand = np.nonzero(A[r:, w] & bit)[0]
        if len(cand) == 0:
            continue
        p = cand[0] + r
        if p != r:
            A[[r, p]] = A[[p, r]]
        rows = np.nonzero(A[:, w] & bit)[0]
        rows = rows[rows != r]
        if len(rows):
            A[rows] ^= A[r]
        r += 1
        if r == M:
            break
    return r

def peg(n, dv_list, dcmax, seed=0, verbose=False):
    """Progressive edge growth.  dv_list: per-variable degree.  Returns H (M,n)."""
    rng = np.random.default_rng(seed)
    E = int(np.sum(dv_list))
    M = int(np.ceil(E / dcmax))
    rows = np.empty(E, dtype=np.int64); cols = np.empty(E, dtype=np.int64)
    ne = 0
    cdeg = np.zeros(M, dtype=np.int64)
    order = np.argsort(-np.asarray(dv_list), kind='stable')  # high degree first
    for v in order:
        for e in range(int(dv_list[v])):
            avail = cdeg < dcmax
            if e == 0:
                cand = np.nonzero(avail)[0]
            else:
                r_ = rows[:ne]; c_ = cols[:ne]
                rc = np.zeros(M, dtype=bool)
                rc[r_[c_ == v]] = True
                prev = np.zeros(M, dtype=bool)
                last = rc.copy()
                while True:
                    rv = np.zeros(n, dtype=bool); rv[c_[rc[r_]]] = True
                    nc = np.zeros(M, dtype=bool); nc[r_[rv[c_]]] = True
                    nc |= rc
                    if nc.sum() == rc.sum():
                        break
                    prev = rc; last = nc & ~rc; rc = nc
                    if (~rc & avail).sum() == 0:
                        break
                unreached = (~rc) & avail
                cand = np.nonzero(unreached)[0]
                if len(cand) == 0:
                    cand = np.nonzero(last & avail)[0]
                    if len(cand) == 0:
                        cand = np.nonzero(avail)[0]
            if len(cand) == 0:
                cand = np.arange(M)
            d = cdeg[cand]
            cand = cand[d == d.min()]
            c = int(cand[rng.integers(len(cand))])
            rows[ne] = c; cols[ne] = v; ne += 1; cdeg[c] += 1
    H = np.zeros((M, n), dtype=np.int8)
    H[rows[:ne], cols[:ne]] = 1
    return H

def rand_H(n, dv, dc, seed=0):
    """Plain random socket permutation: no 4-cycle removal (girth 4 likely)."""
    rng = np.random.default_rng(seed)
    M = n * dv // dc
    sockets = np.repeat(np.arange(n), dv)
    perm = rng.permutation(len(sockets))
    H = np.zeros((M, n), dtype=np.int8)
    for k, s in enumerate(perm):
        H[k // dc, sockets[s]] = 1
    return H

def lift_bg(bg_rows, nrows, ncols, Z, iLS):
    """5G NR base graph lifting: entry V -> Z x Z circulant with shift V mod Z."""
    M = nrows * Z; N = ncols * Z
    H = np.zeros((M, N), dtype=np.int8)
    idx = np.arange(Z)
    for r, c, *sh in bg_rows:
        if r >= nrows or c >= ncols:
            continue
        s = sh[iLS] % Z
        H[r * Z + idx, c * Z + ((idx + s) % Z)] = 1
    return H

def code_props(H):
    Hc = to_csr(H)
    O = (Hc @ Hc.T).tocoo()
    off = O.row != O.col
    maxo = int(O.data[off].max()) if off.any() else 0
    n4 = int((O.data[off] >= 2).sum()) // 2
    girth = 4 if maxo >= 2 else None
    n6 = 0
    if girth is None:
        ei, ej = O.row[off & (O.row < O.col)], O.col[off & (O.row < O.col)]
        A = sp.csr_matrix((np.ones(len(ei)), (ei, ej)), shape=(H.shape[0],) * 2)
        A = A + A.T
        n6 = int((A @ A @ A).diagonal().sum()) // 6
        girth = 6 if n6 > 0 else 8
    d = np.asarray(Hc.sum(1)).ravel()
    dv = np.asarray(Hc.sum(0)).ravel()
    return dict(n=H.shape[1], M=H.shape[0],
                dc_min=int(d.min()), dc_max=int(d.max()), dc_mean=float(d.mean()),
                dv_min=int(dv.min()), dv_max=int(dv.max()), dv_mean=float(dv.mean()),
                girth=girth, n_4cycles=n4, n_6cycles=n6,
                n_overlap_pairs=int(off.sum()))

# ------------------------------------------------- exact mean / variance
def stat_mean(Hc, rho, a, mm=None):
    mm = t_moments(rho, a, 1) if mm is None else mm
    d = np.asarray(Hc.sum(1)).ravel().astype(np.int64)
    return float((mm[1] ** d).mean())

def stat_mean_var_sparse(Hc, rho, a, mm=None, OO=None):
    """Exact mean and variance of S_a, sparse: only overlapping check pairs."""
    mm = t_moments(rho, a, 2) if mm is None else mm
    m1, m2 = mm[1], mm[2]
    d = np.asarray(Hc.sum(1)).ravel().astype(np.int64)
    M = Hc.shape[0]
    O = (Hc @ Hc.T).tocoo() if OO is None else OO
    i, j, o = O.row, O.col, O.data.astype(np.int64)
    D = d[i] + d[j]
    var = float((m2 ** o * m1 ** (D - 2 * o) - m1 ** D).sum()) / M ** 2
    return float((m1 ** d).mean()), var

def stat_var_dense(H, rho, a):
    """Naive dense M x M pair sum (the original implementation)."""
    mm = t_moments(rho, a, 2); m1, m2 = mm[1], mm[2]
    Hi = np.asarray(H, dtype=np.int32)
    d = Hi.sum(1); O = Hi @ Hi.T
    M = len(d); D = d[:, None] + d[None, :]
    return float(((m2 ** O) * (m1 ** (D - 2 * O)) - m1 ** D).sum()) / M ** 2

def stat_var_indep(Hc, rho, a, mm=None):
    """Variance if the checks were independent (diagonal only)."""
    mm = t_moments(rho, a, 2) if mm is None else mm
    d = np.asarray(Hc.sum(1)).ravel().astype(np.int64)
    M = Hc.shape[0]
    return float((mm[2] ** d - mm[1] ** (2 * d)).sum()) / M ** 2

# ------------------------------------------------- exact 4th central moment
POP = np.array([bin(x).count('1') for x in range(16)], dtype=np.int64)

def dep_edges(Hc, OO=None):
    O = (Hc @ Hc.T).tocoo() if OO is None else OO
    m = O.row < O.col
    return O.row[m].astype(np.int64), O.col[m].astype(np.int64)

def _adj_csr(M, ei, ej):
    data = np.ones(2 * len(ei))
    A = sp.csr_matrix((data, (np.concatenate([ei, ej]), np.concatenate([ej, ei]))), shape=(M, M))
    A.sum_duplicates()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64)

def conn_triples(M, indptr, indices, cap=30_000_000):
    deg = (indptr[1:] - indptr[:-1]).astype(np.int64)
    if int((deg * (deg - 1) // 2).sum()) > cap:
        return None
    out = []
    for v in range(M):
        nb = indices[indptr[v]:indptr[v + 1]]
        k = len(nb)
        if k < 2:
            continue
        a, b = np.triu_indices(k, 1)
        T = np.stack([np.full(len(a), v), nb[a], nb[b]], 1)
        out.append(np.sort(T, axis=1))
    if not out:
        return np.zeros((0, 3), dtype=np.int64)
    T = np.concatenate(out)
    return np.unique(T, axis=0)

def conn_quads(M, indptr, indices, T, cap=40_000_000):
    """All connected 4-subsets, generated by extending connected triples."""
    outs = []
    tot = 0
    for s in range(3):
        base = T[:, s]
        deg = (indptr[base + 1] - indptr[base])
        tot += int(deg.sum())
        if tot > cap:
            return None
    for s in range(3):
        base = T[:, s]
        deg = (indptr[base + 1] - indptr[base])
        rep = np.repeat(np.arange(len(T)), deg)
        off = np.arange(deg.sum()) - np.repeat(np.cumsum(deg) - deg, deg)
        nb = indices[np.repeat(indptr[base], deg) + off]
        cand = np.concatenate([T[rep], nb[:, None]], 1)
        keep = (nb != cand[:, 0]) & (nb != cand[:, 1]) & (nb != cand[:, 2])
        outs.append(np.sort(cand[keep], axis=1))
    Q = np.concatenate(outs)
    key = (Q[:, 0] * M + Q[:, 1]) * M * M + Q[:, 2] * M + Q[:, 3]
    _, ix = np.unique(key, return_index=True)
    return Q[ix]

def _tuple_masks(slots, rowlists, K):
    """slots: (B,4) check indices.  rowlists: (M,K4) padded variable index array
    with pad = -1.  Returns N (B,15) mask-count profile."""
    B = slots.shape[0]
    idx = rowlists[slots].reshape(B, -1).astype(np.int64)        # (B, 4*K)
    bit = np.tile(np.repeat(np.array([1, 2, 4, 8]), K), (B, 1)).astype(np.float64)
    pad = idx < 0
    bit[pad] = 0.0
    # make pads unique inside each row so they never merge with anything
    KK = idx.shape[1]
    uniq = -1 - np.arange(KK)[None, :].repeat(B, 0)
    idx = np.where(pad, uniq, idx)
    o = np.argsort(idx, axis=1, kind='stable')
    si = np.take_along_axis(idx, o, 1); sb = np.take_along_axis(bit, o, 1)
    new = np.empty(si.shape, dtype=bool); new[:, 0] = True; new[:, 1:] = si[:, 1:] != si[:, :-1]
    gid = np.cumsum(new, axis=1) - 1
    key = (np.arange(B)[:, None] * KK + gid).ravel()
    msum = np.bincount(key, weights=sb.ravel(), minlength=B * KK).reshape(B, KK)
    msum = np.rint(msum).astype(np.int64)
    return np.stack([(msum == m).sum(1) for m in range(1, 16)], 1)

MASKS = np.arange(1, 16)

def _kappa(N, mm):
    """joint 4th cumulant for each tuple, from its mask profile N (B,15)."""
    mc = np.array([1.0, mm[1], mm[2], mm[3], mm[4]])
    lg = np.log(np.maximum(mc, 1e-300))
    B = N.shape[0]
    E = np.empty((B, 16))
    for A in range(16):
        pc = POP[MASKS & A]
        acc = np.zeros(B)
        for c in range(1, 5):
            sel = pc == c
            if sel.any():
                acc += N[:, sel].sum(1) * lg[c]
        E[:, A] = np.exp(acc)
    d = np.stack([N[:, ((MASKS >> s) & 1) == 1].sum(1) for s in range(4)], 1)
    mu = mm[1] ** d
    E4 = np.zeros(B)
    for A in range(16):
        w = np.ones(B)
        for s in range(4):
            if not (A >> s) & 1:
                w = w * mu[:, s]
        E4 += ((-1) ** (4 - POP[A])) * E[:, A] * w
    def C(a, b):
        return E[:, (1 << a) | (1 << b)] - mu[:, a] * mu[:, b]
    return E4 - (C(0, 1) * C(2, 3) + C(0, 2) * C(1, 3) + C(0, 3) * C(1, 2))

def mu4_exact(Hc, rho, a, pre=None, chunk=150_000, return_pre=False):
    """Exact 4th central moment of S_a.
       mu4 = 3 Var^2 + M^-4 * sum over connected ordered 4-tuples of kappa4."""
    M = Hc.shape[0]
    if pre is None:
        OO = (Hc @ Hc.T).tocoo()
        ei, ej = dep_edges(Hc, OO)
        indptr, indices = _adj_csr(M, ei, ej)
        T = conn_triples(M, indptr, indices)
        Q = None if T is None else conn_quads(M, indptr, indices, T)
        rl = [np.nonzero(Hc[i].toarray().ravel())[0] for i in range(M)]
        K = max(len(r) for r in rl)
        rowlists = -np.ones((M, K), dtype=np.int64)
        for i, r in enumerate(rl):
            rowlists[i, :len(r)] = r
        pre = dict(OO=OO, ei=ei, ej=ej, T=T, Q=Q, rowlists=rowlists, K=K)
    ei, ej, T, Q, rowlists, K = pre['ei'], pre['ej'], pre['T'], pre['Q'], pre['rowlists'], pre['K']
    mm = t_moments(rho, a, 4)
    mean, var = stat_mean_var_sparse(Hc, rho, a, mm, pre['OO'])
    batches = []
    I = np.arange(M)
    batches.append((np.stack([I, I, I, I], 1), 1.0))
    if len(ei):
        batches.append((np.stack([ei, ei, ei, ej], 1), 4.0))
        batches.append((np.stack([ei, ej, ej, ej], 1), 4.0))
        batches.append((np.stack([ei, ei, ej, ej], 1), 6.0))
    if T is not None and len(T):
        for s in range(3):
            batches.append((np.stack([T[:, 0], T[:, 1], T[:, 2], T[:, s]], 1), 12.0))
    if Q is not None and len(Q):
        batches.append((Q, 24.0))
    tot = 0.0
    for slots, w in batches:
        for s0 in range(0, len(slots), chunk):
            sl = slots[s0:s0 + chunk]
            tot += w * float(_kappa(_tuple_masks(sl, rowlists, K), mm).sum())
    exact = (T is not None) and (Q is not None)
    mu4 = 3 * var ** 2 + tot / M ** 4
    r = dict(mean=mean, var=var,
             mu4=(mu4 if exact else None), kurt=(mu4 / var ** 2 if exact else None),
             mu4_partial=mu4, kurt_partial=mu4 / var ** 2, exact=exact,
             n_triples=(None if T is None else len(T)),
             n_quads=(0 if Q is None else len(Q)), n_edges=len(ei),
             triples_capped=(T is None), quads_capped=(Q is None))
    return (r, pre) if return_pre else r

# ---------------------------------------------------------------- estimators
def make_table(Hc, a, grid):
    d = np.asarray(Hc.sum(1)).ravel().astype(np.int64)
    return np.array([float((t_moments(r, a, 1)[1] ** d).mean()) for r in grid])

def invert_table(S, table, grid):
    tb = np.maximum.accumulate(table)
    return np.interp(S, tb, grid)

def rho_of_esn0_db(db):
    s = 10 ** (db / 10.0)
    return s / (1 + s)

def esn0_db_of_rho(rho):
    rho = np.clip(rho, 1e-9, 1 - 1e-9)
    return 10 * np.log10(rho / (1 - rho))

def stat_mean_var_pervar(Hc, m1, m2):
    """Exact mean/variance when the per-variable moments differ
    (asymmetric or position-dependent channels).  Assumes the variables are
    still INDEPENDENT.  m1,m2: length-n arrays."""
    lm1 = np.log(np.clip(np.asarray(m1, float), 1e-300, None))
    lm2 = np.log(np.clip(np.asarray(m2, float), 1e-300, None))
    M = Hc.shape[0]
    s1 = np.asarray(Hc @ lm1).ravel()
    I1 = (Hc.multiply(lm1[None, :]) @ Hc.T).tocoo()
    I2 = (Hc.multiply(lm2[None, :]) @ Hc.T).tocoo()
    assert I1.nnz == I2.nnz
    i, j = I1.row, I1.col
    tot = s1[i] + s1[j]
    cov = np.exp(tot - 2 * I1.data + I2.data) - np.exp(tot)
    return float(np.exp(s1).mean()), float(cov.sum()) / M ** 2


def gf2_nullspace(H):
    """rows of the returned matrix span the code (null space of H over GF(2))."""
    A = (np.asarray(H, dtype=np.uint8) % 2).copy()
    M, n = A.shape
    piv = []; r = 0
    for col in range(n):
        rows = np.nonzero(A[r:, col])[0]
        if len(rows) == 0:
            continue
        p = rows[0] + r
        if p != r:
            A[[r, p]] = A[[p, r]]
        o = np.nonzero(A[:, col])[0]; o = o[o != r]
        if len(o):
            A[o] ^= A[r]
        piv.append(col); r += 1
        if r == M:
            break
    pv = set(piv); free = [c for c in range(n) if c not in pv]
    B = np.zeros((len(free), n), dtype=np.uint8)
    for k, f in enumerate(free):
        B[k, f] = 1
        for i, cc in enumerate(piv):
            B[k, cc] = A[i, f]
    return B
