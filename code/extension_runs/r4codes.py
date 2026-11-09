"""Build / cache the code family used by r4_B, r4_C, r4_D."""
import numpy as np, json, os, sys, time
import r4lib as L

BG = None
def _bg():
    global BG
    if BG is None:
        BG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bg5g.json')))
    return BG

# 5G lifting set index: Z = a * 2^j, a in {2,3,5,7,9,11,13,15} -> iLS 0..7
A_SET = {2: 0, 3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7}
def ils_of_Z(Z):
    z = Z
    while z % 2 == 0:
        z //= 2
    return A_SET[z]

def nr_code(bg, mb, Z):
    """5G NR LDPC: base graph `bg` (bg1 kb=22, bg2 kb=10) truncated to the first
    mb rows and kb+mb columns, lifted by Z (shift = V mod Z, iLS from Z).
    All kb+mb lifted columns are transmitted (no rate matching / no puncturing
    of the first two columns) -> design rate kb/(kb+mb)."""
    rows = _bg()[bg]
    kb = 22 if bg == 'bg1' else 10
    return L.lift_bg(rows, mb, kb + mb, Z, ils_of_Z(Z)), kb

# name -> builder
def build_all(which=None, cache='r4codes.npz', verbose=True):
    specs = {}
    specs['reg_n256_R12']   = lambda: L.peg(256,  [3] * 256, 6, seed=1)
    specs['reg_n1024_R12']  = lambda: L.peg(1024, [3] * 1024, 6, seed=1)
    specs['reg_n4096_R12']  = lambda: L.peg(4096, [3] * 4096, 6, seed=1)
    specs['reg_n1024_R13']  = lambda: L.peg(1024, [4] * 1024, 6, seed=1)     # rate 1/3
    specs['reg_n1024_R56']  = lambda: L.peg(1024, [3] * 1024, 18, seed=1)    # rate 5/6
    specs['reg_n4096_R56']  = lambda: L.peg(4096, [3] * 4096, 18, seed=1)
    specs['irr_n1024_R12']  = lambda: L.peg(1024, _irr_dv(1024), 6, seed=1)
    specs['g4_n1024_R12']   = lambda: L.rand_H(1024, 3, 6, seed=2)           # girth 4
    specs['nr_bg2_R13_n1560'] = lambda: nr_code('bg2', 20, 52)[0]
    specs['nr_bg2_R12_n1040'] = lambda: nr_code('bg2', 10, 52)[0]
    specs['nr_bg1_R12_n4224'] = lambda: nr_code('bg1', 22, 96)[0]
    specs['nr_bg1_R081_n2592'] = lambda: nr_code('bg1', 5, 96)[0]
    if which:
        specs = {k: v for k, v in specs.items() if k in which}
    out = {}
    if cache and os.path.exists(cache):
        z = np.load(cache)
        out = {k: z[k] for k in z.files}
    for k, f in specs.items():
        if k in out:
            continue
        t0 = time.time(); out[k] = f().astype(np.int8)
        if verbose:
            print('built', k, out[k].shape, '%.1fs' % (time.time() - t0), flush=True)
    if cache:
        np.savez_compressed(cache, **out)
    return {k: out[k] for k in specs}

def _irr_dv(n):
    """Variable-degree profile chosen by us (NOT from a published table):
    50% degree 2, 30% degree 3, 15% degree 4, 5% degree 10 -> mean 3.0."""
    f = [(2, 0.50), (3, 0.30), (4, 0.15), (10, 0.05)]
    d = []
    for deg, fr in f:
        d += [deg] * int(round(fr * n))
    while len(d) < n:
        d.append(3)
    d = d[:n]
    rng = np.random.default_rng(3); rng.shuffle(d)
    return d
