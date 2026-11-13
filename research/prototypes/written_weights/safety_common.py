"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Shared pieces for the round-4 safety experiments (W8): decisions, picks under a band,
step-weight policies as plain arrays, and a fit with an adjustable ratio limit.
"""
import numpy as np
from wnet import *
from weights import *
from world import *
from exp_refine_vs_train import thermo, isotonic, spearman

NOISE = 0.5          # sigma of the lognormal noise on an observed outcome


class Pool:
    """A fixed pool of decisions (three candidate peers each) with everything precomputed."""
    def __init__(self, n, world="product", seed=4242):
        self.world = world
        self.C = candidates(np.random.default_rng(seed), n)
        self.X = self.C.reshape(-1, 5)
        self.elig = self.C[..., 2] >= 2
        self.tc = true_cost(self.C, world)                                  # n x 3 true cost
        self.sb = np.asarray(baseline_rule(self.X), float).reshape(n, 3)     # anchor = baseline rule
        self.sb = np.where(self.elig, self.sb, -1.0)
        self.a = self.sb.argmax(1)                                           # anchor pick
        self.ca = self.tc[np.arange(n), self.a]
        self.base = float(self.ca.mean())

    def scores(self, score_fn):
        s = np.asarray(score_fn(self.X), float).reshape(len(self.C), 3)
        return np.where(self.elig, s, -1e18)

    def pick(self, s, delta=None):
        """a policy's pick from its scores s; with a band, only candidates the anchor rates
        within delta of its best are open to it; ties go to the anchor's pick"""
        s = s.copy()
        if delta is not None:
            inband = self.elig & (self.sb >= self.sb.max(1, keepdims=True) - delta)
            s = np.where(inband, s, -1e18)
        s[np.arange(len(s)), self.a] += 1e-6 * (1 + np.abs(s[np.arange(len(s)), self.a]))
        return s.argmax(1)

    def cost_of(self, pick):
        return self.tc[np.arange(len(pick)), pick]

    def harm(self, pick):
        """mean true cost relative to the anchor: +0.05 means 5% worse"""
        return float(self.cost_of(pick).mean() / self.base - 1)


def steps_from(wr, wl, wd):
    """step-weight policy from positive cost weights for rtt (15) and loss (7) steps and
    non-negative benefit weights for the five delivery steps above the gate"""
    st = Steps()
    st.w[RTT] = -np.asarray(wr, np.int64)
    st.w[LOSS] = -np.asarray(wl, np.int64)
    st.w[DELIV][2:] = np.asarray(wd, np.int64)
    st.bias = int(np.sum(wr) + np.sum(wl) + 1)
    return st


def fast_scores(pool, wr, wl, wd):
    """scores of a step-weight policy on the pool, without building the object"""
    cr = np.concatenate([[0], np.cumsum(wr)]); cl = np.concatenate([[0], np.cumsum(wl)])
    cd = np.concatenate([[0, 0, 0], np.cumsum(wd)])
    s = -cr[pool.C[..., 0]] - cl[pool.C[..., 1]] + cd[pool.C[..., 2]]
    return np.where(pool.elig, s.astype(float), -1e18)


def fit_steps_ratio(x, obs, seed, R, limit_delivery=True):
    """fit one weight per step to a log by least squares, force directions, round to integers,
    and hold every priced step inside the ratio limit R (largest step <= R x smallest)"""
    ok = x[:, 2] >= 2
    X, y = thermo(x[ok]), obs[ok]
    rng = np.random.default_rng(4242 + seed); perm = rng.permutation(len(y)); k = max(1, len(y) // 5)
    va, tr = perm[:k], perm[k:]
    best = None
    for name, tf in (("log", np.log), ("raw", lambda v: v)):
        co, *_ = np.linalg.lstsq(X[tr], tf(y[tr]), rcond=1e-6)
        rho = spearman(X[va] @ co, y[va])
        if best is None or rho > best[0]:
            best = (rho, tf)
    co, *_ = np.linalg.lstsq(X, best[1](y), rcond=1e-6)
    cnt = lambda f, lv: np.array([np.count_nonzero(x[ok, f] == v) for v in lv], float)
    c_r = isotonic(np.concatenate([[0], np.cumsum(co[1:16])]), cnt(0, range(16)), True)
    c_l = isotonic(np.concatenate([[0], np.cumsum(co[16:23])]), cnt(1, range(8)), True)
    c_d = isotonic(np.concatenate([[0], np.cumsum(co[23:28])]), cnt(2, range(2, 8)), False)
    d_r, d_l, d_d = np.diff(c_r), np.diff(c_l), -np.diff(c_d)
    prim = np.concatenate([d_r, d_l])
    floor = max(prim.max() / R, 1e-12)                       # the smallest step the limit allows
    K = 8.0 / floor if R >= 8 else 32.0 / floor               # integer resolution
    wr = np.maximum(1, np.rint(K * np.maximum(d_r, floor))).astype(np.int64)
    wl = np.maximum(1, np.rint(K * np.maximum(d_l, floor))).astype(np.int64)
    lo = int(min(wr.min(), wl.min()))
    wr, wl = np.minimum(wr, R * lo), np.minimum(wl, R * lo)
    wd = np.maximum(0, np.rint(K * d_d)).astype(np.int64)
    if limit_delivery:
        wd = np.minimum(wd, R * lo)
    return wr, wl, wd
