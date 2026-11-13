"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W7: a richer decision. Which of three paths should carry this message, given its size?

A 1 kB note wants the path with the shortest round trip. A 32 MB file wants the path with the
highest rate. A rule that looks only at round trip and loss (the fixed rule of the earlier
experiments) cannot tell them apart. Neither can a single points table: the size is the same
for all three candidates, so adding points for it changes nothing.

Each path has a round-trip level (16), a loss level (8), a delivery level (8) and a rate level
(8 levels, 8 kB/s to 1,024 kB/s). Messages come in four sizes: 1 kB (60%), 32 kB (25%),
1 MB (12%), 32 MB (3%).

True time on node i:  ( 2 x rtt + T_i x p ) / (1 - p)  +  size / ( rate_i[level] x (1 - p) ),
times the delivery term. T_i is the node's own timeout (200 ms x exp(s z)); rate_i is what
each rate level really delivers on that node (the label x a node factor x a per-level factor,
kept in order). s is how far nodes depart from the textbook values: 0 = not at all.

Compared (slowdown against the best possible choice for that node; 1.00 is perfect):
  the fixed rule of W1 to W6      : round trip and loss only
  the formula, as code            : the textbook formula with textbook constants (plain code)
  one table (rung 1)              : one points table per node, fitted to its own log
  a table per size, written only  : the formula turned into four points tables; no fitting
  a table per size (rung 2)       : those four tables, refined on the node's own log
The log is gathered while following the formula, with 10% of decisions made at random among
the paths the formula puts within 25% of its best estimate.
Then the band, stated in per cent of estimated time: what can any policy cost, and what does
the fitted policy gain, when it may only choose paths the formula puts within x% of its best?
"""
import json, time
import numpy as np
from safety_common import *

SIZES_KB = np.array([1.0, 32.0, 1024.0, 32768.0]); P_SIZE = np.array([.60, .25, .12, .03])
RATE_KBS = np.array([8.0, 16, 32, 64, 128, 256, 512, 1024])
NODES, POOL = 100, 40_000


def est_time(C, size_kb, T=200.0, rate=RATE_KBS):
    rtt_ms = 20.0 * (C[..., 0] + 1); p = LOSS_PROB[C[..., 1]]
    t = (2 * rtt_ms + T * p) / (1 - p) + 1000.0 * size_kb / (rate[C[..., 3]] * (1 - p))
    return t * (1 + 0.10 * (7 - C[..., 2]))


def make_node(rng, s):
    T = 200.0 * float(np.exp(s * np.clip(rng.standard_normal(), -2.5, 2.5)))
    rate = RATE_KBS * np.exp(s * (0.5 * np.clip(rng.standard_normal(), -2.5, 2.5) + 0.5 * np.clip(rng.standard_normal(8), -2.5, 2.5)))
    return dict(T=T, rate=np.sort(rate))


def thermo4(x):
    cols = [np.ones(len(x))]
    for f, lo, n in ((0, 1, 16), (1, 1, 8), (3, 1, 8), (2, 3, 8)):
        cols += [(x[:, f] >= k).astype(float) for k in range(lo, n)]
    return np.stack(cols, 1)                                    # 1 + 15 + 7 + 7 + 5 = 35 columns


def table(co, x=None):
    """coefficients -> four one-feature curves, each forced to move the declared way"""
    cnt = (lambda f, lv: np.array([np.count_nonzero(x[:, f] == v) for v in lv], float) + 1) if x is not None else (lambda f, lv: np.ones(len(list(lv))))
    c_r = isotonic(np.concatenate([[0], np.cumsum(co[1:16])]), cnt(0, range(16)), True)
    c_l = isotonic(np.concatenate([[0], np.cumsum(co[16:23])]), cnt(1, range(8)), True)
    c_q = isotonic(np.concatenate([[0], np.cumsum(co[23:30])]), cnt(3, range(8)), False)
    c_d = isotonic(np.concatenate([[0], np.cumsum(co[30:35])]), cnt(2, range(2, 8)), False)
    return c_r, c_l, c_q, np.concatenate([[0, 0], c_d])


def score(tb, C):
    c_r, c_l, c_q, c_d = tb
    return -(c_r[C[..., 0]] + c_l[C[..., 1]] + c_q[C[..., 3]] + c_d[C[..., 2]])


GRID = np.array([[r, l, d, q, 0] for r in range(16) for l in range(8) for d in range(2, 8) for q in range(8)], np.int64)
XG = thermo4(GRID)
PRIOR = [np.linalg.lstsq(XG, np.log(est_time(GRID, sz)), rcond=None)[0] for sz in SIZES_KB]      # the formula as four tables
LAM = 10.0


def gather(rng, n, node, band=0.25, explore=0.10):
    C = candidates(rng, n); ctx = rng.choice(4, n, p=P_SIZE); sz = SIZES_KB[ctx][:, None]
    elig = C[..., 2] >= 2
    est = np.where(elig, est_time(C, sz), 1e18)
    mask = elig & (est <= (1 + band) * est.min(1, keepdims=True))
    rnd = (rng.random((n, 3)) * mask).argmax(1); use = (rng.random(n) < explore) & (mask.sum(1) > 0)
    pick = np.where(use, rnd, est.argmin(1))
    x = C[np.arange(n), pick]
    obs = est_time(x, SIZES_KB[ctx], node["T"], node["rate"]) * rng.lognormal(0.0, NOISE, n)
    ok = x[:, 2] >= 2
    return x[ok], ctx[ok], obs[ok]


def fit_per_size(x, ctx, obs):
    tbs = []
    for c in range(4):
        m = ctx == c; X = thermo4(x[m]); y = np.log(obs[m])
        co = np.linalg.solve(X.T @ X + LAM * np.eye(35), X.T @ y + LAM * PRIOR[c])
        tbs.append(table(co, x[m]))
    return tbs


def fit_one(x, ctx, obs):
    X = np.concatenate([thermo4(x)[:, 1:], np.eye(4)[ctx]], 1); y = np.log(obs)
    co = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
    return table(np.concatenate([[0], co[:34]]), x)


if __name__ == "__main__":
    t0 = time.time(); out = []
    rngp = np.random.default_rng(71)
    C = candidates(rngp, POOL); ctx = rngp.choice(4, POOL, p=P_SIZE); sz = SIZES_KB[ctx][:, None]
    elig = C[..., 2] >= 2; idx = np.arange(POOL)
    est = np.where(elig, est_time(C, sz), 1e18); a_formula = est.argmin(1)
    sb = np.where(elig, np.asarray(baseline_rule(C.reshape(-1, 5)), float).reshape(POOL, 3), -1.0); a_fixed = sb.argmax(1)
    written = np.stack([np.where(elig, score(table(PRIOR[c]), C), -1e18) for c in range(4)])[ctx, idx]      # POOL x 3
    a_written = written.argmax(1)
    BANDS = (0.05, 0.10, 0.25, 0.50, None)
    inb = {b: elig & (est <= (1e9 if b is None else 1 + b) * est.min(1, keepdims=True)) for b in BANDS}
    for s in (0.0, 0.5, 1.0, 1.5):
        rng = np.random.default_rng(700 + int(10 * s))
        acc = {}
        def add(name, pick, tt, best, by_size=True):
            r = tt[idx, pick] / best
            d = acc.setdefault(name, dict(typical=[], total=[], by_size=[[] for _ in range(4)]))
            d["typical"].append(float(np.exp(np.mean(np.log(r))))); d["total"].append(float(tt[idx, pick].sum() / best.sum()))
            for c in range(4):
                d["by_size"][c].append(float(np.exp(np.mean(np.log(r[ctx == c])))))
        band_acc = {b: dict(worst=[], best=[], fitted=[], contested=[]) for b in BANDS}
        for i in range(NODES):
            node = make_node(rng, s)
            tt = np.where(elig, est_time(C, sz, node["T"], node["rate"]), 1e18)
            tt = np.where(elig, tt, 3.0 * tt[elig].max())
            best = tt.min(1)
            add("the fixed rule of W1 to W6 (round trip and loss only)", a_fixed, tt, best)
            add("the formula, as code", a_formula, tt, best)
            add("a table per size, written from the formula, no fitting", a_written, tt, best)
            for n in (2_000, 20_000):
                x, cx, obs = gather(np.random.default_rng(100_000 * i + n + int(10 * s)), n, node)
                one = np.where(elig, score(fit_one(x, cx, obs), C), -1e18)
                add(f"one table per node (rung 1), {n} logged decisions", one.argmax(1), tt, best)
                tbs = fit_per_size(x, cx, obs)
                per = np.stack([np.where(elig, score(tbs[c], C), -1e18) for c in range(4)])[ctx, idx]
                add(f"a table per size per node (rung 2), {n} logged decisions", per.argmax(1), tt, best)
                if n == 20_000:
                    base = tt[idx, a_formula]
                    for b in BANDS:
                        m = inb[b]
                        w = np.where(m, tt, -1.0).argmax(1); g = np.where(m, tt, 1e18).argmin(1)
                        sc = np.where(m, per, -1e18); sc[idx, a_formula] += 1e-9          # ties stay with the formula
                        f = sc.argmax(1)
                        geo = lambda p: float(np.exp(np.mean(np.log(tt[idx, p] / base))) - 1)
                        band_acc[b]["worst"].append(geo(w)); band_acc[b]["best"].append(geo(g)); band_acc[b]["fitted"].append(geo(f))
                        band_acc[b]["contested"].append(float((f != a_formula).mean()))
        res = dict(departure_from_textbook=s, nodes=NODES, policies={}, bands=[])
        print(f"\nnodes depart from the textbook by s = {s}   ({time.time()-t0:.0f}s)")
        for k, d in acc.items():
            res["policies"][k] = dict(typical_slowdown=float(np.mean(d["typical"])), worst_node=float(np.max(d["typical"])), total_time_ratio=float(np.mean(d["total"])),
                                      by_size={f"{int(SIZES_KB[c])} kB": float(np.mean(d["by_size"][c])) for c in range(4)})
            v = res["policies"][k]
            print(f"   {k:62s} typical x{v['typical_slowdown']:.3f} (worst node x{v['worst_node']:.3f}) | total time x{v['total_time_ratio']:.3f} | by size " + "  ".join(f"x{t:.2f}" for t in v["by_size"].values()), flush=True)
        for b in BANDS:
            d = band_acc[b]
            row = dict(band_percent_of_estimated_time="none" if b is None else 100 * b, worst_possible_any_policy=float(np.mean(d["worst"])), worst_possible_on_the_worst_node=float(np.max(d["worst"])),
                       best_possible_any_policy=float(np.mean(d["best"])), fitted_per_size_tables=float(np.mean(d["fitted"])), fitted_on_the_worst_node=float(np.max(d["fitted"])), contested_share=float(np.mean(d["contested"])))
            res["bands"].append(row)
            print(f"   band {str(row['band_percent_of_estimated_time']):>5}% of the formula's estimate: worst any policy {100*row['worst_possible_any_policy']:+7.1f}% (worst node {100*row['worst_possible_on_the_worst_node']:+7.1f}%) | best any {100*row['best_possible_any_policy']:+6.1f}% | fitted {100*row['fitted_per_size_tables']:+6.1f}% (worst node {100*row['fitted_on_the_worst_node']:+5.1f}%) | differs on {100*row['contested_share']:.0f}%", flush=True)
        out.append(res)
    json.dump(out, open("w7_results.json", "w"), indent=1)
