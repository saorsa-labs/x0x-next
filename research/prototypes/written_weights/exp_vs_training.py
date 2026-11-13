"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W2: the same function, trained instead of written.

A float MLP is trained by Adam to reproduce the baseline rule from N labelled
examples, then checked over the whole domain with the checks the written network
passes. Two training budgets, 6,000 and 60,000 steps, because the independent
reviewer showed that my first budget was too short to be fair to training.
"""
import json, sys, time
import numpy as np
from wnet import *

ref = baseline_rule(GRID).astype(np.float64)
Xn = GRID / (np.array(LEVELS) - 1.0)          # inputs scaled to 0..1
yn = ref / 80.0


def train(shape, idx, seed, steps=6000, lr=3e-3, batch=256):
    rng = np.random.default_rng(seed)
    sizes = list(shape)
    W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(len(sizes) - 1)]
    B = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
    P = W + B
    M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    X, Y = Xn[idx], yn[idx]
    for t in range(1, steps + 1):
        b = rng.integers(0, len(X), min(batch, len(X)))
        a = [X[b]]
        for i in range(len(W)):
            z = a[-1] @ W[i] + B[i]
            a.append(np.maximum(0, z) if i < len(W) - 1 else z)
        g = (a[-1][:, 0] - Y[b])[:, None] * (2.0 / len(b))
        gW, gB = [None] * len(W), [None] * len(W)
        for i in reversed(range(len(W))):
            gW[i] = a[i].T @ g; gB[i] = g.sum(0)
            if i:
                g = (g @ W[i].T) * (a[i] > 0)
        G = gW + gB
        lr_t = lr * (0.5 * (1 + np.cos(np.pi * t / steps)))
        for k in range(len(P)):
            M[k] = 0.9 * M[k] + 0.1 * G[k]; V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
            P[k] -= lr_t * (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
    def f(Xq):
        a = Xq
        for i in range(len(W)):
            a = a @ W[i] + B[i]
            if i < len(W) - 1:
                a = np.maximum(0, a)
        return a[:, 0] * 80.0
    return f


def evaluate(scores, rng):
    s = np.asarray(scores, float)
    inel = GRID[:, 2] < 2
    worst_eligible_ref = 7.0
    # an ineligible peer the network would rank above some eligible peer
    lowest_eligible = s[~inel].min()
    inel_above = int(np.count_nonzero(s[inel] > lowest_eligible))
    mv = monotone_violations(s, tol=0.5)          # wrong-way steps of more than half a score unit
    worst_step = max([abs(v[4] - v[3]) for v in mv], default=0.0)
    # choices: best of three random candidates; is the pick as good as the rule's pick?
    tri = rng.integers(0, N_CELLS, (200_000, 3))
    pick = tri[np.arange(len(tri)), np.argmax(s[tri], axis=1)]
    best = ref[tri].max(axis=1)
    wrong = ref[pick] < best
    picked_ineligible = inel[pick] & (best > 0)
    wrong_rounded = int(np.count_nonzero(np.rint(s) != ref))
    return dict(max_abs_error=float(np.abs(s - ref).max()), mean_abs_error=float(np.abs(s - ref).mean()),
                cells_wrong_after_rounding=wrong_rounded,
                monotone_violations=len(mv), largest_wrong_way_step=float(worst_step),
                ineligible_cells_ranked_above_an_eligible_cell=inel_above,
                wrong_choices_per_100k=float(1e5 * wrong.mean()),
                ineligible_picked_per_100k=float(1e5 * picked_ineligible.mean()))


if __name__ == "__main__":
    t0 = time.time()
    rows = []
    rng_eval = np.random.default_rng(123)
    from weights import write_baseline
    w = evaluate(write_baseline()(GRID), np.random.default_rng(123))
    rows.append(dict(kind="written", shape="5-8-4-1", n_examples=0, seeds=1, **w))
    print("written  5-8-4-1   N=0     ", {k: round(v, 3) for k, v in w.items()}, flush=True)
    for shape, steps in (((5, 8, 4, 1), 6000), ((5, 8, 4, 1), 60000), ((5, 32, 32, 1), 6000)):
        for N in (256, 1024, 4096, 16384, 65536):
            res = []
            for seed in range(5):   # five independent trainings; the table reports the median, best and worst
                rng = np.random.default_rng(1000 + seed)
                idx = rng.choice(N_CELLS, N, replace=False)
                f = train(shape, idx, seed, steps=steps)
                res.append(evaluate(f(Xn), np.random.default_rng(123)))
            agg = {k: float(np.median([r[k] for r in res])) for k in res[0]}
            agg_min = {k + "_best_seed": float(np.min([r[k] for r in res])) for k in res[0]}
            agg_max = {k + "_worst_seed": float(np.max([r[k] for r in res])) for k in res[0]}
            rows.append(dict(kind="trained", shape="-".join(map(str, shape)), steps=steps, n_examples=N, seeds=5, **agg, **agg_min, **agg_max))
            print(f"trained  {'-'.join(map(str, shape)):9s} steps={steps:<6d} N={N:<6d}", {k: round(v, 2) for k, v in agg.items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open("w2_results.json", "w"), indent=1)
