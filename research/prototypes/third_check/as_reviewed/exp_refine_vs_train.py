"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W3: measure and refine, against training on the same evidence.

A node logs N decisions made under the baseline rule (with the kernel's share of random
choices), seeing only the outcome of the peer it used. Then:

  refit three prices : fit the rule's three existing prices (rtt step, loss step, delivery
                       step) to the log and write them back as integers.
  fit the steps      : fit ONE WEIGHT PER STEP of rtt, loss and delivery to the log by least
                       squares, force each feature's declared direction, round to integers
                       inside the kernel's limits. No network is trained. The result is a
                       policy whose every check can be read off its weights.
  train              : fit a 5-32-32-1 network to the log by gradient descent. Reported three
                       ways: left to decide alone; with the kernel's delivery gate applied to
                       its choices; and with the gate plus early stopping on a held-out fifth.

The first version of this experiment compared the fitted policies with a trained network
that had no gate and no early stopping. The independent reviewer pointed out that this
handicapped training twice, and that a direct fit needs no trained network at all. This
version follows that advice, and adds the reviewer's second world.
"""
import json, sys, time
import numpy as np
from wnet import *
from weights import write_baseline, RTT, LOSS, DELIV
from world import *


def refit_three(x, obs):
    ok = x[:, 2] >= 2
    A = np.stack([x[ok, 0], x[ok, 1], x[ok, 2], np.ones(ok.sum())], 1).astype(float)
    coef, *_ = np.linalg.lstsq(A, np.log(obs[ok]), rcond=None)
    a_rtt, a_loss, a_deliv = max(coef[0], 1e-6), max(coef[1], 0.0), max(-coef[2], 0.0)
    k = 4.0 / a_rtt
    w = (4, int(np.clip(round(a_loss * k), 1, 16)), int(np.clip(round(a_deliv * k), 0, 8)))
    n = write_baseline()
    n.W1[0, RTT], n.W1[0, LOSS], n.W1[0, DELIV] = -w[0], -w[1], w[2]
    n.b1[0] = 15 * w[0] + 7 * w[1] + 1
    n.check_ranges()
    return n, w


def isotonic(y, wts, increasing):
    """pool adjacent violators: the closest curve to y that only moves the declared way"""
    y = np.asarray(y, float) * (1 if increasing else -1)
    blocks = [[v, w, 1] for v, w in zip(y, np.maximum(wts, 1e-9))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            a, b = blocks[i], blocks[i + 1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1], a[2] + b[2]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.concatenate([[b[0]] * b[2] for b in blocks])
    return out * (1 if increasing else -1)


def thermo(x):
    """thermometer features for rtt (15 steps), loss (7 steps) and delivery above the gate (5 steps)"""
    cols = [np.ones(len(x))]
    for f, lo, n in ((0, 1, 16), (1, 1, 8), (2, 3, 8)):
        cols += [(x[:, f] >= k).astype(float) for k in range(lo, n)]
    return np.stack(cols, 1)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def fit_steps(x, obs, seed):
    """one weight per step, from the node's own log; returns a Steps policy and what was chosen"""
    ok = x[:, 2] >= 2
    X, y = thermo(x[ok]), obs[ok]
    rng = np.random.default_rng(4242 + seed); perm = rng.permutation(len(y)); k = max(1, len(y) // 5)
    va, tr = perm[:k], perm[k:]
    best = None
    for scale_name, tf in (("log cost", np.log), ("cost", lambda v: v)):
        co, *_ = np.linalg.lstsq(X[tr] + 0.0, tf(y[tr]), rcond=1e-6)
        rho = spearman(X[va] @ co, y[va])                       # does it rank held-out outcomes?
        if best is None or rho > best[0]:
            best = (rho, scale_name, tf)
    _, scale_name, tf = best
    co, *_ = np.linalg.lstsq(X, tf(y), rcond=1e-6)
    # curves per feature (cost rises with rtt and loss, falls with delivery), made monotone
    counts = lambda f, levels: np.array([np.count_nonzero(x[ok, f] == v) for v in levels], float)
    c_rtt = isotonic(np.concatenate([[0], np.cumsum(co[1:16])]), counts(0, range(16)), True)
    c_loss = isotonic(np.concatenate([[0], np.cumsum(co[16:23])]), counts(1, range(8)), True)
    c_del = isotonic(np.concatenate([[0], np.cumsum(co[23:28])]), counts(2, range(2, 8)), False)
    span = c_rtt[-1] - c_rtt[0]
    K = 240.0 / span if span > 1e-9 else 1.0                    # 240 score units across the rtt range
    st = Steps()
    def primary(curve):
        w = np.maximum(1, np.rint(K * np.diff(curve))).astype(np.int64)     # every step must matter
        return w
    w_r, w_l = primary(c_rtt), primary(c_loss)
    lo = int(min(w_r.min(), w_l.min()))
    w_r, w_l = np.minimum(w_r, MAX_STEP_RATIO * lo), np.minimum(w_l, MAX_STEP_RATIO * lo)
    st.w[RTT] = -w_r; st.w[LOSS] = -w_l
    st.w[DELIV][2:] = np.maximum(0, np.rint(-K * np.diff(c_del))).astype(np.int64)
    st.bias = int(w_r.sum() + w_l.sum() + 1)
    return st, scale_name


def train_mlp(x, obs, seed, steps=6000, lr=3e-3):
    rng = np.random.default_rng(seed)
    X = x / (np.array(LEVELS) - 1.0); Y = np.log(obs)
    mu, sd = Y.mean(), Y.std() + 1e-12; Y = (Y - mu) / sd
    sizes = [5, 32, 32, 1]
    W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])) for i in range(3)]
    B = [np.zeros(sizes[i + 1]) for i in range(3)]
    P = W + B; M = [np.zeros_like(p) for p in P]; V = [np.zeros_like(p) for p in P]
    for t in range(1, steps + 1):
        b = rng.integers(0, len(X), min(256, len(X)))
        a = [X[b]]
        for i in range(3):
            z = a[-1] @ W[i] + B[i]
            a.append(np.maximum(0, z) if i < 2 else z)
        g = (a[-1][:, 0] - Y[b])[:, None] * (2.0 / len(b))
        gW, gB = [None] * 3, [None] * 3
        for i in reversed(range(3)):
            gW[i] = a[i].T @ g; gB[i] = g.sum(0)
            if i:
                g = (g @ W[i].T) * (a[i] > 0)
        G = gW + gB
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * t / steps))
        for k in range(6):
            M[k] = 0.9 * M[k] + 0.1 * G[k]; V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
            P[k] -= lr_t * (M[k] / (1 - 0.9 ** t)) / (np.sqrt(V[k] / (1 - 0.999 ** t)) + 1e-8)
    def score(Xq):                       # higher is better: minus predicted log cost
        a = Xq / (np.array(LEVELS) - 1.0)
        for i in range(3):
            a = a @ W[i] + B[i]
            if i < 2:
                a = np.maximum(0, a)
        return -a[:, 0]
    return score


def train_early_stop(x, obs, seed):
    """choose the number of steps on a held-out fifth of the log, then train on all of it"""
    rng = np.random.default_rng(777 + seed); perm = rng.permutation(len(x)); k = max(1, len(x) // 5)
    va, tr = perm[:k], perm[k:]
    best = None
    for steps in (100, 300, 1000, 3000, 6000):
        f = train_mlp(x[tr], obs[tr], seed, steps=steps)
        pred = -f(x[va]); yv = np.log(obs[va])
        A = np.stack([pred, np.ones_like(pred)], 1); co, *_ = np.linalg.lstsq(A, yv, rcond=None)
        mse = float(np.mean((A @ co - yv) ** 2))
        if best is None or mse < best[0]:
            best = (mse, steps)
    return train_mlp(x, obs, seed, steps=best[1]), best[1]


if __name__ == "__main__":
    t0 = time.time()
    test = candidates(np.random.default_rng(999), 500_000)
    rows = []
    for w in WORLDS:
        ev = lambda fn: evaluate(fn, test, w)
        c_rule = ev(write_baseline())[0]; c_or = ev(lambda X: -true_cost(X, w))[0]
        rows.append(dict(world=w, method="baseline rule, as written", n_logged=0, mean_cost_ms=c_rule))
        rows.append(dict(world=w, method="oracle: knows the true cost", n_logged=0, mean_cost_ms=c_or))
        print(f"[{w}] rule {c_rule:.1f} ms | oracle {c_or:.1f} ms", flush=True)
        for N in (500, 2_000, 20_000, 200_000):
            acc = {k: [] for k in ("refit3", "steps", "steps_ok", "steps_scale", "train", "train_bad", "train_gate", "train_gate_es", "es_steps", "inel_above", "inel_seen")}
            for seed in range(5):
                rng = np.random.default_rng(10 * N + seed)
                x, obs = log_evidence(rng, N, w)
                acc["refit3"].append(ev(refit_three(x, obs)[0])[0])
                st, sc = fit_steps(x, obs, seed)
                acc["steps"].append(ev(st)[0]); acc["steps_ok"].append(st.read_checks()["within_limits"]); acc["steps_scale"].append(sc)
                f = train_mlp(x, obs, seed)
                c, bad = ev(f); acc["train"].append(c); acc["train_bad"].append(bad)
                acc["train_gate"].append(ev(kernel_gate(f))[0])
                fe, n_steps = train_early_stop(x, obs, seed)
                acc["train_gate_es"].append(ev(kernel_gate(fe))[0]); acc["es_steps"].append(n_steps)
                sg = f(GRID); inel = GRID[:, 2] < 2
                acc["inel_above"].append(int(np.count_nonzero(sg[inel] > sg[~inel].min())))
                acc["inel_seen"].append(int(np.count_nonzero(x[:, 2] < 2)))
            med = lambda k: float(np.median(acc[k])); worst = lambda k: float(np.max(acc[k]))
            gap = lambda c: (c_rule - c) / (c_rule - c_or)
            rows += [
                dict(world=w, method="refit three prices", n_logged=N, mean_cost_ms=med("refit3"), worst_seed_ms=worst("refit3"), share_of_gap_closed=gap(med("refit3"))),
                dict(world=w, method="fit one weight per step, written", n_logged=N, mean_cost_ms=med("steps"), worst_seed_ms=worst("steps"), share_of_gap_closed=gap(med("steps")),
                     all_within_limits=bool(all(acc["steps_ok"])), scale_chosen=acc["steps_scale"]),
                dict(world=w, method="trained network, left to decide alone", n_logged=N, mean_cost_ms=med("train"), worst_seed_ms=worst("train"), share_of_gap_closed=gap(med("train")),
                     failing_peer_picked_per_100k=med("train_bad"), ineligible_cells_ranked_above_an_eligible_cell=med("inel_above"), ineligible_examples_in_log=med("inel_seen")),
                dict(world=w, method="trained network + kernel gate", n_logged=N, mean_cost_ms=med("train_gate"), worst_seed_ms=worst("train_gate"), share_of_gap_closed=gap(med("train_gate"))),
                dict(world=w, method="trained network + kernel gate + early stopping", n_logged=N, mean_cost_ms=med("train_gate_es"), worst_seed_ms=worst("train_gate_es"), share_of_gap_closed=gap(med("train_gate_es")), steps_chosen=acc["es_steps"]),
            ]
            print(f"[{w}] N={N:<7d} refit3 {med('refit3'):6.1f} | steps {med('steps'):6.1f} (worst {worst('steps'):6.1f}; limits {all(acc['steps_ok'])}; {acc['steps_scale'][0]}) | "
                  f"train alone {med('train'):6.1f} ({med('train_bad'):.0f}/100k failing) | +gate {med('train_gate'):6.1f} | +gate+es {med('train_gate_es'):6.1f} (worst {worst('train_gate_es'):6.1f}) | {time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open("w3_results.json", "w"), indent=1)
