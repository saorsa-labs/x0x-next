"""
THIRD CHECK -- reviewer's script. Not production code. Not part of x0x.

Is the "fair" re-run of W3 fair to the trained network?
In exp_refine_vs_train.py the fitted step weights are fitted on ELIGIBLE logged examples only and
on THREE features (rtt, loss, delivery), while the trained network is trained on ALL examples
(including the few 3,000 ms timeouts of ineligible peers, which the kernel gate makes irrelevant)
and on FIVE inputs (age and balance are pure noise). Here the same network, optimiser, early
stopping and gate are used, with those two handicaps removed one at a time.
Same logs and seeds as the original (rng = default_rng(10 N + seed)); N = 500 and 2,000.
"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(HERE, 'as_reviewed'))   # path made relative for the zip; was the reviewer's working folder
import numpy as np
from wnet import *
from world import *
from exp_refine_vs_train import fit_steps, train_mlp, train_early_stop

t0 = time.time()


def train_mlp_cols(x, obs, seed, cols, steps=6000, lr=3e-3):
    """exp_refine_vs_train.train_mlp, unchanged except that only the listed input columns are used"""
    rng = np.random.default_rng(seed)
    scale = (np.array(LEVELS) - 1.0)[cols]
    X = x[:, cols] / scale; Y = np.log(obs)
    mu, sd = Y.mean(), Y.std() + 1e-12; Y = (Y - mu) / sd
    sizes = [len(cols), 32, 32, 1]
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
    def score(Xq):
        a = Xq[:, cols] / scale
        for i in range(3):
            a = a @ W[i] + B[i]
            if i < 2:
                a = np.maximum(0, a)
        return -a[:, 0]
    return score


def early_stop_cols(x, obs, seed, cols):
    rng = np.random.default_rng(777 + seed); perm = rng.permutation(len(x)); k = max(1, len(x) // 5)
    va, tr = perm[:k], perm[k:]
    best = None
    for steps in (100, 300, 1000, 3000, 6000):
        f = train_mlp_cols(x[tr], obs[tr], seed, cols, steps=steps)
        pred = -f(x[va]); yv = np.log(obs[va])
        A = np.stack([pred, np.ones_like(pred)], 1); co, *_ = np.linalg.lstsq(A, yv, rcond=None)
        mse = float(np.mean((A @ co - yv) ** 2))
        if best is None or mse < best[0]:
            best = (mse, steps)
    return train_mlp_cols(x, obs, seed, cols, steps=best[1]), best[1]


test = candidates(np.random.default_rng(999), 500_000)
rows = []
ALL5, THREE = [0, 1, 2, 3, 4], [0, 1, 2]
for w in WORLDS:
    ev = lambda fn: evaluate(fn, test, w)[0]
    for N in (500, 2_000):
        acc = {k: [] for k in ("steps", "orig", "elig5", "elig3")}
        for seed in range(5):
            x, obs = log_evidence(np.random.default_rng(10 * N + seed), N, w)
            st, _ = fit_steps(x, obs, seed); acc["steps"].append(ev(st))
            fe, _ = train_early_stop(x, obs, seed); acc["orig"].append(ev(kernel_gate(fe)))                       # the study's "trained + gate + early stopping"
            ok = x[:, 2] >= 2
            f5, _ = early_stop_cols(x[ok], obs[ok], seed, ALL5); acc["elig5"].append(ev(kernel_gate(f5)))          # eligible examples only
            f3, _ = early_stop_cols(x[ok], obs[ok], seed, THREE); acc["elig3"].append(ev(kernel_gate(f3)))         # and the same three inputs the step fit uses
            print(f"   [{w}] N={N} seed {seed}: steps {acc['steps'][-1]:.1f} | trained (study) {acc['orig'][-1]:.1f} | eligible only {acc['elig5'][-1]:.1f} | eligible only, 3 inputs {acc['elig3'][-1]:.1f} | {time.time()-t0:.0f}s", flush=True)
        med = lambda k: float(np.median(acc[k])); worst = lambda k: float(np.max(acc[k]))
        row = dict(world=w, n_logged=N, fitted_steps_ms=med("steps"), trained_gate_es_as_in_study_ms=med("orig"),
                   trained_eligible_examples_only_ms=med("elig5"), trained_eligible_only_three_inputs_ms=med("elig3"),
                   worst_seed=dict(steps=worst("steps"), study=worst("orig"), eligible_only=worst("elig5"), eligible_three=worst("elig3")))
        rows.append(row)
        print(f"[{w}] N={N}: MEDIAN of 5 logs | fitted steps {row['fitted_steps_ms']:.1f} | trained+gate+ES as in the study {row['trained_gate_es_as_in_study_ms']:.1f} | "
              f"trained on eligible examples only {row['trained_eligible_examples_only_ms']:.1f} | and with the three inputs the step fit uses {row['trained_eligible_only_three_inputs_ms']:.1f}", flush=True)
json.dump(rows, open(_os.path.join(HERE, 'check_w3_fair_results.json'), 'w'), indent=1)
