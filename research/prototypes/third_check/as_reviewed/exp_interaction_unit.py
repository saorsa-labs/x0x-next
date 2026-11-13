"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W4: the exception. When does a policy need a unit that combines two features,
and what does that unit cost in checking?

In the "timeout" world a lost packet costs a fixed retransmission timeout, so loss hurts
most on fast paths. No policy made of one-feature steps can say that. Here one hidden unit
is added to the step weights: relu(a_loss * loss - a_rtt * rtt - c), "lossy for its speed".
Its three small integers are chosen from a short list by how well the result ranks held-out
outcomes, and its weight is fitted by least squares with the steps. Then the table is built
and every check is run on it; if a check fails, the unit's weight is shrunk until it passes.
"""
import json, time
import numpy as np
from wnet import *
from weights import write_baseline, RTT, LOSS, DELIV
from world import *
from exp_refine_vs_train import thermo, isotonic, spearman, fit_steps

UNIT_CHOICES = [(al, ar, c) for al in (1, 2, 3) for ar in (1, 2) for c in (-2, 0, 2, 4)]


def unit_col(x, al, ar, c):
    return np.maximum(0, al * x[:, 1] - ar * x[:, 0] - c).astype(float)


def fit_steps_plus(x, obs, seed):
    """Steps first. A unit is added only if it improves the ranking of held-out outcomes in
    every one of five folds on average by more than twice the standard error, and only at
    the largest weight for which the table passes every check."""
    base, scale_name = fit_steps(x, obs, seed)
    tf = np.log if scale_name == "log cost" else (lambda v: v)
    ok = x[:, 2] >= 2
    xs, y = x[ok], obs[ok]
    X = thermo(xs)
    rng = np.random.default_rng(4242 + seed); fold = rng.integers(0, 5, len(y))

    def cv(A):
        out = []
        for k in range(5):
            tr, va = fold != k, fold == k
            co, *_ = np.linalg.lstsq(A[tr], tf(y[tr]), rcond=1e-6)
            out.append(spearman(A[va] @ co, y[va]))
        return np.array(out)

    r0 = cv(X)
    best = None
    for choice in UNIT_CHOICES:
        d = cv(np.column_stack([X, unit_col(xs, *choice)])) - r0
        gain, se = d.mean(), d.std(ddof=1) / np.sqrt(5)
        if gain > 2 * se and gain > 0 and (best is None or gain > best[0]):
            best = (gain, choice)
    info = dict(scale=scale_name, unit=None, v=0, shrunk_by=0, proposed=None)
    if best is None:
        return StepsPlus(base, []), info
    choice = best[1]; info["proposed"] = choice
    A = np.column_stack([X, unit_col(xs, *choice)])
    co, *_ = np.linalg.lstsq(A, tf(y), rcond=1e-6)
    if co[-1] <= 0:
        return StepsPlus(base, []), info
    counts = lambda f, levels: np.array([np.count_nonzero(xs[:, f] == v) for v in levels], float)
    c_rtt = isotonic(np.concatenate([[0], np.cumsum(co[1:16])]), counts(0, range(16)), True)
    c_loss = isotonic(np.concatenate([[0], np.cumsum(co[16:23])]), counts(1, range(8)), True)
    c_del = isotonic(np.concatenate([[0], np.cumsum(co[23:28])]), counts(2, range(2, 8)), False)
    span = c_rtt[-1] - c_rtt[0]
    K = 240.0 / span if span > 1e-9 else 1.0
    st = Steps()
    w_r = np.maximum(1, np.rint(K * np.diff(c_rtt))).astype(np.int64)
    w_l = np.maximum(1, np.rint(K * np.diff(c_loss))).astype(np.int64)
    lo = int(min(w_r.min(), w_l.min()))
    w_r, w_l = np.minimum(w_r, MAX_STEP_RATIO * lo), np.minimum(w_l, MAX_STEP_RATIO * lo)
    st.w[RTT] = -w_r; st.w[LOSS] = -w_l
    st.w[DELIV][2:] = np.maximum(0, np.rint(-K * np.diff(c_del))).astype(np.int64)
    al, ar, c = choice
    a = [-ar, al, 0, 0, 0]
    v = -int(np.rint(K * co[-1])); v0 = v
    st.bias = int(w_r.sum() + w_l.sum() + abs(v) * (al * 7 + abs(min(0, c))) + 1)
    while v < 0:
        pol = StepsPlus(st, [(a, -c, v)])
        if pol.table_checks()["passes"]:
            info.update(unit=choice, v=v, shrunk_by=v - v0)
            return pol, info
        v += 1                                                   # shrink the unit until the table passes
    info["shrunk_by"] = -v0
    return StepsPlus(base, []), info


if __name__ == "__main__":
    t0 = time.time()
    test = candidates(np.random.default_rng(999), 500_000)
    rows = []
    for w in WORLDS:
        ev = lambda fn: evaluate(fn, test, w)[0]
        c_rule = ev(write_baseline()); c_or = ev(lambda X: -true_cost(X, w))
        for N in (500, 2_000, 20_000, 200_000):
            a = dict(steps=[], plus=[], info=[], ok=[], check_s=[])
            for seed in range(5):
                x, obs = log_evidence(np.random.default_rng(10 * N + seed), N, w)
                a["steps"].append(ev(fit_steps(x, obs, seed)[0]))
                pol, info = fit_steps_plus(x, obs, seed)
                t1 = time.time(); chk = pol.table_checks(); a["check_s"].append(time.time() - t1)
                a["plus"].append(ev(pol)); a["info"].append(info); a["ok"].append(chk["passes"])
            gap = lambda c: (c_rule - c) / (c_rule - c_or)
            ms, mp = float(np.median(a["steps"])), float(np.median(a["plus"]))
            rows.append(dict(world=w, n_logged=N, rule_ms=c_rule, oracle_ms=c_or,
                             steps_ms=ms, steps_share_of_gap=gap(ms), steps_worst_seed_ms=float(np.max(a["steps"])),
                             steps_plus_unit_ms=mp, steps_plus_unit_share_of_gap=gap(mp), steps_plus_unit_worst_seed_ms=float(np.max(a["plus"])),
                             unit_used_in_seeds=int(sum(i["unit"] is not None for i in a["info"])), every_policy_passes_the_table=bool(all(a["ok"])),
                             table_check_seconds_numpy=float(np.median(a["check_s"])), detail=a["info"]))
            print(f"[{w}] N={N:<7d} rule {c_rule:.1f} | steps {ms:.2f} ({100*gap(ms):.0f}% of gap) | steps + one unit {mp:.2f} ({100*gap(mp):.0f}%) | unit used in {rows[-1]['unit_used_in_seeds']}/5 seeds "
                  f"{[i['unit'] for i in a['info']]} v={[i['v'] for i in a['info']]} shrunk={[i['shrunk_by'] for i in a['info']]} proposed={sum(i['proposed'] is not None for i in a['info'])} | all pass table: {all(a['ok'])} | oracle {c_or:.1f} | {time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open("w4_results.json", "w"), indent=1, default=str)
