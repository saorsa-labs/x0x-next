"""
THIRD CHECK -- reviewer's attack script. Not production code. Not part of x0x.

Attack 1: "better on the typical decision, worse on the average".
The trial and the watch in exp_trials.py test the LOGARITHM of observed outcomes (mean log ratio),
while every harm figure in the study is the change in the ARITHMETIC mean cost. A policy can win
by ratio on cheap decisions and lose by absolute amount on expensive ones.

Part A: an adversary that knows the true costs (the same adversary the study calls "the worst
        policy that can exist", but with a brain): per decision it picks the in-band option that
        maximises  harm - lam * (log ratio + m)  and abstains otherwise. lam is set so that the
        mean log ratio over contested decisions is -m (so the trial sees a "clear improvement").
Part B: the same policy asks for more room (3 -> 6 -> 9 -> 12 -> 18 -> none), each step tried
        "like any other change: the wider version against the narrower one" (exp_widen.py's rule).
Part C: the watch on the adopted policies.
Part D: a rung-1 points table (monotone, inside the ratio limit) found by search with the same aim.
Part E: what the fix costs: the same trials with a test on raw (arithmetic) outcomes.
"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(HERE, 'as_reviewed'))   # path made relative for the zip; was the reviewer's working folder
import numpy as np
from safety_common import *
from exp_trials import summarise, summarise_watch, LOOKS, Z_ADOPT, Z_STOP, MIN_ADOPT, RUNS

t0 = time.time()
out = dict(partA=[], partB=[], partC=[], partD=[], partE=[])


def inband(P, d):
    return P.elig & (P.sb >= P.sb.max(1, keepdims=True) - (1e9 if d is None else d))


def lagrange_pick(P, mask, ref, lam, m):
    idx = np.arange(len(ref))
    cref = P.tc[idx, ref]
    D = (P.tc - cref[:, None]) / P.base
    L = np.log(P.tc / cref[:, None])
    val = np.where(mask, D - lam * (L + m), -np.inf)
    val[idx, ref] = 0.0                                   # abstain = stay with the reference pick
    return val.argmax(1)


def stats(P, pick, ref):
    idx = np.arange(len(ref)); m = pick != ref
    cp, ca = P.tc[idx, pick][m], P.tc[idx, ref][m]
    L = np.log(cp / ca)
    return m, cp, ca, float(m.mean()), float(L.mean()) if m.any() else 0.0, float(L.std()) if m.any() else 0.0


def tune(P, mask, ref, m):
    """largest harm subject to mean log ratio over contested decisions <= -m (bisection on lam)"""
    lo, hi = 0.0, 50.0
    for _ in range(40):
        lam = 0.5 * (lo + hi)
        pk = lagrange_pick(P, mask, ref, lam, m)
        _, _, _, share, mu, _ = stats(P, pk, ref)
        if share == 0 or mu <= -m:
            hi = lam
        else:
            lo = lam
    return lagrange_pick(P, mask, ref, hi, m), hi


for world in WORLDS:
    P = Pool(600_000, world, seed=31)
    idx = np.arange(len(P.a))
    rng = np.random.default_rng(1234)
    worst3 = P.harm(np.where(inband(P, 3), np.where(P.elig, P.tc, -1.0), -1.0).argmax(1))
    print(f"\n[{world}] worst possible inside band 3 (the study's bound): {100*worst3:+.2f}%", flush=True)

    # ---------- Part A: inside band 3, against the fixed rule
    best = None
    for m in (0.04, 0.06, 0.08, 0.10, 0.15):
        pk, lam = tune(P, inband(P, 3), P.a, m)
        msk, cp, ca, share, mu, sd = stats(P, pk, P.a)
        res = summarise(rng, cp, ca, share, "twin", 0.20, P.base)
        res_split = summarise(rng, cp, ca, share, "split", 0.10, P.base)
        row = dict(world=world, target_mean_log_ratio=-m, lam=lam, harm_arithmetic=P.harm(pk), contested_share=share,
                   mean_log_ratio_on_contested=mu, sd_log_ratio=sd,
                   geometric_mean_change_all_decisions=float(np.exp(np.mean(np.log(P.cost_of(pk) / P.ca))) - 1),
                   twin=res, split=res_split)
        out["partA"].append(row)
        print(f"[{world}] A  m={m:.2f} lam={lam:6.3f} | TRUE harm {100*row['harm_arithmetic']:+5.2f}% | contested {100*share:4.1f}% | mean log ratio {mu:+.3f} (sd {sd:.2f}) | "
              f"twin: adopted {100*res['adopted']:5.1f}% stopped {100*res['stopped']:4.1f}% median {res['median_decisions_to_verdict']:.0f} dec, {res['mean_extra_messages']:.0f} copies | "
              f"split: adopted {100*res_split['adopted']:5.1f}% | {time.time()-t0:.0f}s", flush=True)
        if res["adopted"] >= 0.90 and (best is None or row["harm_arithmetic"] > best[0]):
            best = (row["harm_arithmetic"], pk, m)

    # ---------- Part C (first): the watch on the adopted band-3 attack policy
    if best is not None:
        pk3 = best[1]
        msk, cp, ca, share, mu, sd = stats(P, pk3, P.a)
        for label, mode, control in (("control: 10% stay with the fixed rule", "split", 0.10), ("twin 10%", "twin", 0.10), ("twin 30%", "twin", 0.30)):
            res = summarise_watch(rng, cp, ca, share, mode, control, P.base)
            out["partC"].append(dict(world=world, policy=f"band-3 attack (m={best[2]})", watch=label, harm_while_it_runs=P.harm(pk3), **res))
            print(f"[{world}] C  watch on the band-3 attack ({label}): true harm {100*P.harm(pk3):+.2f}% | reverted in {100*res['reverted']:.1f}% of 400 | median {res['median_decisions_to_revert']:.0f} dec", flush=True)

        # ---------- Part B: earning room with the same trick
        prev = pk3; m = best[2]
        for w_prev, w in ((3, 6), (6, 9), (9, 12), (12, 18), (18, None)):
            cur, lam = tune(P, inband(P, w), prev, m)
            msk, cp, ca, share, mu, sd = stats(P, cur, prev)
            if share < 1e-4:
                print(f"[{world}] B  {w_prev}->{w}: nothing to gain"); break
            res = summarise(rng, cp, ca, share, "twin", 0.20, P.base)
            # the watch afterwards compares the widened policy with the FIXED RULE
            mk2, cp2, ca2, share2, mu2, sd2 = stats(P, cur, P.a)
            resw = summarise_watch(rng, cp2, ca2, share2, "twin", 0.10, P.base)
            row = dict(world=world, from_band=w_prev, to_band="none" if w is None else w, harm_before=P.harm(prev), harm_after=P.harm(cur),
                       differ_on=share, mean_log_ratio_vs_narrower=mu, granted=res["adopted"], stopped=res["stopped"], no_verdict=res["rejected"],
                       median_decisions=res["median_decisions_to_verdict"], copies=res["mean_extra_messages"],
                       mean_log_ratio_vs_fixed_rule=mu2, watch_reverted=resw["reverted"],
                       worst_possible_in_that_band=P.harm(np.where(inband(P, w), np.where(P.elig, P.tc, -1.0), -1.0).argmax(1)))
            out["partB"].append(row)
            print(f"[{world}] B  {w_prev}->{w}: TRUE harm {100*row['harm_before']:+6.2f}% -> {100*row['harm_after']:+6.2f}% (worst possible there {100*row['worst_possible_in_that_band']:+.1f}%) | "
                  f"log ratio vs narrower {mu:+.3f} | granted {100*res['adopted']:5.1f}% stopped {100*res['stopped']:4.1f}% | median {res['median_decisions_to_verdict']:.0f} dec | "
                  f"watch (twin 10%) reverts {100*resw['reverted']:.1f}% | {time.time()-t0:.0f}s", flush=True)
            if res["adopted"] < 0.5:
                break
            prev = cur

    # ---------- Part D: a rung-1 points table with the same aim (coordinate search, as exp_limits_search.py)
    S = Pool(40_000, world, seed=11)
    VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    def objective(pool, w, delta, lam, m):
        pk = pool.pick(fast_scores(pool, *w), delta)
        msk = pk != pool.a
        if not msk.any():
            return 0.0
        L = np.log(pool.cost_of(pk)[msk] / pool.ca[msk])
        return pool.harm(pk) - lam * msk.mean() * (L.mean() + m)
    for delta in (3, 6):
        bestD = None
        for lam in (0.3, 0.6, 1.0, 2.0):
            for restart in range(3):
                r = np.random.default_rng(100 * restart + 7)
                w = [r.choice(VALUES, 15), r.choice(VALUES, 7), r.choice([0] + VALUES, 5)]
                if restart == 0:
                    w = [np.full(15, 32), np.ones(7, int), np.zeros(5, int)]          # latency before everything
                cur = objective(S, w, delta, lam, 0.06); improved = True
                while improved:
                    improved = False
                    for part, vals in ((0, VALUES), (1, VALUES), (2, [0] + VALUES)):
                        for i in range(len(w[part])):
                            keep = w[part][i]
                            for v in vals:
                                if v == keep: continue
                                w[part][i] = v
                                lo = min(w[0].min(), w[1].min()); hi = max(w[0].max(), w[1].max(), w[2].max())
                                if hi > 32 * lo: continue
                                o = objective(S, w, delta, lam, 0.06)
                                if o > cur + 1e-9: cur, keep, improved = o, v, True
                            w[part][i] = keep
                pk = P.pick(fast_scores(P, *w), delta); msk, cp, ca, share, mu, sd = stats(P, pk, P.a)
                h = P.harm(pk)
                if share > 0 and mu < -0.04 and (bestD is None or h > bestD[0]):
                    bestD = (h, [a.copy() for a in w], share, mu, sd, lam)
        if bestD is None:
            print(f"[{world}] D  band {delta}: no points table found with harm and a favourable log ratio"); continue
        h, w, share, mu, sd, lam = bestD
        pk = P.pick(fast_scores(P, *w), delta); msk, cp, ca, share, mu, sd = stats(P, pk, P.a)
        st = steps_from(*w); chk = st.read_checks()
        res = summarise(rng, cp, ca, share, "twin", 0.20, P.base)
        resw = summarise_watch(rng, cp, ca, share, "twin", 0.10, P.base)
        row = dict(world=world, band=delta, weights=dict(rtt=[int(v) for v in w[0]], loss=[int(v) for v in w[1]], delivery=[int(v) for v in w[2]]),
                   passes_version3_read_checks=chk["within_limits"], largest_to_smallest_step_all_features=float(max(w[0].max(), w[1].max(), w[2].max()) / min(w[0].min(), w[1].min())),
                   harm_arithmetic=h, contested_share=share, mean_log_ratio_on_contested=mu, twin=res, watch_twin10=resw)
        out["partD"].append(row)
        print(f"[{world}] D  band {delta}: points table rtt={row['weights']['rtt']} loss={row['weights']['loss']} deliv={row['weights']['delivery']} | V3 read-checks pass: {chk['within_limits']} | "
              f"TRUE harm {100*h:+.2f}% | contested {100*share:.1f}% | mean log ratio {mu:+.3f} | twin: adopted {100*res['adopted']:.1f}% stopped {100*res['stopped']:.1f}% no verdict {100*res['rejected']:.1f}% | "
              f"watch reverts {100*resw['reverted']:.1f}% | {time.time()-t0:.0f}s", flush=True)

    # ---------- Part E: the same trial with a test on raw outcomes (arithmetic mean), on copies
    def run_trial_raw(rng, cp, ca, rho, share):
        K = int(LOOKS[-1] / share * 1.25) + 64
        ix = rng.integers(0, len(cp), K); treat = rng.random(K) < share
        d = np.where(treat, cp[ix] * rng.lognormal(0, NOISE, K) - ca[ix] * rng.lognormal(0, NOISE, K), 0.0)
        nt = np.cumsum(treat); s1, s2 = np.cumsum(d), np.cumsum(d * d)
        verdict, at = "rejected", K - 1
        for n in LOOKS:
            j = int(np.searchsorted(nt, n))
            if j >= K: break
            mm = s1[j] / n; v = max(s2[j] / n - mm * mm, 1e-12); z = -mm / np.sqrt(v / n)
            if z >= Z_ADOPT and n >= MIN_ADOPT: verdict, at = "adopted", j; break
            if z <= -Z_STOP: verdict, at = "stopped", j; break
            at = j
        return verdict, (at + 1) / rho, int(nt[at])
    x, obs = log_evidence(np.random.default_rng(200000), 20_000, world)
    honest = P.pick(fast_scores(P, *fit_steps_ratio(x, obs, 0, 32)), 3)
    worst = np.where(inband(P, 3), np.where(P.elig, P.tc, -1.0), -1.0).argmax(1)
    cases = [("honest fit, band 3", honest), ("worst possible, band 3", worst)]
    if best is not None:
        cases.append((f"the Part A attack (m={best[2]}), band 3", best[1]))
    for name, pk in cases:
        msk, cp, ca, share, mu, sd = stats(P, pk, P.a)
        r = [run_trial_raw(rng, cp, ca, share, 0.20) for _ in range(RUNS)]
        v = [a[0] for a in r]
        row = dict(world=world, policy=name, true_change=P.harm(pk), adopted=v.count("adopted") / RUNS, stopped=v.count("stopped") / RUNS, no_verdict=v.count("rejected") / RUNS,
                   median_decisions=float(np.median([a[1] for a in r])), mean_copies=float(np.mean([a[2] for a in r])))
        out["partE"].append(row)
        print(f"[{world}] E  raw-outcome test, on copies | {name:38s} | true {100*row['true_change']:+5.2f}% | adopted {100*row['adopted']:5.1f}% stopped {100*row['stopped']:5.1f}% no verdict {100*row['no_verdict']:5.1f}% | median {row['median_decisions']:.0f} dec, {row['mean_copies']:.0f} copies", flush=True)

json.dump(out, open(_os.path.join(HERE, "attack_logmean_results.json"), 'w'), indent=1)
print(f"done in {time.time()-t0:.0f}s")
