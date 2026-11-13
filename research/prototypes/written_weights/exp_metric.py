"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W8e: does the trial test the same thing the ceiling is stated in?

The trials of W8c compare the LOGARITHM of outcomes, which measures the typical decision. The
harm ceiling is stated as a change in TOTAL cost. The two can disagree. This experiment builds
the policy that exploits the gap: inside the band it takes the best option on cheap decisions
and the worst option on expensive ones, so the typical decision improves while the total gets
worse. It then runs that policy, and the policies of W8c, under three trial rules:

  typical : the rule of W8c (logarithms)
  total   : the same rule on raw outcomes
  both    : stop if clearly worse on either measure; adopt only if clearly better on both
All trials are on copies, band 3, 20% of contested decisions duplicated, 400 runs.
"""
import json, time
import numpy as np
from safety_common import *

LOOKS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
Z_ADOPT, Z_STOP, MIN_ADOPT, RUNS = 3.09, 2.33, 64, 400


def trial(rng, cp, ca, rho, rule, share=0.20):
    K = int(LOOKS[-1] / share * 1.25) + 64
    idx = rng.integers(0, len(cp), K)
    treat = rng.random(K) < share
    op = cp[idx] * rng.lognormal(0, NOISE, K); oa = ca[idx] * rng.lognormal(0, NOISE, K)
    nt = np.cumsum(treat)
    dl = np.where(treat, np.log(op) - np.log(oa), 0.0); dr = np.where(treat, op - oa, 0.0)
    sl1, sl2, sr1, sr2 = np.cumsum(dl), np.cumsum(dl * dl), np.cumsum(dr), np.cumsum(dr * dr)
    verdict, at = "rejected", K - 1
    for n in LOOKS:
        j = int(np.searchsorted(nt, n))
        if j >= K: break
        zs = []
        for s1, s2 in ((sl1, sl2), (sr1, sr2)):
            m = s1[j] / n; v = max(s2[j] / n - m * m, 1e-12)
            zs.append(-m / np.sqrt(v / n))
        zl, zr = zs
        if rule == "typical": good, bad = zl >= Z_ADOPT, zl <= -Z_STOP
        elif rule == "total": good, bad = zr >= Z_ADOPT, zr <= -Z_STOP
        else: good, bad = (zl >= Z_ADOPT and zr >= Z_ADOPT), (zl <= -Z_STOP or zr <= -Z_STOP)
        if bad: verdict, at = "stopped", j; break
        if good and n >= MIN_ADOPT: verdict, at = "adopted", j; break
        at = j
    return verdict, (at + 1) / rho, int(nt[at])


if __name__ == "__main__":
    t0 = time.time(); out = []; rng = np.random.default_rng(99)
    for world in WORLDS:
        P = Pool(600_000, world, seed=31)
        inb = P.elig & (P.sb >= P.sb.max(1, keepdims=True) - 3)
        best = np.where(inb, P.tc, 1e18).argmin(1); worst = np.where(inb, P.tc, -1.0).argmax(1)
        x, obs = log_evidence(np.random.default_rng(200000), 20_000, world)
        honest = P.pick(fast_scores(P, *fit_steps_ratio(x, obs, 0, 32)), 3)
        # the gap policy: best option on cheap decisions, worst on expensive ones; choose the split that does
        # the most harm to the total while the typical contested decision still improves
        gap, gap_stats = None, None
        for q in np.linspace(0.30, 0.995, 140):
            tau = np.quantile(P.ca, q)
            pk = np.where(P.ca <= tau, best, worst); m = pk != P.a
            lr = float(np.mean(np.log(P.cost_of(pk)[m] / P.ca[m]))); h = P.harm(pk)
            if lr < -0.01 and (gap_stats is None or h > gap_stats["total"]):
                gap, gap_stats = pk, dict(total=h, typical_contested=lr, split_at_quantile=float(q), contested=float(m.mean()))
        cands = {"honest fitted policy": honest, "worst possible policy inside the band": worst,
                 "gap policy: best option on cheap decisions, worst on expensive ones": gap,
                 "careless: random inside the band": np.where(inb, np.random.default_rng(9).random(P.sb.shape), -1).argmax(1)}
        for name, pk in cands.items():
            m = pk != P.a
            cp, ca, rho = P.cost_of(pk)[m], P.ca[m], float(m.mean())
            row = dict(world=world, policy=name, change_in_total_cost=P.harm(pk), change_in_typical_contested_decision=float(np.exp(np.mean(np.log(cp / ca))) - 1), contested_share=rho, rules={})
            for rule in ("typical", "total", "both"):
                r = [trial(rng, cp, ca, rho, rule) for _ in range(RUNS)]
                v = [a for a, _, _ in r]
                row["rules"][rule] = dict(adopted=v.count("adopted") / RUNS, stopped=v.count("stopped") / RUNS, rejected=v.count("rejected") / RUNS,
                                          median_decisions=float(np.median([b for _, b, _ in r])), mean_copies=float(np.mean([c for _, _, c in r])))
            out.append(row)
            print(f"[{world}] {name[:52]:52s} total {100*row['change_in_total_cost']:+5.2f}% typical(contested) {100*row['change_in_typical_contested_decision']:+6.2f}% | " +
                  " | ".join(f"{k}: A {100*d['adopted']:5.1f} S {100*d['stopped']:5.1f} R {100*d['rejected']:5.1f} med {d['median_decisions']:7.0f}" for k, d in row["rules"].items()) + f" | {time.time()-t0:.0f}s", flush=True)
        # the no-effect case, for the error rates of each rule
        m = worst != P.a; cp = ca = P.ca[m]
        row = dict(world=world, policy="no effect: changes picks but not costs", change_in_total_cost=0.0, change_in_typical_contested_decision=0.0, contested_share=float(m.mean()), rules={})
        for rule in ("typical", "total", "both"):
            r = [trial(rng, cp, ca, float(m.mean()), rule) for _ in range(RUNS)]; v = [a for a, _, _ in r]
            row["rules"][rule] = dict(adopted=v.count("adopted") / RUNS, stopped=v.count("stopped") / RUNS, rejected=v.count("rejected") / RUNS, median_decisions=float(np.median([b for _, b, _ in r])), mean_copies=float(np.mean([c for _, _, c in r])))
        out.append(row)
        print(f"[{world}] no effect: " + " | ".join(f"{k}: A {100*d['adopted']:5.1f} S {100*d['stopped']:5.1f} R {100*d['rejected']:5.1f}" for k, d in row["rules"].items()), flush=True)
        print(f"[{world}] gap policy: {gap_stats}")
    json.dump(out, open("w8e_results.json", "w"), indent=1)
