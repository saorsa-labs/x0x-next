"""
THIRD CHECK -- reviewer's attack script. Not production code. Not part of x0x.

 1. "The band applies to every single decision and not to an average": the per-decision cost ratio
    of the worst in-band option against the fixed rule's pick, inside a band of 3.
 2. The watch against SMALL harms. The study measured the watch only on the worst policy the band
    allows (the easiest to see) and reported "total harm before removal ... 45 to 246 ordinary
    decisions' worth of cost in every case". An adopted policy that turns only mildly bad:
    how often is it removed within the watch's 4,096 control samples, and what does it cost?
 3. Copies treated better than originals: by how much must the peers a policy favours speed up
    COPIES for the worst policy the band allows to be adopted?
 4. Trying again: how often is a slightly harmful policy adopted by chance, per trial?
"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(HERE, 'as_reviewed'))   # path made relative for the zip; was the reviewer's working folder
import numpy as np
from safety_common import *
import exp_trials
from exp_trials import summarise, summarise_watch, run_trial, LOOKS, Z_ADOPT, Z_STOP, MIN_ADOPT

t0 = time.time(); out = dict(per_decision=[], watch_small_harm=[], copies=[], retries=[])
rng = np.random.default_rng(4321)
for world in WORLDS:
    P = Pool(600_000, world, seed=31); idx = np.arange(len(P.a))
    inb = P.elig & (P.sb >= P.sb.max(1, keepdims=True) - 3)
    worst = np.where(inb, np.where(P.elig, P.tc, -1.0), -1.0).argmax(1)
    m = worst != P.a
    ratio = P.tc[idx, worst][m] / P.ca[m]
    row = dict(world=world, share_of_decisions_contested=float(m.mean()), mean_harm_all_decisions=P.harm(worst),
               per_contested_decision_cost_ratio=dict(median=float(np.median(ratio)), p90=float(np.quantile(ratio, .9)), p99=float(np.quantile(ratio, .99)), max=float(ratio.max())),
               share_of_contested_decisions_at_least_50_percent_worse=float((ratio >= 1.5).mean()), share_at_least_twice_the_cost=float((ratio >= 2.0).mean()))
    out["per_decision"].append(row)
    print(f"1 [{world}] band 3, worst in-band option against the rule's pick, per contested decision: median x{row['per_contested_decision_cost_ratio']['median']:.2f}, "
          f"90th pct x{row['per_contested_decision_cost_ratio']['p90']:.2f}, 99th x{row['per_contested_decision_cost_ratio']['p99']:.2f}, max x{row['per_contested_decision_cost_ratio']['max']:.2f}; "
          f"{100*row['share_of_contested_decisions_at_least_50_percent_worse']:.0f}% of contested decisions cost at least 50% more, {100*row['share_at_least_twice_the_cost']:.0f}% at least double", flush=True)

    # ---- 2. the watch against mild harm: the adversary takes the in-band option that is worse, but by no more than a cap
    for cap in (1.03, 1.06, 1.10, 1.20, 1.50, None):
        allowed = inb & (P.tc > P.ca[:, None]) & ((P.tc <= P.ca[:, None] * cap) if cap else True)
        pk = np.where(allowed.any(1), np.where(allowed, P.tc, -1.0).argmax(1), P.a)
        mm = pk != P.a
        if mm.mean() < 1e-4: continue
        cp, ca, rho = P.cost_of(pk)[mm], P.ca[mm], float(mm.mean())
        for label, mode, control in (("one contested decision in ten stays with the fixed rule", "split", 0.10), ("one in ten sent both ways", "twin", 0.10), ("three in ten sent both ways", "twin", 0.30)):
            res = summarise_watch(rng, cp, ca, rho, mode, control, P.base)
            horizon = (int(LOOKS[-1] / control * 1.25) + 64) / rho
            row = dict(world=world, per_decision_cap="none (worst in band)" if cap is None else cap, harm_while_it_runs=P.harm(pk), contested_share=rho, watch=label,
                       removed_within_the_watch_window=res["reverted"], median_decisions_until_removed_or_window_end=res["median_decisions_to_revert"],
                       mean_total_harm_in_ordinary_decisions=res["mean_excess_cost_decisions"], window_decisions=horizon)
            out["watch_small_harm"].append(row)
            print(f"2 [{world}] turns mildly bad (each pick at most {('x%.2f' % cap) if cap else 'any amount'} worse): harm {100*row['harm_while_it_runs']:+.2f}% on {100*rho:.1f}% of decisions | watch: {label:55s} | "
                  f"removed in {100*res['reverted']:5.1f}% of 400 within {horizon:,.0f} decisions | median {res['median_decisions_to_revert']:,.0f} dec | total harm {res['mean_excess_cost_decisions']:+.0f} decisions' worth", flush=True)

    # ---- 3. copies treated better than originals (trial on copies, worst policy the band allows)
    cp, ca, rho = P.cost_of(worst)[m], P.ca[m], float(m.mean())
    for speedup in (1.0, 0.9, 0.8, 0.75, 0.7, 0.6):
        res = summarise(rng, cp * speedup, ca, rho, "twin", 0.20, P.base)
        out["copies"].append(dict(world=world, policy="worst possible inside band 3", true_harm_if_adopted=P.harm(worst), copies_served_at=speedup, adopted=res["adopted"], stopped=res["stopped"], no_verdict=res["rejected"], median_decisions=res["median_decisions_to_verdict"]))
        print(f"3 [{world}] worst policy the band allows (true harm {100*P.harm(worst):+.1f}%), peers serve its COPIES at x{speedup:.2f} of the true cost: adopted {100*res['adopted']:5.1f}% | stopped {100*res['stopped']:5.1f}% | no verdict {100*res['rejected']:5.1f}% | median {res['median_decisions_to_verdict']:,.0f} dec", flush=True)

    # ---- 4. trying again: per-trial chance that a mildly harmful policy is adopted (4,000 trials each)
    exp_trials.RUNS = 4000
    for cap in (1.03, 1.06):
        allowed = inb & (P.tc > P.ca[:, None]) & (P.tc <= P.ca[:, None] * cap)
        pk = np.where(allowed.any(1), np.where(allowed, P.tc, -1.0).argmax(1), P.a); mm = pk != P.a
        res = exp_trials.summarise(rng, P.cost_of(pk)[mm], P.ca[mm], float(mm.mean()), "twin", 0.20, P.base)
        out["retries"].append(dict(world=world, per_decision_cap=cap, true_harm=P.harm(pk), adopted_per_trial=res["adopted"], stopped=res["stopped"], no_verdict=res["rejected"], trials=4000))
        print(f"4 [{world}] mildly harmful policy (each pick at most x{cap:.2f} worse, true harm {100*P.harm(pk):+.2f}%): adopted in {100*res['adopted']:.2f}% of 4,000 trials on copies, stopped {100*res['stopped']:.1f}%, no verdict {100*res['rejected']:.1f}%", flush=True)
    mm = worst != P.a
    res = exp_trials.summarise(rng, P.ca[mm], P.ca[mm], float(mm.mean()), "twin", 0.20, P.base)
    out["retries"].append(dict(world=world, per_decision_cap="no effect", true_harm=0.0, adopted_per_trial=res["adopted"], stopped=res["stopped"], no_verdict=res["rejected"], trials=4000))
    print(f"4 [{world}] policy with no effect: adopted in {100*res['adopted']:.2f}% of 4,000 trials on copies ({time.time()-t0:.0f}s)", flush=True)
    exp_trials.RUNS = 400
json.dump(out, open(_os.path.join(HERE, 'attack_watch_and_copies_results.json'), 'w'), indent=1)
