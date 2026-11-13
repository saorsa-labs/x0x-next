"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W8b: the band. The kernel lets a policy choose only among candidates that the
anchor (here the baseline rule) rates within delta points of its own best. How much can ANY
policy then cost, whatever its weights or form, and how much of the available gain is left?

  worst possible : an adversary that knows the true costs picks the worst candidate in the band
  best possible  : an oracle picks the best candidate in the band
Named policies are run inside the same band. Baseline points: 3 per rtt level, 4 per loss level.
The figures belong to these two worlds, where candidates are spread evenly over every rtt level.
exp_calibrate.py shows how they move with the mix of candidates.
"""
import json
import numpy as np
from safety_common import *

if __name__ == "__main__":
    out = []
    w8a = json.load(open("w8a_results.json"))
    for world in WORLDS:
        P = Pool(600_000, world, seed=31)        # the same pool as exp_trials.py, so the figures agree
        base_net = write_baseline()
        x, obs = log_evidence(np.random.default_rng(200000), 20_000, world)
        honest = fast_scores(P, *fit_steps_ratio(x, obs, 0, 32))
        ww = next(r for r in w8a if r["world"] == world and r["ratio_limit"] == 32 and not r["delivery_under_the_limit"])["worst_weights"]
        named = {
            "honest fit (20,000 logged decisions)": honest,
            "hostile: one loss step priced at 32 rtt steps (Version 3's worst)": P.scores(biased_inside_limits(base_net)),
            "hostile: worst step-weight patch inside Version 3's limits": fast_scores(P, ww["rtt"], ww["loss"], ww["delivery"]),
            "hostile: latency and loss stop mattering": P.scores(hostile_flatten(base_net)),
            "hostile: top age with top balance wins (reviewer's club)": P.scores(hostile_club(base_net)),
        }
        big = np.where(P.elig, P.tc, -1.0); small = np.where(P.elig, P.tc, 1e18)
        for delta in (0, 3, 6, 9, 12, 18, 24, 40, None):
            inband = P.elig & (P.sb >= P.sb.max(1, keepdims=True) - (1e9 if delta is None else delta))
            worst = P.harm(np.where(inband, big, -1.0).argmax(1)); best = P.harm(np.where(inband, small, 1e18).argmin(1))
            row = dict(world=world, band_points="none" if delta is None else delta, decisions_with_a_choice=float((inband.sum(1) >= 2).mean()),
                       worst_possible_any_policy=worst, best_possible_any_policy=best, named={})
            for k, s in named.items():
                pk = P.pick(s, delta)
                row["named"][k] = dict(change_in_cost=P.harm(pk), contested_share=float((pk != P.a).mean()))
            out.append(row)
            print(f"[{world}] band {str(row['band_points']):>4}: choice in {100*row['decisions_with_a_choice']:5.1f}% | worst any {100*worst:+6.1f}% | best any {100*best:+5.2f}% | "
                  + " | ".join(f"{100*v['change_in_cost']:+5.1f}% ({100*v['contested_share']:.0f}%)" for v in row["named"].values()))
    json.dump(out, open("w8b_results.json", "w"), indent=1)
