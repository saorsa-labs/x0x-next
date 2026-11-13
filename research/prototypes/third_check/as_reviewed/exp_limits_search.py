"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W8a: how much harm can a step-weight patch do while passing the static limits?

An adversary searches the step weights (coordinate ascent, several restarts) for the policy
that costs the node most, subject to the limits: every weight has its declared sign, every
rtt and loss step counts, and no step is worth more than R times the smallest. Two regimes:
  as shipped in Version 3 : delivery steps had a direction but no ratio limit
  tightened               : every priced feature is under the same ratio limit
For each R the same limit is applied to an honest fit, to see what the limit costs.
"""
import json, time
import numpy as np
from safety_common import *

VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 512, 4096]


def worst_patch(pool, R, limit_delivery, restarts=6, seed=0):
    rng = np.random.default_rng(seed)
    prim_vals = [v for v in VALUES if v <= R]
    del_vals = [0] + ([v for v in VALUES if v <= R] if limit_delivery else VALUES)
    best = (-1, None)
    for r in range(restarts):
        w = [rng.choice(prim_vals, 15), rng.choice(prim_vals, 7), rng.choice(del_vals, 5)]
        if r == 0:
            w = [np.ones(15, int), np.full(7, max(prim_vals)), np.zeros(5, int)]
        cur = pool.harm(pool.pick(fast_scores(pool, *w)))
        improved = True
        while improved:
            improved = False
            for part, vals in ((0, prim_vals), (1, prim_vals), (2, del_vals)):
                for i in range(len(w[part])):
                    keep = w[part][i]
                    for v in vals:
                        if v == keep:
                            continue
                        w[part][i] = v
                        if min(w[0].min(), w[1].min()) != 1 and max(w[0].max(), w[1].max()) > R * min(w[0].min(), w[1].min()):
                            continue
                        h = pool.harm(pool.pick(fast_scores(pool, *w)))
                        if h > cur + 1e-9:
                            cur, keep, improved = h, v, True
                    w[part][i] = keep
        if cur > best[0]:
            best = (cur, [a.copy() for a in w])
    return best


if __name__ == "__main__":
    t0 = time.time()
    out = []
    for world in WORLDS:
        search = Pool(40_000, world, seed=11); test = Pool(400_000, world, seed=12)
        oracle = test.harm(test.tc.__add__(np.where(test.elig, 0, 1e9)).argmin(1))
        fits = {}
        for R in (32, 16, 8, 4, 2):
            gains = []
            for seed in range(5):
                x, obs = log_evidence(np.random.default_rng(200000 + seed), 20_000, world)
                wr, wl, wd = fit_steps_ratio(x, obs, seed, R)
                gains.append(test.harm(test.pick(fast_scores(test, wr, wl, wd))))
            fits[R] = float(np.median(gains))
        for limit_delivery in (False, True):
            for R in (32, 16, 8, 4, 2):
                h, w = worst_patch(search, R, limit_delivery)
                ht = test.harm(test.pick(fast_scores(test, *w)))
                row = dict(world=world, ratio_limit=R, delivery_under_the_limit=limit_delivery, worst_patch_harm=ht,
                           honest_fit_change=fits[R], oracle_change=oracle,
                           worst_weights=dict(rtt=[int(v) for v in w[0]], loss=[int(v) for v in w[1]], delivery=[int(v) for v in w[2]]))
                out.append(row)
                print(f"[{world}] R={R:<3d} delivery limited={str(limit_delivery):5s} worst patch {100*ht:+6.1f}% | honest fit {100*fits[R]:+5.2f}% | oracle {100*oracle:+5.2f}% | {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open("w8a_results.json", "w"), indent=1)
