"""
THIRD CHECK -- reviewer's script. Not production code. Not part of x0x.

Is "tightening the ratio limit to 4 or 8 times is not enough" a fair test of Grok Heavy's proposal?
Grok proposed: "no PRIMARY step > 4-8x another". The study's rebuttal figures (18% at 4, 39% at 8)
come from runs where the DELIVERY feature (not a primary feature; the fixed rule does not price it
at all) is allowed steps R times the smallest primary step, and every worst patch found is
"delivery outranks latency and loss". Here the same search is run with the primary ratio at R and
delivery held to (a) at most the smallest primary step, (b) at most twice it, (c) not priced.
Also: verify with wnet.Steps.read_checks() that the study's 62.5% / 47.9% patches pass Version 3's limits.
"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(HERE, 'as_reviewed'))   # path made relative for the zip; was the reviewer's working folder
import numpy as np
from safety_common import *

VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
t0 = time.time()
out = dict(v3_patch_checks=[], searches=[])

# ---- 1. the stored worst patches against Version 3's own limit check
w8a = json.load(open(_os.path.join(HERE, 'as_reviewed', 'w8a_results.json')))
for r in w8a:
    if r["ratio_limit"] == 32:
        w = r["worst_weights"]; st = steps_from(w["rtt"], w["loss"], w["delivery"]); chk = st.read_checks()
        tab = st(GRID); lim = limit_report(tab)
        row = dict(world=r["world"], delivery_under_the_limit=r["delivery_under_the_limit"], reported_harm=r["worst_patch_harm"],
                   read_checks_within_limits=chk["within_limits"], monotone=chk["monotone"], primary_ratio=chk["largest_to_smallest_primary_step"],
                   table_limit_report_within_limits=lim["within_limits"], table_monotone_violations=len(monotone_violations(tab)), table_gate=gate_violations(tab))
        out["v3_patch_checks"].append(row)
        print(f"[{r['world']}] stored worst patch (delivery limited={r['delivery_under_the_limit']}): harm {100*r['worst_patch_harm']:+.1f}% | Steps.read_checks within_limits={chk['within_limits']} "
              f"(monotone {chk['monotone']}, primary ratio {chk['largest_to_smallest_primary_step']:.0f}) | table limit_report within_limits={lim['within_limits']}, wrong-way pairs {row['table_monotone_violations']}, gate {row['table_gate']}", flush=True)


def worst_patch(pool, R, deliv_cap, restarts=6, seed=0):
    """deliv_cap: multiple of the smallest primary step that a delivery step may reach (0 = not priced)"""
    rng = np.random.default_rng(seed)
    prim_vals = [v for v in VALUES if v <= R]
    del_vals = [0] + [v for v in VALUES if v <= R * max(deliv_cap, 0)] if deliv_cap > 0 else [0]
    best = (-1, None)
    def ok(w):
        lo = min(w[0].min(), w[1].min()); hi = max(w[0].max(), w[1].max())
        return hi <= R * lo and w[2].max() <= deliv_cap * lo
    for r in range(restarts):
        w = [rng.choice(prim_vals, 15), rng.choice(prim_vals, 7), np.zeros(5, int)]
        if r == 0:
            w = [np.ones(15, int), np.full(7, max(prim_vals)), np.zeros(5, int)]
        if r == 1:
            w = [np.full(15, max(prim_vals)), np.ones(7, int), np.zeros(5, int)]
        if not ok(w):
            w = [np.ones(15, int), np.ones(7, int), np.zeros(5, int)]
        cur = pool.harm(pool.pick(fast_scores(pool, *w)))
        improved = True
        while improved:
            improved = False
            for part, vals in ((0, prim_vals), (1, prim_vals), (2, del_vals)):
                for i in range(len(w[part])):
                    keep = w[part][i]
                    for v in vals:
                        if v == keep: continue
                        w[part][i] = v
                        if not ok(w): continue
                        h = pool.harm(pool.pick(fast_scores(pool, *w)))
                        if h > cur + 1e-9: cur, keep, improved = h, v, True
                    w[part][i] = keep
        if cur > best[0]:
            best = (cur, [a.copy() for a in w])
    return best


for world in WORLDS:
    search = Pool(40_000, world, seed=11); test = Pool(400_000, world, seed=12)
    for deliv_cap, label in ((1, "delivery step <= smallest primary step"), (2, "delivery step <= 2 x smallest primary step"), (0, "delivery not priced")):
        for R in (32, 8, 4, 2):
            h, w = worst_patch(search, R, deliv_cap)
            ht = test.harm(test.pick(fast_scores(test, *w)))
            gains = []
            for seed in range(5):
                x, obs = log_evidence(np.random.default_rng(200000 + seed), 20_000, world)
                wr, wl, wd = fit_steps_ratio(x, obs, seed, R, limit_delivery=False)
                lo = int(min(wr.min(), wl.min())); wd = np.minimum(wd, deliv_cap * lo)
                gains.append(test.harm(test.pick(fast_scores(test, wr, wl, wd))))
            row = dict(world=world, primary_ratio_limit=R, delivery_rule=label, worst_patch_harm=ht, honest_fit_change=float(np.median(gains)),
                       worst_weights=dict(rtt=[int(v) for v in w[0]], loss=[int(v) for v in w[1]], delivery=[int(v) for v in w[2]]))
            out["searches"].append(row)
            print(f"[{world}] primary ratio {R:<2d} | {label:42s} | worst patch found {100*ht:+6.1f}% | honest fit {100*row['honest_fit_change']:+5.2f}% | rtt {row['worst_weights']['rtt']} loss {row['worst_weights']['loss']} deliv {row['worst_weights']['delivery']} | {time.time()-t0:.0f}s", flush=True)
json.dump(out, open(_os.path.join(HERE, 'check_ratio_limit_results.json'), 'w'), indent=1)
