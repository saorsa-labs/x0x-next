"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W8d: three questions the band raises.

 A. Evidence. Experiment W3 gathered its evidence by making 10% of decisions at random among
    all candidates and never charged for it. What does each way of gathering evidence cost,
    and how good is a policy fitted from evidence gathered only inside the band?
 B. Widening. A band caps the gain as well as the harm. A policy that has been adopted at one
    width asks for the next width; that request is tried like any other change (the wider
    version against the narrower one, on the decisions where they differ). Does an honest
    policy get its room, and is a "sleeper" (honest inside the narrow band, as bad as possible
    in the ring just outside it) stopped?
 C. Capture. A hostile policy that could recognise an attacker's peers perfectly: what share of
    decisions can it steer to them inside the band?
"""
import json, time
import numpy as np
from safety_common import *
from exp_trials import summarise, RUNS


def log_band(rng, n, world, delta, explore=0.10):
    """follow the baseline rule; a share of decisions is made at random among the candidates
    inside the band (delta None: among all eligible; delta 'all': among all three, as W3 did)"""
    C = candidates(rng, n)
    elig = C[..., 2] >= 2
    sb = np.where(elig, np.asarray(baseline_rule(C.reshape(-1, 5)), float).reshape(n, 3), -1.0)
    a = sb.argmax(1)
    if delta == "all":
        mask = np.ones((n, 3), bool)
    else:
        mask = elig & (sb >= sb.max(1, keepdims=True) - (1e9 if delta is None else delta))
    rnd = (rng.random((n, 3)) * mask).argmax(1)
    use = (rng.random(n) < explore) & (mask.sum(1) > 0)
    pick = np.where(use, rnd, a)
    x = C[np.arange(n), pick]
    return x, true_cost(x, world) * rng.lognormal(0.0, NOISE, n)


if __name__ == "__main__":
    t0 = time.time(); out = dict(evidence=[], widening=[], capture=[])
    for world in WORLDS:
        P = Pool(600_000, world, seed=31)
        inb = lambda d: P.elig & (P.sb >= P.sb.max(1, keepdims=True) - (1e9 if d is None else d))
        # ---------------- A. evidence
        r = np.random.default_rng(5)
        def standing(mask, share=0.10):
            rnd = (r.random(P.sb.shape) * mask).argmax(1); use = (r.random(len(P.C)) < share) & (mask.sum(1) > 0)
            return P.harm(np.where(use, rnd, P.a))
        ways = (("10% at random among all candidates (as in W3)", "all", np.ones_like(P.elig)),
                ("10% at random among eligible candidates", None, P.elig),
                ("10% at random inside band 12", 12, inb(12)), ("10% at random inside band 6", 6, inb(6)), ("10% at random inside band 3", 3, inb(3)))
        for label, d, mask in ways:
            cost = standing(mask)
            for n in (2_000, 20_000, 200_000):
                g = {3: [], 6: [], None: []}
                for seed in range(5):
                    x, obs = log_band(np.random.default_rng(1000 * n + seed), n, world, d)
                    s = fast_scores(P, *fit_steps_ratio(x, obs, seed, 32))
                    for band in g:
                        g[band].append(P.harm(P.pick(s, band)))
                row = dict(world=world, evidence=label, logged_decisions=n, standing_cost_of_gathering=cost,
                           fitted_policy_inside_band_3=float(np.median(g[3])), fitted_policy_inside_band_6=float(np.median(g[6])),
                           fitted_policy_with_no_band=float(np.median(g[None])), worst_seed_with_no_band=float(np.max(g[None])))
                out["evidence"].append(row)
                print(f"[{world}] {label:46s} n={n:7d} | gathering costs {100*cost:+6.2f}% | fitted: band 3 {100*row['fitted_policy_inside_band_3']:+5.2f}%  band 6 {100*row['fitted_policy_inside_band_6']:+5.2f}%  no band {100*row['fitted_policy_with_no_band']:+5.2f}% (worst seed {100*row['worst_seed_with_no_band']:+5.2f}%) | {time.time()-t0:.0f}s", flush=True)
        # ---------------- B. widening
        rng = np.random.default_rng(77)
        x, obs = log_band(np.random.default_rng(424242), 20_000, world, 6)
        honest = fast_scores(P, *fit_steps_ratio(x, obs, 0, 32))
        widths = [3, 6, 9, 12, 18, None]
        prev = P.pick(honest, widths[0])
        for w_prev, w in zip(widths[:-1], widths[1:]):
            cur = P.pick(honest, w); m = cur != prev
            res = summarise(rng, P.cost_of(cur)[m], P.cost_of(prev)[m], float(m.mean()), "twin", 0.20, P.base)
            row = dict(world=world, policy="honest fit (evidence gathered inside band 6)", from_band=w_prev, to_band="none" if w is None else w,
                       cost_vs_fixed_rule_before=P.harm(prev), cost_vs_fixed_rule_after=P.harm(cur), differ_on=float(m.mean()), **res)
            out["widening"].append(row)
            print(f"[{world}] honest  {w_prev}->{w}: {100*P.harm(prev):+5.2f}% -> {100*P.harm(cur):+5.2f}% | differ on {100*m.mean():4.1f}% | adopt {100*res['adopted']:5.1f} stop {100*res['stopped']:5.1f} rej {100*res['rejected']:5.1f} | median {res['median_decisions_to_verdict']:.0f} dec, {res['mean_extra_messages']:.0f} copies", flush=True)
            prev = cur
        # sleeper: honest inside the narrow band, worst possible in the ring just outside
        for w_prev, w in ((3, 6), (6, 9), (9, 12)):
            narrow = P.pick(honest, w_prev)
            ring = inb(w) & ~inb(w_prev)
            worst_ring = np.where(ring, P.tc, -1.0).argmax(1)
            cur = np.where(ring.any(1), worst_ring, narrow); m = cur != narrow
            for label, mode, share in (("twin, 20% of the decisions where they differ are duplicated", "twin", 0.20), ("split, 10% of the decisions where they differ", "split", 0.10)):
                res = summarise(rng, P.cost_of(cur)[m], P.cost_of(narrow)[m], float(m.mean()), mode, share, P.base)
                row = dict(world=world, policy="sleeper: honest inside the narrow band, worst possible in the ring outside it", from_band=w_prev, to_band=w, trial=label,
                           cost_vs_fixed_rule_before=P.harm(narrow), cost_vs_fixed_rule_if_widened=P.harm(cur), differ_on=float(m.mean()), **res)
                out["widening"].append(row)
                print(f"[{world}] sleeper {w_prev}->{w} ({mode}): {100*P.harm(narrow):+5.2f}% -> {100*P.harm(cur):+5.2f}% | differ on {100*m.mean():4.1f}% | adopt {100*res['adopted']:5.1f} stop {100*res['stopped']:5.1f} | median {res['median_decisions_to_verdict']:.0f} dec | harm rate {100*res['mean_harm_rate_while_on_trial']:+.2f}% | excess {res['mean_excess_cost_decisions']:+.1f}", flush=True)
        # ---------------- C. capture
        rr = np.random.default_rng(11)
        for q in (0.05, 0.20):
            att = rr.random(P.sb.shape) < q                       # the attacker's peers, no better or worse than others
            fixed = float(att[np.arange(len(P.a)), P.a].mean())
            for d in (0, 3, 6, 9, 12, None):
                mask = inb(d)
                s = np.where(mask, att.astype(float), -1e18)       # prefers an attacker's peer whenever it may
                pk = P.pick(s, None)
                row = dict(world=world, attacker_share_of_peers=q, band_points="none" if d is None else d, share_of_decisions_under_the_fixed_rule=fixed,
                           share_of_decisions_under_a_hostile_policy=float(att[np.arange(len(pk)), pk].mean()), cost_of_that_policy=P.harm(pk))
                out["capture"].append(row)
                print(f"[{world}] attacker holds {100*q:.0f}% of peers | band {str(row['band_points']):>4} | fixed rule gives it {100*fixed:.1f}% of decisions | hostile policy {100*row['share_of_decisions_under_a_hostile_policy']:.1f}% | cost {100*row['cost_of_that_policy']:+.1f}%", flush=True)
    json.dump(out, open("w8d_results.json", "w"), indent=1)
