"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W8c: how long can a harmful patch run, and what does it cost before it is caught?

Three ways of trying a candidate policy against the anchor (the baseline rule), all sequential:
  old   : Version 3's trial. The candidate decides 10% of ALL decisions; outcomes of the two
          arms are compared over all decisions. No band.
  split : the candidate is held inside the band and decides 10% of the CONTESTED decisions
          (those where it would choose differently from the anchor). Only contested decisions
          are compared: the others carry no information.
  twin  : as split, but the candidate's choice is tried on a duplicate of the message while the
          user's message still goes the anchor's way. The pair is compared. No user-facing harm;
          the cost is one extra message for each duplicated decision.
At each look (8, 16, 32 ... 4,096 samples) two things are compared: the typical outcome (the
mean of the logarithm) and the total (the mean of the raw outcome). The candidate is STOPPED if
it is worse on EITHER with z <= -2.33, ADOPTED if it is better on BOTH with z >= 3.09 on at least
64 samples, and REJECTED if no verdict by the last look. Testing both closes a gap that W8e shows:
a policy can improve the typical decision while making the total worse.
Version 3's rule ("old") is kept as it was: logarithms only, all decisions compared.
Then the watch: an adopted policy that turns harmful (a shift in traffic, or a dormant
trigger) is compared with a control share of contested decisions and reverted.
Outcomes are true cost times lognormal noise (sigma 0.5). Costs are in units of one ordinary
decision's cost under the anchor. 400 runs per case.
"""
import json, time
import numpy as np
from safety_common import *

LOOKS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
Z_ADOPT, Z_STOP = 3.09, 2.33
MIN_ADOPT = 64        # no adoption on fewer samples: with 8 or 16 the z test is too generous
RUNS = 400


def run_trial(rng, cp, ca, rho, mode, share, base, both=True):
    """cp, ca: true costs of the candidate's and the anchor's pick on the decisions being compared.
    rho: share of all decisions that are being compared. Returns verdict, decisions elapsed,
    excess cost paid by the user (decision-costs), extra messages sent."""
    K = int(LOOKS[-1] / share * 1.25) + 64
    idx = rng.integers(0, len(cp), K)
    tp, ta = cp[idx], ca[idx]
    treat = rng.random(K) < share
    op = np.log(tp * rng.lognormal(0, NOISE, K)); oa = np.log(ta * rng.lognormal(0, NOISE, K))
    nt = np.cumsum(treat)
    verdict, at = "rejected", K - 1
    if mode == "twin":
        d = np.where(treat, op - oa, 0.0); dr = np.where(treat, np.exp(op) - np.exp(oa), 0.0)
        s1, s2, r1, r2 = np.cumsum(d), np.cumsum(d * d), np.cumsum(dr), np.cumsum(dr * dr)
        for n in LOOKS:
            j = int(np.searchsorted(nt, n))
            if j >= K:
                break
            m = s1[j] / n; v = max(s2[j] / n - m * m, 1e-12)
            z = -m / np.sqrt(v / n)
            mr = r1[j] / n; vr = max(r2[j] / n - mr * mr, 1e-12)
            zr = -mr / np.sqrt(vr / n)
            if z <= -Z_STOP or zr <= -Z_STOP: verdict, at = "stopped", j; break
            if z >= Z_ADOPT and zr >= Z_ADOPT and n >= MIN_ADOPT: verdict, at = "adopted", j; break
            at = j
        excess = 0.0
        extra = int(nt[at])
    else:
        nc = np.cumsum(~treat)
        def sums(a_t, a_c):
            xt = np.where(treat, a_t, 0.0); xc = np.where(~treat, a_c, 0.0)
            return np.cumsum(xt), np.cumsum(xt * xt), np.cumsum(xc), np.cumsum(xc * xc)
        def zof(S, j, n):
            st1, st2, sc1, sc2 = S
            mt, mc = st1[j] / n, sc1[j] / nc[j]
            vt = max(st2[j] / n - mt * mt, 1e-12); vc = max(sc2[j] / nc[j] - mc * mc, 1e-12)
            return (mc - mt) / np.sqrt(vt / n + vc / nc[j])
        SL = sums(op, oa); SR = sums(np.exp(op), np.exp(oa))
        for n in LOOKS:
            j = int(np.searchsorted(nt, n))
            if j >= K or nc[j] < 8:
                if j >= K: break
                continue
            z = zof(SL, j, n); zr = zof(SR, j, n) if both else z
            if z <= -Z_STOP or zr <= -Z_STOP: verdict, at = "stopped", j; break
            if z >= Z_ADOPT and zr >= Z_ADOPT and n >= MIN_ADOPT: verdict, at = "adopted", j; break
            at = j
        excess = float(((tp - ta) * treat)[: at + 1].sum() / base)
        extra = 0
    return verdict, (at + 1) / rho, excess, extra


def summarise(rng, cp, ca, rho, mode, share, base, both=True):
    r = [run_trial(rng, cp, ca, rho, mode, share, base, both) for _ in range(RUNS)]
    v = [x[0] for x in r]; dec = np.array([x[1] for x in r]); ex = np.array([x[2] for x in r]); extra = np.array([x[3] for x in r])
    return dict(adopted=v.count("adopted") / RUNS, stopped=v.count("stopped") / RUNS, rejected=v.count("rejected") / RUNS,
                median_decisions_to_verdict=float(np.median(dec)), p90_decisions_to_verdict=float(np.quantile(dec, 0.9)),
                mean_excess_cost_decisions=float(ex.mean()), worst_run_excess_cost_decisions=float(ex.max()),
                mean_harm_rate_while_on_trial=float((ex / dec).mean()), mean_extra_messages=float(extra.mean()))


def run_watch(rng, cp, ca, rho, mode, control, base):
    """the candidate decides every contested decision except a control share; revert when the
    control (or the duplicate) shows it is worse"""
    K = int(LOOKS[-1] / control * 1.25) + 64
    idx = rng.integers(0, len(cp), K)
    tp, ta = cp[idx], ca[idx]
    ctl = rng.random(K) < control
    op = np.log(tp * rng.lognormal(0, NOISE, K)); oa = np.log(ta * rng.lognormal(0, NOISE, K))
    nc = np.cumsum(ctl)
    at, verdict = K - 1, "missed"
    if mode == "twin":
        d = np.where(ctl, op - oa, 0.0); s1, s2 = np.cumsum(d), np.cumsum(d * d)
        dr = np.where(ctl, np.exp(op) - np.exp(oa), 0.0); r1, r2 = np.cumsum(dr), np.cumsum(dr * dr)
        for n in LOOKS:
            j = int(np.searchsorted(nc, n))
            if j >= K: break
            m = s1[j] / n; v = max(s2[j] / n - m * m, 1e-12)
            mr = r1[j] / n; vr = max(r2[j] / n - mr * mr, 1e-12)
            if -m / np.sqrt(v / n) <= -Z_STOP or -mr / np.sqrt(vr / n) <= -Z_STOP: verdict, at = "reverted", j; break
        excess = float((tp - ta)[: at + 1].sum() / base)
    else:
        ntr = np.cumsum(~ctl)
        def sums(a_t, a_c):
            xt = np.where(~ctl, a_t, 0.0); xc = np.where(ctl, a_c, 0.0)
            return np.cumsum(xt), np.cumsum(xt * xt), np.cumsum(xc), np.cumsum(xc * xc)
        def zof(S, j, n):
            st1, st2, sc1, sc2 = S
            mt, mc = st1[j] / ntr[j], sc1[j] / n
            vt = max(st2[j] / ntr[j] - mt * mt, 1e-12); vc = max(sc2[j] / n - mc * mc, 1e-12)
            return (mc - mt) / np.sqrt(vt / ntr[j] + vc / n)
        SL = sums(op, oa); SR = sums(np.exp(op), np.exp(oa))
        for n in LOOKS:
            j = int(np.searchsorted(nc, n))
            if j >= K: break
            if ntr[j] < 8: continue
            if zof(SL, j, n) <= -Z_STOP or zof(SR, j, n) <= -Z_STOP: verdict, at = "reverted", j; break
        excess = float(((tp - ta) * ~ctl)[: at + 1].sum() / base)
    return verdict, (at + 1) / rho, excess


def summarise_watch(rng, cp, ca, rho, mode, control, base):
    r = [run_watch(rng, cp, ca, rho, mode, control, base) for _ in range(RUNS)]
    dec = np.array([x[1] for x in r]); ex = np.array([x[2] for x in r])
    return dict(reverted=sum(x[0] == "reverted" for x in r) / RUNS, median_decisions_to_revert=float(np.median(dec)),
                p90_decisions_to_revert=float(np.quantile(dec, 0.9)), mean_excess_cost_decisions=float(ex.mean()),
                mean_harm_rate_until_reverted=float((ex / dec).mean()))


if __name__ == "__main__":
    t0 = time.time(); rng = np.random.default_rng(88)
    w8a = json.load(open("w8a_results.json"))
    out = dict(trials=[], watch=[], exploration=[])
    for world in WORLDS:
        P = Pool(600_000, world, seed=31)
        base_net = write_baseline()
        x, obs = log_evidence(np.random.default_rng(200000), 20_000, world)
        ww = next(r for r in w8a if r["world"] == world and r["ratio_limit"] == 32 and not r["delivery_under_the_limit"])["worst_weights"]
        big = np.where(P.elig, P.tc, -1.0)
        def worst_in(delta):
            inband = P.elig & (P.sb >= P.sb.max(1, keepdims=True) - (1e9 if delta is None else delta))
            return np.where(inband, big, -1.0)                     # scores = true cost: picks the worst allowed
        cands = {
            "honest fit": fast_scores(P, *fit_steps_ratio(x, obs, 0, 32)),
            "hostile: Version 3's worst (one loss step = 32 rtt steps)": P.scores(biased_inside_limits(base_net)),
            "hostile: worst step-weight patch inside Version 3's limits": fast_scores(P, ww["rtt"], ww["loss"], ww["delivery"]),
            "hostile: worst possible policy of any form": None,
            "careless: picks at random among the candidates it may choose": "random",
            "no effect: changes picks but not costs (measures the error rates)": "null",
        }
        for name, s in cands.items():
            for label, mode, delta, share in (("old: 10% of all decisions, no band", "old", None, 0.10),
                                              ("split: band 3, 10% of contested decisions", "split", 3, 0.10),
                                              ("twin: band 3, 20% of contested decisions duplicated", "twin", 3, 0.20),
                                              ("split: band 6, 10% of contested decisions", "split", 6, 0.10),
                                              ("twin: band 6, 20% of contested decisions duplicated", "twin", 6, 0.20)):
                if isinstance(s, str) and s == "random":
                    inb = P.elig & (P.sb >= P.sb.max(1, keepdims=True) - (1e9 if delta is None else delta))
                    sc = np.where(inb, np.random.default_rng(9).random(P.sb.shape), -1e18)
                elif isinstance(s, str) and s == "null":
                    sc = worst_in(delta)
                else:
                    sc = worst_in(delta) if s is None else s
                pk = P.pick(sc, delta)
                full = P.harm(pk)
                if isinstance(s, str) and s == "null":
                    m = pk != P.a
                    cp = ca = P.ca[m]; rho = float(m.mean()); full = 0.0
                    res = summarise(rng, cp, ca, 1.0 if mode == "old" else rho, "split" if mode == "old" else mode, share, P.base, both=(mode != "old"))
                elif mode == "old":
                    cp, ca, rho = P.cost_of(pk), P.ca, 1.0
                    res = summarise(rng, cp, ca, rho, "split", share, P.base, both=False)
                else:
                    m = pk != P.a
                    if m.mean() < 1e-4:
                        continue
                    cp, ca, rho = P.cost_of(pk)[m], P.ca[m], float(m.mean())
                    res = summarise(rng, cp, ca, rho, mode, share, P.base)
                row = dict(world=world, candidate=name, trial=label, change_in_cost_if_it_decided_everything_it_may=full, contested_share=float((pk != P.a).mean()), **res)
                out["trials"].append(row)
                print(f"[{world}] {name[:44]:44s} | {label[:34]:34s} | full {100*full:+6.1f}% | adopt {100*res['adopted']:3.0f}% stop {100*res['stopped']:3.0f}% rej {100*res['rejected']:3.0f}% | "
                      f"median {res['median_decisions_to_verdict']:8.0f} dec | harm rate {100*res['mean_harm_rate_while_on_trial']:+5.2f}% | excess {res['mean_excess_cost_decisions']:+6.1f} | extra msgs {res['mean_extra_messages']:.0f} | {time.time()-t0:.0f}s", flush=True)
        # the watch: an adopted policy turns into the worst the band allows; and, for scale, the same with no band
        hf = fast_scores(P, *fit_steps_ratio(x, obs, 0, 32))
        for name, s, delta in (("honest fit inside band 3 (a revert here is a mistake)", hf, 3), ("honest fit inside band 6 (a revert here is a mistake)", hf, 6),
                               ("worst possible inside band 3", None, 3), ("worst possible inside band 6", None, 6),
                               ("Version 3's worst, no band", P.scores(biased_inside_limits(base_net)), None),
                               ("worst inside Version 3's limits, no band", fast_scores(P, ww["rtt"], ww["loss"], ww["delivery"]), None)):
            sc = worst_in(delta) if s is None else s
            pk = P.pick(sc, delta); m = pk != P.a
            for label, mode, control in (("control: 10% of contested decisions stay with the anchor", "split", 0.10),
                                         ("twin: 10% of contested decisions also sent the anchor's way", "twin", 0.10),
                                         ("twin: 30%", "twin", 0.30)):
                res = summarise_watch(rng, P.cost_of(pk)[m], P.ca[m], float(m.mean()), mode, control, P.base)
                out["watch"].append(dict(world=world, policy=name, watch=label, harm_while_it_runs=P.harm(pk), **res))
                print(f"[{world}] WATCH {name[:40]:40s} | {label[:28]:28s} | harm {100*P.harm(pk):+6.1f}% | reverted {100*res['reverted']:3.0f}% | median {res['median_decisions_to_revert']:7.0f} dec | excess {res['mean_excess_cost_decisions']:+7.1f}", flush=True)
        # ---- the three further cases below were first run by the third independent check, against the first trial rule
        inb3 = P.elig & (P.sb >= P.sb.max(1, keepdims=True) - 3)
        worst3 = np.where(inb3, big, -1.0).argmax(1)
        # (a) the watch against a policy that turns only mildly bad: each pick worse, but by no more than a cap
        for cap in (1.06, 1.10, 1.20, 1.50, None):
            allowed = inb3 & (P.tc > P.ca[:, None]) & ((P.tc <= P.ca[:, None] * cap) if cap else True)
            pk = np.where(allowed.any(1), np.where(allowed, P.tc, -1.0).argmax(1), P.a); m = pk != P.a
            for label, mode, control in (("control: 10% of contested decisions stay with the anchor", "split", 0.10), ("twin: 10% of contested decisions also sent the anchor's way", "twin", 0.10), ("twin: 30%", "twin", 0.30)):
                res = summarise_watch(rng, P.cost_of(pk)[m], P.ca[m], float(m.mean()), mode, control, P.base)
                out.setdefault("watch_mild_harm", []).append(dict(world=world, each_pick_at_most="any amount worse" if cap is None else f"{cap:.2f} times the cost", watch=label,
                                                                   harm_while_it_runs=P.harm(pk), contested_share=float(m.mean()), **res))
                print(f"[{world}] MILD each pick at most {str(cap):5s} | {label[:26]:26s} | harm {100*P.harm(pk):+5.2f}% | removed {100*res['reverted']:5.1f}% | median {res['median_decisions_to_revert']:9.0f} dec | total harm {res['mean_excess_cost_decisions']:+7.1f}", flush=True)
        # (b) peers that serve COPIES faster than originals: the worst policy the band allows, on trial on copies
        m = worst3 != P.a
        for speed in (1.0, 0.9, 0.8, 0.7, 0.6):
            res = summarise(rng, P.cost_of(worst3)[m] * speed, P.ca[m], float(m.mean()), "twin", 0.20, P.base)
            out.setdefault("copies_served_faster", []).append(dict(world=world, copies_cost=speed, true_harm_if_adopted=P.harm(worst3), **res))
            print(f"[{world}] COPIES served at {speed:.1f} of the true cost | adopted {100*res['adopted']:5.1f}% stopped {100*res['stopped']:5.1f}% no verdict {100*res['rejected']:5.1f}%", flush=True)
        # (c) trying again: the chance per trial that a policy with no effect, or a very slightly harmful one, is adopted
        import exp_trials as _self
        keep_runs = _self.RUNS; _self.RUNS = 4000
        for cap in (None, 1.03, 1.06):
            if cap is None:
                cp_ = ca_ = P.ca[m]; rho_ = float(m.mean()); harm_ = 0.0
            else:
                allowed = inb3 & (P.tc > P.ca[:, None]) & (P.tc <= P.ca[:, None] * cap)
                pk = np.where(allowed.any(1), np.where(allowed, P.tc, -1.0).argmax(1), P.a); mm = pk != P.a
                cp_, ca_, rho_, harm_ = P.cost_of(pk)[mm], P.ca[mm], float(mm.mean()), P.harm(pk)
            res = _self.summarise(rng, cp_, ca_, rho_, "twin", 0.20, P.base)
            out.setdefault("repeated_tries", []).append(dict(world=world, policy="no effect" if cap is None else f"each pick at most {cap:.2f} times the cost", true_harm=harm_, trials=4000,
                                                              adopted=res["adopted"], stopped=res["stopped"], rejected=res["rejected"]))
            print(f"[{world}] TRIES {'no effect' if cap is None else cap}: true harm {100*harm_:+.3f}% | adopted in {100*res['adopted']:.2f}% of 4,000 trials", flush=True)
        _self.RUNS = keep_runs
        # what exploration costs: a share of decisions made at random
        n = len(P.C); r = np.random.default_rng(5)
        def explore(share, mask):
            w = r.random((n, 3)) * mask; rnd = w.argmax(1)
            use = (r.random(n) < share) & (mask.sum(1) > 0)
            return P.harm(np.where(use, rnd, P.a))
        allc = np.ones((n, 3), bool)
        for label, share, mask in (("10% of decisions at random among all candidates (as in experiment W3)", 0.10, allc),
                                   ("10% at random among eligible candidates", 0.10, P.elig),
                                   ("2% at random among eligible candidates", 0.02, P.elig),
                                   ("10% at random inside band 3", 0.10, P.elig & (P.sb >= P.sb.max(1, keepdims=True) - 3)),
                                   ("10% at random inside band 6", 0.10, P.elig & (P.sb >= P.sb.max(1, keepdims=True) - 6))):
            h = explore(share, mask)
            out["exploration"].append(dict(world=world, exploration=label, standing_cost=h))
            print(f"[{world}] EXPLORE {label}: {100*h:+.2f}%", flush=True)
    json.dump(out, open("w8c_results.json", "w"), indent=1)
