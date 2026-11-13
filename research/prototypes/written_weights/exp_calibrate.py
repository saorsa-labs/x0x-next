"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W6b: keeping the harm ceiling when the fixed rule fits a node badly.

A band of 3 points around the fixed rule holds any policy under about 4% in the two toy worlds
of W8b. That figure belongs to those worlds. It moves with how well the rule fits the node (W6)
and with the mix of candidates the node meets (part 2 here: where every candidate is fast, one
level of round-trip time is a large relative difference, and 3 points lets through far more).

Remedy A: the node sets its own margin. It fits a cost table to its 2,000 logged decisions and
  estimates, over the candidate sets of decisions it has seen, what the worst choice inside each
  margin would cost. It takes the widest margin whose estimate is under a target. The TRUE worst
  case is then measured on a separate pool of decisions the node did not use.
  (The first version of this experiment chose the margin on the same pool it was scored on. The
  third independent check found that, and found that 5 to 10% of nodes then exceed the ceiling.
  This version chooses and scores on separate decisions, and tries a target below the ceiling.)
Part 3: an attacker shapes the evidence. It holds a share of the peers behind the node's logged
  decisions. While the node gathers evidence, its peers that measure fast deliver slowly and its
  others deliver as fast as the best path, so that the node's own cost table comes out too flat
  and its self-set margin too wide. Afterwards everyone behaves normally. (The third check's attack.)
Remedy B: a table that has earned its room becomes the node's yardstick. Later candidates are
  held within 5% of ITS estimate of cost and not the fixed rule's.
"""
import json, time
import numpy as np
from safety_common import *
from exp_unlike_nodes import node_cost, log_node, NODES, POOL

CEILING = 0.05
WIDTHS = (0, 1, 2, 3, 4, 6, 9, 12)


def fit_cost_table(x, obs):
    """log of cost as a sum of three one-feature curves, each forced to move the declared way"""
    ok = x[:, 2] >= 2
    X, y = thermo(x[ok]), np.log(obs[ok])
    co = np.linalg.solve(X.T @ X + 1e-3 * np.eye(X.shape[1]), X.T @ y)
    cnt = lambda f, lv: np.array([np.count_nonzero(x[ok, f] == v) for v in lv], float) + 1
    c_r = isotonic(np.concatenate([[0], np.cumsum(co[1:16])]), cnt(0, range(16)), True)
    c_l = isotonic(np.concatenate([[0], np.cumsum(co[16:23])]), cnt(1, range(8)), True)
    c_d = isotonic(np.concatenate([[0], np.cumsum(co[23:28])]), cnt(2, range(2, 8)), False)
    return c_r, c_l, np.concatenate([[0, 0], c_d])


class Sets:
    """candidate sets with the fixed rule's scores and pick"""
    def __init__(self, C):
        n = len(C); self.C = C; self.idx = np.arange(n)
        self.elig = C[..., 2] >= 2
        self.sb = np.where(self.elig, np.asarray(baseline_rule(C.reshape(-1, 5)), float).reshape(n, 3), -1.0)
        self.a = self.sb.argmax(1)
    def inb(self, d):
        return self.elig & (self.sb >= self.sb.max(1, keepdims=True) - d)


def est_cost(tbl, C):
    c_r, c_l, c_d = tbl
    return np.exp(c_r[C[..., 0]] + c_l[C[..., 1]] + c_d[C[..., 2]])


def choose_margin(tbl, S, target):
    ec = est_cost(tbl, S.C); base = ec[S.idx, S.a].sum(); chosen = 0
    for d in WIDTHS:
        w = np.where(S.inb(d), ec, -1.0).argmax(1)
        if ec[S.idx, w].sum() / base - 1 <= target:
            chosen = d
    return chosen


def mix(rng, n, rtt_levels, loss_levels):
    C = candidates(rng, n)
    C[..., 0] = rng.choice(np.asarray(rtt_levels), (n, 3))
    if loss_levels is not None:
        p = P_LOSS[np.asarray(loss_levels)]; C[..., 1] = rng.choice(np.asarray(loss_levels), (n, 3), p=p / p.sum())
    return C


def log_from(rng, C, cost_fn, delta=6, explore=0.10):
    S = Sets(C); n = len(C); mask = S.inb(delta)
    rnd = (rng.random((n, 3)) * mask).argmax(1); use = (rng.random(n) < explore) & (mask.sum(1) > 0)
    pick = np.where(use, rnd, S.a)
    x = C[S.idx, pick]
    return x, cost_fn(C)[S.idx, pick] * rng.lognormal(0.0, NOISE, n)


if __name__ == "__main__":
    t0 = time.time(); out = dict(unlike_nodes=[], candidate_mixes=[])
    q = lambda v: dict(mean=float(np.mean(v)), nine_in_ten_nodes_under=float(np.quantile(v, 0.9)), worst_node=float(np.max(v)), share_of_nodes_over_the_ceiling=float(np.mean(np.asarray(v) > CEILING)))
    f = lambda d: f"mean {100*d['mean']:+5.1f}%, 9 in 10 under {100*d['nine_in_ten_nodes_under']:+5.1f}%, worst node {100*d['worst_node']:+5.1f}%, over the ceiling {100*d['share_of_nodes_over_the_ceiling']:3.0f}%"
    E = Sets(candidates(np.random.default_rng(61), POOL))                      # evaluation pool: never used to choose
    VARIANTS = (("2,000 decisions seen, target 5%", 2_000, 0.05), ("2,000 decisions seen, target 4%", 2_000, 0.04),
                ("20,000 decisions seen, target 5%", 20_000, 0.05), ("20,000 decisions seen, target 4%", 20_000, 0.04))
    # ---------------- part 1: nodes that differ in what things cost them
    for world in ("loss penalty", "exponents", "three ways"):
        for s in (0.0, 0.5, 1.0, 1.5):
            zs = np.clip(np.random.default_rng(600 + int(10 * s)).standard_normal((NODES, 3)), -2.5, 2.5)
            fixed3 = []; A = {v[0]: dict(width=[], worst=[], gain=[]) for v in VARIANTS}; B = dict(worst=[], table=[])
            for i in range(NODES):
                tc = node_cost(E.C, world, zs[i], s)
                base_true = tc[E.idx, E.a].sum()
                x, obs = log_node(np.random.default_rng(10_000 * i + 2_000 + int(100 * s)), 2_000, world, zs[i], s)
                tbl = fit_cost_table(x, obs)
                fixed3.append(float(tc[E.idx, np.where(E.inb(3), tc, -1.0).argmax(1)].sum() / base_true - 1))
                seen = candidates(np.random.default_rng(777_000 + i), 20_000)          # candidate sets the node has seen; no outcomes needed
                for name, n_seen, target in VARIANTS:
                    d = choose_margin(tbl, Sets(seen[:n_seen]), target)
                    w = np.where(E.inb(d), tc, -1.0).argmax(1)
                    own = np.where(E.inb(d), -np.log(est_cost(tbl, E.C)), -1e18); own[E.idx, E.a] += 1e-9
                    A[name]["width"].append(d); A[name]["worst"].append(float(tc[E.idx, w].sum() / base_true - 1)); A[name]["gain"].append(float(tc[E.idx, own.argmax(1)].sum() / base_true - 1))
                ec = np.where(E.elig, est_cost(tbl, E.C), 1e18); a2 = ec.argmin(1)
                band = E.elig & (ec <= ec.min(1, keepdims=True) * (1 + CEILING))
                B["worst"].append(float(tc[E.idx, np.where(band, tc, -1.0).argmax(1)].sum() / tc[E.idx, a2].sum() - 1)); B["table"].append(float(tc[E.idx, a2].sum() / base_true - 1))
            row = dict(world=world, spread=s, nodes=NODES, ceiling=CEILING, fixed_margin_of_3_true_worst_case=q(fixed3), self_set_margin={},
                       own_table_as_yardstick=dict(true_worst_case_of_any_policy_within_5_percent_of_its_estimate=q(B["worst"]), the_table_itself_against_the_fixed_rule=q(B["table"])))
            print(f"\n[{world}] spread {s} ({time.time()-t0:.0f}s)\n   fixed margin of 3, true worst case              : {f(row['fixed_margin_of_3_true_worst_case'])}")
            for name, _, _ in VARIANTS:
                a = A[name]
                row["self_set_margin"][name] = dict(widths_chosen={str(d): int(np.sum(np.array(a["width"]) == d)) for d in WIDTHS}, true_worst_case=q(a["worst"]), own_table_inside_it=q(a["gain"]))
                print(f"   self-set, {name:34s}: {f(row['self_set_margin'][name]['true_worst_case'])} | own table inside it: mean {100*np.mean(a['gain']):+.1f}%")
            print(f"   own table as yardstick, 5% band: true worst     : {f(row['own_table_as_yardstick']['true_worst_case_of_any_policy_within_5_percent_of_its_estimate'])}", flush=True)
            out["unlike_nodes"].append(row)
    # ---------------- part 2: nodes that meet a different mix of candidates (costs as in world one)
    MIXES = (("every level of round trip (the mix of W8b)", range(16), None), ("fast candidates only: round-trip levels 0 to 3", range(4), None),
             ("fast and clean: round-trip levels 0 to 3, loss levels 0 or 1", range(4), (0, 1)), ("very fast: round-trip levels 0 or 1", range(2), None), ("slow candidates only: round-trip levels 8 to 15", range(8, 16), None))
    for world in WORLDS:
        cost_fn = lambda C, world=world: true_cost(C, world)
        for label, rl, ll in MIXES:
            Em = Sets(mix(np.random.default_rng(91), POOL, rl, ll)); tc = cost_fn(Em.C); base_true = tc[Em.idx, Em.a].sum()
            worst_at = {d: float(tc[Em.idx, np.where(Em.inb(d), tc, -1.0).argmax(1)].sum() / base_true - 1) for d in WIDTHS}
            res = {v[0]: dict(width=[], worst=[]) for v in VARIANTS}
            for i in range(40):
                r = np.random.default_rng(50_000 + i)
                x, obs = log_from(r, mix(r, 2_000, rl, ll), cost_fn)
                tbl = fit_cost_table(x, obs); seen = mix(np.random.default_rng(60_000 + i), 20_000, rl, ll)
                for name, n_seen, target in VARIANTS:
                    d = choose_margin(tbl, Sets(seen[:n_seen]), target)
                    res[name]["width"].append(d); res[name]["worst"].append(worst_at[d])
            row = dict(world=world, candidate_mix=label, true_worst_case_at_each_margin={str(d): worst_at[d] for d in WIDTHS},
                       self_set_margin={n_: dict(widths_chosen={str(d): int(np.sum(np.array(v_["width"]) == d)) for d in WIDTHS}, true_worst_case=q(v_["worst"])) for n_, v_ in res.items()})
            out["candidate_mixes"].append(row)
            print(f"\n[{world}] {label}: fixed margin of 3 lets through {100*worst_at[3]:+.1f}%; margin 0: {100*worst_at[0]:+.1f}%, 1: {100*worst_at[1]:+.1f}%, 2: {100*worst_at[2]:+.1f}%")
            for n_, v_ in row["self_set_margin"].items():
                print(f"   self-set, {n_:34s}: {f(v_['true_worst_case'])} | widths {v_['widths_chosen']}", flush=True)
    # ---------------- part 3: evidence shaped by an attacker
    def log_shaped(rng, n, world, z, s, share, M, delta=6, explore=0.10):
        C = candidates(rng, n); S = Sets(C); mask = S.inb(delta)
        rnd = (rng.random((n, 3)) * mask).argmax(1); use = (rng.random(n) < explore) & (mask.sum(1) > 0)
        pick = np.where(use, rnd, S.a); x = C[S.idx, pick]
        tcx = node_cost(C, world, z, s)[S.idx, pick]
        att = rng.random(n) < share
        best_case = node_cost(np.array([[[0, 0, 7, 0, 0]]]), world, z, s)[0, 0]
        shaped = np.where(x[:, 0] <= 3, tcx * M, np.minimum(tcx, best_case))
        tcx = np.where(att & (x[:, 2] >= 2), shaped, tcx)
        return x, tcx * rng.lognormal(0.0, NOISE, n)
    out["shaped_evidence"] = []
    for world, s in (("three ways", 0.0), ("loss penalty", 0.0), ("three ways", 1.0)):
        zs = np.clip(np.random.default_rng(600 + int(10 * s)).standard_normal((NODES, 3)), -2.5, 2.5)
        for share, M in ((0.0, 1), (0.10, 8), (0.20, 8), (0.10, 30), (0.20, 30)):
            A = {v[0]: dict(width=[], worst=[]) for v in VARIANTS}
            for i in range(NODES):
                tc = node_cost(E.C, world, zs[i], s); base_true = tc[E.idx, E.a].sum()
                x, obs = log_shaped(np.random.default_rng(10_000 * i + 2_000 + int(100 * s) + 7), 2_000, world, zs[i], s, share, M)
                tbl = fit_cost_table(x, obs); seen = candidates(np.random.default_rng(777_000 + i), 20_000)
                for name, n_seen, target in VARIANTS:
                    d = choose_margin(tbl, Sets(seen[:n_seen]), target)
                    A[name]["width"].append(d); A[name]["worst"].append(float(tc[E.idx, np.where(E.inb(d), tc, -1.0).argmax(1)].sum() / base_true - 1))
            row = dict(world=world, spread=s, attacker_share_of_the_evidence=share, slow_factor_on_fast_looking_peers=M,
                       self_set_margin={k: dict(widths_chosen={str(d): int(np.sum(np.array(v["width"]) == d)) for d in WIDTHS}, true_worst_case=q(v["worst"])) for k, v in A.items()})
            out["shaped_evidence"].append(row)
            print(f"\n[{world}] spread {s}: attacker supplies {100*share:.0f}% of the evidence, fast-looking peers {M}x slower ({time.time()-t0:.0f}s)")
            for k, v in row["self_set_margin"].items():
                print(f"   self-set, {k:34s}: {f(v['true_worst_case'])}", flush=True)
    json.dump(out, open("w6b_results.json", "w"), indent=1)
