"""
THIRD CHECK -- reviewer's script. Not production code. Not part of x0x.

Checks and attacks on W6 / W6b (unlike nodes, self-set margin):
 A. What does gathering evidence cost on unlike nodes? (Claim: "Measure mode costs about 0.1%".)
    W6 gathered inside band 6, not band 3.
 B. Is "one table for all" the best single rule? (i) a single table tuned directly on the true
    costs of all nodes; (ii) ordinary code with one to three constants estimated per node.
 C. The self-set margin (W6b): (i) the world W6b left out ("exponents"); (ii) the margin chosen
    from the node's own 2,000 logged candidate sets and not from the 60,000-decision evaluation
    pool; (iii) an attacker shapes the node's log ("probe fast, deliver slow") so that the node's
    own cost table is too flat and its self-set margin too wide.
 D. How much of "3.9%" is the candidate mix? The same band on nodes whose candidates are mostly fast.
"""
import sys, json, time
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _os.path.join(HERE, 'as_reviewed'))   # path made relative for the zip; was the reviewer's working folder
import numpy as np
from safety_common import *
from exp_unlike_nodes import node_cost, log_node, NODES, POOL
from exp_calibrate import fit_cost_table, CEILING, WIDTHS

t0 = time.time()
out = dict(A=[], B=[], C=[], D=[])
P = Pool(POOL, "product", seed=61); idx = np.arange(POOL)
inb = {d: P.elig & (P.sb >= P.sb.max(1, keepdims=True) - d) for d in set(WIDTHS) | {3, 6}}
PART = sys.argv[1] if len(sys.argv) > 1 else "ABCD"

# ---------------------------------------------------------------- A
if "A" in PART:
    r = np.random.default_rng(5)
    rnd = {d: (r.random(P.sb.shape) * inb[d]).argmax(1) for d in (3, 6)}
    use = (r.random(POOL) < 0.10)
    for world in ("loss penalty", "exponents", "three ways"):
        for s in (0.0, 0.5, 1.0, 1.5):
            zs = np.clip(np.random.default_rng(600 + int(10 * s)).standard_normal((NODES, 3)), -2.5, 2.5)
            row = dict(world=world, spread=s)
            for d in (3, 6):
                pk = np.where(use & (inb[d].sum(1) > 0), rnd[d], P.a)
                v = []
                for i in range(NODES):
                    tc = node_cost(P.C, world, zs[i], s)
                    v.append(tc[idx, pk].mean() / tc[idx, P.a].mean() - 1)
                v = np.array(v)
                row[f"gathering_inside_band_{d}"] = dict(mean=float(v.mean()), nine_in_ten_under=float(np.quantile(v, .9)), worst_node=float(v.max()))
            out["A"].append(row)
            a, b = row["gathering_inside_band_3"], row["gathering_inside_band_6"]
            print(f"A [{world}] spread {s}: 10% random inside band 3 costs mean {100*a['mean']:+.2f}% (worst node {100*a['worst_node']:+.2f}%) | inside band 6 (what W6 used): mean {100*b['mean']:+.2f}% (worst node {100*b['worst_node']:+.2f}%)", flush=True)

# ---------------------------------------------------------------- B
if "B" in PART:
    LOSS_TERM = np.log(1 + 0.35 * np.arange(8)); RTT_TERM = np.log(np.arange(16) + 1.0)
    def code_rule_three(x, obs):
        """ordinary code: cost = k (rtt+1)^a (1+0.35 loss)^b (1+0.1(7-deliv))^d with a, b, d estimated per node (3 numbers)"""
        ok = x[:, 2] >= 2
        A = np.stack([RTT_TERM[x[ok, 0]], LOSS_TERM[x[ok, 1]], np.log(1 + 0.10 * (7 - x[ok, 2])), np.ones(ok.sum())], 1)
        co, *_ = np.linalg.lstsq(A, np.log(obs[ok]), rcond=None)
        a, b, d = max(co[0], 0), max(co[1], 0), max(co[2], 0)
        est = a * RTT_TERM[P.C[..., 0]] + b * LOSS_TERM[P.C[..., 1]] + d * np.log(1 + 0.10 * (7 - P.C[..., 2]))
        return np.where(P.elig, est, 1e18).argmin(1)
    def code_rule_timeout(x, obs):
        """ordinary code: cost = (20 (rtt+1) + T p)/(1-p) x delivery term, with the node's own T estimated (1 number)"""
        ok = x[:, 2] >= 2
        p = LOSS_PROB[x[ok, 1]]; dl = 1 + 0.10 * (7 - x[ok, 2]); y = np.log(obs[ok])
        best = None
        for T in 200.0 * np.exp(np.linspace(-4, 4, 161)):
            pred = np.log((20.0 * (x[ok, 0] + 1) + T * p) / (1 - p) * dl)
            e = np.mean((y - pred - np.mean(y - pred)) ** 2)
            if best is None or e < best[0]: best = (e, T)
        T = best[1]; pp = LOSS_PROB[P.C[..., 1]]
        est = (20.0 * (P.C[..., 0] + 1) + T * pp) / (1 - pp) * (1 + 0.10 * (7 - P.C[..., 2]))
        return np.where(P.elig, est, 1e18).argmin(1)
    def best_single_table(tcs_sub, fixed_sub, Psub, restarts=3):
        VALUES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
        ii = np.arange(len(Psub.a))
        def score(w):
            pk = Psub.pick(fast_scores(Psub, *w), None)
            return float((tcs_sub[:, ii, pk].mean(1) / fixed_sub - 1).mean())
        best = (1e9, None)
        for r in range(restarts):
            rg = np.random.default_rng(r)
            w = [rg.choice(VALUES, 15), rg.choice(VALUES, 7), rg.choice([0] + VALUES, 5)]
            if r == 0: w = [np.full(15, 3), np.full(7, 4), np.zeros(5, int)]
            cur = score(w); improved = True
            while improved:
                improved = False
                for part, vals in ((0, VALUES), (1, VALUES), (2, [0] + VALUES)):
                    for i in range(len(w[part])):
                        keep = w[part][i]
                        for v in vals:
                            if v == keep: continue
                            w[part][i] = v
                            c = score(w)
                            if c < cur - 1e-9: cur, keep, improved = c, v, True
                        w[part][i] = keep
            if cur < best[0]: best = (cur, [a.copy() for a in w])
        return best[1]
    Psub = Pool(6_000, "product", seed=62)
    for world in ("loss penalty", "exponents", "three ways"):
        for s in (1.0, 1.5):
            zs = np.clip(np.random.default_rng(600 + int(10 * s)).standard_normal((NODES, 3)), -2.5, 2.5)
            tcs = [node_cost(P.C, world, z, s) for z in zs]
            fixed = np.array([tc[idx, P.a].mean() for tc in tcs])
            rel = lambda pick_by_node: np.array([tcs[i][idx, pick_by_node(i)].mean() for i in range(NODES)]) / fixed - 1
            oracle = rel(lambda i: np.where(P.elig, tcs[i], 1e18).argmin(1))
            worst_served = np.argsort(oracle)[: NODES // 10]
            res = dict(world=world, spread=s)
            def put(name, v):
                res[name] = dict(mean=float(v.mean()), worst_served_tenth=float(v[worst_served].mean()), share_made_worse=float((v > 0.002).mean()), worst_node=float(v.max()))
                print(f"B [{world}] spread {s}: {name:62s} mean {100*v.mean():+6.2f}% | worst-served tenth {100*v[worst_served].mean():+6.2f}% | made worse {100*(v>0.002).mean():3.0f}% | worst node {100*v.max():+6.2f}% | {time.time()-t0:.0f}s", flush=True)
            put("best possible", oracle)
            for n in (500, 2_000):
                logs = [log_node(np.random.default_rng(10_000 * i + n + int(100 * s)), n, world, zs[i], s) for i in range(NODES)]
                own = [fast_scores(P, *fit_steps_ratio(logs[i][0], logs[i][1], i, 32)) for i in range(NODES)]
                put(f"a table per node (27 numbers), {n} logged decisions [the study's]", rel(lambda i: P.pick(own[i], None)))
                if world == "loss penalty":
                    picks = [code_rule_timeout(*logs[i]) for i in range(NODES)]
                    put(f"ordinary code + ONE constant estimated per node, {n} logged decisions", rel(lambda i: picks[i]))
                picks3 = [code_rule_three(*logs[i]) for i in range(NODES)]
                put(f"ordinary code + THREE constants estimated per node, {n} logged decisions", rel(lambda i: picks3[i]))
                if n == 2_000:
                    xs = np.concatenate([l[0] for l in logs]); os_ = np.concatenate([l[1] / np.exp(np.mean(np.log(l[1]))) for l in logs])
                    one = fast_scores(P, *fit_steps_ratio(xs, os_, 0, 32))
                    put("one table for all nodes [the study's: least squares on pooled logs]", rel(lambda i: P.pick(one, None)))
            # (i) a single table tuned directly on true costs (the best a single table could do, searched)
            tcs_sub = np.stack([node_cost(Psub.C, world, z, s) for z in zs]); fixed_sub = tcs_sub[:, np.arange(len(Psub.a)), Psub.a].mean(1)
            w = best_single_table(tcs_sub, fixed_sub, Psub)
            sc = fast_scores(P, *w)
            put("one table for all nodes, tuned directly on every node's true costs", rel(lambda i: P.pick(sc, None)))
            res["tuned_single_table_weights"] = dict(rtt=[int(v) for v in w[0]], loss=[int(v) for v in w[1]], delivery=[int(v) for v in w[2]])
            out["B"].append(res)

# ---------------------------------------------------------------- C
if "C" in PART:
    def choose_width(est, Cset_elig, Cset_sb, Cset_a):
        """the study's rule: widest width whose ESTIMATED worst case is under the ceiling, over a given set of decisions"""
        ec = np.exp(est); ii = np.arange(len(Cset_a)); base_est = ec[ii, Cset_a].sum(); chosen = 0
        for d in WIDTHS:
            m = Cset_elig & (Cset_sb >= Cset_sb.max(1, keepdims=True) - d)
            w = np.where(m, est, -1e18).argmax(1)
            if ec[ii, w].sum() / base_est - 1 <= CEILING: chosen = d
        return chosen
    def true_worst(tc, d):
        return float(tc[idx, np.where(inb[d], tc, -1.0).argmax(1)].sum() / tc[idx, P.a].sum() - 1)
    def est_on(C, c_r, c_l, c_d):
        return c_r[C[..., 0]] + c_l[C[..., 1]] + c_d[C[..., 2]]
    q = lambda v: dict(mean=float(np.mean(v)), nine_in_ten_nodes_under=float(np.quantile(v, 0.9)), worst_node=float(np.max(v)), share_of_nodes_over_ceiling=float(np.mean(np.array(v) > CEILING)))
    f = lambda d: f"mean {100*d['mean']:+5.1f}%, 9 in 10 under {100*d['nine_in_ten_nodes_under']:+5.1f}%, worst node {100*d['worst_node']:+5.1f}%, over 5%: {100*d['share_of_nodes_over_ceiling']:.0f}% of nodes"
    for world in ("loss penalty", "exponents", "three ways"):
        for s in (0.0, 0.5, 1.0, 1.5):
            zs = np.clip(np.random.default_rng(600 + int(10 * s)).standard_normal((NODES, 3)), -2.5, 2.5)
            acc = dict(pool=[], own=[], own_w=[], pool_w=[])
            for i in range(NODES):
                tc = node_cost(P.C, world, zs[i], s)
                seed = 10_000 * i + 2_000 + int(100 * s)
                x, obs = log_node(np.random.default_rng(seed), 2_000, world, zs[i], s)
                c_r, c_l, c_d = fit_cost_table(x, obs)
                # as the study did it: margin chosen on the evaluation pool
                dpool = choose_width(est_on(P.C, c_r, c_l, c_d), P.elig, P.sb, P.a)
                # as a node would have to do it: margin chosen on its own 2,000 logged candidate sets
                Cl = candidates(np.random.default_rng(seed), 2_000); el = Cl[..., 2] >= 2
                sbl = np.where(el, np.asarray(baseline_rule(Cl.reshape(-1, 5)), float).reshape(2_000, 3), -1.0)
                down = choose_width(est_on(Cl, c_r, c_l, c_d), el, sbl, sbl.argmax(1))
                acc["pool"].append(true_worst(tc, dpool)); acc["own"].append(true_worst(tc, down)); acc["pool_w"].append(dpool); acc["own_w"].append(down)
            row = dict(world=world, spread=s, margin_chosen_on_evaluation_pool=q(acc["pool"]), margin_chosen_on_own_logged_decisions=q(acc["own"]),
                       widths_on_pool={str(d): int(np.sum(np.array(acc["pool_w"]) == d)) for d in WIDTHS}, widths_on_own_log={str(d): int(np.sum(np.array(acc["own_w"]) == d)) for d in WIDTHS})
            out["C"].append(row)
            print(f"C [{world}] spread {s}: self-set margin, TRUE worst case | chosen on the evaluation pool (the study's): {f(row['margin_chosen_on_evaluation_pool'])}")
            print(f"C {'':28s}                                    | chosen on the node's own 2,000 decisions:      {f(row['margin_chosen_on_own_logged_decisions'])} | {time.time()-t0:.0f}s", flush=True)

    # (iii) shaped evidence: an attacker holds a share of peers. While the node gathers evidence, the attacker's peers
    #       that MEASURE fast (rtt level <= 3) deliver slowly (outcome x M); the others deliver as fast as the best path.
    #       Afterwards everybody behaves normally: the TRUE worst case is computed on the unshaped costs.
    def log_shaped(rng, n, world, z, s, share, M, delta=6, explore=0.10):
        C = candidates(rng, n); elig = C[..., 2] >= 2
        sb = np.where(elig, np.asarray(baseline_rule(C.reshape(-1, 5)), float).reshape(n, 3), -1.0)
        mask = elig & (sb >= sb.max(1, keepdims=True) - delta)
        rnd = (rng.random((n, 3)) * mask).argmax(1); use = (rng.random(n) < explore) & (mask.sum(1) > 0)
        pick = np.where(use, rnd, sb.argmax(1)); x = C[np.arange(n), pick]
        tcx = node_cost(C, world, z, s)[np.arange(n), pick]
        att = rng.random(n) < share
        best_case = node_cost(np.array([[[0, 0, 7, 0, 0]]]), world, z, s)[0, 0]
        shaped = np.where(x[:, 0] <= 3, tcx * M, np.minimum(tcx, best_case))
        tcx = np.where(att & (x[:, 2] >= 2), shaped, tcx)
        return x, tcx * rng.lognormal(0.0, NOISE, n), C
    for world, s in (("three ways", 0.0), ("loss penalty", 0.0), ("three ways", 1.0)):
        zs = np.clip(np.random.default_rng(600 + int(10 * s)).standard_normal((NODES, 3)), -2.5, 2.5)
        for share, M in ((0.0, 1), (0.05, 8), (0.10, 8), (0.20, 8), (0.10, 30), (0.20, 30)):
            tw, ws = [], []
            for i in range(NODES):
                tc = node_cost(P.C, world, zs[i], s)
                x, obs, Cl = log_shaped(np.random.default_rng(10_000 * i + 2_000 + int(100 * s) + 7), 2_000, world, zs[i], s, share, M)
                c_r, c_l, c_d = fit_cost_table(x, obs)
                d = choose_width(est_on(P.C, c_r, c_l, c_d), P.elig, P.sb, P.a)
                tw.append(true_worst(tc, d)); ws.append(d)
            row = dict(world=world, spread=s, attacker_share_of_logged_decisions=share, slow_factor_on_fast_looking_peers=M, true_worst_case_at_self_set_margin=q(tw),
                       widths_chosen={str(d): int(np.sum(np.array(ws) == d)) for d in WIDTHS})
            out["C"].append(row)
            print(f"C shaped log [{world}] spread {s}: attacker supplies {100*share:4.1f}% of the evidence, factor {M:2d} | widths {row['widths_chosen']} | TRUE worst case at the self-set margin: {f(row['true_worst_case_at_self_set_margin'])} | {time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- D
if "D" in PART:
    def band_worst(C, world, delta):
        n = len(C); ii = np.arange(n); elig = C[..., 2] >= 2
        tc = true_cost(C, world); sb = np.where(elig, np.asarray(baseline_rule(C.reshape(-1, 5)), float).reshape(n, 3), -1.0)
        a = sb.argmax(1); m = elig & (sb >= sb.max(1, keepdims=True) - delta)
        w = np.where(m, tc, -1.0).argmax(1); b = np.where(m, tc, 1e18).argmin(1)
        return float(tc[ii, w].mean() / tc[ii, a].mean() - 1), float(tc[ii, b].mean() / tc[ii, a].mean() - 1), float((m.sum(1) >= 2).mean())
    rng = np.random.default_rng(99)
    for label, rtt_levels, loss_p in (("the study's mix: rtt uniform over 16 levels", range(16), P_LOSS),
                                      ("fast candidates only: rtt levels 0 to 3", range(4), P_LOSS),
                                      ("fast and clean: rtt levels 0 to 3, loss levels 0 or 1", range(4), np.array([.6, .4, 0, 0, 0, 0, 0, 0])),
                                      ("very fast: rtt levels 0 to 1", range(2), P_LOSS),
                                      ("slow candidates only: rtt levels 8 to 15", range(8, 16), P_LOSS)):
        C = candidates(rng, 300_000); C[..., 0] = rng.choice(list(rtt_levels), (300_000, 3)); C[..., 1] = rng.choice(8, (300_000, 3), p=loss_p)
        for world in WORLDS:
            row = dict(world=world, candidate_mix=label)
            for d in (0, 3, 6):
                w, b, ch = band_worst(C, world, d); row[f"band_{d}"] = dict(worst_possible=w, best_possible=b, decisions_with_a_choice=ch)
            out["D"].append(row)
            print(f"D [{world}] {label:55s} | worst possible in band 0: {100*row['band_0']['worst_possible']:+5.1f}% | band 3: {100*row['band_3']['worst_possible']:+5.1f}% (choice in {100*row['band_3']['decisions_with_a_choice']:.0f}% of decisions) | band 6: {100*row['band_6']['worst_possible']:+5.1f}%", flush=True)

json.dump(out, open(_os.path.join(HERE, f'check_w6_results_{PART}.json'), 'w'), indent=1)
print(f"done in {time.time()-t0:.0f}s")
