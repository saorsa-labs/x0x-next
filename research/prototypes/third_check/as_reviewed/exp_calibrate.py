"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W6b: keeping the harm ceiling when the fixed rule fits a node badly.

Experiment W6 showed that a band of 3 points around the fixed rule holds any policy under about
4% on nodes the rule fits, and lets through far more on the nodes it fits worst. Two remedies
are tested here, both using only what the node has measured itself (2,000 logged decisions).

 A. A band the node sets for itself. The node fits a cost table to its own log and uses it to
    estimate what the worst choice inside each band width would cost. It takes the widest band
    whose estimate is under the ceiling (5%). Reported: the true worst case at the width chosen.
 B. The node's own table as the yardstick. Once a table has earned its room (experiment W8d),
    later candidates are held within 5% of ITS estimate of cost and not the fixed rule's.
    Reported: the true worst case of any policy inside that band, against the table's own pick.
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


if __name__ == "__main__":
    t0 = time.time(); out = []
    P = Pool(POOL, "product", seed=61); idx = np.arange(POOL)
    inb = {d: P.elig & (P.sb >= P.sb.max(1, keepdims=True) - d) for d in WIDTHS}
    for world in ("loss penalty", "three ways"):
        for s in (0.0, 0.5, 1.0, 1.5):
            zs = np.clip(np.random.default_rng(600 + int(10 * s)).standard_normal((NODES, 3)), -2.5, 2.5)
            A = dict(width=[], true_worst=[], fixed3_worst=[], gain_inside=[]); B = dict(true_worst=[], est_gain=[], anchor_vs_fixed=[])
            for i in range(NODES):
                tc = node_cost(P.C, world, zs[i], s)
                x, obs = log_node(np.random.default_rng(10_000 * i + 2_000 + int(100 * s)), 2_000, world, zs[i], s)
                c_r, c_l, c_d = fit_cost_table(x, obs)
                est = c_r[P.C[..., 0]] + c_l[P.C[..., 1]] + c_d[P.C[..., 2]]           # estimated log cost
                ec = np.exp(est)
                base_true, base_est = tc[idx, P.a].sum(), ec[idx, P.a].sum()
                # A: the widest band whose estimated worst case is under the ceiling
                chosen = 0
                for d in WIDTHS:
                    w = np.where(inb[d], est, -1e18).argmax(1)
                    if ec[idx, w].sum() / base_est - 1 <= CEILING:
                        chosen = d
                w = np.where(inb[chosen], tc, -1.0).argmax(1)
                A["width"].append(chosen); A["true_worst"].append(float(tc[idx, w].sum() / base_true - 1))
                A["fixed3_worst"].append(float(tc[idx, np.where(inb[3], tc, -1.0).argmax(1)].sum() / base_true - 1))
                own = np.where(inb[chosen], -est, -1e18); own[idx, P.a] += 1e-9
                A["gain_inside"].append(float(tc[idx, own.argmax(1)].sum() / base_true - 1))
                # B: the node's own table as the yardstick, band = 5% of its estimated cost
                e2 = np.where(P.elig, est, 1e18); a2 = e2.argmin(1)
                band = P.elig & (e2 <= e2.min(1, keepdims=True) + np.log(1 + CEILING))
                w2 = np.where(band, tc, -1.0).argmax(1)
                B["true_worst"].append(float(tc[idx, w2].sum() / tc[idx, a2].sum() - 1))
                B["anchor_vs_fixed"].append(float(tc[idx, a2].sum() / base_true - 1))
            q = lambda v: dict(mean=float(np.mean(v)), nine_in_ten_nodes_under=float(np.quantile(v, 0.9)), worst_node=float(np.max(v)))
            row = dict(world=world, spread=s, nodes=NODES, ceiling=CEILING,
                       fixed_band_3_true_worst_case=q(A["fixed3_worst"]),
                       self_set_band=dict(widths_chosen={str(d): int(np.sum(np.array(A["width"]) == d)) for d in WIDTHS}, true_worst_case=q(A["true_worst"]), own_table_inside_it=q(A["gain_inside"])),
                       own_table_as_yardstick=dict(true_worst_case_of_any_policy_within_5_percent_of_its_estimate=q(B["true_worst"]), the_table_itself_against_the_fixed_rule=q(B["anchor_vs_fixed"])))
            out.append(row)
            f = lambda d: f"mean {100*d['mean']:+5.1f}%, 9 in 10 under {100*d['nine_in_ten_nodes_under']:+5.1f}%, worst node {100*d['worst_node']:+5.1f}%"
            print(f"\n[{world}] spread {s} ({time.time()-t0:.0f}s)")
            print(f"   fixed band of 3 points, true worst case         : {f(row['fixed_band_3_true_worst_case'])}")
            print(f"   A. self-set band: widths chosen {row['self_set_band']['widths_chosen']}")
            print(f"      true worst case at the chosen width           : {f(row['self_set_band']['true_worst_case'])}")
            print(f"      the node's own table inside it                : {f(row['self_set_band']['own_table_inside_it'])}")
            print(f"   B. own table as yardstick, 5% band: true worst   : {f(row['own_table_as_yardstick']['true_worst_case_of_any_policy_within_5_percent_of_its_estimate'])}")
            print(f"      the table itself against the fixed rule       : {f(row['own_table_as_yardstick']['the_table_itself_against_the_fixed_rule'])}", flush=True)
    json.dump(out, open("w6b_results.json", "w"), indent=1)
