"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W6: does it matter that nodes are not alike?

Experiments W3 to W5 gave every node the same world, and the fixed rule was close to right
for all of them: the most any policy could gain was about 3%. Here the nodes differ in what
a lost packet costs them (a phone on a radio link waits seconds for a retransmission, a
server on a wired link waits milliseconds). One fixed rule has to serve them all.

 world "loss penalty": cost = (20 (rtt+1) + T_i p) / (1 - p) x (1 + 0.1 (7 - deliv)),
                       T_i = 200 ms x exp(s z_i), z_i standard normal (cut at 2.5).
 world "exponents"   : cost = 20 (rtt+1)^a_i x (1 + 0.35 loss)^b_i x (1 + 0.1 (7 - deliv)),
                       a_i = exp(-s z_i / 2), b_i = exp(+s z_i / 2).
 world "three ways"  : as "exponents", but the three exponents (on latency, loss and delivery)
                       vary independently: a_i = exp(s z1/2), b_i = exp(s z2/2), d_i = exp(s z3)
                       (d_i held between 0.1 and 8). Nodes differ in more than one way.
 s is the spread: 0 means all nodes alike, 1.5 means very unlike.

Compared, for each node, with the fixed rule:
  one table for all   : the best single points table, fitted to the pooled logs of all nodes
  three profiles      : nodes sorted into three groups by their true parameters (the most
                        favourable grouping: a real network would have to guess), one table each
  a table per node    : each node fits its own table from its own log (500 / 2,000 / 20,000)
  best possible       : each node picks with knowledge of its own true costs
Evidence is gathered inside band 6 (10% of decisions at random among near-equal candidates).
The per-node tables are also run inside bands, to show what the band costs unlike nodes.
"""
import json, time
import numpy as np
from safety_common import *

NODES, POOL = 200, 60_000


def node_cost(C, world, z, s):
    rtt, loss, deliv = C[..., 0], C[..., 1], C[..., 2]
    dl = 1 + 0.10 * (7 - deliv)
    if world == "loss penalty":
        p = LOSS_PROB[loss]; T = 200.0 * np.exp(s * z[0])
        c = (20.0 * (rtt + 1) + T * p) / (1 - p) * dl
    elif world == "exponents":
        a, b = np.exp(-s * z[0] / 2), np.exp(s * z[0] / 2)
        c = 20.0 * (rtt + 1.0) ** a * (1 + 0.35 * loss) ** b * dl
    else:
        a, b, d = np.exp(s * z[0] / 2), np.exp(s * z[1] / 2), float(np.clip(np.exp(s * z[2]), 0.1, 8.0))
        c = 20.0 * (rtt + 1.0) ** a * (1 + 0.35 * loss) ** b * dl ** d
    return np.where(deliv < 2, 3.0 * c.max(), c)


def kmeans(V, k, seed=0, rounds=30):
    rng = np.random.default_rng(seed); cen = V[rng.choice(len(V), k, replace=False)]
    for _ in range(rounds):
        lab = ((V[:, None, :] - cen[None]) ** 2).sum(-1).argmin(1)
        cen = np.array([V[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(k)])
    return lab


def log_node(rng, n, world, z, s, delta=6, explore=0.10):
    C = candidates(rng, n)
    elig = C[..., 2] >= 2
    sb = np.where(elig, np.asarray(baseline_rule(C.reshape(-1, 5)), float).reshape(n, 3), -1.0)
    mask = elig & (sb >= sb.max(1, keepdims=True) - delta)
    rnd = (rng.random((n, 3)) * mask).argmax(1); use = (rng.random(n) < explore) & (mask.sum(1) > 0)
    pick = np.where(use, rnd, sb.argmax(1))
    x = C[np.arange(n), pick]
    tc = node_cost(C, world, z, s)[np.arange(n), pick]
    return x, tc * rng.lognormal(0.0, NOISE, n)


if __name__ == "__main__":
    t0 = time.time(); out = []
    P = Pool(POOL, "product", seed=61)                # candidates and the fixed rule; costs are per node below
    idx = np.arange(POOL)
    inb = {d: P.elig & (P.sb >= P.sb.max(1, keepdims=True) - (1e9 if d is None else d)) for d in (3, 6, 12, None)}
    for world in ("loss penalty", "exponents", "three ways"):
        for s in (0.0, 0.5, 1.0, 1.5):
            zs = np.clip(np.random.default_rng(600 + int(10 * s)).standard_normal((NODES, 3)), -2.5, 2.5)
            tcs = [node_cost(P.C, world, z, s) for z in zs]
            fixed = np.array([tc[idx, P.a].mean() for tc in tcs])
            def rel(pick_by_node):
                return np.array([tcs[i][idx, pick_by_node(i)].mean() for i in range(NODES)]) / fixed - 1
            oracle = rel(lambda i: np.where(P.elig, tcs[i], 1e18).argmin(1))
            res = dict(world=world, spread=s, nodes=NODES)
            def put(name, v):
                worst_served = np.argsort(oracle)[: NODES // 10]            # the tenth of nodes the fixed rule serves worst
                res[name] = dict(mean=float(v.mean()), median=float(np.median(v)), tenth_the_fixed_rule_serves_worst=float(v[worst_served].mean()),
                                 share_of_nodes_made_worse=float((v > 0.002).mean()), worst_node=float(v.max()))
            put("best possible", oracle)
            for d in (3, 6, 12):
                put(f"best possible inside band {d}", rel(lambda i: np.where(inb[d], tcs[i], 1e18).argmin(1)))
            for d in (3, 6, 12, None):
                put(f"worst possible inside band {d}" if d else "worst possible with no band", rel(lambda i: np.where(inb[d], tcs[i], -1.0).argmax(1)))
            logs = {}
            for n in (500, 2_000, 20_000):
                logs[n] = [log_node(np.random.default_rng(10_000 * i + n + int(100 * s)), n, world, zs[i], s) for i in range(NODES)]
            # one table for all nodes: pooled logs, each node's observations divided by its own typical value
            n = 2_000
            xs = np.concatenate([l[0] for l in logs[n]]); os_ = np.concatenate([l[1] / np.exp(np.mean(np.log(l[1]))) for l in logs[n]])
            one = fast_scores(P, *fit_steps_ratio(xs, os_, 0, 32))
            put("one table for all nodes", rel(lambda i: P.pick(one, None)))
            if world == "three ways":
                lab = kmeans(zs, 3); groups = [np.flatnonzero(lab == j) for j in range(3)]
            else:
                groups = np.array_split(np.argsort(zs[:, 0]), 3)
            prof = {}
            for g in groups:
                if len(g) == 0: continue
                xs = np.concatenate([logs[n][i][0] for i in g]); os_ = np.concatenate([logs[n][i][1] / np.exp(np.mean(np.log(logs[n][i][1]))) for i in g])
                sc = fast_scores(P, *fit_steps_ratio(xs, os_, 0, 32))
                for i in g: prof[i] = sc
            put("three profiles", rel(lambda i: P.pick(prof[i], None)))
            if world == "three ways":
                lab = kmeans(zs, 9); prof9 = {}
                for j in range(9):
                    g = np.flatnonzero(lab == j)
                    if len(g) == 0: continue
                    xs = np.concatenate([logs[n][i][0] for i in g]); os_ = np.concatenate([logs[n][i][1] / np.exp(np.mean(np.log(logs[n][i][1]))) for i in g])
                    sc = fast_scores(P, *fit_steps_ratio(xs, os_, 0, 32))
                    for i in g: prof9[i] = sc
                put("nine profiles", rel(lambda i: P.pick(prof9[i], None)))
            for n in (500, 2_000, 20_000):
                for R in (32, 1024):
                    own = [fast_scores(P, *fit_steps_ratio(logs[n][i][0], logs[n][i][1], i, R)) for i in range(NODES)]
                    put(f"a table per node, {n} logged decisions, ratio limit {R}", rel(lambda i: P.pick(own[i], None)))
                    if n == 20_000 and R == 1024:
                        for d in (3, 6, 12):
                            put(f"a table per node, 20000 logged decisions, ratio limit 1024, inside band {d}", rel(lambda i: P.pick(own[i], d)))
            out.append(res)
            print(f"\n[{world}] spread {s}  ({time.time()-t0:.0f}s)")
            for k, v in res.items():
                if isinstance(v, dict):
                    print(f"   {k:78s} mean {100*v['mean']:+6.1f}% | median {100*v['median']:+6.1f}% | worst-served tenth {100*v['tenth_the_fixed_rule_serves_worst']:+6.1f}% | made worse {100*v['share_of_nodes_made_worse']:4.0f}% | worst node {100*v['worst_node']:+6.1f}%", flush=True)
    json.dump(out, open("w6_results.json", "w"), indent=1)
