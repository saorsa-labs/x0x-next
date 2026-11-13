#!/usr/bin/env python3
"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.

Question tested
---------------
Does learned peer selection help a broadcast overlay, and what does it take to
keep it safe when some peers collude?

Setup (a Perigee-style toy, see Mao et al., PODC 2020)
------------------------------------------------------
* N nodes on a unit square. Link latency grows with distance. Each node has a
  processing delay before it forwards.
* Every node keeps D_OUT outgoing connections; links carry messages both ways.
* Each round, B messages start at random honest nodes and flood the overlay.
  A node scores each neighbour by how late that neighbour's copy arrives
  relative to the first copy (90th percentile over the round; lower is better).
* A fraction f of nodes collude. They have a private 5 ms backbone and 1 ms
  processing, so in the "bait" phase they look like excellent peers.
  After the bait phase they attack in one of two ways:
    - "blackhole": they stop forwarding everything (loud).
    - "censor":    they silently drop the messages of a few chosen honest
                   sources (2 of the 30 messages each round) and keep
                   forwarding the rest quickly. This is quiet: their 90th
                   percentile score stays excellent, so score-based eviction
                   never fires.

Policies compared (all replace 2 outgoing peers per round, so churn is equal)
---------------------------------------------------------------------------
* random        fixed protocol baseline: rotate 2 random slots per round.
* greedy        learned, as published: drop the 2 worst-scored peers, add 2
                peers drawn from a uniform sampler the adversary cannot bias.
* greedy_open   greedy, but one of the two new peers each round is whoever
                dialled in. Colluders dial every honest node with their
                nearest member, every round.
* shared        greedy, but new peers come from neighbours' recommendations
                (honest nodes recommend their best-scored peers truthfully;
                colluders recommend their members nearest to the asker).
* shield_slots  3 of 8 slots are kernel-owned random slots, filled only from
                the uniform sampler, that scores can never touch. The other
                5 slots are learned and open to dial-ins (worst case).
* shield_full   shield_slots + evict dead peers + fall back to random
                selection for 5 rounds if half the learned slots went dead.
* shield_closed (follow-up run, not in the default set) shield_full, but the
                learned slots are filled only from the uniform sampler and
                never from dial-ins. This is the configuration the design
                recommends.

This is a toy. It tests the direction of the claims, not their size in a real
network. Numbers are means over seeds; see results.json for spreads.
"""
import argparse, json, itertools, sys, time
from multiprocessing import Pool
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

N = 400
D_OUT = 8
IN_CAP = 20
B = 30               # messages per round
BAIT, ATK = 40, 20   # rounds
R_MAX = 2.0          # penalty (s) for a copy that never arrives
DEADLINE = 3.0       # a node counts as reached if it gets the message within this
RAND_SLOTS = 3
CENSOR_SHARE = 0.03   # share of honest nodes the colluders censor
CENSOR_MSGS = 2        # victim-origin messages per attack round (of B)

POLICIES = ["random", "greedy", "greedy_open", "shared", "shield_slots", "shield_full"]


def run(args):
    policy, f, attack, seed = args
    rng = np.random.default_rng(seed)
    pos = rng.random((N, 2))
    dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    L = 0.010 + 0.150 * dist / np.sqrt(2)
    adv = np.zeros(N, bool)
    n_adv = int(round(f * N))
    if n_adv:
        adv[rng.choice(N, n_adv, replace=False)] = True
    honest = ~adv
    hon_idx = np.flatnonzero(honest)
    proc = rng.lognormal(mean=np.log(0.020), sigma=0.6, size=N)
    proc[adv] = 0.001
    if n_adv:
        L[np.ix_(adv, adv)] = 0.005
    victims = np.zeros(N, bool)
    victims[rng.choice(hon_idx, max(1, int(round(CENSOR_SHARE * len(hon_idx)))), replace=False)] = True

    out = [set() for _ in range(N)]
    rand_slot = [set() for _ in range(N)]   # only used by shield policies
    indeg = np.zeros(N, int)
    freeze = np.zeros(N, int)

    def can_add(v, w):
        return w != v and w not in out[v] and v not in out[w] and (adv[w] or indeg[w] < IN_CAP)

    def sample_peer(v):
        for _ in range(60):
            w = int(rng.integers(N))
            if can_add(v, w):
                return w
        return None

    def add(v, w, as_rand=False):
        out[v].add(w); indeg[w] += 1
        if as_rand:
            rand_slot[v].add(w)

    def drop(v, u):
        out[v].discard(u); indeg[u] -= 1; rand_slot[v].discard(u)

    adv_idx = np.flatnonzero(adv)
    near_adv = {int(v): adv_idx[np.argsort(L[v, adv_idx])].tolist() for v in hon_idx} if n_adv else {}

    def offer(v):
        """A peer that dialled in: the nearest colluder not yet connected, if any."""
        for a_ in near_adv.get(v, []):
            if can_add(v, a_):
                return a_
        return sample_peer(v)

    shield = policy.startswith("shield")
    open_learn = policy in ("greedy_open", "shield_slots", "shield_full")
    for v in rng.permutation(N):
        v = int(v)
        while len(out[v]) < D_OUT:
            if adv[v]:
                w = int(rng.choice(hon_idx))
                if not can_add(v, w):
                    continue
            else:
                w = sample_peer(v)
                if w is None:
                    break
            add(v, w, as_rand=(shield and honest[v] and len(rand_slot[v]) < RAND_SLOTS))

    hist = []
    for rnd in range(BAIT + ATK):
        attacking = rnd >= BAIT and n_adv > 0
        rows, cols = [], []
        for v in range(N):
            for u in out[v]:
                rows += [v, u]; cols += [u, v]
        rows = np.array(rows); cols = np.array(cols)
        w = proc[rows] + L[rows, cols]
        G_full = sp.csr_matrix((w, (rows, cols)), shape=(N, N))
        keep = ~adv[rows]
        G_cut = sp.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=(N, N))

        if attacking and attack == "censor":
            vic_idx = np.flatnonzero(victims); oth_idx = np.flatnonzero(honest & ~victims)
            src = np.concatenate([rng.choice(vic_idx, CENSOR_MSGS), rng.choice(oth_idx, B - CENSOR_MSGS)])
        else:
            src = rng.choice(np.flatnonzero(honest & ~victims), B)
        T = np.full((B, N), np.inf)
        fwd = np.ones((B, N), bool)          # does node u forward message m?
        if not attacking:
            T = dijkstra(G_full, directed=True, indices=src)
        elif attack == "blackhole":
            T = dijkstra(G_cut, directed=True, indices=src)
            fwd[:, adv] = False
        else:  # censor
            vm = victims[src]
            if (~vm).any():
                T[~vm] = dijkstra(G_full, directed=True, indices=src[~vm])
            if vm.any():
                T[vm] = dijkstra(G_cut, directed=True, indices=src[vm])
                fwd[np.ix_(vm, adv)] = False
        fwd &= np.isfinite(T)

        # --- network-level metrics over honest nodes
        Th = np.minimum(T[:, honest], DEADLINE + 1)
        reached = Th <= DEADLINE
        reach = reached.mean(axis=1)
        t90 = np.quantile(np.where(reached, Th, DEADLINE), 0.9, axis=1)
        vm = victims[src]
        rec = dict(round=rnd, reach=float(reach.mean()), t90=float(np.median(t90)),
                   reach_victim=float(reach[vm].mean()) if vm.any() else None,
                   reach_other=float(reach[~vm].mean()) if (~vm).any() else None)
        fr, cap, iso = [], 0, 0
        inn = [set() for _ in range(N)]
        for v in range(N):
            for u in out[v]:
                inn[u].add(v)
        for v in hon_idx:
            o = out[v]
            a = sum(adv[u] for u in o)
            fr.append(a / max(1, len(o)))
            cap += (a == len(o) and len(o) > 0)
            nb = o | inn[v]
            iso += all(adv[u] for u in nb) if nb else 1
        rec.update(adv_frac_out=float(np.mean(fr)), all_chosen_adv=cap / len(hon_idx), eclipsed=iso / len(hon_idx))
        hist.append(rec)

        # --- each honest node scores its neighbours from what it saw
        score = [None] * N
        deliv = [None] * N
        best = [None] * N
        for v in hon_idx:
            nb = np.array(sorted(out[v] | inn[v]))
            if len(nb) == 0:
                continue
            A = np.where(fwd[:, nb], T[:, nb] + proc[nb] + L[nb, v], np.inf)
            tmin = A.min(axis=1)
            got = np.isfinite(tmin)
            if got.sum() == 0:
                sc = np.full(len(nb), R_MAX); dr = np.zeros(len(nb))
            else:
                R = A[got] - tmin[got][:, None]
                dr = np.isfinite(R).mean(axis=0)
                R[~np.isfinite(R)] = R_MAX
                sc = np.quantile(R, 0.9, axis=0)
            score[v] = dict(zip(nb.tolist(), sc.tolist()))
            deliv[v] = dict(zip(nb.tolist(), dr.tolist()))
            best[v] = [int(x) for x in nb[np.argsort(sc)[:3]]]

        def recommended(v):
            pool = []
            for u in out[v] | inn[v]:
                if adv[u]:
                    pool += near_adv[v][:3]
                elif best[u]:
                    pool += best[u]
            pool = [w for w in pool if can_add(v, w)]
            return int(rng.choice(pool)) if pool else sample_peer(v)

        # --- policy update (honest nodes only; colluders keep their links)
        for v in hon_idx:
            v = int(v)
            if score[v] is None:
                continue
            sc, dr = score[v], deliv[v]
            outs = list(out[v])
            if policy == "random":
                for u in rng.permutation(outs)[:2]:
                    drop(v, int(u))
            elif policy in ("greedy", "greedy_open", "shared"):
                for u in sorted(outs, key=lambda u: -sc[u])[:2]:
                    drop(v, u)
            else:  # shield_*
                learned = [u for u in outs if u not in rand_slot[v]]
                rslots = list(rand_slot[v])
                n_dead_learned = 0
                if policy in ("shield_full", "shield_closed"):
                    for u in outs:
                        if dr[u] < 0.5:
                            n_dead_learned += u not in rand_slot[v]
                            drop(v, u)
                    if learned and n_dead_learned >= 0.5 * len(learned):
                        freeze[v] = 5
                        for u in [u for u in out[v] if u not in rand_slot[v]]:
                            drop(v, u)
                    learned = [u for u in out[v] if u not in rand_slot[v]]
                    rslots = list(rand_slot[v])
                if rslots and len(rand_slot[v]) >= RAND_SLOTS:
                    drop(v, int(rng.choice(rslots)))
                if learned and len(out[v]) - len(rand_slot[v]) >= D_OUT - RAND_SLOTS:
                    if freeze[v] > 0:
                        drop(v, int(rng.choice(learned)))
                    else:
                        drop(v, max(learned, key=lambda u: sc[u]))
                if freeze[v] > 0:
                    freeze[v] -= 1
            # refill
            first_learned = True
            while len(out[v]) < D_OUT:
                if shield and len(rand_slot[v]) < RAND_SLOTS:
                    w = sample_peer(v)
                    if w is None: break
                    add(v, w, as_rand=True)
                else:
                    if policy == "shared":
                        w = recommended(v)
                    elif open_learn and first_learned and freeze[v] == 0:
                        w = offer(v)
                    else:
                        w = sample_peer(v)
                    first_learned = False
                    if w is None: break
                    add(v, w)
    return dict(policy=policy, f=f, attack=attack, seed=seed, hist=hist)


def summarise(results):
    """Mean over seeds of phase-level metrics."""
    import collections
    g = collections.defaultdict(list)
    for r in results:
        h = r["hist"]
        bait_tail = h[BAIT - 10:BAIT]
        first = h[BAIT]
        atk = h[BAIT:BAIT + ATK]
        m = dict(
            t90_benign_ms=1000 * np.mean([x["t90"] for x in bait_tail]),
            adv_frac_out=np.mean([x["adv_frac_out"] for x in bait_tail]),
            all_chosen_adv=np.mean([x["all_chosen_adv"] for x in bait_tail]),
            eclipsed=np.mean([x["eclipsed"] for x in bait_tail]),
            reach_first_attack_round=first["reach"],
            reach_attack_mean=np.mean([x["reach"] for x in atk]),
            reach_victim_attack_mean=np.mean([x["reach_victim"] for x in atk if x["reach_victim"] is not None]) if any(x["reach_victim"] is not None for x in atk) else float("nan"),
            t90_attack_ms=1000 * np.mean([x["t90"] for x in atk]),
            rounds_to_99=next((i for i, x in enumerate(atk) if x["reach"] >= 0.99), ATK),
        )
        g[(r["policy"], r["f"], r["attack"])].append(m)
    out = []
    for (p, f, a), ms in sorted(g.items()):
        row = dict(policy=p, f=f, attack=a, seeds=len(ms))
        for k in ms[0]:
            vals = np.array([m[k] for m in ms], float)
            row[k] = float(np.nanmean(vals)); row[k + "_sd"] = float(np.nanstd(vals))
        out.append(row)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--bait", type=int, default=BAIT, help="rounds before the attack starts")
    ap.add_argument("--atk", type=int, default=ATK, help="attack rounds")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--fs", default="0.1,0.2,0.3", help="colluder shares")
    a = ap.parse_args()
    BAIT, ATK = a.bait, a.atk            # module globals; workers are forked after this
    POLICIES = a.policies.split(",")
    FS = [float(x) for x in a.fs.split(",")]
    jobs = []
    for p in POLICIES:
        for s in range(a.seeds):
            jobs.append((p, 0.0, "none", s))
            for f in FS:
                for atk in ("blackhole", "censor"):
                    jobs.append((p, f, atk, s))
    t0 = time.time()
    with Pool(a.procs) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(run, jobs, chunksize=2)):
            res.append(r)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{len(jobs)} runs, {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    summ = summarise(res)
    json.dump(dict(params=dict(N=N, D_OUT=D_OUT, B=B, BAIT=BAIT, ATK=ATK, RAND_SLOTS=RAND_SLOTS,
                               CENSOR_SHARE=CENSOR_SHARE, seeds=a.seeds),
                   summary=summ,
                   curves={f"{r['policy']}|{r['f']}|{r['attack']}|{r['seed']}":
                           [[x["reach"], x["t90"], x["adv_frac_out"], x["reach_victim"], x["all_chosen_adv"], x["eclipsed"]] for x in r["hist"]]
                           for r in res}),
              open(a.out, "w"))
    print(f"done in {time.time()-t0:.0f}s -> {a.out}")
