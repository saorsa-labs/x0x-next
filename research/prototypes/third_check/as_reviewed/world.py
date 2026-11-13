"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.

Two toy worlds in which a node picks the best of three candidate peers and pays a cost.
Both differ from what the baseline rule assumed.

  "product": cost = 20 (rtt+1) x (1 + 0.35 loss) x (1 + 0.1 (7 - deliv)). A product of
             one-feature terms, so its logarithm is a sum of one-feature terms.
  "timeout": a lost packet costs a fixed retransmission timeout, so loss hurts most on
             fast paths. Not a sum of one-feature terms in any simple scale. This world
             was suggested by the independent reviewer of the study.

In both, a peer whose delivery bucket is below 2 times out (3,000 ms).
"""
import numpy as np
from wnet import *
from weights import write_baseline

P_LOSS = np.array([.40, .20, .13, .09, .07, .05, .04, .02])
P_DELIV = np.array([.03, .04, .06, .09, .13, .20, .22, .23])
TIMEOUT = 3000.0
LOSS_PROB = np.array([0, .02, .05, .10, .15, .20, .30, .40])
WORLDS = ("product", "timeout")


def true_cost(x, world="product"):
    rtt, loss, deliv = x[..., 0], x[..., 1], x[..., 2]
    if world == "product":
        cost = 20.0 * (rtt + 1) * (1 + 0.35 * loss) * (1 + 0.10 * (7 - deliv))
    else:
        p = LOSS_PROB[loss]
        cost = (20.0 * (rtt + 1) + 200.0 * p) / (1 - p) * (1 + 0.10 * (7 - deliv))
    return np.where(deliv < 2, TIMEOUT, cost)


def candidates(rng, n):
    """n decisions, each with three candidate peers"""
    c = np.empty((n, 3, 5), np.int64)
    c[..., 0] = rng.integers(0, 16, (n, 3))
    c[..., 1] = rng.choice(8, (n, 3), p=P_LOSS)
    c[..., 2] = rng.choice(8, (n, 3), p=P_DELIV)
    c[..., 3] = rng.integers(0, 8, (n, 3))
    c[..., 4] = rng.integers(0, 8, (n, 3))
    return c


def choose(score_fn, C, rng=None):
    """highest score wins; the kernel breaks ties at random"""
    s = np.asarray(score_fn(C.reshape(-1, 5)), float).reshape(C.shape[0], 3)
    if rng is not None:
        s = s + rng.random(s.shape) * 1e-6
    return np.argmax(s, axis=1)


def kernel_gate(score_fn):
    """the gate as the kernel applies it: an ineligible candidate is never preferred"""
    return lambda X: np.where(X[:, 2] < 2, -1e18, np.asarray(score_fn(X), float))


def evaluate(score_fn, C, world="product"):
    pick = choose(score_fn, C)
    x = C[np.arange(len(C)), pick]
    cost = true_cost(x, world)
    any_eligible = (C[..., 2] >= 2).any(axis=1)
    bad = (x[:, 2] < 2) & any_eligible
    return float(cost.mean()), float(1e5 * bad.mean())


def log_evidence(rng, n, world="product", explore=0.10):
    """what one node would have on disk: the peer it used and what happened.
    It follows the baseline rule, with a share of random choices (the kernel's random share)."""
    C = candidates(rng, n)
    pick = choose(write_baseline(), C)
    rnd = rng.random(n) < explore
    pick = np.where(rnd, rng.integers(0, 3, n), pick)
    x = C[np.arange(n), pick]
    obs = true_cost(x, world) * rng.lognormal(0.0, 0.5, n)
    return x, obs
