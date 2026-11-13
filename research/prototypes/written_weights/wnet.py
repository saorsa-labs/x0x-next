"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.

A policy as a tiny integer network whose weights are WRITTEN, not trained.
This module holds the pieces shared by the experiments:

  * the decision point's feature grid (every input the network can ever see),
  * a fixed-shape integer network and its evaluator,
  * the checks a node would run before it lets a set of weights decide anything:
    equality with a reference rule, monotonicity in each declared direction,
    the gate, and an exact "atlas diff" between two sets of weights.

Everything is exact integer arithmetic over the whole input domain. Nothing is
sampled, so nothing can hide between samples.
"""
import itertools
import numpy as np

# ---- the decision point: scoring a candidate peer or relay ------------------
# name, number of levels, direction (+1: score must not fall as the feature rises,
#                                    -1: score must not rise as the feature rises)
FEATURES = [("rtt", 16, -1),     # round-trip time bucket, 0 fastest
            ("loss", 8, -1),     # loss bucket, 0 none
            ("deliv", 8, +1),    # delivery-rate bucket, 7 best
            ("age", 8, +1),      # how long this node has known the peer
            ("bal", 8, +1)]      # what the peer has given this node, net (reciprocity)
NAMES = [f[0] for f in FEATURES]
LEVELS = [f[1] for f in FEATURES]
DIRS = [f[2] for f in FEATURES]
GRID = np.array(list(itertools.product(*[range(n) for n in LEVELS])), dtype=np.int64)  # 65,536 x 5
N_CELLS = len(GRID)
SHAPE = (5, 8, 4, 1)     # fixed for this decision point: 5 inputs, 8 and 4 hidden units, 1 score


def baseline_rule(X):
    """The fixed protocol's rule, as ordinary code. This is the reference."""
    rtt, loss, deliv = X[:, 0], X[:, 1], X[:, 2]
    score = np.maximum(0, 80 - 3 * rtt - 4 * loss)
    return np.where(deliv < 2, 0, score)


class Net:
    """5 -> 8 -> 4 -> 1, ReLU, int8 weights, int16 biases, 64-bit accumulators.
    With weights and biases at their limits the output can reach about 19.7 billion, which
    does not fit in 32 bits, so a specification must either fix 64-bit accumulators (as
    here) or have the checker reject any policy whose table exceeds a 32-bit bound.
    The independent reviewer found this; an earlier comment here was wrong."""

    def __init__(self):
        self.W1 = np.zeros((8, 5), np.int64); self.b1 = np.zeros(8, np.int64)
        self.W2 = np.zeros((4, 8), np.int64); self.b2 = np.zeros(4, np.int64)
        self.W3 = np.zeros((1, 4), np.int64); self.b3 = np.zeros(1, np.int64)
        self.names = {}          # (layer, unit) -> what the unit means

    def copy(self):
        n = Net()
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(n, k, getattr(self, k).copy())
        n.names = dict(self.names)
        return n

    def check_ranges(self):
        for k in ("W1", "W2", "W3"):
            w = getattr(self, k)
            assert w.min() >= -128 and w.max() <= 127, f"{k} outside int8"
        for k in ("b1", "b2", "b3"):
            b = getattr(self, k)
            assert b.min() >= -32768 and b.max() <= 32767, f"{k} outside int16"

    def __call__(self, X):
        h1 = np.maximum(0, X @ self.W1.T + self.b1)
        h2 = np.maximum(0, h1 @ self.W2.T + self.b2)
        return np.maximum(0, h2 @ self.W3.T + self.b3)[:, 0]

    def n_weights(self):
        return sum(getattr(self, k).size for k in ("W1", "b1", "W2", "b2", "W3", "b3"))

    def nonzero(self):
        return sum(int(np.count_nonzero(getattr(self, k))) for k in ("W1", "b1", "W2", "b2", "W3", "b3"))

    def flat(self):
        return np.concatenate([getattr(self, k).ravel() for k in ("W1", "b1", "W2", "b2", "W3", "b3")])


def diff_weights(a, b):
    """The patch: every position where two sets of weights differ."""
    out = []
    for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
        A, B = getattr(a, k), getattr(b, k)
        for idx in zip(*np.nonzero(A != B)):
            out.append((k, tuple(int(i) for i in idx), int(A[idx]), int(B[idx])))
    return out


# ---- checks over the whole domain -------------------------------------------
def cell_index():
    """strides so that neighbours along one feature can be found by index arithmetic"""
    strides, s = [], 1
    for n in reversed(LEVELS):
        strides.append(s); s *= n
    return list(reversed(strides))


STRIDES = cell_index()


def monotone_violations(scores, tol=0):
    """Every pair of neighbouring cells along a declared direction where the score
    moves the wrong way. Returns a list of (feature, cell, neighbour, score, score')."""
    S = np.asarray(scores).reshape(LEVELS)
    out = []
    for ax, (name, n, d) in enumerate(FEATURES):
        lo = np.take(S, range(0, n - 1), axis=ax)
        hi = np.take(S, range(1, n), axis=ax)
        bad = (hi - lo) * d < -tol          # d=+1: must not fall; d=-1: must not rise
        for idx in zip(*np.nonzero(bad)):
            cell = list(idx); nb = list(idx); nb[ax] += 1
            out.append((name, tuple(int(c) for c in cell), tuple(int(c) for c in nb),
                        float(lo[idx]), float(hi[idx])))
    return out


def gate_violations(scores):
    """Cells with deliv < 2 that score above zero, plus any eligible cell at zero."""
    s = np.asarray(scores)
    inel = GRID[:, 2] < 2
    return int(np.count_nonzero(s[inel] > 0)), int(np.count_nonzero(s[~inel] <= 0))


def pair_stats(old, new):
    """Exact counts over all unordered pairs of cells:
       reversed  : old ranks A above B, new ranks B above A
       ties_broken: old ties them, new orders them
       merged    : old orders them, new ties them
    Computed from the joint histogram, not by looping over 2.1 billion pairs."""
    old = np.asarray(old); new = np.asarray(new)
    ov, oi = np.unique(old, return_inverse=True)
    nv, ni = np.unique(new, return_inverse=True)
    H = np.zeros((len(ov), len(nv)), dtype=object)
    np.add.at(H, (oi, ni), 1)
    H = H.astype(object)
    n = len(old)
    total = n * (n - 1) // 2
    # cells with strictly smaller old and strictly larger new than (o, v)
    C = np.zeros_like(H)
    # prefix over old (exclusive), suffix over new (exclusive)
    pre = np.cumsum(H, axis=0) - H
    suf = np.cumsum(pre[:, ::-1], axis=1)[:, ::-1] - pre
    reversed_pairs = int((H * suf).sum())
    row = H.sum(axis=1); col = H.sum(axis=0)
    same_old = int(sum(int(r) * (int(r) - 1) // 2 for r in row))
    same_new = int(sum(int(c) * (int(c) - 1) // 2 for c in col))
    same_both = int(sum(int(h) * (int(h) - 1) // 2 for h in H.ravel()))
    return dict(total_pairs=total, reversed=reversed_pairs,
                ties_broken=same_old - same_both, merged=same_new - same_both)


def describe(cell):
    return ", ".join(f"{n}={v}" for n, v in zip(NAMES, cell))


# ---- roles and limits the kernel declares for this decision point --------------------
PRIMARY = [0, 1]      # rtt, loss: must matter. One step worse must lower an eligible peer's score.
TIEBREAK = [3, 4]     # age, bal: may only order peers that are level on every other feature.
MAX_STEP_RATIO = 32   # no primary step may be worth more than 32 times another


def tiebreak_violations(scores):
    """Group cells by their (rtt, loss, deliv) setting. Over the tie-break features each
    group spans a band [lowest, highest]. Bands of different groups must not overlap,
    except where two groups have identical bands. Returns the number of offending groups."""
    S = np.asarray(scores).reshape(LEVELS)
    other = [a for a in range(len(LEVELS)) if a not in TIEBREAK]
    lo = S.min(axis=tuple(TIEBREAK)).ravel(); hi = S.max(axis=tuple(TIEBREAK)).ravel()
    order = np.lexsort((hi, lo)); lo, hi = lo[order], hi[order]
    bad = 0
    for i in range(len(lo) - 1):
        same = lo[i] == lo[i + 1] and hi[i] == hi[i + 1]
        if not same and not hi[i] < lo[i + 1]:
            bad += 1
    return bad


def primary_steps(scores):
    """every one-step difference along a primary feature, over eligible cells, as score lost"""
    S = np.asarray(scores).reshape(LEVELS)
    out = []
    for ax in PRIMARY:
        n = LEVELS[ax]
        better = np.take(S, range(0, n - 1), axis=ax); worse = np.take(S, range(1, n), axis=ax)
        el = np.take(GRID[:, 2].reshape(LEVELS), range(0, n - 1), axis=ax) >= 2
        out.append((better - worse)[el])
    return np.concatenate([o.ravel() for o in out])


def limit_report(scores):
    st = primary_steps(scores)
    flat = int(np.count_nonzero(st <= 0))
    ratio = float(st.max() / st.min()) if flat == 0 else float("inf")
    return dict(tiebreak_violations=tiebreak_violations(scores), primary_steps_that_do_not_matter=flat,
                largest_to_smallest_primary_step=ratio, within_limits=bool(tiebreak_violations(scores) == 0 and flat == 0 and ratio <= MAX_STEP_RATIO))


class Steps:
    """The simplest network: one linear layer over thermometer-coded features, so there is
    ONE WEIGHT PER STEP of each feature. score = bias + the sum of the steps a candidate has
    passed. The delivery gate is the kernel's: an ineligible candidate is never offered.
    Everything about such a policy can be read off its weights, with no enumeration:
    it is monotone if every step has the declared sign, and limits are inequalities."""

    def __init__(self):
        self.w = [np.zeros(n - 1, np.int64) for n in LEVELS]
        self.bias = 0

    def __call__(self, X):
        s = np.full(len(X), self.bias, np.int64)
        for f in range(len(LEVELS)):
            cum = np.concatenate([[0], np.cumsum(self.w[f])])
            s += cum[X[:, f]]
        return np.where(X[:, 2] < 2, 0, np.maximum(1, s))        # kernel gate; eligible peers stay above zero

    def n_weights(self):
        return 1 + sum(len(w) for w in self.w)

    def nonzero(self):
        return int(self.bias != 0) + sum(int(np.count_nonzero(w)) for w in self.w)

    def read_checks(self):
        """checks that need only the weights"""
        mono = all(np.all(self.w[f] * DIRS[f] >= 0) for f in range(len(LEVELS)))
        prim = np.concatenate([-self.w[f] for f in PRIMARY])
        matter = bool(np.all(prim > 0))
        ratio = float(prim.max() / prim.min()) if matter else float("inf")
        other = [f for f in range(len(LEVELS)) if f not in TIEBREAK]
        g = int(np.gcd.reduce(np.concatenate([np.abs(self.w[f]) for f in other] + [[0]])))
        swing = int(sum(np.abs(self.w[f]).sum() for f in TIEBREAK))
        return dict(monotone=bool(mono), primary_steps_matter=matter, largest_to_smallest_primary_step=ratio,
                    tiebreak_swing=swing, smallest_gap_between_other_settings=g,
                    tiebreak_only_breaks_ties=bool(swing < g or swing == 0),
                    within_limits=bool(mono and matter and ratio <= MAX_STEP_RATIO and (swing < g or swing == 0)))

    @staticmethod
    def from_table(scores):
        """read step weights off an additive table; raises if the table is not additive"""
        S = np.asarray(scores).reshape(LEVELS)
        st = Steps()
        ref = [0, 0, 7, 0, 0]                       # an eligible reference cell
        for f in range(len(LEVELS)):
            idx = list(ref); vals = []
            for k in range(LEVELS[f]):
                idx[f] = k; vals.append(int(S[tuple(idx)]))
            if f == 2:                              # below the gate the table is zero; steps there are not defined
                vals[0] = vals[1] = vals[2]
            st.w[f] = np.diff(np.array(vals, np.int64))
        st.bias = int(S[tuple(ref)]) - int(np.concatenate([[0], np.cumsum(st.w[2])])[7])
        if not np.array_equal(st(GRID), np.where(GRID[:, 2] < 2, 0, np.maximum(1, np.asarray(scores)))):
            raise ValueError("table is not a sum of one-feature steps")
        return st


class StepsPlus:
    """The exception to the default form: step weights plus a few hidden units that can
    combine features. Each unit is relu(a . x + c) over the raw feature levels, added to the
    score with weight v. Once a unit combines two features, directions and limits can no
    longer be read off the weights, so every check runs on the table (the answer on every
    input). The kernel fixes how many such units a decision point may have."""

    def __init__(self, steps, units=()):
        self.steps = steps
        self.units = list(units)            # each: (a: 5 integers, c: integer, v: integer)

    def __call__(self, X):
        s = np.full(len(X), self.steps.bias, np.int64)
        for f in range(len(LEVELS)):
            cum = np.concatenate([[0], np.cumsum(self.steps.w[f])])
            s += cum[X[:, f]]
        for a, c, v in self.units:
            s += int(v) * np.maximum(0, X @ np.asarray(a, np.int64) + int(c))
        return np.where(X[:, 2] < 2, 0, np.maximum(1, s))

    def n_weights(self):
        return self.steps.n_weights() + sum(len(a) + 2 for a, c, v in self.units)

    def table_checks(self):
        t = self(GRID)
        lim = limit_report(t)
        return dict(wrong_way_pairs=len(monotone_violations(t)), gate=gate_violations(t), **lim,
                    passes=bool(len(monotone_violations(t)) == 0 and gate_violations(t) == (0, 0) and lim["within_limits"]))
