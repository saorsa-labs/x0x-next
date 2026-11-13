"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W5: how many live decisions does a bounded trial need?

A node trials a candidate policy on a share of real decisions and compares what it observes
with the incumbent. Outcomes are noisy (here: true cost times lognormal noise, sigma 0.5).
This asks how many decisions per arm a one-sided test needs (5% false alarm, 90% power) to
tell the two apart, for a harmful patch that passes every check and for an honest gain.
Normal approximation on the logarithm of the observed cost; arithmetic, not a field result.
"""
import json
import numpy as np
from wnet import *
from weights import *
from world import *
from exp_refine_vs_train import fit_steps

Z = 1.645 + 1.282


def arm(score_fn, C, world, rng):
    pick = choose(score_fn, C)
    x = C[np.arange(len(C)), pick]
    return true_cost(x, world) * rng.lognormal(0.0, 0.5, len(C))


if __name__ == "__main__":
    rng = np.random.default_rng(31337)
    C = candidates(rng, 400_000)
    out = []
    base = write_baseline()
    x, obs = log_evidence(np.random.default_rng(200000), 20_000, "product")
    fitted, _ = fit_steps(x, obs, 0)
    cases = [("harmful patch inside every limit (loss priced at 32 rtt steps)", biased_inside_limits(base)),
             ("reviewer's flatten patch (fails the limits; shown for scale)", hostile_flatten(base)),
             ("honest gain: step weights fitted to 20,000 logged decisions", fitted)]
    for name, pol in cases:
        a = arm(base, C, "product", rng); b = arm(pol, C, "product", rng)
        la, lb = np.log(a), np.log(b)
        d = lb.mean() - la.mean()
        n = Z ** 2 * (la.var() + lb.var()) / d ** 2
        out.append(dict(case=name, incumbent_mean_ms=float(a.mean()), candidate_mean_ms=float(b.mean()),
                        change_in_mean_cost=float(b.mean() / a.mean() - 1), decisions_per_arm_to_tell_apart=int(np.ceil(n))))
        print(f"{name}: {a.mean():.1f} -> {b.mean():.1f} ms ({100*(b.mean()/a.mean()-1):+.1f}%), about {int(np.ceil(n)):,} decisions per arm")
    json.dump(out, open("w5_results.json", "w"), indent=1)
