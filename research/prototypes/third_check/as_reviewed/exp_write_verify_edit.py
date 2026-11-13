"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.
Experiment W1: write, check, edit, diff, and attack.
"""
import json, time
import numpy as np
from wnet import *
from weights import *
import world

out = {}
t0 = time.time()
ref = baseline_rule(GRID)
test = world.candidates(np.random.default_rng(999), 500_000)
cost = lambda net: world.evaluate(net, test)[0]


def report(net, prev=None, s_prev=None):
    s = net(GRID)
    r = dict(nonzero_numbers=net.nonzero(), monotone_violations=len(monotone_violations(s)),
             gate_violations=gate_violations(s), limits=limit_report(s), mean_cost_ms=cost(net))
    r["passes_direction_and_gate"] = bool(r["monotone_violations"] == 0 and r["gate_violations"] == (0, 0))
    r["passes_every_check"] = bool(r["passes_direction_and_gate"] and r["limits"]["within_limits"])
    if prev is not None:
        ps = pair_stats(s_prev, s)
        r.update(patch=diff_weights(prev, net), numbers_changed=len(diff_weights(prev, net)),
                 cells_with_changed_score=int(np.count_nonzero(s_prev != s)), pairs=ps,
                 share_of_pairs_reordered=ps["reversed"] / ps["total_pairs"])
    return s, r


# 1. WRITE the baseline as weights; equal to the rule on every input?
base = write_baseline()
s_base, r = report(base)
r["cells_differing_from_rule"] = int(np.count_nonzero(s_base != ref))
out["domain_cells"] = int(N_CELLS); out["numbers_in_network"] = base.n_weights(); out["baseline"] = r

# 2. EDIT 1 (structural): prefer the longer-known peer among baseline ties
e1 = edit_prefer_known(base)
s_e1, out["edit1_prefer_known"] = report(e1, base, s_base)

# 3. EDIT 2 (calibration): the price of a loss step, -4 -> -6; first as I wrote it, then corrected
e2a = edit_loss_weight_naive(e1, -6); _, out["edit2a_loss_weight_naive"] = report(e2a, e1, s_e1)
e2 = edit_loss_weight(e1, -6); _, out["edit2b_loss_weight"] = report(e2, e1, s_e1)

# 4. ATTACKS
p_find = lambda cells, k: 1 - (1 - cells / N_CELLS) ** k
out["random_testing"] = {
    "my_editing_mistake_192_cells": {str(k): p_find(192, k) for k in (100, 1_000, 10_000)},
    "one_trigger_cell": {str(k): p_find(1, k) for k in (1_000, 10_000, 100_000)}}
attacks = {}
for name, fn in (("trigger_one_cell", plant_trigger), ("reviewer_balance_lock", hostile_balance_lock),
                 ("reviewer_club_age_and_balance", hostile_club), ("reviewer_flatten", hostile_flatten),
                 ("mine_biased_inside_limits", biased_inside_limits)):
    net = fn(e1); s, r = report(net, e1, s_e1)
    if name == "trigger_one_cell":
        cells = np.flatnonzero(s != s_e1)
        r["trigger_cells"] = [describe(GRID[i]) for i in cells]; r["score_at_trigger"] = [int(s[i]) for i in cells]
        r["top_honest_score"] = int(s_e1.max())
    attacks[name] = r
out["attacks"] = attacks
out["rule_mean_cost_ms"] = cost(base)

# 5. The same policies with ONE WEIGHT PER STEP (a single linear layer over thermometer-coded features)
sb, s1 = steps_baseline(), steps_prefer_known()
out["one_weight_per_step"] = dict(
    numbers_in_form=sb.n_weights(),
    baseline=dict(equal_to_network_on_every_input=bool(np.array_equal(sb(GRID), s_base)), nonzero_numbers=sb.nonzero(), checks_read_from_weights=sb.read_checks()),
    prefer_known=dict(equal_to_network_on_every_input=bool(np.array_equal(s1(GRID), s_e1)), nonzero_numbers=s1.nonzero(), checks_read_from_weights=s1.read_checks()),
    can_express=dict(trigger_one_cell=False, reviewer_club_age_and_balance=False,
                     note="both need a unit that fires on a combination of features; this form has none"))
for nm, fn in (("trigger_one_cell", plant_trigger), ("reviewer_club_age_and_balance", hostile_club), ("reviewer_balance_lock", hostile_balance_lock)):
    try:
        Steps.from_table(fn(e1)(GRID)); ok = True
    except ValueError:
        ok = False
    out["one_weight_per_step"]["can_express"][nm] = ok
out["seconds_total"] = time.time() - t0
json.dump(out, open("w1_results.json", "w"), indent=1)

b = out["baseline"]; e = out["edit1_prefer_known"]; a = out["edit2a_loss_weight_naive"]; c = out["edit2b_loss_weight"]
print(f"domain: {N_CELLS:,} cells; network: {base.n_weights()} numbers; rule costs {out['rule_mean_cost_ms']:.1f} ms in the toy world")
print(f"BASELINE as weights: {b['nonzero_numbers']} non-zero; differs from the rule on {b['cells_differing_from_rule']} cells; every check {b['passes_every_check']}")
print(f"EDIT 1 prefer-known: {e['numbers_changed']} numbers; pairs reversed {e['pairs']['reversed']:,}; ties broken {e['pairs']['ties_broken']:,} of {e['pairs']['total_pairs']:,}; every check {e['passes_every_check']}")
print(f"EDIT 2a as first written: {a['numbers_changed']} number; gate violations {a['gate_violations']} -> rejected")
print(f"EDIT 2b corrected: {c['numbers_changed']} numbers; cells changed {c['cells_with_changed_score']:,}; pairs reversed {100*c['share_of_pairs_reordered']:.2f}%; every check {c['passes_every_check']}")
print("random testing finds:", {k: {n: round(v, 3) for n, v in d.items()} for k, d in out["random_testing"].items()})
for name, r in attacks.items():
    print(f"ATTACK {name:32s} numbers {r['numbers_changed']} | direction+gate {'pass' if r['passes_direction_and_gate'] else 'FAIL'} | limits {'pass' if r['limits']['within_limits'] else 'FAIL'}"
          f" | pairs reordered {100*r['share_of_pairs_reordered']:.2f}% | cost {r['mean_cost_ms']:.1f} ms")
o = out["one_weight_per_step"]
print(f"ONE WEIGHT PER STEP: {o['numbers_in_form']} numbers; baseline equal {o['baseline']['equal_to_network_on_every_input']} ({o['baseline']['nonzero_numbers']} non-zero), "
      f"prefer-known equal {o['prefer_known']['equal_to_network_on_every_input']} ({o['prefer_known']['nonzero_numbers']} non-zero); can express: {o['can_express']}")
print(f"total {out['seconds_total']:.2f}s")
