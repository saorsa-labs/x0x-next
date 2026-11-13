//! Twin, trial and watch tests (poc-spec §7–§8).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use policy_poc::fit::{fit_table, log_evidence};
use policy_poc::trial::{self, DEFAULT_MARGIN};
use policy_poc::twin::{mean_expected_cost, DEFAULT_CANDIDATES};
use policy_poc::watch::Removal;
use policy_poc::*;

fn fitted(world: WorldParams, seed: u64) -> PointsTable {
    fit_table(&log_evidence(world, seed, 50_000))
}

fn costs(world: WorldParams, t: &PointsTable, margin: i64) -> (f64, f64, f64) {
    let r = mean_expected_cost(world, 99, 20_000, |_, d| {
        band_pick(&FixedRule, None::<&PointsTable>, &d.candidates, 0).chosen
    });
    let f = mean_expected_cost(world, 99, 20_000, |_, d| {
        band_pick(&FixedRule, Some(t), &d.candidates, margin).chosen
    });
    let o = mean_expected_cost(world, 99, 20_000, |tw, d| tw.oracle_pick(d));
    (r, f, o)
}

#[test]
fn twin_is_deterministic() {
    let a = log_evidence(WorldParams::world_one(), 3, 1_000);
    let b = log_evidence(WorldParams::world_one(), 3, 1_000);
    assert_eq!(a, b);
    assert_eq!(fit_table(&a), fit_table(&b));
}

#[test]
fn fitted_tables_pass_the_checker() {
    for w in [WorldParams::world_one(), WorldParams::world_two()] {
        assert!(Checker::check(&fitted(w, 1)).is_ok());
    }
}

#[test]
fn world_one_fit_approaches_oracle_world_two_does_not() {
    let w1 = WorldParams::world_one();
    let w2 = WorldParams::world_two();
    let (r1, f1, o1) = costs(w1, &fitted(w1, 1), i64::MAX);
    let (r2, f2, o2) = costs(w2, &fitted(w2, 1), i64::MAX);
    // Both fitted tables beat the rule.
    assert!(f1 < r1);
    assert!(f2 < r2);
    // World one: within 0.2% of the oracle. World two: a clear gap.
    let gap1 = (f1 - o1) / o1;
    let gap2 = (f2 - o2) / o2;
    assert!(gap1 < 0.002, "world one gap {gap1}");
    assert!(gap2 > 0.005, "world two gap {gap2}");
    assert!(gap2 > 10.0 * gap1);
}

#[test]
fn trial_adopts_fitted_and_rejects_worse() {
    let cfg = TrialConfig::default();
    let w = WorldParams::world_one();
    let good = trial::run(&fitted(w, 1), w, 5, &cfg).unwrap();
    assert_eq!(good.verdict, Verdict::Adopt);
    assert!(good.mean_log_gain > 0.0 && good.mean_cost_gain > 0.0);
    assert!(good.gain_percent() > 1.0);

    let mut bad = PointsTable::from_scorer(&FixedRule);
    bad.steps.rtt = [0; 15];
    bad.steps.loss = [0; 7];
    assert!(Checker::check(&bad).is_ok());
    let rep = trial::run(&bad, w, 5, &cfg).unwrap();
    assert_eq!(rep.verdict, Verdict::RejectWorse);
    assert!(rep.decisions < cfg.max_decisions, "should stop early");
}

#[test]
fn trial_rejects_a_copy_of_the_rule_as_inconclusive() {
    let cfg = TrialConfig {
        max_decisions: 5_000,
        ..TrialConfig::default()
    };
    let same = PointsTable::from_scorer(&FixedRule);
    let rep = trial::run(&same, WorldParams::world_one(), 5, &cfg).unwrap();
    assert_eq!(rep.verdict, Verdict::RejectInconclusive);
    assert_eq!(rep.contested, 0);
    assert_eq!(rep.decisions, 5_000);
}

#[test]
fn trial_config_validated() {
    let bad = TrialConfig {
        check_every: 0,
        ..TrialConfig::default()
    };
    assert!(trial::run(&FixedRule, WorldParams::world_one(), 1, &bad).is_err());
    let bad = TrialConfig {
        min_decisions: 10,
        max_decisions: 5,
        ..TrialConfig::default()
    };
    assert!(bad.validate().is_err());
    assert!(Watch::new(WatchConfig {
        window: 1,
        ..WatchConfig::default()
    })
    .is_err());
}

#[test]
fn watch_removes_policy_that_turns_bad_after_world_shift() {
    let w = WorldParams::world_one();
    let policy = fitted(w, 2);
    // Good during the trial.
    let rep = trial::run(&policy, w, 5, &TrialConfig::default()).unwrap();
    assert_eq!(rep.verdict, Verdict::Adopt);
    // Then the world shifts.
    let mut point = DecisionPoint::new(policy, DEFAULT_MARGIN, WatchConfig::default()).unwrap();
    let mut twin = Twin::new(WorldParams::world_one_shifted(), 11, DEFAULT_CANDIDATES);
    let mut removal: Option<Removal> = None;
    for _ in 0..200_000 {
        if let WatchEvent::Removed(r) = point.decide(&mut twin).event {
            removal = Some(r);
            break;
        }
    }
    let r = removal.expect("policy should be removed");
    assert!(r.window_mean < r.bound && r.bound < 0.0);
    assert!(!point.policy_active());
    // The decision point is back on the fixed rule.
    for _ in 0..2_000 {
        let rec = point.decide(&mut twin);
        assert!(!rec.contested);
        assert_eq!(rec.cost, rec.rule_cost);
    }
}

#[test]
fn watch_keeps_a_good_policy() {
    let w = WorldParams::world_one();
    let mut point =
        DecisionPoint::new(fitted(w, 1), DEFAULT_MARGIN, WatchConfig::default()).unwrap();
    let mut twin = Twin::new(w, 11, DEFAULT_CANDIDATES);
    let (mut used, mut rule) = (0.0, 0.0);
    for _ in 0..30_000 {
        let rec = point.decide(&mut twin);
        used += rec.cost;
        rule += rec.rule_cost;
    }
    assert!(point.policy_active());
    assert!(point.watch().samples() > 100);
    assert!(used < rule);
}

#[test]
fn watch_samples_one_in_ten_contested() {
    let mut watch = Watch::new(WatchConfig::default()).unwrap();
    let mut removed_at = None;
    for i in 1..=10_000u64 {
        // The policy is always twice as costly as the rule.
        if let WatchEvent::Removed(r) = watch.observe_contested(10.0, 20.0) {
            removed_at = Some((i, r));
            break;
        }
    }
    let (i, r) = removed_at.unwrap();
    // Removed as soon as the 50-sample window fills: 50 samples × 10.
    assert_eq!(i, 500);
    assert_eq!(r.samples, 50);
    assert!((r.running_margin + 50.0 * 2f64.ln()).abs() < 1e-9);
    assert_eq!(watch.observe_contested(10.0, 20.0), WatchEvent::NotSampled);
}
