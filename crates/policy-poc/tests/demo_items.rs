//! One assertion group per evidence-log item (poc-spec §10).
#![cfg(feature = "pqc")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::sync::OnceLock;

use policy_poc::demo::{self, DemoReport};
use policy_poc::*;

fn report() -> &'static DemoReport {
    static REPORT: OnceLock<DemoReport> = OnceLock::new();
    REPORT.get_or_init(|| demo::run().unwrap())
}

#[test]
fn demo_is_deterministic_apart_from_signatures() {
    let again = demo::run().unwrap();
    // ML-DSA signing is hedged, but sizes and every decision are fixed.
    assert_eq!(again.lines, report().lines);
    assert_eq!(report().public_key_len, 1952);
}

#[test]
fn item1_rule_as_weights_equal_on_all_inputs() {
    let s = &report().step1;
    assert_eq!(s.inputs_checked, 65_536);
    assert_eq!(s.mismatches, 0);
    assert_eq!(s.table, PointsTable::from_scorer(&FixedRule));
}

#[test]
fn item2_tie_edit_reorders_only_ties() {
    let s = &report().step2;
    assert!(s.check_passed);
    assert_eq!(s.diff.reversed_pairs, 0);
    assert_eq!(s.diff.newly_tied_pairs, 0);
    assert!(s.diff.newly_ordered_pairs > 0);
    assert!(s.examples_are_primary_ties);
    assert!(s.diff.preserves_existing_preferences());
}

#[test]
fn item3_trigger_rejected_by_direction_and_limit_checks() {
    let s = &report().step3;
    assert!(s.trigger_bonus >= 100 * QUANT);
    let PatchDecision::RejectedByChecker(failures) = &s.decision else {
        panic!("trigger patch not rejected by checker: {:?}", s.decision);
    };
    assert!(failures
        .iter()
        .any(|f| matches!(f, CheckFailure::Direction { .. })));
    assert!(failures
        .iter()
        .any(|f| matches!(f, CheckFailure::StepLimit { .. })));
}

#[test]
fn item4_worse_patch_rejected_by_trial_and_drifting_patch_removed_by_watch() {
    let s = &report().step4;
    assert!(Checker::check(&s.worse.table).is_ok());
    let PatchDecision::RejectedByTrial { trial, .. } = &s.worse_decision else {
        panic!("worse patch not rejected by trial");
    };
    assert_eq!(trial.verdict, Verdict::RejectWorse);
    assert!(trial.mean_cost_gain < 0.0);

    assert!(matches!(s.drifting_decision, PatchDecision::Adopted { .. }));
    let r = s.removal.expect("watch should remove the drifting policy");
    assert!(r.window_mean < r.bound);
    assert!(s.rule_restored);
}

#[test]
fn item5_fitted_patch_adopted_and_survives_with_gain() {
    let s = &report().step5;
    let PatchDecision::Adopted { trial, .. } = &s.decision else {
        panic!("fitted patch not adopted");
    };
    assert!(trial.gain_percent() > 1.0);
    assert!(s.survived_watch);
    assert!(s.watch_samples > 100);
    assert!(s.watch_policy_mean < s.watch_rule_mean);
    let (r1, f1, o1) = s.world_one;
    assert!(o1 <= f1 && f1 < r1);
    let (r2, f2, o2) = s.world_two;
    assert!(
        o2 < f2 && f2 < r2,
        "world two: fitted beats rule, not oracle"
    );
    let (_, fu1, ou1) = s.world_one_unbanded;
    let (_, fu2, ou2) = s.world_two_unbanded;
    assert!((fu2 - ou2) / ou2 > 10.0 * (fu1 - ou1) / ou1);
}

#[test]
fn item6_band_bounds_awful_policy() {
    let s = &report().step6;
    assert!(s.max_shortfall <= s.margin);
    let (rule, banded, free) = s.mean_costs;
    assert!(rule <= banded && banded < free);
    assert!(free > 3.0 * banded);
    let (worst_banded, worst_free) = s.worst_ratio;
    assert!(worst_banded < worst_free / 10.0);
}
