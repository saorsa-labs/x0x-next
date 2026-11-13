//! Rule, table and checker tests (poc-spec §1–§5).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use policy_poc::checker::{MAX_BASE, MAX_STEP, MAX_TIE_STEP};
use policy_poc::*;

#[test]
fn input_space_is_65536() {
    assert_eq!(INPUT_SPACE, 65_536);
    assert_eq!(Features::enumerate_all().count(), 65_536);
}

#[test]
fn rule_as_table_equal_on_all_65536_inputs() {
    let table = PointsTable::from_scorer(&FixedRule);
    let mut checked = 0;
    for f in Features::enumerate_all() {
        assert_eq!(table.score(f), FixedRule.score(f), "mismatch at {f}");
        checked += 1;
    }
    assert_eq!(checked, 65_536);
    assert_eq!(PointsTable::from_scorer_exact(&FixedRule).unwrap(), table);
    assert_eq!(FeatureSteps::COUNT, 43);
}

#[test]
fn rule_table_passes_checker() {
    let table = PointsTable::from_scorer(&FixedRule);
    let report = Checker::check(&table).unwrap();
    assert!(report.tie_spread < QUANT);
}

#[test]
fn direction_failures_detected() {
    let mut t = PointsTable::from_scorer(&FixedRule);
    t.steps.rtt[0] = QUANT;
    t.steps.loss[3] = QUANT;
    t.steps.delivery[6] = -QUANT;
    let failures = Checker::check(&t).unwrap_err();
    let dirs: Vec<_> = failures
        .iter()
        .filter_map(|f| match f {
            CheckFailure::Direction { feature, step, .. } => Some((*feature, *step)),
            _ => None,
        })
        .collect();
    assert_eq!(
        dirs,
        vec![
            (Feature::Rtt, 0),
            (Feature::Loss, 3),
            (Feature::Delivery, 6)
        ]
    );
}

#[test]
fn role_failures_detected() {
    // Primary step that is not a multiple of QUANT.
    let mut t = PointsTable::from_scorer(&FixedRule);
    t.steps.rtt[2] = -150;
    assert!(Checker::check(&t)
        .unwrap_err()
        .iter()
        .any(|f| matches!(f, CheckFailure::PrimaryNotQuantized { .. })));

    // Tie features whose combined spread reaches QUANT, each step in limits.
    let mut t = PointsTable::from_scorer(&FixedRule);
    t.steps.age = [MAX_TIE_STEP; 7];
    let failures = Checker::check(&t).unwrap_err();
    assert!(failures
        .iter()
        .any(|f| matches!(f, CheckFailure::TieSpreadTooLarge { spread } if *spread >= QUANT)));
    assert!(!failures
        .iter()
        .any(|f| matches!(f, CheckFailure::StepLimit { .. })));

    // A negative excursion counts too: +16, -16, +16 ... spread is still bounded.
    let mut t = PointsTable::from_scorer(&FixedRule);
    t.steps.age = [16, -16, 16, -16, 16, -16, 16];
    assert!(Checker::check(&t).is_ok());
}

#[test]
fn limit_failures_detected() {
    let mut t = PointsTable::from_scorer(&FixedRule);
    t.steps.rtt[1] = -(MAX_STEP + QUANT);
    t.steps.balance[0] = MAX_TIE_STEP + 1;
    t.base = MAX_BASE + 1;
    let failures = Checker::check(&t).unwrap_err();
    assert!(failures.iter().any(|f| matches!(
        f,
        CheckFailure::StepLimit {
            feature: Feature::Rtt,
            step: 1,
            ..
        }
    )));
    assert!(failures.iter().any(|f| matches!(
        f,
        CheckFailure::StepLimit {
            feature: Feature::Balance,
            step: 0,
            ..
        }
    )));
    assert!(failures
        .iter()
        .any(|f| matches!(f, CheckFailure::BaseLimit { .. })));
}

#[test]
fn diff_of_identical_scorers_is_empty() {
    let t = PointsTable::from_scorer(&FixedRule);
    let d = Checker::diff(&t, &FixedRule);
    assert_eq!(d.deltas.len(), 65_536);
    assert_eq!(d.changed_inputs, 0);
    assert_eq!(d.reversed_pairs, 0);
    assert_eq!(d.newly_ordered_pairs, 0);
    assert_eq!(d.newly_tied_pairs, 0);
}

#[test]
fn diff_counts_are_exact_for_negation_and_flattening() {
    let n = 65_536u64;
    let all_pairs = n * (n - 1) / 2;
    // Pairs the rule ties, counted directly.
    let mut scores: Vec<i64> = Features::enumerate_all()
        .map(|f| FixedRule.score(f))
        .collect();
    scores.sort_unstable();
    let mut ties = 0u64;
    let mut i = 0;
    while i < scores.len() {
        let mut j = i;
        while j < scores.len() && scores[j] == scores[i] {
            j += 1;
        }
        let k = (j - i) as u64;
        ties += k * (k - 1) / 2;
        i = j;
    }

    // Negation reverses every strictly ordered pair and keeps every tie.
    let neg = FnScorer(|f: Features| -FixedRule.score(f));
    let d = Checker::diff(&neg, &FixedRule);
    assert_eq!(d.reversed_pairs, all_pairs - ties);
    assert_eq!(d.newly_ordered_pairs, 0);
    assert_eq!(d.newly_tied_pairs, 0);
    assert!(!d.reversed_examples.is_empty());
    for (a, b) in &d.reversed_examples {
        assert!(FixedRule.score(*a) > FixedRule.score(*b));
        assert!(neg.score(*b) > neg.score(*a));
    }

    // A constant scorer ties everything the rule ordered.
    let flat = FnScorer(|_f: Features| 0);
    let d = Checker::diff(&flat, &FixedRule);
    assert_eq!(d.newly_tied_pairs, all_pairs - ties);
    assert_eq!(d.reversed_pairs, 0);
}

#[test]
fn tie_edit_only_reorders_exact_ties() {
    let mut t = PointsTable::from_scorer(&FixedRule);
    t.steps.age = [1; 7];
    assert!(Checker::check(&t).is_ok());
    let d = Checker::diff(&t, &FixedRule);
    assert!(d.preserves_existing_preferences());
    assert!(d.newly_ordered_pairs > 0);
    assert_eq!(d.changed_inputs, 65_536 - 65_536 / 8);
    for (a, b) in &d.newly_ordered_examples {
        assert_eq!(FixedRule.score(*a), FixedRule.score(*b));
        assert!(a.age() > b.age());
    }
}

#[test]
fn hostile_table_cannot_overflow_score() {
    let mut t = PointsTable::zero();
    t.base = i64::MAX;
    t.steps.rtt = [i64::MAX; 15];
    let f = Features::new(15, 0, 0, 0, 0).unwrap();
    assert_eq!(t.score(f), i64::MAX);
}
