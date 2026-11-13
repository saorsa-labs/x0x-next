//! The band property (poc-spec §6): for any policy, any candidates and any
//! margin, the pick's rule score is within the margin of the rule's best.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use policy_poc::rng::SplitMix64;
use policy_poc::*;

fn random_features(rng: &mut SplitMix64) -> Features {
    Features::from_index(rng.below(INPUT_SPACE as u64) as usize).unwrap()
}

fn random_candidates(rng: &mut SplitMix64) -> Vec<Candidate> {
    let n = 1 + rng.below(8) as usize;
    (0..n)
        .map(|_| Candidate {
            // Small id space so duplicate ids and ties occur.
            id: rng.below(4) as u32,
            features: random_features(rng),
        })
        .collect()
}

/// An arbitrary, unchecked table: any weight, any sign.
fn random_table(rng: &mut SplitMix64) -> PointsTable {
    let mut t = PointsTable::zero();
    t.base = rng.next_u64() as i64;
    for feature in Feature::ALL {
        for s in t.steps.of_mut(feature) {
            *s = (rng.next_u64() as i64) >> rng.below(64);
        }
    }
    t
}

fn random_margin(rng: &mut SplitMix64) -> i64 {
    match rng.below(4) {
        0 => 0,
        1 => -(rng.below(10_000) as i64),
        2 => rng.below(10_000) as i64,
        _ => rng.next_u64() as i64,
    }
}

fn assert_band(out: &PickOutcome, cands: &[Candidate], margin: i64) {
    let rule_best = cands.iter().map(|c| FixedRule.score(c.features)).max();
    let chosen = out.chosen.unwrap();
    let chosen_score = FixedRule.score(cands[chosen].features);
    let best = rule_best.unwrap();
    assert_eq!(out.rule_best_score, best);
    assert_eq!(out.chosen_rule_score, chosen_score);
    assert!(
        chosen_score >= best.saturating_sub(margin.max(0)),
        "pick outside band: best {best}, chosen {chosen_score}, margin {margin}"
    );
}

#[test]
fn band_property_random_tables() {
    let mut rng = SplitMix64::new(0xBA5E);
    for _ in 0..20_000 {
        let cands = random_candidates(&mut rng);
        let policy = random_table(&mut rng);
        let margin = random_margin(&mut rng);
        let out = band_pick(&FixedRule, Some(&policy), &cands, margin);
        assert_band(&out, &cands, margin);
    }
}

#[test]
fn band_property_arbitrary_closures() {
    let mut rng = SplitMix64::new(0xC105E);
    for round in 0..20_000u64 {
        let cands = random_candidates(&mut rng);
        let margin = random_margin(&mut rng);
        let salt = rng.next_u64();
        // A pseudo-random, non-monotone policy (hash of the input).
        let policy = FnScorer(move |f: Features| {
            let mut h = SplitMix64::new(salt ^ f.index() as u64 ^ round);
            h.next_u64() as i64
        });
        let out = band_pick(&FixedRule, Some(&policy), &cands, margin);
        assert_band(&out, &cands, margin);
        // The adversary: always the rule's worst.
        let worst = FnScorer(|f: Features| -FixedRule.score(f));
        let out = band_pick(&FixedRule, Some(&worst), &cands, margin);
        assert_band(&out, &cands, margin);
    }
}

#[test]
fn no_policy_is_rule_choice_and_policy_tie_goes_to_rule() {
    let mut rng = SplitMix64::new(42);
    for _ in 0..5_000 {
        let cands = random_candidates(&mut rng);
        let plain = band_pick(&FixedRule, None::<&PointsTable>, &cands, 1_000);
        assert_eq!(plain.chosen, plain.rule_choice);
        assert!(!plain.contested());
        // Rule's choice has the best rule score and the lowest id among ties.
        let r = plain.rule_choice.unwrap();
        for c in &cands {
            let s = FixedRule.score(c.features);
            let rs = FixedRule.score(cands[r].features);
            assert!(s < rs || (s == rs && c.id >= cands[r].id));
        }
        // A flat policy ties everything, so the rule's choice stands.
        let flat = FnScorer(|_f: Features| 7);
        let out = band_pick(&FixedRule, Some(&flat), &cands, i64::MAX);
        assert_eq!(out.chosen, plain.rule_choice);
    }
}
