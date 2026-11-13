//! The fixed rule, rung 0 of the policy plane (design study §5; poc-spec §2–§3).
//!
//! [`FixedRule`] is ordinary code. It is the baseline every policy is measured
//! against, and the decision point falls back to it whenever a policy is
//! removed. It is monotone by construction:
//!
//! * the score never rises as rtt rises and never rises as loss rises;
//! * the score never falls as delivery rises;
//! * age and balance only break ties: their total contribution spans less
//!   than [`QUANT`], while primary contributions are multiples of `QUANT`.
//!
//! The rule is additive per feature on purpose. Rung 1 (the points table)
//! cannot express interactions between features, so a rule with interactions
//! could not be written exactly as weights.

use crate::features::Features;

/// Primary features contribute in multiples of this quantum; tie features
/// contribute strictly less than this in total (poc-spec §2).
pub const QUANT: i64 = 100;

/// Anything that turns a feature vector into a score. Higher is better.
pub trait Scorer {
    /// Score one candidate's features.
    fn score(&self, f: Features) -> i64;
}

impl<S: Scorer + ?Sized> Scorer for &S {
    fn score(&self, f: Features) -> i64 {
        (**self).score(f)
    }
}

/// Wraps any closure as a [`Scorer`] (used for adversarial policies in tests
/// and in the band demonstration).
#[derive(Debug, Clone, Copy)]
pub struct FnScorer<F>(pub F);

impl<F: Fn(Features) -> i64> Scorer for FnScorer<F> {
    fn score(&self, f: Features) -> i64 {
        (self.0)(f)
    }
}

/// The rung-0 rule: hand-written, additive, monotone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FixedRule;

impl FixedRule {
    /// Per-level rtt term: each level costs 15 quanta.
    #[must_use]
    pub fn rtt_term(level: u8) -> i64 {
        -15 * QUANT * i64::from(level)
    }

    /// Per-level loss term: each level costs 12 quanta.
    #[must_use]
    pub fn loss_term(level: u8) -> i64 {
        -12 * QUANT * i64::from(level)
    }

    /// Per-level delivery term: very poor delivery (levels 0 and 1) is
    /// punished hard, then each better level gains a little.
    #[must_use]
    pub fn delivery_term(level: u8) -> i64 {
        let quanta = match level {
            0 => -200,
            1 => -150,
            2 => -40,
            3 => -30,
            4 => -20,
            5 => -12,
            6 => -6,
            _ => 0,
        };
        quanta * QUANT
    }

    /// Per-level balance term: a tie-break of 8 points per level (at most 56,
    /// below one quantum).
    #[must_use]
    pub fn balance_term(level: u8) -> i64 {
        8 * i64::from(level)
    }

    /// Per-level age term: the fixed rule ignores age.
    #[must_use]
    pub fn age_term(_level: u8) -> i64 {
        0
    }

    /// The primary part of the score (multiple of [`QUANT`]).
    #[must_use]
    pub fn primary(f: Features) -> i64 {
        Self::rtt_term(f.rtt()) + Self::loss_term(f.loss()) + Self::delivery_term(f.delivery())
    }
}

impl Scorer for FixedRule {
    fn score(&self, f: Features) -> i64 {
        Self::primary(f) + Self::age_term(f.age()) + Self::balance_term(f.balance())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::Feature;

    #[test]
    fn rule_is_monotone_and_tie_bounded() {
        let rule = FixedRule;
        for f in Features::enumerate_all() {
            let s = rule.score(f);
            if let Some(g) = f.with(Feature::Rtt, f.rtt() + 1) {
                assert!(rule.score(g) <= s);
            }
            if let Some(g) = f.with(Feature::Loss, f.loss() + 1) {
                assert!(rule.score(g) <= s);
            }
            if let Some(g) = f.with(Feature::Delivery, f.delivery() + 1) {
                assert!(rule.score(g) >= s);
            }
            assert_eq!(FixedRule::primary(f) % QUANT, 0);
            let tie = s - FixedRule::primary(f);
            assert!((0..QUANT).contains(&tie));
        }
    }
}
