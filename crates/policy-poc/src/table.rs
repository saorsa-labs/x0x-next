//! The points table, rung 1 of the policy plane (design study §5, "written
//! weights"; poc-spec §4).
//!
//! A points table is a base plus, for every feature, one weight per step
//! between adjacent levels: 15 + 7 + 7 + 7 + 7 = 43 weights. The score of an
//! input is the base plus, per feature, the cumulative sum of that feature's
//! steps up to the input's level.
//!
//! Rung 1 is additive per feature by construction. It cannot express an
//! interaction such as "loss hurts more on fast paths"; the twin's world two
//! (see [`crate::twin`]) exists to show that limit honestly.

use serde::{Deserialize, Serialize};

use crate::error::PolicyError;
use crate::features::{Feature, Features};
use crate::rule::Scorer;

/// Step weights for each feature: `steps[k]` is added when the level goes
/// from `k` to `k + 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSteps {
    /// 15 rtt steps (16 levels).
    pub rtt: [i64; 15],
    /// 7 loss steps (8 levels).
    pub loss: [i64; 7],
    /// 7 delivery steps (8 levels).
    pub delivery: [i64; 7],
    /// 7 age steps (8 levels).
    pub age: [i64; 7],
    /// 7 balance steps (8 levels).
    pub balance: [i64; 7],
}

impl FeatureSteps {
    /// All step weights zero.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            rtt: [0; 15],
            loss: [0; 7],
            delivery: [0; 7],
            age: [0; 7],
            balance: [0; 7],
        }
    }

    /// The step weights of one feature.
    #[must_use]
    pub fn of(&self, feature: Feature) -> &[i64] {
        match feature {
            Feature::Rtt => &self.rtt,
            Feature::Loss => &self.loss,
            Feature::Delivery => &self.delivery,
            Feature::Age => &self.age,
            Feature::Balance => &self.balance,
        }
    }

    /// Mutable step weights of one feature.
    pub fn of_mut(&mut self, feature: Feature) -> &mut [i64] {
        match feature {
            Feature::Rtt => &mut self.rtt,
            Feature::Loss => &mut self.loss,
            Feature::Delivery => &mut self.delivery,
            Feature::Age => &mut self.age,
            Feature::Balance => &mut self.balance,
        }
    }

    /// Total number of step weights (43).
    pub const COUNT: usize = 15 + 7 + 7 + 7 + 7;
}

/// A rung-1 policy: a base plus 43 step weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointsTable {
    /// Constant added to every score. Irrelevant to ordering, bounded by the
    /// checker.
    pub base: i64,
    /// Per-feature step weights.
    pub steps: FeatureSteps,
}

impl PointsTable {
    /// A table whose every weight is zero.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            base: 0,
            steps: FeatureSteps::zero(),
        }
    }

    /// Contribution of one feature at `level`: the sum of its first `level`
    /// steps (saturating, so a hostile table cannot overflow).
    #[must_use]
    pub fn contribution(&self, feature: Feature, level: u8) -> i64 {
        self.steps
            .of(feature)
            .iter()
            .take(level as usize)
            .fold(0i64, |acc, s| acc.saturating_add(*s))
    }

    /// Express any scorer as a table by reading its per-feature step
    /// differences along the axes through the all-zero input.
    ///
    /// This is exact if and only if the scorer is additive per feature; for
    /// [`crate::FixedRule`] it is exact (proved on all 65,536 inputs by the
    /// tests). Use [`PointsTable::from_scorer_exact`] to have that checked.
    #[must_use]
    pub fn from_scorer(scorer: &impl Scorer) -> Self {
        let origin = Features::zero();
        let base = scorer.score(origin);
        let mut steps = FeatureSteps::zero();
        for feature in Feature::ALL {
            let mut prev = base;
            for (k, slot) in steps.of_mut(feature).iter_mut().enumerate() {
                // `k + 1 < levels <= 16`, so the conversion cannot fail.
                let level = u8::try_from(k + 1).unwrap_or(u8::MAX);
                let here = origin
                    .with(feature, level)
                    .map_or(prev, |f| scorer.score(f));
                *slot = here.saturating_sub(prev);
                prev = here;
            }
        }
        Self { base, steps }
    }

    /// [`PointsTable::from_scorer`], then verify equality on all 65,536
    /// inputs.
    ///
    /// # Errors
    /// [`PolicyError::NotAdditive`] when the scorer has feature
    /// interactions that rung 1 cannot express.
    pub fn from_scorer_exact(scorer: &impl Scorer) -> Result<Self, PolicyError> {
        let table = Self::from_scorer(scorer);
        let mismatches = Features::enumerate_all()
            .filter(|f| table.score(*f) != scorer.score(*f))
            .count();
        if mismatches == 0 {
            Ok(table)
        } else {
            Err(PolicyError::NotAdditive { mismatches })
        }
    }
}

impl Scorer for PointsTable {
    fn score(&self, f: Features) -> i64 {
        Feature::ALL.iter().fold(self.base, |acc, feature| {
            acc.saturating_add(self.contribution(*feature, f.get(*feature)))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::FnScorer;

    #[test]
    fn non_additive_scorer_is_refused() {
        let interacting = FnScorer(|f: Features| i64::from(f.rtt()) * i64::from(f.loss()));
        assert!(matches!(
            PointsTable::from_scorer_exact(&interacting),
            Err(PolicyError::NotAdditive { .. })
        ));
    }
}
