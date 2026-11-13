//! The band and the pick (design study §5, "the band"; poc-spec §6).
//!
//! The kernel, not the policy, decides which candidates a policy may choose
//! from. Eligible candidates are those whose **rule** score is within
//! `margin` of the rule's best. A policy only reorders the eligible set, so
//! whatever it does, the pick's rule score is within `margin` of the rule's
//! best. That property is enforced here and tested for arbitrary policies.

use crate::features::Features;
use crate::rule::Scorer;

/// A candidate peer: an identifier plus the features the node measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// Stable identifier; ties break toward the lower id.
    pub id: u32,
    /// Features measured by this node.
    pub features: Features,
}

/// Result of one banded pick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PickOutcome {
    /// Index (into the candidate slice) of the pick; `None` if there were no
    /// candidates.
    pub chosen: Option<usize>,
    /// Index of the rule's own choice; `None` if there were no candidates.
    pub rule_choice: Option<usize>,
    /// The rule's best score over all candidates.
    pub rule_best_score: i64,
    /// The rule's score of the chosen candidate.
    pub chosen_rule_score: i64,
    /// Number of candidates inside the band.
    pub eligible: usize,
}

impl PickOutcome {
    /// True when a policy chose a different candidate than the rule would.
    #[must_use]
    pub fn contested(&self) -> bool {
        self.chosen != self.rule_choice
    }
}

/// Pick a candidate. With no policy, the rule's best (ties to the lower id).
/// With a policy, the policy's best among candidates whose rule score is at
/// least `rule_best - margin`; a policy tie that includes the rule's choice
/// goes to the rule's choice, other ties to the lower id. A negative
/// `margin` is treated as zero.
pub fn band_pick(
    rule: &impl Scorer,
    policy: Option<&impl Scorer>,
    candidates: &[Candidate],
    margin: i64,
) -> PickOutcome {
    let margin = margin.max(0);
    let rule_scores: Vec<i64> = candidates.iter().map(|c| rule.score(c.features)).collect();

    let rule_choice = best_index(candidates, &rule_scores, |_| true, None);
    let Some(rc) = rule_choice else {
        return PickOutcome {
            chosen: None,
            rule_choice: None,
            rule_best_score: 0,
            chosen_rule_score: 0,
            eligible: 0,
        };
    };
    let rule_best = rule_scores[rc];
    let floor = rule_best.saturating_sub(margin);
    let eligible = rule_scores.iter().filter(|s| **s >= floor).count();

    let chosen = match policy {
        None => rc,
        Some(p) => {
            let policy_scores: Vec<i64> = candidates.iter().map(|c| p.score(c.features)).collect();
            // The band filter is applied by the kernel before the policy's
            // scores are consulted at all.
            best_index(
                candidates,
                &policy_scores,
                |i| rule_scores[i] >= floor,
                Some(rc),
            )
            .unwrap_or(rc)
        }
    };

    PickOutcome {
        chosen: Some(chosen),
        rule_choice: Some(rc),
        rule_best_score: rule_best,
        chosen_rule_score: rule_scores[chosen],
        eligible,
    }
}

/// Highest score among indices passing `allowed`; ties go to `prefer` if it is
/// among the tied best, else to the lowest id (then lowest index).
fn best_index(
    candidates: &[Candidate],
    scores: &[i64],
    allowed: impl Fn(usize) -> bool,
    prefer: Option<usize>,
) -> Option<usize> {
    let best_score = (0..candidates.len())
        .filter(|&i| allowed(i))
        .map(|i| scores[i])
        .max()?;
    if let Some(p) = prefer {
        if allowed(p) && scores[p] == best_score {
            return Some(p);
        }
    }
    (0..candidates.len())
        .filter(|&i| allowed(i) && scores[i] == best_score)
        .min_by_key(|&i| (candidates[i].id, i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule::{FixedRule, FnScorer};
    use crate::table::PointsTable;

    fn cand(id: u32, rtt: u8) -> Candidate {
        Candidate {
            id,
            features: Features::new(rtt, 0, 7, 0, 0).unwrap(),
        }
    }

    #[test]
    fn no_policy_picks_rule_best_and_breaks_ties_on_id() {
        let cs = [cand(9, 3), cand(2, 1), cand(1, 1)];
        let out = band_pick(&FixedRule, None::<&PointsTable>, &cs, 0);
        assert_eq!(out.chosen, Some(2));
    }

    #[test]
    fn policy_restricted_to_band() {
        let cs = [cand(1, 0), cand(2, 1), cand(3, 15)];
        let slow_lover = FnScorer(|f: Features| i64::from(f.rtt()));
        let out = band_pick(&FixedRule, Some(&slow_lover), &cs, 1500);
        assert_eq!(out.chosen, Some(1));
        assert!(out.contested());
    }

    #[test]
    fn policy_tie_goes_to_rule() {
        let cs = [cand(1, 1), cand(0, 0)];
        let flat = FnScorer(|_f: Features| 0);
        let out = band_pick(&FixedRule, Some(&flat), &cs, 10_000);
        assert_eq!(out.chosen, Some(1));
        assert!(!out.contested());
    }

    #[test]
    fn empty_candidates() {
        let out = band_pick(&FixedRule, None::<&PointsTable>, &[], 10);
        assert_eq!(out.chosen, None);
    }
}
