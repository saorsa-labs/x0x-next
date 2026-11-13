//! One Measure-mode decision (plan §1, "what it does per decision").
//!
//! Pure code, shared by the live loop and by replay, so a trace is checked
//! against exactly the logic that produced it:
//!
//! 1. [`sample_slots`] draws up to [`CANDIDATES_PER_DECISION`] distinct peers
//!    (partial Fisher–Yates over the configured peer slots).
//! 2. [`choose`] runs [`band_pick`] with the fixed rule and no policy to get
//!    the rule's choice and the band at [`MARGIN`] (6 quanta).
//! 3. One decision in [`EXPLORE_ONE_IN`] picks uniformly among the in-band
//!    candidates instead.
//! 4. One decision in [`COPY_ONE_IN`] also sends a copy: to the rule's
//!    choice when the pick differs from it, otherwise to a uniformly drawn
//!    other in-band candidate (no copy if the band holds only the pick).
//!
//! Randomness: decision `k` uses its own window of the node's single
//! SplitMix64 stream, positions `[k·D, (k+1)·D)` with `D` =
//! [`DRAWS_PER_DECISION`]; unused positions are skipped. A decision draws at
//! most 4 + 4 values, well inside the window.

use policy_poc::rng::SplitMix64;
use policy_poc::twin::DEFAULT_CANDIDATES;
use policy_poc::{band_pick, Candidate, FixedRule, Scorer, QUANT};

/// Stream positions reserved per decision.
pub const DRAWS_PER_DECISION: u64 = 16;
/// SplitMix64's state increment (one draw advances the state by this).
pub const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// Candidates offered per decision (`DEFAULT_CANDIDATES` in twin.rs).
pub const CANDIDATES_PER_DECISION: usize = DEFAULT_CANDIDATES;
/// Band margin for exploration and copies: 6 quanta.
pub const MARGIN: i64 = 6 * QUANT;
/// One decision in this many explores inside the band.
pub const EXPLORE_ONE_IN: u64 = 10;
/// One decision in this many sends a copy.
pub const COPY_ONE_IN: u64 = 10;

/// The generator positioned at decision `decision`'s window of the stream
/// seeded with `seed`.
#[must_use]
pub fn decision_rng(seed: u64, decision: u64) -> SplitMix64 {
    let offset = decision
        .wrapping_mul(DRAWS_PER_DECISION)
        .wrapping_mul(SPLITMIX_GAMMA);
    SplitMix64::new(seed.wrapping_add(offset))
}

fn below(rng: &mut SplitMix64, n: usize) -> usize {
    usize::try_from(rng.below(n as u64)).unwrap_or(0)
}

/// Draw `min(CANDIDATES_PER_DECISION, peer_count)` distinct peer slots.
pub fn sample_slots(rng: &mut SplitMix64, peer_count: usize) -> Vec<usize> {
    let k = CANDIDATES_PER_DECISION.min(peer_count);
    let mut slots: Vec<usize> = (0..peer_count).collect();
    for i in 0..k {
        let j = i + below(rng, peer_count - i);
        slots.swap(i, j);
    }
    slots.truncate(k);
    slots
}

/// What the decision chose. Indices are into the candidate slice.
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// The fixed rule's choice.
    pub rule_choice: usize,
    /// The candidate actually used.
    pub pick: usize,
    /// Whether the exploration draw fired (the pick may still equal the
    /// rule's choice).
    pub explored: bool,
    /// Candidates inside the band, in index order.
    pub eligible: Vec<usize>,
    /// Probability that this procedure picks `pick`.
    pub propensity: f64,
    /// Candidate that also received a copy, if any.
    pub copy: Option<usize>,
}

/// Probability that [`choose`] picks candidate `i`, given the rule's choice
/// and the band.
#[must_use]
pub fn propensity(i: usize, rule_choice: usize, eligible: &[usize]) -> f64 {
    let explore = 1.0 / EXPLORE_ONE_IN as f64;
    let mut p = 0.0;
    if i == rule_choice {
        p += 1.0 - explore;
    }
    if eligible.contains(&i) {
        p += explore / eligible.len() as f64;
    }
    p
}

/// Make the decision. `None` only for an empty candidate set.
pub fn choose(rng: &mut SplitMix64, candidates: &[Candidate]) -> Option<Choice> {
    let out = band_pick(&FixedRule, None::<&FixedRule>, candidates, MARGIN);
    let rule_choice = out.rule_choice?;
    let floor = out.rule_best_score.saturating_sub(MARGIN);
    let eligible: Vec<usize> = (0..candidates.len())
        .filter(|&i| FixedRule.score(candidates[i].features) >= floor)
        .collect();

    let explored = rng.below(EXPLORE_ONE_IN) == 0;
    let pick = if explored {
        eligible
            .get(below(rng, eligible.len()))
            .copied()
            .unwrap_or(rule_choice)
    } else {
        rule_choice
    };

    let copy = if rng.below(COPY_ONE_IN) != 0 {
        None
    } else if pick != rule_choice {
        Some(rule_choice)
    } else {
        // No draw when the band holds only the pick.
        let others: Vec<usize> = eligible.iter().copied().filter(|&i| i != pick).collect();
        if others.is_empty() {
            None
        } else {
            others.get(below(rng, others.len())).copied()
        }
    };

    Some(Choice {
        rule_choice,
        pick,
        explored,
        propensity: propensity(pick, rule_choice, &eligible),
        eligible,
        copy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use policy_poc::Features;

    #[test]
    fn decision_windows_are_one_stream() {
        let seed = 0xDEAD_BEEF;
        let mut one = SplitMix64::new(seed);
        for k in 0..20u64 {
            let mut w = decision_rng(seed, k);
            for _ in 0..DRAWS_PER_DECISION {
                assert_eq!(w.next_u64(), one.next_u64());
            }
        }
    }

    #[test]
    fn shares_match_policy_poc() {
        assert!((1.0 / EXPLORE_ONE_IN as f64 - policy_poc::fit::EXPLORE_SHARE).abs() < 1e-12);
        assert_eq!(MARGIN, 600);
    }

    #[test]
    fn samples_are_distinct_and_bounded() {
        for n in 0..9 {
            for k in 0..50 {
                let s = sample_slots(&mut decision_rng(1, k), n);
                assert_eq!(s.len(), n.min(4));
                let mut d = s.clone();
                d.sort_unstable();
                d.dedup();
                assert_eq!(d.len(), s.len());
                assert!(s.iter().all(|&i| i < n));
            }
        }
    }

    fn cands(seed: u64) -> Vec<Candidate> {
        let mut r = SplitMix64::new(seed);
        (0..4u32)
            .map(|id| Candidate {
                id,
                features: Features::new(
                    (r.below(3)) as u8,
                    (r.below(2)) as u8,
                    5 + (r.below(3)) as u8,
                    0,
                    4,
                )
                .unwrap(),
            })
            .collect()
    }

    #[test]
    fn picks_stay_in_band_and_propensities_sum_to_one() {
        let (mut explored, mut contested, mut copies) = (0, 0, 0);
        for k in 0..5_000u64 {
            let cs = cands(k);
            let c = choose(&mut decision_rng(7, k), &cs).unwrap();
            let best = cs
                .iter()
                .map(|x| FixedRule.score(x.features))
                .max()
                .unwrap();
            assert!(FixedRule.score(cs[c.pick].features) >= best - MARGIN);
            assert!(c.eligible.contains(&c.rule_choice));
            let total: f64 = (0..cs.len())
                .map(|i| propensity(i, c.rule_choice, &c.eligible))
                .sum();
            assert!((total - 1.0).abs() < 1e-12);
            assert!(c.propensity > 0.0);
            if let Some(cp) = c.copy {
                assert_ne!(cp, c.pick);
                assert!(c.eligible.contains(&cp));
                copies += 1;
            }
            explored += usize::from(c.explored);
            contested += usize::from(c.pick != c.rule_choice);
        }
        assert!((400..600).contains(&explored), "explored {explored}");
        assert!(contested > 0 && contested < explored);
        assert!(copies > 100 && copies < 600, "copies {copies}");
    }
}
