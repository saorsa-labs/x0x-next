//! The checker (design study §5, "checks read off the weights"; poc-spec §5).
//!
//! The checker reads everything off the weights and trusts nothing a patch
//! says about itself. Its static checks are:
//!
//! * **Directions** — rtt steps ≤ 0, loss steps ≤ 0, delivery steps ≥ 0.
//! * **Roles** — primary steps are multiples of [`QUANT`]; the tie features'
//!   combined spread (max minus min partial sum of age, plus the same for
//!   balance) is below `QUANT`. Together these make the order lexicographic:
//!   tie features can only reorder inputs whose primary scores are equal.
//! * **Limits** — every |step| ≤ [`MAX_STEP`] (primary) or ≤ [`MAX_TIE_STEP`]
//!   (tie), and |base| ≤ [`MAX_BASE`].
//!
//! [`Checker::diff`] computes the exact decision difference between two
//! scorers over all 65,536 inputs: the per-input score delta and exact counts
//! of reversed, newly ordered and newly tied input pairs (an O(n log n)
//! discordant-pair count over all 2,147,450,880 unordered pairs), plus a
//! capped list of example pairs.

use crate::features::{Feature, Features};
use crate::rule::{Scorer, QUANT};
use crate::table::PointsTable;

/// Largest allowed |step| for a primary feature (200 quanta).
pub const MAX_STEP: i64 = 200 * QUANT;
/// Largest allowed |step| for a tie feature.
pub const MAX_TIE_STEP: i64 = 16;
/// Largest allowed |base|.
pub const MAX_BASE: i64 = 1_000_000;
/// Maximum number of example pairs reported per category by [`Checker::diff`].
pub const MAX_EXAMPLE_PAIRS: usize = 8;

/// One reason a table fails the static checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckFailure {
    /// A step points the wrong way (e.g. a positive rtt step).
    Direction {
        /// Feature whose step is wrong.
        feature: Feature,
        /// Step index (level `step` to `step + 1`).
        step: usize,
        /// The offending weight.
        weight: i64,
    },
    /// A primary step is not a multiple of [`QUANT`].
    PrimaryNotQuantized {
        /// Feature whose step is wrong.
        feature: Feature,
        /// Step index.
        step: usize,
        /// The offending weight.
        weight: i64,
    },
    /// Tie features could reorder inputs whose primary scores differ.
    TieSpreadTooLarge {
        /// Combined spread of the tie features' partial sums.
        spread: i64,
    },
    /// A step exceeds its magnitude limit.
    StepLimit {
        /// Feature whose step is too large.
        feature: Feature,
        /// Step index.
        step: usize,
        /// The offending weight.
        weight: i64,
        /// The limit it broke.
        limit: i64,
    },
    /// The base exceeds [`MAX_BASE`].
    BaseLimit {
        /// The offending base.
        base: i64,
    },
}

impl std::fmt::Display for CheckFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckFailure::Direction {
                feature,
                step,
                weight,
            } => write!(
                f,
                "direction: {feature} step {step} = {weight} points the wrong way"
            ),
            CheckFailure::PrimaryNotQuantized {
                feature,
                step,
                weight,
            } => write!(
                f,
                "roles: {feature} step {step} = {weight} is not a multiple of {QUANT}"
            ),
            CheckFailure::TieSpreadTooLarge { spread } => write!(
                f,
                "roles: tie features span {spread} >= {QUANT}, could override primary order"
            ),
            CheckFailure::StepLimit {
                feature,
                step,
                weight,
                limit,
            } => write!(
                f,
                "limit: {feature} step {step} = {weight} exceeds |{limit}|"
            ),
            CheckFailure::BaseLimit { base } => {
                write!(f, "limit: base {base} exceeds |{MAX_BASE}|")
            }
        }
    }
}

/// What a passing table looks like, read off its weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckReport {
    /// Combined spread of the tie features (< [`QUANT`]).
    pub tie_spread: i64,
    /// Largest |primary step|.
    pub max_primary_step: i64,
    /// Largest |tie step|.
    pub max_tie_step: i64,
}

/// An ordered pair of inputs `(a, b)`.
pub type InputPair = (Features, Features);

/// Exact decision difference between a candidate scorer and an incumbent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiffReport {
    /// `candidate(x) - incumbent(x)` for every input, in index order.
    pub deltas: Vec<i64>,
    /// Number of inputs whose score changed.
    pub changed_inputs: usize,
    /// Unordered pairs the incumbent strictly orders one way and the
    /// candidate strictly orders the other way.
    pub reversed_pairs: u64,
    /// Pairs the incumbent ties and the candidate strictly orders.
    pub newly_ordered_pairs: u64,
    /// Pairs the incumbent strictly orders and the candidate ties.
    pub newly_tied_pairs: u64,
    /// Example reversed pairs `(a, b)`: incumbent prefers `a`, candidate `b`.
    pub reversed_examples: Vec<InputPair>,
    /// Example newly ordered pairs `(a, b)`: candidate now prefers `a`.
    pub newly_ordered_examples: Vec<InputPair>,
}

impl DiffReport {
    /// True when no pair the incumbent strictly ordered changes order or
    /// becomes tied.
    #[must_use]
    pub fn preserves_existing_preferences(&self) -> bool {
        self.reversed_pairs == 0 && self.newly_tied_pairs == 0
    }
}

/// The static checker and exact differ.
#[derive(Debug, Clone, Copy, Default)]
pub struct Checker;

impl Checker {
    /// Run every static check on `table`.
    ///
    /// # Errors
    /// Every failure found (not just the first).
    pub fn check(table: &PointsTable) -> Result<CheckReport, Vec<CheckFailure>> {
        let mut failures = Vec::new();
        let mut max_primary_step = 0i64;
        let mut max_tie_step = 0i64;

        for feature in Feature::ALL {
            for (step, &weight) in table.steps.of(feature).iter().enumerate() {
                let wrong_way = match feature {
                    Feature::Rtt | Feature::Loss => weight > 0,
                    Feature::Delivery => weight < 0,
                    Feature::Age | Feature::Balance => false,
                };
                if wrong_way {
                    failures.push(CheckFailure::Direction {
                        feature,
                        step,
                        weight,
                    });
                }
                let magnitude = weight.saturating_abs();
                let limit = if feature.is_primary() {
                    max_primary_step = max_primary_step.max(magnitude);
                    if weight % QUANT != 0 {
                        failures.push(CheckFailure::PrimaryNotQuantized {
                            feature,
                            step,
                            weight,
                        });
                    }
                    MAX_STEP
                } else {
                    max_tie_step = max_tie_step.max(magnitude);
                    MAX_TIE_STEP
                };
                if magnitude > limit {
                    failures.push(CheckFailure::StepLimit {
                        feature,
                        step,
                        weight,
                        limit,
                    });
                }
            }
        }

        let tie_spread = spread(table.steps.of(Feature::Age))
            .saturating_add(spread(table.steps.of(Feature::Balance)));
        if tie_spread >= QUANT {
            failures.push(CheckFailure::TieSpreadTooLarge { spread: tie_spread });
        }
        if table.base.saturating_abs() > MAX_BASE {
            failures.push(CheckFailure::BaseLimit { base: table.base });
        }

        if failures.is_empty() {
            Ok(CheckReport {
                tie_spread,
                max_primary_step,
                max_tie_step,
            })
        } else {
            Err(failures)
        }
    }

    /// Exact decision difference of `candidate` against `incumbent` over all
    /// 65,536 inputs.
    #[must_use]
    pub fn diff(candidate: &impl Scorer, incumbent: &impl Scorer) -> DiffReport {
        let inputs: Vec<Features> = Features::enumerate_all().collect();
        let inc: Vec<i64> = inputs.iter().map(|f| incumbent.score(*f)).collect();
        let cand: Vec<i64> = inputs.iter().map(|f| candidate.score(*f)).collect();
        let deltas: Vec<i64> = cand
            .iter()
            .zip(&inc)
            .map(|(c, i)| c.saturating_sub(*i))
            .collect();
        let changed_inputs = deltas.iter().filter(|d| **d != 0).count();

        // Sort input indices by (incumbent, candidate) ascending.
        let mut order: Vec<usize> = (0..inputs.len()).collect();
        order.sort_by_key(|&i| (inc[i], cand[i]));

        let inc_ties = tie_pairs(order.iter().map(|&i| inc[i]));
        let both_ties = tie_pairs(order.iter().map(|&i| (inc[i], cand[i])));
        let mut by_cand = cand.clone();
        by_cand.sort_unstable();
        let cand_ties = tie_pairs(by_cand.iter().copied());

        // Strict inversions of the candidate sequence = reversed pairs.
        let mut seq: Vec<i64> = order.iter().map(|&i| cand[i]).collect();
        let reversed_pairs = count_strict_inversions(&mut seq);

        let newly_ordered_pairs = inc_ties.saturating_sub(both_ties);
        let newly_tied_pairs = cand_ties.saturating_sub(both_ties);

        let newly_ordered_examples = newly_ordered_examples(&order, &inc, &cand, &inputs);
        let reversed_examples = reversed_examples(&order, &inc, &cand, &inputs);

        DiffReport {
            deltas,
            changed_inputs,
            reversed_pairs,
            newly_ordered_pairs,
            newly_tied_pairs,
            reversed_examples,
            newly_ordered_examples,
        }
    }
}

/// Max minus min of the partial sums (including the empty sum 0).
fn spread(steps: &[i64]) -> i64 {
    let mut acc = 0i64;
    let mut lo = 0i64;
    let mut hi = 0i64;
    for s in steps {
        acc = acc.saturating_add(*s);
        lo = lo.min(acc);
        hi = hi.max(acc);
    }
    hi.saturating_sub(lo)
}

/// Number of equal pairs in a sorted sequence.
fn tie_pairs<T: PartialEq>(sorted: impl Iterator<Item = T>) -> u64 {
    let mut total = 0u64;
    let mut run = 0u64;
    let mut prev: Option<T> = None;
    for v in sorted {
        if prev.as_ref() == Some(&v) {
            run += 1;
        } else {
            total += run * run.saturating_sub(1) / 2;
            run = 1;
        }
        prev = Some(v);
    }
    total + run * run.saturating_sub(1) / 2
}

/// Count pairs `i < j` with `seq[i] > seq[j]` (merge sort, O(n log n)).
fn count_strict_inversions(seq: &mut [i64]) -> u64 {
    let n = seq.len();
    if n < 2 {
        return 0;
    }
    let mid = n / 2;
    let mut count = {
        let (left, right) = seq.split_at_mut(mid);
        count_strict_inversions(left) + count_strict_inversions(right)
    };
    let mut merged = Vec::with_capacity(n);
    let (mut i, mut j) = (0, mid);
    while i < mid && j < n {
        if seq[i] <= seq[j] {
            merged.push(seq[i]);
            i += 1;
        } else {
            // seq[i..mid] are all > seq[j].
            count += (mid - i) as u64;
            merged.push(seq[j]);
            j += 1;
        }
    }
    merged.extend_from_slice(&seq[i..mid]);
    merged.extend_from_slice(&seq[j..n]);
    seq.copy_from_slice(&merged);
    count
}

/// Up to [`MAX_EXAMPLE_PAIRS`] newly ordered pairs: one per incumbent tie
/// group that the candidate splits, higher candidate score first.
fn newly_ordered_examples(
    order: &[usize],
    inc: &[i64],
    cand: &[i64],
    inputs: &[Features],
) -> Vec<InputPair> {
    order
        .chunk_by(|&a, &b| inc[a] == inc[b])
        .filter_map(|group| {
            let first = *group.first()?;
            let other = *group.iter().find(|&&j| cand[j] != cand[first])?;
            Some(if cand[other] > cand[first] {
                (inputs[other], inputs[first])
            } else {
                (inputs[first], inputs[other])
            })
        })
        .take(MAX_EXAMPLE_PAIRS)
        .collect()
}

/// Up to [`MAX_EXAMPLE_PAIRS`] reversed pairs, found by tracking the largest
/// candidate score among strictly lower incumbent groups.
fn reversed_examples(
    order: &[usize],
    inc: &[i64],
    cand: &[i64],
    inputs: &[Features],
) -> Vec<InputPair> {
    let mut out = Vec::new();
    let mut best_lower: Option<usize> = None; // over strictly lower inc groups
    for group in order.chunk_by(|&a, &b| inc[a] == inc[b]) {
        for &j in group {
            if let Some(b) = best_lower {
                if cand[b] > cand[j] {
                    // Incumbent prefers j (higher), candidate prefers b.
                    out.push((inputs[j], inputs[b]));
                    if out.len() == MAX_EXAMPLE_PAIRS {
                        return out;
                    }
                }
            }
        }
        for &j in group {
            if best_lower.is_none_or(|b| cand[j] > cand[b]) {
                best_lower = Some(j);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inversion_counter_matches_brute_force() {
        let mut v = vec![3, 1, 2, 2, 5, 0, 3];
        let brute = {
            let mut c = 0;
            for i in 0..v.len() {
                for j in i + 1..v.len() {
                    if v[i] > v[j] {
                        c += 1;
                    }
                }
            }
            c
        };
        assert_eq!(count_strict_inversions(&mut v), brute);
    }

    #[test]
    fn spread_counts_negative_excursions() {
        assert_eq!(spread(&[5, -10, 3]), 10);
        assert_eq!(spread(&[1; 7]), 7);
    }
}
