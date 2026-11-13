//! Fitting a points table from a node's own evidence (design study §5,
//! "written weights"; poc-spec §7 and demo §10.5).
//!
//! A node logs what it has: the peer it used and the realized cost. It
//! follows the fixed rule with a small share of random choices (the kernel's
//! exploration share). The fit is an additive model of `ln(cost)` with one
//! effect per (feature, level), solved by backfitting. Effects are made
//! monotone in the checker's directions (pool-adjacent-violators), scaled to
//! points (`FIT_SCALE` quanta per unit of log cost), and written as steps
//! that respect the checker's limits. Tie-feature weights are copied from the
//! fixed rule: the fit only touches the primary features.

use crate::band::band_pick;
use crate::checker::MAX_STEP;
use crate::features::{Feature, Features};
use crate::rule::{FixedRule, QUANT};
use crate::table::PointsTable;
use crate::twin::{Twin, WorldParams, DEFAULT_CANDIDATES};

/// Quanta of score per unit of `ln(cost)`.
pub const FIT_SCALE: f64 = 100.0;
/// Share of decisions where the logging node picks at random.
pub const EXPLORE_SHARE: f64 = 0.10;
/// Backfitting sweeps.
pub const BACKFIT_SWEEPS: usize = 50;

/// One logged observation: the features of the peer used and its realized
/// cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evidence {
    /// Features of the peer used.
    pub features: Features,
    /// Realized cost, ms.
    pub cost: f64,
}

/// Log `n` decisions in `world` the way a node running the fixed rule would.
#[must_use]
pub fn log_evidence(world: WorldParams, seed: u64, n: usize) -> Vec<Evidence> {
    let mut twin = Twin::new(world, seed, DEFAULT_CANDIDATES);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let d = twin.next_decision();
        let rule_pick = band_pick(&FixedRule, None::<&FixedRule>, &d.candidates, 0)
            .chosen
            .unwrap_or(0);
        let pick = if twin.uniform() < EXPLORE_SHARE {
            twin.below(d.candidates.len())
        } else {
            rule_pick
        };
        if let Some(c) = d.candidates.get(pick) {
            out.push(Evidence {
                features: c.features,
                cost: twin.realized_cost(&d, pick),
            });
        }
    }
    out
}

/// Fit a rung-1 table to evidence.
#[must_use]
pub fn fit_table(evidence: &[Evidence]) -> PointsTable {
    let primaries = [Feature::Rtt, Feature::Loss, Feature::Delivery];
    let y: Vec<f64> = evidence.iter().map(|e| e.cost.max(1e-9).ln()).collect();
    let n = evidence.len().max(1) as f64;
    let mut effects: Vec<Vec<f64>> = primaries
        .iter()
        .map(|f| vec![0.0; f.levels() as usize])
        .collect();
    let mut counts: Vec<Vec<f64>> = effects.iter().map(|e| vec![0.0; e.len()]).collect();
    for e in evidence {
        for (k, f) in primaries.iter().enumerate() {
            if let Some(c) = counts[k].get_mut(e.features.get(*f) as usize) {
                *c += 1.0;
            }
        }
    }
    let mut mu = y.iter().sum::<f64>() / n;

    for _ in 0..BACKFIT_SWEEPS {
        for k in 0..primaries.len() {
            let mut sums = vec![0.0; effects[k].len()];
            for (e, yi) in evidence.iter().zip(&y) {
                let mut r = yi - mu;
                for (j, f) in primaries.iter().enumerate() {
                    if j != k {
                        r -= effects[j]
                            .get(e.features.get(*f) as usize)
                            .copied()
                            .unwrap_or(0.0);
                    }
                }
                if let Some(s) = sums.get_mut(e.features.get(primaries[k]) as usize) {
                    *s += r;
                }
            }
            let mut fresh: Vec<f64> = sums
                .iter()
                .zip(&counts[k])
                .map(|(s, c)| if *c > 0.0 { s / c } else { f64::NAN })
                .collect();
            fill_gaps(&mut fresh);
            // Anchor level 0 at zero; the offset moves into mu.
            let anchor = fresh.first().copied().unwrap_or(0.0);
            for v in &mut fresh {
                *v -= anchor;
            }
            mu += anchor;
            effects[k] = fresh;
        }
    }

    let mut table = PointsTable::from_scorer(&FixedRule);
    table.base = 0;
    for (k, feature) in primaries.iter().enumerate() {
        // Score is minus log cost. Directions: rtt and loss scores fall with
        // level, delivery score rises with level.
        let scores: Vec<f64> = effects[k].iter().map(|e| -e * FIT_SCALE).collect();
        let increasing = matches!(feature, Feature::Delivery);
        let mono = isotonic(&scores, &counts[k], increasing);
        let steps = table.steps.of_mut(*feature);
        let mut prev = quantize(mono.first().copied().unwrap_or(0.0));
        for (k_step, slot) in steps.iter_mut().enumerate() {
            let here = quantize(mono.get(k_step + 1).copied().unwrap_or(0.0));
            *slot = (here - prev).clamp(-MAX_STEP, MAX_STEP);
            prev = here;
        }
    }
    table
}

/// Round a score in quanta to a multiple of [`QUANT`].
fn quantize(quanta: f64) -> i64 {
    // Scores are bounded by FIT_SCALE × |ln cost| (well under 1e6 quanta), so
    // the cast is in range.
    (quanta.round() as i64).saturating_mul(QUANT)
}

/// Replace NaN entries (unseen levels) with the nearest seen value.
fn fill_gaps(v: &mut [f64]) {
    let mut last = None;
    for x in v.iter_mut() {
        if x.is_nan() {
            if let Some(l) = last {
                *x = l;
            }
        } else {
            last = Some(*x);
        }
    }
    let mut next = None;
    for x in v.iter_mut().rev() {
        if x.is_nan() {
            *x = next.unwrap_or(0.0);
        } else {
            next = Some(*x);
        }
    }
}

/// Weighted isotonic regression (pool adjacent violators).
fn isotonic(values: &[f64], weights: &[f64], increasing: bool) -> Vec<f64> {
    let sign = if increasing { 1.0 } else { -1.0 };
    // Blocks of (mean, weight, length) on sign-adjusted values.
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
    for (v, w) in values.iter().zip(weights) {
        blocks.push((v * sign, w.max(1e-9), 1));
        while blocks.len() >= 2 {
            let len = blocks.len();
            let (m2, w2, l2) = blocks[len - 1];
            let (m1, w1, l1) = blocks[len - 2];
            if m1 <= m2 {
                break;
            }
            blocks.truncate(len - 2);
            blocks.push(((m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, l1 + l2));
        }
    }
    blocks
        .iter()
        .flat_map(|(m, _, l)| std::iter::repeat_n(m * sign, *l))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotonic_enforces_order() {
        let out = isotonic(&[3.0, 1.0, 2.0], &[1.0, 1.0, 1.0], true);
        assert!(out.windows(2).all(|w| w[0] <= w[1]));
        let out = isotonic(&[1.0, 3.0, 2.0], &[1.0, 1.0, 1.0], false);
        assert!(out.windows(2).all(|w| w[0] >= w[1]));
    }
}
