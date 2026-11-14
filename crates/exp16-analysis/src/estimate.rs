//! Off-policy cost estimates from logged Measure decisions (plan §4,
//! "scoring on week 4").
//!
//! # The estimator, honestly
//!
//! A trace shows the realized cost of the candidate the node used (and of
//! the copy, when one was sent), never of the others. To estimate what a
//! different scorer would have cost on the same decisions we use
//! **self-normalized inverse-propensity weighting** (SNIPW) over the logged
//! picks:
//!
//! ```text
//! w_i  = π(pick_i | decision_i) / p_i          p_i = logged propensity of pick_i
//! Ĉ(π) = Σ w_i · cost_i / Σ w_i
//! ```
//!
//! For a deterministic scorer, `π(pick_i)` is 1 when the scorer's banded pick
//! equals the logged pick and 0 otherwise, so only matching decisions count,
//! each weighted by `1 / p_i`.
//!
//! What this does and does not give:
//!
//! * **Support.** The Measure procedure gives positive propensity only to the
//!   rule's choice and to the candidates inside the band at 6 quanta. A
//!   scorer evaluated at margin ≤ 6 quanta therefore always has support, and
//!   SNIPW is consistent for it. A target that puts mass outside that band
//!   (a margin above 6) has none there; those decisions get zero weight and
//!   the estimate is biased toward the supported part. [`IpwEstimate::coverage`]
//!   reports the share of target mass with support.
//! * **Variance.** Picks that differ from the rule's choice have propensity
//!   about 0.1 / (band size), so a scorer that often disagrees with the rule
//!   is estimated from the ~10% explored decisions only, at weights of 20 to
//!   40. [`IpwEstimate::effective_sample_size`] is `(Σw)² / Σw²`.
//! * **Self-normalization** trades a small bias (vanishing with sample size)
//!   for much lower variance than the plain `Σ w·c / n` form, and makes the
//!   estimate a weighted mean of observed costs.
//! * **Timeouts** enter as the request timeout (a censored cost).
//!
//! The logged copies give a cross-check that needs no weights: on a
//! duplicated decision both the pick's and the copy's costs are observed, so
//! when two scorers' picks both fall among {pick, copy} their costs can be
//! compared directly ([`paired_check`]). Copies are not a random sample of
//! decisions (one in ten, and only where the band holds two or more
//! candidates), so the paired result is a sanity check on direction and
//! size, not an unbiased estimate of the holdout cost.

use measure_agent::decide::EXPLORE_ONE_IN;
use policy_poc::rng::SplitMix64;
use serde::Serialize;

use crate::load::LoggedDecision;

/// One decision's contribution to an SNIPW estimate.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Term {
    /// `π(pick) / p`.
    pub weight: f64,
    /// Realized cost of the logged pick, ms.
    pub cost: f64,
    /// Target probability mass on candidates with zero logged propensity.
    pub unsupported: f64,
}

/// An SNIPW estimate of a target's mean cost.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct IpwEstimate {
    /// Estimated mean cost, ms; `None` when no decision has weight.
    pub mean_cost_ms: Option<f64>,
    /// Decisions in the sample.
    pub decisions: usize,
    /// Decisions with positive weight.
    pub matched: usize,
    /// Sum of weights.
    pub weight_sum: f64,
    /// `(Σw)² / Σw²`.
    pub effective_sample_size: f64,
    /// Share of the target's probability mass that had support.
    pub coverage: f64,
}

/// Probability that the Measure procedure picks candidate `i`
/// (measure-agent's [`measure_agent::decide::propensity`]), recomputed from
/// the logged band.
#[must_use]
pub fn logged_propensity(d: &LoggedDecision, i: usize, eligible: &[usize]) -> f64 {
    measure_agent::decide::propensity(i, d.rule_choice, eligible)
}

/// Term of a deterministic target that picks `target` on decision `d`.
#[must_use]
pub fn deterministic_term(d: &LoggedDecision, target: usize, eligible: &[usize]) -> Term {
    let supported = logged_propensity(d, target, eligible) > 0.0;
    Term {
        weight: if target == d.pick {
            1.0 / d.propensity
        } else {
            0.0
        },
        cost: d.pick_cost_ms,
        unsupported: if supported { 0.0 } else { 1.0 },
    }
}

/// Term of a stochastic target with probabilities `pi` over the candidates.
#[must_use]
pub fn stochastic_term(d: &LoggedDecision, pi: &[f64], eligible: &[usize]) -> Term {
    let unsupported = pi
        .iter()
        .enumerate()
        .filter(|(i, _)| logged_propensity(d, *i, eligible) <= 0.0)
        .map(|(_, p)| p)
        .sum();
    Term {
        weight: pi.get(d.pick).copied().unwrap_or(0.0) / d.propensity,
        cost: d.pick_cost_ms,
        unsupported,
    }
}

/// Combine terms into an estimate.
#[must_use]
pub fn estimate(terms: &[Term]) -> IpwEstimate {
    let (mut w, mut wc, mut w2, mut unsupported) = (0.0, 0.0, 0.0, 0.0);
    let mut matched = 0;
    for t in terms {
        w += t.weight;
        wc += t.weight * t.cost;
        w2 += t.weight * t.weight;
        unsupported += t.unsupported;
        matched += usize::from(t.weight > 0.0);
    }
    let n = terms.len();
    IpwEstimate {
        mean_cost_ms: (w > 0.0).then(|| wc / w),
        decisions: n,
        matched,
        weight_sum: w,
        effective_sample_size: if w2 > 0.0 { w * w / w2 } else { 0.0 },
        coverage: if n == 0 {
            0.0
        } else {
            1.0 - unsupported / n as f64
        },
    }
}

/// Relative gain of `challenger` over `baseline`: `(base − chal) / base`.
#[must_use]
pub fn gain(baseline: f64, challenger: f64) -> Option<f64> {
    (baseline > 0.0 && baseline.is_finite() && challenger.is_finite())
        .then(|| (baseline - challenger) / baseline)
}

/// A gain with its bootstrap interval.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GainEstimate {
    /// Point estimate of `(C_base − C_chal) / C_base`.
    pub gain: f64,
    /// 5th percentile of the bootstrap gains: the lower end of the two-sided
    /// 90% percentile interval (plan §4's "90% bootstrap lower bound").
    pub lower_90: Option<f64>,
    /// 95th percentile of the bootstrap gains.
    pub upper_90: Option<f64>,
    /// Bootstrap resamples in which both estimates existed.
    pub resamples: usize,
}

/// Gain of `chal` over `base` (two term lists over the same decisions, in
/// the same order) with a percentile bootstrap over decisions: each resample
/// draws decisions with replacement and recomputes both SNIPW estimates on
/// it, so the pairing of the two scorers on a decision is kept.
#[must_use]
pub fn bootstrap_gain(
    base: &[Term],
    chal: &[Term],
    resamples: usize,
    seed: u64,
) -> Option<GainEstimate> {
    let point = gain(estimate(base).mean_cost_ms?, estimate(chal).mean_cost_ms?)?;
    let n = base.len().min(chal.len());
    let mut rng = SplitMix64::new(seed);
    let mut gains = Vec::with_capacity(resamples);
    for _ in 0..resamples {
        let (mut wb, mut wcb, mut wch, mut wcch) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..n {
            let i = usize::try_from(rng.below(n as u64)).unwrap_or(0);
            let (b, c) = (base[i], chal[i]);
            wb += b.weight;
            wcb += b.weight * b.cost;
            wch += c.weight;
            wcch += c.weight * c.cost;
        }
        if wb > 0.0 && wch > 0.0 {
            if let Some(g) = gain(wcb / wb, wcch / wch) {
                gains.push(g);
            }
        }
    }
    gains.sort_by(f64::total_cmp);
    Some(GainEstimate {
        gain: point,
        lower_90: quantile(&gains, 0.05),
        upper_90: quantile(&gains, 0.95),
        resamples: gains.len(),
    })
}

/// Quantile of sorted values (nearest rank); `None` when empty.
#[must_use]
pub fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64).round();
    // rank is within 0..len by construction.
    sorted.get(rank as usize).copied()
}

/// The copy cross-check of two scorers.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct PairedCheck {
    /// Duplicated decisions where both scorers' picks were observed (each is
    /// the pick or the copy).
    pub pairs: usize,
    /// Of those, decisions where the two scorers picked differently.
    pub disagreements: usize,
    /// Mean observed cost of the baseline's picks on the pairs, ms.
    pub mean_base_ms: f64,
    /// Mean observed cost of the challenger's picks on the pairs, ms.
    pub mean_chal_ms: f64,
    /// `(base − chal) / base` on the pairs.
    pub gain: Option<f64>,
}

/// Compare two scorers' picks (`base[i]`, `chal[i]` on `decisions[i]`)
/// directly on duplicated decisions where both picks were observed.
#[must_use]
pub fn paired_check(
    decisions: &[&LoggedDecision],
    base: &[usize],
    chal: &[usize],
) -> Option<PairedCheck> {
    let (mut pairs, mut disagreements, mut sb, mut sc) = (0usize, 0usize, 0.0, 0.0);
    for ((d, &b), &c) in decisions.iter().zip(base).zip(chal) {
        let Some((copy, copy_cost)) = d.copy else {
            continue;
        };
        let observed = |i: usize| {
            if i == d.pick {
                Some(d.pick_cost_ms)
            } else if i == copy {
                Some(copy_cost)
            } else {
                None
            }
        };
        if let (Some(cb), Some(cc)) = (observed(b), observed(c)) {
            pairs += 1;
            disagreements += usize::from(b != c);
            sb += cb;
            sc += cc;
        }
    }
    (pairs > 0).then(|| {
        let (mb, mc) = (sb / pairs as f64, sc / pairs as f64);
        PairedCheck {
            pairs,
            disagreements,
            mean_base_ms: mb,
            mean_chal_ms: mc,
            gain: gain(mb, mc),
        }
    })
}

/// The Measure procedure's exploration share (one in [`EXPLORE_ONE_IN`]).
#[must_use]
pub fn explore_share() -> f64 {
    1.0 / EXPLORE_ONE_IN as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use measure_agent::decide::{choose, decision_rng, MARGIN};
    use policy_poc::fit::fit_table;
    use policy_poc::twin::{Twin, WorldParams, DEFAULT_CANDIDATES};
    use policy_poc::{band_pick, FixedRule, Scorer};

    /// Log `n` decisions of the real Measure procedure in a twin world,
    /// recording for each the realized cost every candidate would have had
    /// (the counterfactuals a real trace never shows).
    fn measure_log(world: WorldParams, n: usize, seed: u64) -> Vec<(LoggedDecision, Vec<f64>)> {
        let mut twin = Twin::new(world, seed, DEFAULT_CANDIDATES);
        let mut out = Vec::with_capacity(n);
        for k in 0..n as u64 {
            let d = twin.next_decision();
            let c = choose(&mut decision_rng(seed, k), &d.candidates).unwrap();
            let expected: Vec<f64> = (0..d.candidates.len())
                .map(|i| twin.realized_cost(&d, i))
                .collect();
            let copy = c.copy.map(|cp| (cp, twin.realized_cost(&d, cp)));
            out.push((
                LoggedDecision {
                    decision: k,
                    utc_ms: 0,
                    candidates: d.candidates.clone(),
                    rule_choice: c.rule_choice,
                    pick: c.pick,
                    propensity: c.propensity,
                    explored: c.explored,
                    pick_cost_ms: twin.realized_cost(&d, c.pick),
                    pick_failed: false,
                    copy,
                },
                expected,
            ));
        }
        out
    }

    fn eligible(d: &LoggedDecision) -> Vec<usize> {
        let best = d
            .candidates
            .iter()
            .map(|c| FixedRule.score(c.features))
            .max()
            .unwrap();
        (0..d.candidates.len())
            .filter(|&i| FixedRule.score(d.candidates[i].features) >= best - MARGIN)
            .collect()
    }

    #[test]
    fn snipw_recovers_the_true_mean_of_a_banded_target() {
        let world = WorldParams::world_one();
        // A target that often disagrees with the rule: a fitted table at the
        // logged margin.
        let target = fit_table(&policy_poc::fit::log_evidence(world, 11, 20_000));
        let log = measure_log(world, 60_000, 5);
        let mut terms = Vec::new();
        let mut truth = 0.0;
        let mut disagree = 0;
        for (d, expected) in &log {
            let pick = band_pick(&FixedRule, Some(&target), &d.candidates, MARGIN)
                .chosen
                .unwrap();
            disagree += usize::from(pick != d.rule_choice);
            truth += expected[pick];
            terms.push(deterministic_term(d, pick, &eligible(d)));
        }
        truth /= log.len() as f64;
        let est = estimate(&terms);
        let m = est.mean_cost_ms.unwrap();
        assert!(
            disagree > 500,
            "target must disagree with the rule: {disagree}"
        );
        assert!(
            (m - truth).abs() / truth < 0.03,
            "estimate {m} truth {truth}"
        );
        assert!((est.coverage - 1.0).abs() < 1e-12);

        // The rule itself: the logged picks mostly are the rule's.
        let rule_terms: Vec<Term> = log
            .iter()
            .map(|(d, _)| deterministic_term(d, d.rule_choice, &eligible(d)))
            .collect();
        let rule_truth: f64 =
            log.iter().map(|(d, e)| e[d.rule_choice]).sum::<f64>() / log.len() as f64;
        let rm = estimate(&rule_terms).mean_cost_ms.unwrap();
        assert!(
            (rm - rule_truth).abs() / rule_truth < 0.02,
            "rule {rm} vs {rule_truth}"
        );
    }

    #[test]
    fn snipw_recovers_a_stochastic_target() {
        let log = measure_log(WorldParams::world_two(), 60_000, 9);
        let mut terms = Vec::new();
        let mut truth = 0.0;
        for (d, expected) in &log {
            let el = eligible(d);
            let mut pi = vec![0.0; d.candidates.len()];
            for &i in &el {
                pi[i] = 1.0 / el.len() as f64;
            }
            truth += el.iter().map(|&i| expected[i]).sum::<f64>() / el.len() as f64;
            terms.push(stochastic_term(d, &pi, &el));
        }
        truth /= log.len() as f64;
        let m = estimate(&terms).mean_cost_ms.unwrap();
        assert!(
            (m - truth).abs() / truth < 0.03,
            "estimate {m} truth {truth}"
        );
    }

    #[test]
    fn bootstrap_brackets_the_point_and_pairs_agree() {
        let world = WorldParams::world_one();
        let log = measure_log(world, 20_000, 21);
        let target = fit_table(&policy_poc::fit::log_evidence(world, 3, 20_000));
        let (mut base, mut chal, mut bp, mut cp) = (vec![], vec![], vec![], vec![]);
        for (d, _) in &log {
            let el = eligible(d);
            let pick = band_pick(&FixedRule, Some(&target), &d.candidates, MARGIN)
                .chosen
                .unwrap();
            base.push(deterministic_term(d, d.rule_choice, &el));
            chal.push(deterministic_term(d, pick, &el));
            bp.push(d.rule_choice);
            cp.push(pick);
        }
        let g = bootstrap_gain(&base, &chal, 300, 1).unwrap();
        let (lo, hi) = (g.lower_90.unwrap(), g.upper_90.unwrap());
        assert!(lo <= g.gain && g.gain <= hi, "{lo} {} {hi}", g.gain);
        assert!(g.gain > 0.0);
        let refs: Vec<&LoggedDecision> = log.iter().map(|(d, _)| d).collect();
        let pc = paired_check(&refs, &bp, &cp).unwrap();
        assert!(pc.pairs > 100 && pc.disagreements > 0);
    }

    #[test]
    fn empty_and_unmatched_samples_have_no_estimate() {
        assert_eq!(estimate(&[]).mean_cost_ms, None);
        let t = Term {
            weight: 0.0,
            cost: 5.0,
            unsupported: 1.0,
        };
        let e = estimate(&[t, t]);
        assert_eq!(e.mean_cost_ms, None);
        assert_eq!(e.coverage, 0.0);
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[1.0, 2.0, 3.0], 0.0), Some(1.0));
    }
}
