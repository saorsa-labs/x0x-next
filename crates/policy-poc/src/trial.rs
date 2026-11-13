//! The trial (design study §5, "trial before adoption"; poc-spec §8).
//!
//! A paired, sequential comparison of a candidate policy against the fixed
//! rule on copies: for every simulated decision both the rule and the banded
//! policy pick from the same candidate set and are charged the same realized
//! outcomes. Per decision two paired gains are recorded:
//!
//! * the **typical outcome**: `ln(rule cost) − ln(policy cost)`;
//! * the **total**: `rule cost − policy cost`.
//!
//! Every `check_every` decisions after `min_decisions`, the trial stops and
//! **rejects** if either mean gain is below `−z × standard error`, and
//! **adopts** if both are above `+z × standard error`. If neither happens by
//! `max_decisions` the policy is rejected as inconclusive: a patch must earn
//! adoption. The constants are plain and documented; `z = 3` is deliberately
//! conservative because the test is repeated at every checkpoint.

use crate::band::band_pick;
use crate::error::PolicyError;
use crate::rule::{FixedRule, Scorer, QUANT};
use crate::twin::{Twin, WorldParams, DEFAULT_CANDIDATES};

/// Default band margin: 40 quanta of rule score.
pub const DEFAULT_MARGIN: i64 = 40 * QUANT;

/// Trial constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrialConfig {
    /// No verdict before this many decisions.
    pub min_decisions: usize,
    /// Give up (reject as inconclusive) after this many decisions.
    pub max_decisions: usize,
    /// Look at the statistics every this many decisions.
    pub check_every: usize,
    /// Number of standard errors that counts as "clearly".
    pub z: f64,
    /// Band margin applied by the kernel to the policy.
    pub margin: i64,
}

impl Default for TrialConfig {
    fn default() -> Self {
        Self {
            min_decisions: 2_000,
            max_decisions: 40_000,
            check_every: 1_000,
            z: 3.0,
            margin: DEFAULT_MARGIN,
        }
    }
}

impl TrialConfig {
    /// Validate the constants.
    ///
    /// # Errors
    /// [`PolicyError::InvalidConfig`] for zero sizes, `min > max`, or a
    /// non-positive `z`.
    pub fn validate(&self) -> Result<(), PolicyError> {
        if self.check_every == 0 || self.min_decisions == 0 {
            return Err(PolicyError::InvalidConfig("trial sizes must be positive"));
        }
        if self.min_decisions > self.max_decisions {
            return Err(PolicyError::InvalidConfig("min_decisions > max_decisions"));
        }
        if self.z.is_nan() || self.z <= 0.0 {
            return Err(PolicyError::InvalidConfig("z must be positive"));
        }
        Ok(())
    }
}

/// Outcome of a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Clearly better on both the typical outcome and the total.
    Adopt,
    /// Clearly worse on at least one; stopped early.
    RejectWorse,
    /// Not clearly better by `max_decisions`.
    RejectInconclusive,
}

/// Running mean and variance (Welford).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Running {
    n: u64,
    mean: f64,
    m2: f64,
}

impl Running {
    /// Add one observation.
    pub fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    /// Number of observations.
    #[must_use]
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Mean (0 when empty).
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Standard error of the mean (0 with fewer than two observations).
    #[must_use]
    pub fn std_err(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let var = self.m2 / (self.n - 1) as f64;
        (var / self.n as f64).sqrt()
    }
}

/// Everything the trial measured.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrialReport {
    /// The verdict.
    pub verdict: Verdict,
    /// Decisions simulated.
    pub decisions: usize,
    /// Decisions where policy and rule picked differently.
    pub contested: usize,
    /// Total realized cost under the rule, ms.
    pub rule_total_cost: f64,
    /// Total realized cost under the policy, ms.
    pub policy_total_cost: f64,
    /// Mean of `ln(rule) − ln(policy)` (positive = policy better).
    pub mean_log_gain: f64,
    /// Standard error of `mean_log_gain`.
    pub log_gain_se: f64,
    /// Mean of `rule − policy`, ms (positive = policy better).
    pub mean_cost_gain: f64,
    /// Standard error of `mean_cost_gain`.
    pub cost_gain_se: f64,
}

impl TrialReport {
    /// Policy's total-cost improvement over the rule, in percent.
    #[must_use]
    pub fn gain_percent(&self) -> f64 {
        if self.rule_total_cost > 0.0 {
            100.0 * (self.rule_total_cost - self.policy_total_cost) / self.rule_total_cost
        } else {
            0.0
        }
    }
}

/// Run a trial of `policy` against the fixed rule in `world`.
///
/// # Errors
/// [`PolicyError::InvalidConfig`] if `config` is invalid.
pub fn run(
    policy: &impl Scorer,
    world: WorldParams,
    seed: u64,
    config: &TrialConfig,
) -> Result<TrialReport, PolicyError> {
    config.validate()?;
    let mut twin = Twin::new(world, seed, DEFAULT_CANDIDATES);
    let mut log_gain = Running::default();
    let mut cost_gain = Running::default();
    let mut contested = 0usize;
    let mut rule_total = 0.0;
    let mut policy_total = 0.0;
    let mut verdict = Verdict::RejectInconclusive;
    let mut decisions = 0usize;

    while decisions < config.max_decisions {
        let d = twin.next_decision();
        decisions += 1;
        let out = band_pick(&FixedRule, Some(policy), &d.candidates, config.margin);
        let (Some(p), Some(r)) = (out.chosen, out.rule_choice) else {
            continue;
        };
        if out.contested() {
            contested += 1;
        }
        let rc = twin.realized_cost(&d, r);
        let pc = twin.realized_cost(&d, p);
        rule_total += rc;
        policy_total += pc;
        log_gain.push(rc.ln() - pc.ln());
        cost_gain.push(rc - pc);

        if decisions >= config.min_decisions && decisions.is_multiple_of(config.check_every) {
            let z = config.z;
            let lg_hi = log_gain.mean() > z * log_gain.std_err();
            let cg_hi = cost_gain.mean() > z * cost_gain.std_err();
            let lg_lo = log_gain.mean() < -z * log_gain.std_err();
            let cg_lo = cost_gain.mean() < -z * cost_gain.std_err();
            if lg_lo || cg_lo {
                verdict = Verdict::RejectWorse;
                break;
            }
            if lg_hi && cg_hi {
                verdict = Verdict::Adopt;
                break;
            }
        }
    }

    Ok(TrialReport {
        verdict,
        decisions,
        contested,
        rule_total_cost: rule_total,
        policy_total_cost: policy_total,
        mean_log_gain: log_gain.mean(),
        log_gain_se: log_gain.std_err(),
        mean_cost_gain: cost_gain.mean(),
        cost_gain_se: cost_gain.std_err(),
    })
}
