//! The watch (design study §5, "watch after adoption, automatic removal";
//! poc-spec §8).
//!
//! Adoption is not the end. After a policy is adopted, the decision point
//! keeps comparing it with the fixed rule on 1 in [`WatchConfig::sample_every`]
//! *contested* decisions (where the policy and the rule pick differently),
//! using copies so both outcomes are known. It keeps a sliding window of the
//! per-sample gain `ln(rule cost) − ln(policy cost)` and a running margin (the
//! sum of all sampled gains). When the window is full and its mean falls below
//! `−bound`, with `bound = max(z × standard error, min_bound)`, the policy is
//! **removed automatically** and the decision point returns to the fixed
//! rule. Removal needs no one's approval and cannot be vetoed by the policy.

use std::collections::VecDeque;

use crate::band::band_pick;
use crate::error::PolicyError;
use crate::rule::{FixedRule, Scorer};
use crate::trial::Running;
use crate::twin::Twin;

/// Watch constants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WatchConfig {
    /// Sample one in this many contested decisions.
    pub sample_every: u64,
    /// Number of samples in the sliding window.
    pub window: usize,
    /// Standard errors below zero that count as "behind".
    pub z: f64,
    /// Smallest bound, in log-cost units, so a tiny window variance cannot
    /// remove a policy on noise.
    pub min_bound: f64,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            sample_every: 10,
            window: 50,
            z: 3.0,
            min_bound: 0.05,
        }
    }
}

/// Why and when a policy was removed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Removal {
    /// Contested decisions seen before removal.
    pub contested_seen: u64,
    /// Samples taken before removal.
    pub samples: u64,
    /// Window mean gain at removal (negative).
    pub window_mean: f64,
    /// The bound it fell below (as a negative number: `-bound`).
    pub bound: f64,
    /// Running margin (sum of all sampled gains) at removal.
    pub running_margin: f64,
}

/// What one contested observation did.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WatchEvent {
    /// Not one of the 1-in-N sampled decisions.
    NotSampled,
    /// Sampled; the policy stays.
    Sampled,
    /// Sampled and the policy was removed.
    Removed(Removal),
}

/// The sliding-window watch.
#[derive(Debug, Clone, PartialEq)]
pub struct Watch {
    config: WatchConfig,
    contested_seen: u64,
    samples: u64,
    window: VecDeque<f64>,
    running_margin: f64,
    removal: Option<Removal>,
}

impl Watch {
    /// A new watch.
    ///
    /// # Errors
    /// [`PolicyError::InvalidConfig`] for a zero sample rate or window, or
    /// non-positive `z`.
    pub fn new(config: WatchConfig) -> Result<Self, PolicyError> {
        if config.sample_every == 0 || config.window < 2 {
            return Err(PolicyError::InvalidConfig(
                "watch needs sample_every >= 1 and window >= 2",
            ));
        }
        if config.z.is_nan()
            || config.z <= 0.0
            || config.min_bound.is_nan()
            || config.min_bound < 0.0
        {
            return Err(PolicyError::InvalidConfig("watch z must be positive"));
        }
        Ok(Self {
            config,
            contested_seen: 0,
            samples: 0,
            window: VecDeque::with_capacity(config.window),
            running_margin: 0.0,
            removal: None,
        })
    }

    /// Record one contested decision's outcomes on copies.
    pub fn observe_contested(&mut self, rule_cost: f64, policy_cost: f64) -> WatchEvent {
        if self.removal.is_some() {
            return WatchEvent::NotSampled;
        }
        self.contested_seen += 1;
        if !self.contested_seen.is_multiple_of(self.config.sample_every) {
            return WatchEvent::NotSampled;
        }
        self.samples += 1;
        let gain = rule_cost.ln() - policy_cost.ln();
        self.running_margin += gain;
        if self.window.len() == self.config.window {
            self.window.pop_front();
        }
        self.window.push_back(gain);
        if self.window.len() < self.config.window {
            return WatchEvent::Sampled;
        }
        let mut stats = Running::default();
        for g in &self.window {
            stats.push(*g);
        }
        let bound = (self.config.z * stats.std_err()).max(self.config.min_bound);
        if stats.mean() < -bound {
            let removal = Removal {
                contested_seen: self.contested_seen,
                samples: self.samples,
                window_mean: stats.mean(),
                bound: -bound,
                running_margin: self.running_margin,
            };
            self.removal = Some(removal);
            return WatchEvent::Removed(removal);
        }
        WatchEvent::Sampled
    }

    /// The removal record, if the policy has been removed.
    #[must_use]
    pub fn removal(&self) -> Option<Removal> {
        self.removal
    }

    /// Samples taken so far.
    #[must_use]
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Contested decisions seen so far.
    #[must_use]
    pub fn contested_seen(&self) -> u64 {
        self.contested_seen
    }

    /// Sum of all sampled gains so far.
    #[must_use]
    pub fn running_margin(&self) -> f64 {
        self.running_margin
    }
}

/// One decision taken by a [`DecisionPoint`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionRecord {
    /// Index of the candidate used.
    pub used: usize,
    /// Realized cost of the candidate used, ms.
    pub cost: f64,
    /// Realized cost the fixed rule's pick would have had, ms.
    pub rule_cost: f64,
    /// Whether an active policy disagreed with the rule.
    pub contested: bool,
    /// What the watch did.
    pub event: WatchEvent,
}

/// A decision point running an adopted policy under the band and the watch.
/// Once the watch removes the policy, every later decision uses the fixed rule.
#[derive(Debug, Clone)]
pub struct DecisionPoint<P> {
    policy: Option<P>,
    margin: i64,
    watch: Watch,
}

impl<P: Scorer> DecisionPoint<P> {
    /// Install an adopted `policy` with band `margin`.
    ///
    /// # Errors
    /// [`PolicyError::InvalidConfig`] if the watch config is invalid.
    pub fn new(policy: P, margin: i64, config: WatchConfig) -> Result<Self, PolicyError> {
        Ok(Self {
            policy: Some(policy),
            margin,
            watch: Watch::new(config)?,
        })
    }

    /// True while the policy is still installed.
    #[must_use]
    pub fn policy_active(&self) -> bool {
        self.policy.is_some()
    }

    /// The watch.
    #[must_use]
    pub fn watch(&self) -> &Watch {
        &self.watch
    }

    /// Take the next decision from `twin`.
    pub fn decide(&mut self, twin: &mut Twin) -> DecisionRecord {
        let d = twin.next_decision();
        let out = band_pick(&FixedRule, self.policy.as_ref(), &d.candidates, self.margin);
        let rule_idx = out.rule_choice.unwrap_or(0);
        let used = out.chosen.unwrap_or(rule_idx);
        let cost = twin.realized_cost(&d, used);
        let rule_cost = twin.realized_cost(&d, rule_idx);
        let contested = self.policy.is_some() && out.contested();
        let event = if contested {
            self.watch.observe_contested(rule_cost, cost)
        } else {
            WatchEvent::NotSampled
        };
        if matches!(event, WatchEvent::Removed(_)) {
            // Removal returns the decision point to the fixed rule.
            self.policy = None;
        }
        DecisionRecord {
            used,
            cost,
            rule_cost,
            contested,
            event,
        }
    }
}
