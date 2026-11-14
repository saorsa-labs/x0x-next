//! # exp16-analysis — the Experiment 16 analysis (plan §4)
//!
//! Reads exported `x0x-measure/1` traces (one data directory per node, or
//! several), re-verifies every file with measure-agent's own replay and
//! manifest checks ([`load`]), and runs the analysis of
//! `docs/experiment-16-plan.md` §4 ([`analysis`]):
//!
//! 1. **Best single table**: [`policy_poc::fit::fit_table`] on the pooled fit
//!    window of all nodes, or the fixed rule if that has the lower pooled
//!    fit-window cost.
//! 2. **Formula with per-node constants**: [`policy_poc::fit::fit_formula`]
//!    on the node's own fit window.
//! 3. **Per-node table**: `fit_table` on the node's own fit window.
//!
//! Every fitted table must pass [`policy_poc::Checker::check`]. Each scorer
//! picks inside the band at the logged margin (6 quanta) and is scored on
//! the node's holdout by self-normalized inverse-propensity weighting over
//! the logged picks, cross-checked against the logged copies ([`estimate`]
//! documents the estimator and its limits). The pass metric, the
//! formula-versus-table match and the band-margin spread use the thresholds
//! the plan fixes ([`analysis::GAIN_THRESHOLD`], [`analysis::PASS_SHARE`],
//! [`analysis::MATCH_TOLERANCE`]).
//!
//! Runs with few nodes, few holdout decisions, or a holdout-fraction split
//! are labelled small-sample: NOT the experiment's verdict.

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod analysis;
pub mod error;
pub mod estimate;
pub mod load;
pub mod report;

pub use analysis::{analyze, Options, Report, Split, Verdict};
pub use error::AnalysisError;
pub use load::{load_nodes, LoggedDecision, NodeTrace};
