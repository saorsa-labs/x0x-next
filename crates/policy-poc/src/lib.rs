//! # policy-poc — the rung-1 policy plane, end to end
//!
//! A proof of concept of the policy plane from the x0x design study
//! (`docs/design-study-2026-11-13.pdf`, §5 "The policy plane"), specified in
//! `docs/poc-spec.md`. One decision point — scoring a candidate peer — is
//! carried through every element of the plane:
//!
//! | Spec | Study §5 element | Module |
//! |------|------------------|--------|
//! | §1 | kernel-fixed features (65,536 inputs) | [`features`] |
//! | §2–§3 | quantized scores, the fixed rule (rung 0) | [`rule`] |
//! | §4 | the points table (rung 1, "written weights") | [`table`] |
//! | §5 | the checker: directions, roles, limits, exact diff | [`checker`] |
//! | §6 | the band and the kernel-enforced pick | [`band`] |
//! | §7 | the twin: world one and world two | [`twin`], [`fit`], [`rng`] |
//! | §8 | the trial and the watch with automatic removal | [`trial`], [`watch`] |
//! | §9 | signed patches (ML-DSA-65, `pqc` feature) | [`patch`] |
//! | §10 | the six-step evidence log | [`demo`] |
//!
//! Everything is deterministic: no wall clock, no I/O in library code, and
//! every random source is seeded.

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod band;
pub mod checker;
#[cfg(feature = "pqc")]
pub mod demo;
pub mod error;
pub mod features;
pub mod fit;
pub mod patch;
pub mod rng;
pub mod rule;
pub mod table;
pub mod trial;
pub mod twin;
pub mod watch;

pub use band::{band_pick, Candidate, PickOutcome};
pub use checker::{CheckFailure, CheckReport, Checker, DiffReport};
pub use error::PolicyError;
pub use features::{Feature, Features, INPUT_SPACE};
pub use patch::{evaluate_patch, Patch, PatchDecision};
#[cfg(feature = "pqc")]
pub use patch::{receive_patch, PatchKeypair, SignedPatch};
pub use rule::{FixedRule, FnScorer, Scorer, QUANT};
pub use table::{FeatureSteps, PointsTable};
pub use trial::{TrialConfig, TrialReport, Verdict};
pub use twin::{Twin, WorldKind, WorldParams};
pub use watch::{DecisionPoint, Watch, WatchConfig, WatchEvent};
