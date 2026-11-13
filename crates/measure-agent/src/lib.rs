//! # measure-agent — the Experiment 16 instrument (v0.1)
//!
//! Runs one decision point of the policy plane in **Measure mode** and
//! nothing else: no adoption, no weights, no trials, no patches. The plan is
//! `docs/experiment-16-plan.md`: §1 describes the instrument and §3 the run
//! protocol and trace schema implemented here. The rule and the band are the
//! exact `policy-poc` code ([`policy_poc::FixedRule`],
//! [`policy_poc::band_pick`], [`policy_poc::rng::SplitMix64`]), so a trace
//! replays through the same code the analysis uses.
//!
//! ## One decision
//!
//! Each tick the agent probes every configured peer and updates that peer's
//! estimators ([`estimator`]). It then draws up to
//! [`policy_poc::twin::DEFAULT_CANDIDATES`] candidates, cuts their estimators
//! to the five feature levels with the fixed boundaries in [`levels`], and
//! picks with the fixed rule ([`decide`]). On one decision in ten it picks
//! uniformly among the candidates inside the band at margin 6 quanta; on one
//! decision in ten it also sends a copy, so the trace holds paired outcomes.
//! The realized outcome (round trip in microseconds, or a timeout) is written
//! to an append-only daily JSONL trace ([`trace`]).
//!
//! ## Deviations from the plan
//!
//! * **Transport (sanctioned by the lead).** The plan's instrument is
//!   "standalone over QUIC". v0.1 uses a minimal UDP request/echo frame
//!   protocol instead ([`frame`]): a frame carries a nonce, a send timestamp
//!   and a sequence number, and the reflector echoes it unchanged except for
//!   its kind byte. QUIC would pull in all of `ant-quic`, and the five
//!   features need only rtt, loss and delivery probing. Standalone over QUIC
//!   remains the v0.2 target. Default port: [`DEFAULT_PORT`] (UDP 7483).
//! * **Balance is not measured.** Request/echo traffic is symmetric, so
//!   balance is the constant mid level [`levels::BALANCE_CONSTANT_LEVEL`]
//!   and the trace header marks it as unmeasured.
//! * **Age is per process.** First successful contact is remembered only
//!   while the agent runs; a restart resets every peer's age to level 0.
//! * **Fixed request size.** Each decision places one request of the
//!   configured size; the plan's size shaping from x0xd `/diagnostics` is not
//!   in v0.1.
//! * **No NTP offset.** The `ntp_offset_ms` field is always `null` in v0.1.
//!
//! ## Determinism
//!
//! All randomness comes from one SplitMix64 stream seeded from the experiment
//! seed and a hash of the node name. Decision `k` owns stream positions
//! `[k·D, (k+1)·D)` with `D` = [`decide::DRAWS_PER_DECISION`], so a restart
//! continues the same stream past the decisions already logged, and
//! [`trace::replay_trace`] re-derives every choice from the header alone.

#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used, clippy::panic))]

pub mod agent;
pub mod clock;
pub mod config;
pub mod decide;
pub mod error;
pub mod estimator;
pub mod frame;
pub mod levels;
pub mod privacy;
pub mod probe;
pub mod reflector;
pub mod selftest;
pub mod trace;

pub use agent::{run_agent, RunSummary};
pub use config::{Config, PeerConfig};
pub use error::MeasureError;
pub use trace::{replay_trace, verify_trace, VerifyReport};

/// Trace schema id written into every header.
pub const SCHEMA_ID: &str = "x0x-measure/1";

/// Probe transport id written into every header (v0.1: UDP request/echo).
pub const TRANSPORT_ID: &str = "udp-echo/1";

/// Default UDP port of the reflector.
pub const DEFAULT_PORT: u16 = 7483;
