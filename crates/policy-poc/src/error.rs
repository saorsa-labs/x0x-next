//! Error type for the policy plane POC (design study §5).

use thiserror::Error;

/// Errors raised by the policy plane POC.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// A feature level was outside its kernel-fixed range.
    #[error("feature {feature} level {value} is outside 0..{levels}")]
    FeatureOutOfRange {
        /// Feature name.
        feature: &'static str,
        /// Offending level.
        value: u8,
        /// Number of levels the feature has.
        levels: u8,
    },
    /// A scorer is not additive per feature, so rung 1 cannot express it.
    #[error(
        "scorer is not additive per feature: {mismatches} of 65536 inputs differ from the table"
    )]
    NotAdditive {
        /// Number of inputs where the derived table disagrees with the scorer.
        mismatches: usize,
    },
    /// Patch (de)serialization failed.
    #[error("patch serialization failed: {0}")]
    Serialization(String),
    /// A cryptographic operation failed.
    #[error("signature operation failed: {0}")]
    Crypto(String),
    /// The patch signature did not verify against the given key.
    #[error("patch signature does not verify")]
    BadSignature,
    /// A configuration value was invalid.
    #[error("invalid configuration: {0}")]
    InvalidConfig(&'static str),
}
