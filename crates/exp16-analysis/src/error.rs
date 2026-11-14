//! Error type of the Experiment 16 analysis.

use thiserror::Error;

/// Errors raised by the analysis.
#[derive(Debug, Error)]
pub enum AnalysisError {
    /// A trace failed verification (format, replay or manifest), or could
    /// not be read. Tampered traces are refused through this variant.
    #[error("trace {file} refused: {source}")]
    Trace {
        /// The trace file.
        file: String,
        /// What measure-agent's checks reported.
        #[source]
        source: measure_agent::MeasureError,
    },
    /// A trace verified but its content cannot be analysed.
    #[error("trace {file}: {message}")]
    Content {
        /// The trace file.
        file: String,
        /// What was wrong.
        message: String,
    },
    /// An I/O operation failed.
    #[error("{context}: {source}")]
    Io {
        /// What the analysis was doing.
        context: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The command line or analysis options are invalid.
    #[error("usage: {0}")]
    Usage(String),
    /// There is nothing to analyse.
    #[error("no data: {0}")]
    NoData(String),
    /// The JSON report could not be serialized.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl AnalysisError {
    /// Wrap an I/O error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}
