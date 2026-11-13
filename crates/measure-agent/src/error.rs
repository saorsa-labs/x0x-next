//! Error type of the measure agent.

use thiserror::Error;

/// Errors raised by the measure agent.
#[derive(Debug, Error)]
pub enum MeasureError {
    /// An I/O operation failed.
    #[error("{context}: {source}")]
    Io {
        /// What the agent was doing.
        context: String,
        /// The underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The configuration is invalid.
    #[error("invalid configuration: {0}")]
    Config(String),
    /// A frame could not be decoded.
    #[error("bad frame: {0}")]
    Frame(&'static str),
    /// A trace line could not be parsed or serialized.
    #[error("trace {file} line {line}: {message}")]
    TraceFormat {
        /// File name.
        file: String,
        /// One-based line number (0 when not tied to a line).
        line: usize,
        /// What was wrong.
        message: String,
    },
    /// A replayed decision did not match its logged record.
    #[error("replay mismatch at decision {decision}: {message}")]
    Replay {
        /// Decision number.
        decision: u64,
        /// What differed.
        message: String,
    },
    /// The manifest of a closed daily file did not match the file.
    #[error("manifest mismatch for {file}: {message}")]
    Manifest {
        /// Trace file name.
        file: String,
        /// What differed.
        message: String,
    },
    /// The self-test failed one of its own checks.
    #[error("self-test failed: {0}")]
    SelfTest(String),
}

impl MeasureError {
    /// Wrap an I/O error with context.
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}
