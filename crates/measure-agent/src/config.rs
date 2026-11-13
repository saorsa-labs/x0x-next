//! Agent configuration (TOML).
//!
//! ```toml
//! node = "studio1"
//! data_dir = "/var/lib/measure-agent"
//! listen_port = 7483
//! tick_ms = 10000
//!
//! [[peers]]
//! label = "nyc"
//! addr = "192.0.2.10:7483"
//! ```
//!
//! Optional keys: `listen_ip` (default `0.0.0.0`), `experiment_seed`,
//! `node_class`, `request_bytes`, `probe_timeout_ms`, `request_timeout_ms`.
//! Peer labels and addresses are local only: labels reach a trace only as a
//! salted hash, and addresses never do.

use std::collections::BTreeSet;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::MeasureError;
use crate::frame::{HEADER_LEN, MAX_FRAME};
use crate::DEFAULT_PORT;

/// Default experiment seed (Experiment 16).
pub const DEFAULT_EXPERIMENT_SEED: u64 = 16;

/// One configured peer.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeerConfig {
    /// Local label; hashed before it reaches a trace.
    pub label: String,
    /// The peer's reflector address. Never written to a trace.
    pub addr: SocketAddr,
}

/// The whole configuration.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Node name; hashed before it reaches a trace.
    pub node: String,
    /// Directory holding the salt, traces and manifests.
    pub data_dir: PathBuf,
    /// Reflector UDP port.
    #[serde(default = "default_port")]
    pub listen_port: u16,
    /// Reflector bind address.
    #[serde(default = "default_listen_ip")]
    pub listen_ip: IpAddr,
    /// Milliseconds between decisions.
    pub tick_ms: u64,
    /// Experiment seed, mixed with the node name into the stream seed.
    #[serde(default = "default_seed")]
    pub experiment_seed: u64,
    /// Node class (e.g. `vps`, `mac`, `pi`, `phone`), copied into headers.
    #[serde(default)]
    pub node_class: Option<String>,
    /// Size of a decision request frame in bytes.
    #[serde(default = "default_request_bytes")]
    pub request_bytes: usize,
    /// How long to wait for probe echoes each tick.
    #[serde(default = "default_probe_timeout")]
    pub probe_timeout_ms: u64,
    /// How long to wait for a decision request's echo.
    #[serde(default = "default_request_timeout")]
    pub request_timeout_ms: u64,
    /// Peers to measure.
    pub peers: Vec<PeerConfig>,
}

fn default_port() -> u16 {
    DEFAULT_PORT
}
fn default_listen_ip() -> IpAddr {
    IpAddr::V4(Ipv4Addr::UNSPECIFIED)
}
fn default_seed() -> u64 {
    DEFAULT_EXPERIMENT_SEED
}
fn default_request_bytes() -> usize {
    1024
}
fn default_probe_timeout() -> u64 {
    1_000
}
fn default_request_timeout() -> u64 {
    2_000
}

impl Config {
    /// Parse and validate TOML text.
    ///
    /// # Errors
    /// [`MeasureError::Config`] on a parse error or an invalid value.
    pub fn from_toml(text: &str) -> Result<Self, MeasureError> {
        let cfg: Self = toml::from_str(text).map_err(|e| MeasureError::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read, parse and validate a TOML file.
    ///
    /// # Errors
    /// [`MeasureError::Io`] if unreadable, else as [`Config::from_toml`].
    pub fn load(path: &Path) -> Result<Self, MeasureError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| MeasureError::io(format!("read config {}", path.display()), e))?;
        Self::from_toml(&text)
    }

    /// Check every value.
    ///
    /// # Errors
    /// [`MeasureError::Config`] naming the first invalid value.
    pub fn validate(&self) -> Result<(), MeasureError> {
        let bad = |m: &str| Err(MeasureError::Config(m.to_string()));
        if self.node.trim().is_empty() {
            return bad("node must not be empty");
        }
        if self.peers.is_empty() {
            return bad("at least one peer is required");
        }
        let mut labels = BTreeSet::new();
        for p in &self.peers {
            if p.label.trim().is_empty() {
                return bad("peer labels must not be empty");
            }
            if !labels.insert(p.label.as_str()) {
                return Err(MeasureError::Config(format!(
                    "duplicate peer label {:?}",
                    p.label
                )));
            }
        }
        let v4 = self.peers[0].addr.is_ipv4();
        if self.peers.iter().any(|p| p.addr.is_ipv4() != v4) {
            return bad("v0.1 needs every peer in the same address family");
        }
        if self.request_bytes < HEADER_LEN || self.request_bytes > MAX_FRAME {
            return Err(MeasureError::Config(format!(
                "request_bytes must be in {HEADER_LEN}..={MAX_FRAME}"
            )));
        }
        if self.probe_timeout_ms == 0 || self.request_timeout_ms == 0 {
            return bad("timeouts must be positive");
        }
        Ok(())
    }

    /// The reflector bind address.
    #[must_use]
    pub fn listen_addr(&self) -> SocketAddr {
        SocketAddr::new(self.listen_ip, self.listen_port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
node = "n1"
data_dir = "/tmp/x"
tick_ms = 10000
[[peers]]
label = "a"
addr = "127.0.0.1:7483"
[[peers]]
label = "b"
addr = "127.0.0.1:7484"
"#;

    #[test]
    fn parses_with_defaults() {
        let c = Config::from_toml(GOOD).unwrap();
        assert_eq!(c.listen_port, 7483);
        assert_eq!(c.request_bytes, 1024);
        assert_eq!(c.peers.len(), 2);
        assert_eq!(c.experiment_seed, DEFAULT_EXPERIMENT_SEED);
    }

    #[test]
    fn rejects_bad_configs() {
        assert!(Config::from_toml(&GOOD.replace("\"b\"", "\"a\"")).is_err());
        assert!(Config::from_toml(&GOOD.replace("127.0.0.1:7484", "[::1]:7484")).is_err());
        assert!(Config::from_toml(&format!("{GOOD}\nbogus = 1")).is_err());
        assert!(Config::from_toml(&GOOD.replace("tick_ms", "request_bytes = 4\ntick_ms")).is_err());
    }
}
