//! Salted hashing of names, and the stream seed (plan §1 "peer ids are
//! hashed with a per-node salt", §3 "seeded from the experiment seed and a
//! hash of the node id").
//!
//! The salt is 32 random bytes created on first start and kept in the data
//! directory, so a peer's hash is stable across restarts. The salt is also
//! written into every trace header, as the lead specified; anyone holding a
//! trace can therefore test a guessed label against a hash, but a trace
//! alone never names a peer or carries an address.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::error::MeasureError;

/// Salt length in bytes.
pub const SALT_LEN: usize = 32;
/// File in the data directory holding the hex salt.
pub const SALT_FILE: &str = "salt.hex";

/// Lower-case hex of `bytes`.
#[must_use]
pub fn to_hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from(DIGITS[usize::from(b >> 4)]));
        s.push(char::from(DIGITS[usize::from(b & 0x0f)]));
    }
    s
}

/// Parse lower- or upper-case hex.
#[must_use]
pub fn from_hex(s: &str) -> Option<Vec<u8>> {
    let s = s.trim();
    if !s.len().is_multiple_of(2) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok())
        .collect()
}

/// Read `n` bytes from the OS random source.
///
/// # Errors
/// [`MeasureError::Io`] when `/dev/urandom` cannot be read.
pub fn os_random(n: usize) -> Result<Vec<u8>, MeasureError> {
    let mut buf = vec![0u8; n];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut f| f.read_exact(&mut buf))
        .map_err(|e| MeasureError::io("read /dev/urandom", e))?;
    Ok(buf)
}

/// A random `u64` from the OS.
///
/// # Errors
/// As [`os_random`].
pub fn os_random_u64() -> Result<u64, MeasureError> {
    let b = os_random(8)?;
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    Ok(u64::from_le_bytes(a))
}

/// Load the node's salt from `dir`, creating it on first use.
///
/// # Errors
/// [`MeasureError::Io`] on a filesystem error, [`MeasureError::Config`] if
/// the stored salt is malformed.
pub fn load_or_create_salt(dir: &Path) -> Result<Vec<u8>, MeasureError> {
    let path = dir.join(SALT_FILE);
    match std::fs::read_to_string(&path) {
        Ok(text) => match from_hex(&text) {
            Some(s) if s.len() == SALT_LEN => Ok(s),
            _ => Err(MeasureError::Config(format!(
                "{} is not {SALT_LEN} bytes of hex",
                path.display()
            ))),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let salt = os_random(SALT_LEN)?;
            std::fs::write(&path, to_hex(&salt))
                .map_err(|e| MeasureError::io(format!("write {}", path.display()), e))?;
            Ok(salt)
        }
        Err(e) => Err(MeasureError::io(format!("read {}", path.display()), e)),
    }
}

fn salted(domain: &[u8], salt: &[u8], name: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(domain);
    h.update([0u8]);
    h.update(salt);
    h.update(name.as_bytes());
    h.finalize().into()
}

/// Salted hash of a peer label: SHA-256 over a domain tag, the salt and the
/// label, first 16 bytes as hex.
#[must_use]
pub fn peer_hash(salt: &[u8], label: &str) -> String {
    to_hex(&salted(b"x0x-measure/1 peer", salt, label)[..16])
}

/// Salted hash of the node name, full SHA-256 as hex.
#[must_use]
pub fn node_hash(salt: &[u8], node: &str) -> String {
    to_hex(&salted(b"x0x-measure/1 node", salt, node))
}

/// The node's stream seed: first 8 bytes (little endian) of SHA-256 over a
/// domain tag, the experiment seed and SHA-256 of the node name. Unsalted on
/// purpose, so the experiment can re-derive every node's seed.
#[must_use]
pub fn stream_seed(experiment_seed: u64, node: &str) -> u64 {
    let node_digest: [u8; 32] = Sha256::digest(node.as_bytes()).into();
    let mut h = Sha256::new();
    h.update(b"x0x-measure/1 seed");
    h.update(experiment_seed.to_le_bytes());
    h.update(node_digest);
    let d: [u8; 32] = h.finalize().into();
    let mut a = [0u8; 8];
    a.copy_from_slice(&d[..8]);
    u64::from_le_bytes(a)
}

/// SHA-256 of `bytes` as hex.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    to_hex(&Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let b = [0u8, 1, 0xab, 0xff];
        assert_eq!(to_hex(&b), "0001abff");
        assert_eq!(from_hex("0001ABff").unwrap(), b);
        assert!(from_hex("abc").is_none());
        assert!(from_hex("zz").is_none());
    }

    #[test]
    fn hashes_depend_on_salt_and_hide_label() {
        let a = peer_hash(&[1; 32], "saorsa-2");
        let b = peer_hash(&[2; 32], "saorsa-2");
        assert_ne!(a, b);
        assert_eq!(a.len(), 32);
        assert!(!a.contains("saorsa"));
        assert_ne!(node_hash(&[1; 32], "x"), node_hash(&[1; 32], "y"));
        assert_eq!(stream_seed(16, "n"), stream_seed(16, "n"));
        assert_ne!(stream_seed(16, "n"), stream_seed(17, "n"));
    }
}
