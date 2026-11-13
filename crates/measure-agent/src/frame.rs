//! The v0.1 probe frame: a minimal UDP request/echo protocol.
//!
//! This is the sanctioned deviation from the plan's QUIC transport (see the
//! crate docs). A prober sends a [`Kind::Request`] frame; a reflector sends
//! the same bytes back with the kind byte set to [`Kind::Echo`]. The echo is
//! never larger than the request, so a reflector cannot amplify traffic.
//!
//! Layout (big endian), [`HEADER_LEN`] = 32 bytes followed by `pad_len` zero
//! bytes:
//!
//! | Offset | Size | Field |
//! |-------:|-----:|-------|
//! | 0 | 2 | magic `b"XM"` |
//! | 2 | 1 | version, [`VERSION`] |
//! | 3 | 1 | kind: 0 request, 1 echo |
//! | 4 | 1 | class: 0 measurement probe, 1 decision request |
//! | 5 | 1 | reserved, 0 |
//! | 6 | 2 | `pad_len` |
//! | 8 | 8 | sequence number |
//! | 16 | 8 | nonce (per prober run) |
//! | 24 | 8 | send timestamp, prober monotonic microseconds |

use crate::error::MeasureError;

/// Frame magic.
pub const MAGIC: [u8; 2] = *b"XM";
/// Frame version.
pub const VERSION: u8 = 1;
/// Length of the fixed header.
pub const HEADER_LEN: usize = 32;
/// Largest frame this protocol sends (fits one UDP datagram over IPv4).
pub const MAX_FRAME: usize = 65_507;

/// Whether a frame asks for an echo or is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Sent by a prober.
    Request,
    /// Sent back by a reflector.
    Echo,
}

/// What a request is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// A per-tick measurement probe (feeds rtt and loss).
    Probe,
    /// A decision's request (feeds delivery and the realized outcome).
    Decision,
}

/// One frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// Request or echo.
    pub kind: Kind,
    /// Probe or decision request.
    pub class: Class,
    /// Number of zero padding bytes after the header.
    pub pad_len: u16,
    /// Sequence number, unique per prober run.
    pub seq: u64,
    /// Nonce of the prober run; echoes with another nonce are ignored.
    pub nonce: u64,
    /// Prober monotonic send time in microseconds.
    pub send_ts_us: u64,
}

impl Frame {
    /// Total encoded length.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        HEADER_LEN + usize::from(self.pad_len)
    }

    /// Encode into a fresh buffer.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.encoded_len()];
        out[0..2].copy_from_slice(&MAGIC);
        out[2] = VERSION;
        out[3] = match self.kind {
            Kind::Request => 0,
            Kind::Echo => 1,
        };
        out[4] = match self.class {
            Class::Probe => 0,
            Class::Decision => 1,
        };
        out[6..8].copy_from_slice(&self.pad_len.to_be_bytes());
        out[8..16].copy_from_slice(&self.seq.to_be_bytes());
        out[16..24].copy_from_slice(&self.nonce.to_be_bytes());
        out[24..32].copy_from_slice(&self.send_ts_us.to_be_bytes());
        out
    }

    /// Decode a datagram.
    ///
    /// # Errors
    /// [`MeasureError::Frame`] when the magic, version, kind, class, reserved
    /// byte or length is wrong.
    pub fn decode(buf: &[u8]) -> Result<Self, MeasureError> {
        if buf.len() < HEADER_LEN {
            return Err(MeasureError::Frame("shorter than the header"));
        }
        if buf[0..2] != MAGIC {
            return Err(MeasureError::Frame("bad magic"));
        }
        if buf[2] != VERSION {
            return Err(MeasureError::Frame("unknown version"));
        }
        let kind = match buf[3] {
            0 => Kind::Request,
            1 => Kind::Echo,
            _ => return Err(MeasureError::Frame("unknown kind")),
        };
        let class = match buf[4] {
            0 => Class::Probe,
            1 => Class::Decision,
            _ => return Err(MeasureError::Frame("unknown class")),
        };
        if buf[5] != 0 {
            return Err(MeasureError::Frame("reserved byte set"));
        }
        let pad_len = u16::from_be_bytes([buf[6], buf[7]]);
        if buf.len() != HEADER_LEN + usize::from(pad_len) {
            return Err(MeasureError::Frame("length does not match pad_len"));
        }
        Ok(Self {
            kind,
            class,
            pad_len,
            seq: read_u64(&buf[8..16]),
            nonce: read_u64(&buf[16..24]),
            send_ts_us: read_u64(&buf[24..32]),
        })
    }
}

fn read_u64(bytes: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(bytes);
    u64::from_be_bytes(a)
}

/// Turn a received request datagram into its echo in place. Returns `false`
/// (and leaves the buffer untouched) for anything that is not a valid
/// request, which the reflector then drops.
pub fn echo_in_place(buf: &mut [u8]) -> bool {
    match Frame::decode(buf) {
        Ok(f) if f.kind == Kind::Request => {
            buf[3] = 1;
            true
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(pad: u16) -> Frame {
        Frame {
            kind: Kind::Request,
            class: Class::Decision,
            pad_len: pad,
            seq: 0x0102_0304_0506_0708,
            nonce: u64::MAX - 3,
            send_ts_us: 1_234_567,
        }
    }

    #[test]
    fn roundtrip() {
        for pad in [0u16, 1, 992, 8160] {
            let f = sample(pad);
            let bytes = f.encode();
            assert_eq!(bytes.len(), HEADER_LEN + usize::from(pad));
            assert_eq!(Frame::decode(&bytes).unwrap(), f);
        }
        let echo = Frame {
            kind: Kind::Echo,
            class: Class::Probe,
            ..sample(0)
        };
        assert_eq!(Frame::decode(&echo.encode()).unwrap(), echo);
    }

    #[test]
    fn echo_flips_kind_only() {
        let f = sample(16);
        let mut bytes = f.encode();
        assert!(echo_in_place(&mut bytes));
        let e = Frame::decode(&bytes).unwrap();
        assert_eq!(e.kind, Kind::Echo);
        assert_eq!(
            Frame {
                kind: Kind::Request,
                ..e
            },
            f
        );
        // An echo is not echoed again.
        assert!(!echo_in_place(&mut bytes));
    }

    #[test]
    fn rejects_malformed() {
        let good = sample(4).encode();
        assert!(Frame::decode(&good[..HEADER_LEN - 1]).is_err());
        assert!(Frame::decode(&good[..good.len() - 1]).is_err());
        for (i, v) in [(0usize, b'Y'), (2, 9), (3, 7), (4, 9), (5, 1)] {
            let mut bad = good.clone();
            bad[i] = v;
            assert!(Frame::decode(&bad).is_err(), "byte {i}");
        }
    }
}
