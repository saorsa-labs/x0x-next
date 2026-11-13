//! The prober side of the v0.1 UDP request/echo transport.
//!
//! [`Prober::exchange`] sends one request to each target, then collects
//! echoes until all have arrived or the timeout passes. An echo counts only
//! if it comes from the target's address and carries this run's nonce, a
//! pending sequence number and the timestamp that was sent. The round trip
//! is measured on this node's monotonic clock.

use std::collections::HashMap;
use std::io::ErrorKind;
use std::net::{SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

use crate::clock::Clock;
use crate::error::MeasureError;
use crate::frame::{Class, Frame, Kind, HEADER_LEN};

/// One request to send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Target {
    /// Reflector address.
    pub addr: SocketAddr,
    /// Probe or decision request.
    pub class: Class,
    /// Total frame size in bytes (at least [`HEADER_LEN`]).
    pub frame_bytes: usize,
}

/// A prober bound to one local UDP socket.
#[derive(Debug)]
pub struct Prober {
    socket: UdpSocket,
    nonce: u64,
    next_seq: u64,
    clock: Clock,
}

impl Prober {
    /// Bind a prober to `local` (usually a wildcard address, port 0).
    ///
    /// # Errors
    /// [`MeasureError::Io`] if the socket cannot be bound.
    pub fn bind(local: SocketAddr, nonce: u64, clock: Clock) -> Result<Self, MeasureError> {
        let socket = UdpSocket::bind(local)
            .map_err(|e| MeasureError::io(format!("bind prober on {local}"), e))?;
        Ok(Self {
            socket,
            nonce,
            next_seq: 0,
            clock,
        })
    }

    /// Send one request per target and wait up to `timeout` for the echoes.
    /// Returns, per target, the round trip in microseconds or `None`.
    ///
    /// # Errors
    /// [`MeasureError::Io`] only if the socket's read timeout cannot be set;
    /// send and receive failures count as lost requests.
    pub fn exchange(
        &mut self,
        targets: &[Target],
        timeout: Duration,
    ) -> Result<Vec<Option<u64>>, MeasureError> {
        let mut results = vec![None; targets.len()];
        let mut pending: HashMap<u64, (usize, u64)> = HashMap::new();
        for (i, t) in targets.iter().enumerate() {
            let seq = self.next_seq;
            self.next_seq = self.next_seq.wrapping_add(1);
            let pad = t.frame_bytes.saturating_sub(HEADER_LEN);
            let frame = Frame {
                kind: Kind::Request,
                class: t.class,
                pad_len: u16::try_from(pad).unwrap_or(u16::MAX),
                seq,
                nonce: self.nonce,
                send_ts_us: self.clock.mono_us(),
            };
            if self.socket.send_to(&frame.encode(), t.addr).is_ok() {
                pending.insert(seq, (i, frame.send_ts_us));
            }
        }

        let deadline = Instant::now() + timeout;
        let mut buf = vec![0u8; 65_536];
        while !pending.is_empty() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            self.socket
                .set_read_timeout(Some(remaining))
                .map_err(|e| MeasureError::io("set prober read timeout", e))?;
            let (n, from) = match self.socket.recv_from(&mut buf) {
                Ok(v) => v,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => break,
                // ICMP errors, interrupts: keep waiting until the deadline.
                Err(_) => continue,
            };
            let now_us = self.clock.mono_us();
            let Ok(echo) = Frame::decode(&buf[..n]) else {
                continue;
            };
            if echo.kind != Kind::Echo || echo.nonce != self.nonce {
                continue;
            }
            if let Some(&(i, ts)) = pending.get(&echo.seq) {
                if ts == echo.send_ts_us && targets[i].addr == from {
                    results[i] = Some(now_us.saturating_sub(ts));
                    pending.remove(&echo.seq);
                }
            }
        }
        Ok(results)
    }
}
