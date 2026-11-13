//! The reflector side of the v0.1 UDP request/echo transport.
//!
//! The reflector echoes every valid request frame back to its sender with
//! the kind byte flipped ([`crate::frame::echo_in_place`]) and drops
//! everything else. It keeps no state about senders and writes no trace.
//! An echo is exactly as large as its request, so there is no amplification.

use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::error::MeasureError;
use crate::frame::echo_in_place;

/// How often a blocked reflector checks its stop flag.
const POLL: Duration = Duration::from_millis(100);

/// Fault injection, used by the self-test to give peers different features.
/// A production reflector uses [`Faults::NONE`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Faults {
    /// Extra delay before each echo.
    pub delay: Duration,
    /// Drop every n-th valid request (counting from 1); `None` drops nothing.
    pub drop_every: Option<u64>,
}

impl Faults {
    /// No faults.
    pub const NONE: Self = Self {
        delay: Duration::ZERO,
        drop_every: None,
    };
}

/// A running reflector thread.
#[derive(Debug)]
pub struct Reflector {
    local: SocketAddr,
    stop: Arc<AtomicBool>,
    echoed: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl Reflector {
    /// Bind to `addr` and start echoing on a background thread.
    ///
    /// # Errors
    /// [`MeasureError::Io`] if the socket cannot be bound or configured, or
    /// the thread cannot be spawned.
    pub fn spawn(addr: SocketAddr, faults: Faults) -> Result<Self, MeasureError> {
        let socket = UdpSocket::bind(addr)
            .map_err(|e| MeasureError::io(format!("bind reflector on {addr}"), e))?;
        socket
            .set_read_timeout(Some(POLL))
            .map_err(|e| MeasureError::io("set reflector read timeout", e))?;
        let local = socket
            .local_addr()
            .map_err(|e| MeasureError::io("reflector local address", e))?;
        let stop = Arc::new(AtomicBool::new(false));
        let echoed = Arc::new(AtomicU64::new(0));
        let (s, n) = (Arc::clone(&stop), Arc::clone(&echoed));
        let handle = std::thread::Builder::new()
            .name(format!("reflector-{}", local.port()))
            .spawn(move || serve(&socket, faults, &s, &n))
            .map_err(|e| MeasureError::io("spawn reflector thread", e))?;
        Ok(Self {
            local,
            stop,
            echoed,
            handle: Some(handle),
        })
    }

    /// The bound address (useful when bound to port 0).
    #[must_use]
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }

    /// Number of echoes sent so far.
    #[must_use]
    pub fn echoed(&self) -> u64 {
        self.echoed.load(Ordering::Relaxed)
    }

    /// Block until the reflector thread exits (it only exits after
    /// [`Reflector::stop`] or on drop).
    pub fn join(mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }

    /// Stop the thread and wait for it.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for Reflector {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn serve(socket: &UdpSocket, faults: Faults, stop: &AtomicBool, echoed: &AtomicU64) {
    let mut buf = vec![0u8; 65_536];
    let mut valid: u64 = 0;
    while !stop.load(Ordering::Relaxed) {
        let (n, from) = match socket.recv_from(&mut buf) {
            Ok(v) => v,
            // Timeouts let the loop check the stop flag; other errors (ICMP
            // reports, interrupts) are not fatal to a reflector.
            Err(_) => continue,
        };
        let pkt = &mut buf[..n];
        if !echo_in_place(pkt) {
            continue;
        }
        valid += 1;
        if faults
            .drop_every
            .is_some_and(|k| k > 0 && valid.is_multiple_of(k))
        {
            continue;
        }
        if !faults.delay.is_zero() {
            std::thread::sleep(faults.delay);
        }
        if socket.send_to(pkt, from).is_ok() {
            echoed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::Clock;
    use crate::frame::Class;
    use crate::probe::{Prober, Target};

    #[test]
    fn echoes_over_loopback_and_drops_on_schedule() {
        let r = Reflector::spawn(
            "127.0.0.1:0".parse().unwrap(),
            Faults {
                delay: Duration::ZERO,
                drop_every: Some(3),
            },
        )
        .unwrap();
        let mut p = Prober::bind("127.0.0.1:0".parse().unwrap(), 42, Clock::start()).unwrap();
        let t = Target {
            addr: r.local_addr(),
            class: Class::Probe,
            frame_bytes: 512,
        };
        let mut got = Vec::new();
        for _ in 0..6 {
            got.push(p.exchange(&[t], Duration::from_millis(300)).unwrap()[0].is_some());
        }
        assert_eq!(got, [true, true, false, true, true, false]);
        assert_eq!(r.echoed(), 4);
        r.stop();
    }
}
