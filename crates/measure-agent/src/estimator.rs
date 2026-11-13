//! Per-peer estimators (plan §1, "the five features, measured by the node").
//!
//! * rtt: smoothed like QUIC/TCP srtt, `srtt = 7/8·srtt + 1/8·sample`, in
//!   integer microseconds; seeded by the first sample. Fed by probes and
//!   decision requests.
//! * loss: share of the last [`LOSS_WINDOW`] measurement probes that got no
//!   echo before the probe timeout.
//! * delivery: share of this node's decision requests to the peer that
//!   completed, with a prior of [`DELIVERY_PRIOR_OK`] successes in
//!   [`DELIVERY_PRIOR_N`] requests so a new peer starts at 90%.
//! * age: time since the first echo of any kind, per process.

use std::collections::VecDeque;

use crate::levels::RawFeatures;

/// Sliding window of measurement probes used for loss.
pub const LOSS_WINDOW: usize = 32;
/// Prior successes in the delivery estimate.
pub const DELIVERY_PRIOR_OK: u64 = 9;
/// Prior request count in the delivery estimate.
pub const DELIVERY_PRIOR_N: u64 = 10;

/// Estimators of one peer.
#[derive(Debug, Clone, Default)]
pub struct PeerEstimator {
    srtt_us: Option<u64>,
    probes: VecDeque<bool>,
    requests_sent: u64,
    requests_completed: u64,
    first_seen_ns: Option<u64>,
}

impl PeerEstimator {
    /// A peer never contacted.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn on_rtt(&mut self, rtt_us: u64, now_ns: u64) {
        self.srtt_us = Some(match self.srtt_us {
            None => rtt_us,
            Some(s) => s - s / 8 + rtt_us / 8,
        });
        if self.first_seen_ns.is_none() {
            self.first_seen_ns = Some(now_ns);
        }
    }

    /// Record a measurement probe: `Some(rtt)` if echoed, `None` if lost.
    pub fn on_probe(&mut self, outcome: Option<u64>, now_ns: u64) {
        if self.probes.len() == LOSS_WINDOW {
            self.probes.pop_front();
        }
        self.probes.push_back(outcome.is_none());
        if let Some(rtt) = outcome {
            self.on_rtt(rtt, now_ns);
        }
    }

    /// Record a decision request: `Some(rtt)` if completed, `None` on timeout.
    pub fn on_request(&mut self, outcome: Option<u64>, now_ns: u64) {
        self.requests_sent += 1;
        if let Some(rtt) = outcome {
            self.requests_completed += 1;
            self.on_rtt(rtt, now_ns);
        }
    }

    /// Current raw values at monotonic time `now_ns`.
    #[must_use]
    pub fn raw(&self, now_ns: u64) -> RawFeatures {
        let lost = self.probes.iter().filter(|l| **l).count() as u64;
        let loss_ppm = if self.probes.is_empty() {
            0
        } else {
            lost * 1_000_000 / self.probes.len() as u64
        };
        let delivery_ppm = (self.requests_completed + DELIVERY_PRIOR_OK) * 1_000_000
            / (self.requests_sent + DELIVERY_PRIOR_N);
        RawFeatures {
            srtt_us: self.srtt_us,
            loss_ppm,
            delivery_ppm,
            age_s: self
                .first_seen_ns
                .map(|t| now_ns.saturating_sub(t) / 1_000_000_000),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srtt_smooths_and_loss_windows() {
        let mut e = PeerEstimator::new();
        assert_eq!(e.raw(0).srtt_us, None);
        assert_eq!(e.raw(0).delivery_ppm, 900_000);
        e.on_probe(Some(8_000), 5);
        assert_eq!(e.raw(5).srtt_us, Some(8_000));
        e.on_probe(Some(16_000), 6);
        assert_eq!(e.raw(6).srtt_us, Some(9_000));
        e.on_probe(None, 7);
        assert_eq!(e.raw(7).loss_ppm, 333_333);
        for _ in 0..LOSS_WINDOW {
            e.on_probe(Some(9_000), 8);
        }
        assert_eq!(e.raw(8).loss_ppm, 0);
        assert_eq!(e.raw(5 + 3_000_000_000).age_s, Some(3));
    }

    #[test]
    fn delivery_counts_requests() {
        let mut e = PeerEstimator::new();
        e.on_request(None, 0);
        assert_eq!(e.raw(0).delivery_ppm, 818_181);
        assert_eq!(e.raw(0).age_s, None);
        e.on_request(Some(100), 1);
        assert_eq!(e.raw(1).delivery_ppm, 833_333);
    }
}
