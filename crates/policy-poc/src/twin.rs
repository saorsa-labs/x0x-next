//! The Twin: a deterministic simulator with two worlds (design study §5, "the
//! twin"; poc-spec §7).
//!
//! The twin produces candidate sets and realized outcomes from a seed. Each
//! decision offers a few candidate peers; picking one costs a realized amount
//! (milliseconds) = the world's expected cost for those features × a per
//! candidate log-normal noise factor. Noise is drawn per candidate, so two
//! scorers compared on the same decision ("copies") see the same outcome when
//! they pick the same peer.
//!
//! * **World one** — the expected cost is a product of one-feature terms. In
//!   log space it is a sum of one-feature terms, which rung 1 can express, so
//!   a fitted table can approach the oracle.
//! * **World two** — a lost packet costs a fixed timeout, so loss hurts most
//!   on fast paths. That is an rtt × loss interaction that rung 1 cannot
//!   express; a fitted table beats the rule but not the oracle.

use crate::band::Candidate;
use crate::features::{Features, AGE_LEVELS, BALANCE_LEVELS, RTT_LEVELS};
use crate::rng::SplitMix64;

/// Loss probability per loss level (world two).
pub const LOSS_PROB: [f64; 8] = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40];
/// How often each loss level occurs among candidates.
pub const LOSS_FREQ: [f64; 8] = [0.40, 0.20, 0.13, 0.09, 0.07, 0.05, 0.04, 0.02];
/// How often each delivery level occurs among candidates.
pub const DELIVERY_FREQ: [f64; 8] = [0.03, 0.04, 0.06, 0.09, 0.13, 0.20, 0.22, 0.23];
/// Cost multiplier per delivery level (both worlds).
pub const DELIVERY_MULT: [f64; 8] = [6.0, 4.0, 1.7, 1.5, 1.35, 1.2, 1.1, 1.0];

/// Which family of cost model a world uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorldKind {
    /// Product of one-feature terms (additive in log space).
    One,
    /// Fixed retransmission timeout per lost packet (rtt × loss interaction).
    Two,
}

/// Parameters of a world. The watch demo shifts them after adoption.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldParams {
    /// Cost model family.
    pub kind: WorldKind,
    /// Cost of the fastest, cleanest path, in ms.
    pub base_ms: f64,
    /// rtt term is `(rtt + 1)^rtt_power × rtt_growth^rtt`.
    pub rtt_power: f64,
    /// See `rtt_power`.
    pub rtt_growth: f64,
    /// World one: loss term is `1 + loss_slope × loss`.
    pub loss_slope: f64,
    /// World two: cost of one lost packet, in ms.
    pub loss_timeout_ms: f64,
    /// Standard deviation of the log-normal noise on realized costs.
    pub noise_sigma: f64,
}

impl WorldParams {
    /// World one as specified: `20 (rtt+1) (1 + 0.35 loss) × delivery`.
    #[must_use]
    pub const fn world_one() -> Self {
        Self {
            kind: WorldKind::One,
            base_ms: 20.0,
            rtt_power: 1.0,
            rtt_growth: 1.0,
            loss_slope: 0.35,
            loss_timeout_ms: 0.0,
            noise_sigma: 0.5,
        }
    }

    /// World two as specified: `(20 (rtt+1) + 200 p) / (1 − p) × delivery`.
    #[must_use]
    pub const fn world_two() -> Self {
        Self {
            kind: WorldKind::Two,
            base_ms: 20.0,
            rtt_power: 1.0,
            rtt_growth: 1.0,
            loss_slope: 0.0,
            loss_timeout_ms: 200.0,
            noise_sigma: 0.5,
        }
    }

    /// World one after a shift: latency now compounds (each rtt level costs
    /// ×1.3) and loss no longer matters. Used to turn an adopted policy bad.
    #[must_use]
    pub const fn world_one_shifted() -> Self {
        Self {
            kind: WorldKind::One,
            base_ms: 20.0,
            rtt_power: 0.0,
            rtt_growth: 1.3,
            loss_slope: 0.0,
            loss_timeout_ms: 0.0,
            noise_sigma: 0.5,
        }
    }

    /// Expected (noise-free) cost of using a peer with features `f`, in ms.
    #[must_use]
    pub fn expected_cost(&self, f: Features) -> f64 {
        let rtt = f64::from(f.rtt());
        let rtt_term = (rtt + 1.0).powf(self.rtt_power) * self.rtt_growth.powf(rtt);
        let delivery = DELIVERY_MULT
            .get(f.delivery() as usize)
            .copied()
            .unwrap_or(1.0);
        match self.kind {
            WorldKind::One => {
                let loss = 1.0 + self.loss_slope * f64::from(f.loss());
                self.base_ms * rtt_term * loss * delivery
            }
            WorldKind::Two => {
                let p = LOSS_PROB.get(f.loss() as usize).copied().unwrap_or(0.0);
                (self.base_ms * rtt_term + self.loss_timeout_ms * p) / (1.0 - p) * delivery
            }
        }
    }
}

/// One decision: the candidate set and each candidate's noise factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    /// Candidates offered.
    pub candidates: Vec<Candidate>,
    /// Multiplicative noise factor per candidate (same order).
    pub noise: Vec<f64>,
}

/// The deterministic simulator.
#[derive(Debug, Clone)]
pub struct Twin {
    /// Current world parameters (may be shifted between decisions).
    pub world: WorldParams,
    rng: SplitMix64,
    per_decision: usize,
    next_id: u32,
}

/// Default number of candidates offered per decision.
pub const DEFAULT_CANDIDATES: usize = 4;

impl Twin {
    /// A twin over `world`, seeded with `seed`, offering `per_decision`
    /// candidates per decision (at least one).
    #[must_use]
    pub fn new(world: WorldParams, seed: u64, per_decision: usize) -> Self {
        Self {
            world,
            rng: SplitMix64::new(seed),
            per_decision: per_decision.max(1),
            next_id: 0,
        }
    }

    /// Draw one random candidate's features.
    pub fn draw_features(&mut self) -> Features {
        let rtt = self.rng.below(u64::from(RTT_LEVELS));
        let loss = self.rng.weighted(&LOSS_FREQ);
        let delivery = self.rng.weighted(&DELIVERY_FREQ);
        let age = self.rng.below(u64::from(AGE_LEVELS));
        let balance = self.rng.below(u64::from(BALANCE_LEVELS));
        // Every value is drawn below its level count, so construction cannot
        // fail; fall back to the origin rather than panic.
        Features::new(
            u8::try_from(rtt).unwrap_or(0),
            u8::try_from(loss).unwrap_or(0),
            u8::try_from(delivery).unwrap_or(0),
            u8::try_from(age).unwrap_or(0),
            u8::try_from(balance).unwrap_or(0),
        )
        .unwrap_or(Features::zero())
    }

    /// Generate the next decision.
    pub fn next_decision(&mut self) -> Decision {
        let mut candidates = Vec::with_capacity(self.per_decision);
        let mut noise = Vec::with_capacity(self.per_decision);
        for _ in 0..self.per_decision {
            let features = self.draw_features();
            candidates.push(Candidate {
                id: self.next_id,
                features,
            });
            self.next_id = self.next_id.wrapping_add(1);
            noise.push((self.world.noise_sigma * self.rng.normal()).exp());
        }
        Decision { candidates, noise }
    }

    /// Uniform random index in `0..n` from the twin's stream (for exploration).
    pub fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.rng.below(n as u64)).unwrap_or(0)
    }

    /// Uniform in `[0, 1)` from the twin's stream.
    pub fn uniform(&mut self) -> f64 {
        self.rng.next_f64()
    }

    /// Realized cost of picking candidate `index` in `decision`.
    #[must_use]
    pub fn realized_cost(&self, decision: &Decision, index: usize) -> f64 {
        match (decision.candidates.get(index), decision.noise.get(index)) {
            (Some(c), Some(n)) => self.world.expected_cost(c.features) * n,
            _ => f64::INFINITY,
        }
    }

    /// Expected (noise-free) cost of picking candidate `index`.
    #[must_use]
    pub fn expected_cost(&self, decision: &Decision, index: usize) -> f64 {
        decision
            .candidates
            .get(index)
            .map_or(f64::INFINITY, |c| self.world.expected_cost(c.features))
    }

    /// The oracle: the candidate with the lowest expected cost (ties to the
    /// lower id). `None` for an empty decision.
    #[must_use]
    pub fn oracle_pick(&self, decision: &Decision) -> Option<usize> {
        let mut best: Option<(f64, u32, usize)> = None;
        for (i, c) in decision.candidates.iter().enumerate() {
            let cost = self.world.expected_cost(c.features);
            let better = match best {
                None => true,
                Some((bc, bid, _)) => cost < bc || (cost == bc && c.id < bid),
            };
            if better {
                best = Some((cost, c.id, i));
            }
        }
        best.map(|b| b.2)
    }
}

/// Mean expected cost over `n` decisions of a picker (a function from a
/// decision to a chosen index). Uses noise-free costs to reduce variance.
pub fn mean_expected_cost(
    world: WorldParams,
    seed: u64,
    n: usize,
    mut picker: impl FnMut(&Twin, &Decision) -> Option<usize>,
) -> f64 {
    let mut twin = Twin::new(world, seed, DEFAULT_CANDIDATES);
    let mut total = 0.0;
    for _ in 0..n {
        let d = twin.next_decision();
        let pick = picker(&twin, &d).unwrap_or(0);
        total += twin.expected_cost(&d, pick);
    }
    if n == 0 {
        0.0
    } else {
        total / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_decisions() {
        let mut a = Twin::new(WorldParams::world_one(), 3, 4);
        let mut b = Twin::new(WorldParams::world_one(), 3, 4);
        for _ in 0..50 {
            assert_eq!(a.next_decision(), b.next_decision());
        }
    }

    #[test]
    fn world_two_has_interaction() {
        let w = WorldParams::world_two();
        let f = |r, l| w.expected_cost(Features::new(r, l, 7, 0, 0).unwrap());
        let fast_ratio = f(0, 7) / f(0, 0);
        let slow_ratio = f(15, 7) / f(15, 0);
        assert!(fast_ratio > slow_ratio * 1.5);
        let w1 = WorldParams::world_one();
        let g = |r, l| w1.expected_cost(Features::new(r, l, 7, 0, 0).unwrap());
        assert!(((g(0, 7) / g(0, 0)) - (g(15, 7) / g(15, 0))).abs() < 1e-9);
    }
}
