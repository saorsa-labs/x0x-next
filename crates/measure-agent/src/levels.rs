//! Fixed level boundaries of the five features (plan §1, "level boundaries
//! fixed in the trace schema").
//!
//! Each feature is cut by an ascending list of edges: the level is the number
//! of edges the raw value is at or above. So a list of `n − 1` edges yields
//! levels `0..n`. Every header repeats these lists ([`Boundaries`]); changing
//! any of them changes the schema version.

use policy_poc::features::{AGE_LEVELS, BALANCE_LEVELS, DELIVERY_LEVELS, LOSS_LEVELS, RTT_LEVELS};
use policy_poc::Features;
use serde::{Deserialize, Serialize};

/// Smoothed rtt edges in microseconds: log-spaced from 1 ms by a factor of
/// 1.6 (rounded), 15 edges for 16 levels. Level 15 is 720 ms and slower; a
/// peer with no rtt sample yet is also level 15.
pub const RTT_EDGES_US: [u64; 15] = [
    1_000, 1_600, 2_560, 4_096, 6_554, 10_486, 16_777, 26_844, 42_950, 68_719, 109_951, 175_922,
    281_475, 450_360, 720_576,
];

/// Probe loss-rate edges in parts per million, 7 edges for 8 levels.
pub const LOSS_EDGES_PPM: [u64; 7] = [5_000, 20_000, 50_000, 100_000, 150_000, 250_000, 400_000];

/// Request delivery-rate edges in parts per million, 7 edges for 8 levels
/// (higher = better delivery).
pub const DELIVERY_EDGES_PPM: [u64; 7] = [
    300_000, 500_000, 700_000, 800_000, 900_000, 950_000, 990_000,
];

/// Age edges in seconds since first successful contact, log-spaced:
/// 1 min, 10 min, 1 h, 6 h, 1 day, 3 days, 7 days.
pub const AGE_EDGES_S: [u64; 7] = [60, 600, 3_600, 21_600, 86_400, 259_200, 604_800];

/// Balance level used for every candidate in v0.1 (not measured).
pub const BALANCE_CONSTANT_LEVEL: u8 = 4;

/// Level of `value` under ascending `edges`: the count of edges `<= value`.
#[must_use]
pub fn cut(value: u64, edges: &[u64]) -> u8 {
    let n = edges.iter().take_while(|e| value >= **e).count();
    u8::try_from(n).unwrap_or(u8::MAX)
}

/// Raw estimator values of one candidate, before cutting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawFeatures {
    /// Smoothed rtt in microseconds; `None` before the first sample.
    pub srtt_us: Option<u64>,
    /// Probe loss rate over the sliding window, parts per million.
    pub loss_ppm: u64,
    /// Request delivery rate (with prior), parts per million.
    pub delivery_ppm: u64,
    /// Seconds since first successful contact; `None` if never contacted.
    pub age_s: Option<u64>,
}

impl RawFeatures {
    /// Cut to the five feature levels with the fixed boundaries.
    #[must_use]
    pub fn levels(&self) -> Features {
        let rtt = match self.srtt_us {
            Some(us) => cut(us, &RTT_EDGES_US),
            None => RTT_LEVELS - 1,
        };
        let loss = cut(self.loss_ppm, &LOSS_EDGES_PPM);
        let delivery = cut(self.delivery_ppm, &DELIVERY_EDGES_PPM);
        let age = self.age_s.map_or(0, |s| cut(s, &AGE_EDGES_S));
        // Every cut is bounded by its edge count, so this cannot fail; fall
        // back to the origin rather than panic.
        Features::new(rtt, loss, delivery, age, BALANCE_CONSTANT_LEVEL).unwrap_or(Features::zero())
    }
}

/// The boundaries as written into every trace header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Boundaries {
    /// rtt edges (µs); 16 levels.
    pub rtt_us: Vec<u64>,
    /// Loss edges (ppm); 8 levels.
    pub loss_ppm: Vec<u64>,
    /// Delivery edges (ppm); 8 levels.
    pub delivery_ppm: Vec<u64>,
    /// Age edges (s); 8 levels.
    pub age_s: Vec<u64>,
    /// Balance: not measured in v0.1.
    pub balance: BalanceSchema,
}

/// How balance is produced in this schema version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BalanceSchema {
    /// Always `false` in v0.1.
    pub measured: bool,
    /// The constant level every candidate gets.
    pub constant_level: u8,
    /// Number of balance levels.
    pub levels: u8,
}

impl Boundaries {
    /// The fixed boundaries of this schema version.
    #[must_use]
    pub fn current() -> Self {
        Self {
            rtt_us: RTT_EDGES_US.to_vec(),
            loss_ppm: LOSS_EDGES_PPM.to_vec(),
            delivery_ppm: DELIVERY_EDGES_PPM.to_vec(),
            age_s: AGE_EDGES_S.to_vec(),
            balance: BalanceSchema {
                measured: false,
                constant_level: BALANCE_CONSTANT_LEVEL,
                levels: BALANCE_LEVELS,
            },
        }
    }

    /// True when the edge counts match the kernel-fixed level counts and
    /// every list is strictly ascending.
    #[must_use]
    pub fn is_well_formed(&self) -> bool {
        let asc = |v: &[u64]| v.windows(2).all(|w| w[0] < w[1]);
        self.rtt_us.len() + 1 == usize::from(RTT_LEVELS)
            && self.loss_ppm.len() + 1 == usize::from(LOSS_LEVELS)
            && self.delivery_ppm.len() + 1 == usize::from(DELIVERY_LEVELS)
            && self.age_s.len() + 1 == usize::from(AGE_LEVELS)
            && self.balance.constant_level < BALANCE_LEVELS
            && asc(&self.rtt_us)
            && asc(&self.loss_ppm)
            && asc(&self.delivery_ppm)
            && asc(&self.age_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_boundaries_are_well_formed() {
        assert!(Boundaries::current().is_well_formed());
    }

    #[test]
    fn rtt_cuts_at_edges() {
        assert_eq!(cut(0, &RTT_EDGES_US), 0);
        assert_eq!(cut(999, &RTT_EDGES_US), 0);
        assert_eq!(cut(1_000, &RTT_EDGES_US), 1);
        assert_eq!(cut(1_599, &RTT_EDGES_US), 1);
        assert_eq!(cut(1_600, &RTT_EDGES_US), 2);
        assert_eq!(cut(720_575, &RTT_EDGES_US), 14);
        assert_eq!(cut(720_576, &RTT_EDGES_US), 15);
        assert_eq!(cut(u64::MAX, &RTT_EDGES_US), 15);
        // Every edge starts its own level.
        for (i, e) in RTT_EDGES_US.iter().enumerate() {
            assert_eq!(usize::from(cut(*e, &RTT_EDGES_US)), i + 1);
            assert_eq!(usize::from(cut(*e - 1, &RTT_EDGES_US)), i);
        }
    }

    #[test]
    fn eight_level_cuts_at_edges() {
        for edges in [LOSS_EDGES_PPM, DELIVERY_EDGES_PPM, AGE_EDGES_S] {
            assert_eq!(cut(0, &edges), 0);
            for (i, e) in edges.iter().enumerate() {
                assert_eq!(usize::from(cut(*e, &edges)), i + 1);
                assert_eq!(usize::from(cut(*e - 1, &edges)), i);
            }
            assert_eq!(cut(u64::MAX, &edges), 7);
        }
        assert_eq!(cut(1_000_000, &LOSS_EDGES_PPM), 7);
        assert_eq!(cut(1_000_000, &DELIVERY_EDGES_PPM), 7);
    }

    #[test]
    fn raw_to_levels() {
        let unknown = RawFeatures {
            srtt_us: None,
            loss_ppm: 0,
            delivery_ppm: 900_000,
            age_s: None,
        };
        let f = unknown.levels();
        assert_eq!(
            (f.rtt(), f.loss(), f.delivery(), f.age(), f.balance()),
            (15, 0, 5, 0, 4)
        );
        let known = RawFeatures {
            srtt_us: Some(3_000),
            loss_ppm: 250_000,
            delivery_ppm: 1_000_000,
            age_s: Some(3_600),
        };
        let f = known.levels();
        assert_eq!(
            (f.rtt(), f.loss(), f.delivery(), f.age(), f.balance()),
            (3, 6, 7, 3, 4)
        );
    }
}
