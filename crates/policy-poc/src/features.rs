//! Kernel-fixed features of a candidate peer (design study §5, "The policy
//! plane"; poc-spec §1).
//!
//! The kernel fixes which facts a policy may see and their resolution. Every
//! feature is a small level measured by the node itself; no peer ever reports
//! its own features. The whole input space is 16 × 8 × 8 × 8 × 8 = 65,536
//! points, small enough that every check in this crate enumerates it in full.

use crate::error::PolicyError;

/// Number of rtt levels (`0..16`, higher = slower round trip).
pub const RTT_LEVELS: u8 = 16;
/// Number of loss levels (`0..8`, higher = more loss).
pub const LOSS_LEVELS: u8 = 8;
/// Number of delivery levels (`0..8`, higher = better delivery rate).
pub const DELIVERY_LEVELS: u8 = 8;
/// Number of age levels (`0..8`, higher = relationship known longer).
pub const AGE_LEVELS: u8 = 8;
/// Number of balance levels (`0..8`, higher = better reciprocity balance).
pub const BALANCE_LEVELS: u8 = 8;

/// Size of the whole input space: 16 × 8 × 8 × 8 × 8.
pub const INPUT_SPACE: usize = RTT_LEVELS as usize
    * LOSS_LEVELS as usize
    * DELIVERY_LEVELS as usize
    * AGE_LEVELS as usize
    * BALANCE_LEVELS as usize;

/// One of the five kernel-fixed features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Feature {
    /// Round-trip time level. Primary feature.
    Rtt,
    /// Packet loss level. Primary feature.
    Loss,
    /// Delivery rate level. Primary feature.
    Delivery,
    /// Relationship age level. Tie feature.
    Age,
    /// Reciprocity balance level. Tie feature.
    Balance,
}

impl Feature {
    /// All five features in canonical order.
    pub const ALL: [Feature; 5] = [
        Feature::Rtt,
        Feature::Loss,
        Feature::Delivery,
        Feature::Age,
        Feature::Balance,
    ];

    /// Number of levels this feature takes.
    #[must_use]
    pub const fn levels(self) -> u8 {
        match self {
            Feature::Rtt => RTT_LEVELS,
            Feature::Loss => LOSS_LEVELS,
            Feature::Delivery => DELIVERY_LEVELS,
            Feature::Age => AGE_LEVELS,
            Feature::Balance => BALANCE_LEVELS,
        }
    }

    /// True for the primary features (rtt, loss, delivery), whose
    /// contributions are multiples of [`crate::QUANT`].
    #[must_use]
    pub const fn is_primary(self) -> bool {
        matches!(self, Feature::Rtt | Feature::Loss | Feature::Delivery)
    }

    /// Lower-case name used in reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Feature::Rtt => "rtt",
            Feature::Loss => "loss",
            Feature::Delivery => "delivery",
            Feature::Age => "age",
            Feature::Balance => "balance",
        }
    }
}

impl std::fmt::Display for Feature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A measured feature vector. Construction validates every level, so a value
/// of this type is always inside the 65,536-point input space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Features {
    rtt: u8,
    loss: u8,
    delivery: u8,
    age: u8,
    balance: u8,
}

impl Features {
    /// Build a feature vector, rejecting any level outside its range.
    ///
    /// # Errors
    /// [`PolicyError::FeatureOutOfRange`] when a level is too large.
    pub fn new(rtt: u8, loss: u8, delivery: u8, age: u8, balance: u8) -> Result<Self, PolicyError> {
        for (feature, value) in [
            (Feature::Rtt, rtt),
            (Feature::Loss, loss),
            (Feature::Delivery, delivery),
            (Feature::Age, age),
            (Feature::Balance, balance),
        ] {
            if value >= feature.levels() {
                return Err(PolicyError::FeatureOutOfRange {
                    feature: feature.name(),
                    value,
                    levels: feature.levels(),
                });
            }
        }
        Ok(Self {
            rtt,
            loss,
            delivery,
            age,
            balance,
        })
    }

    /// The all-zero feature vector (every level at its lowest).
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            rtt: 0,
            loss: 0,
            delivery: 0,
            age: 0,
            balance: 0,
        }
    }

    /// Round-trip time level, `0..16`.
    #[must_use]
    pub const fn rtt(self) -> u8 {
        self.rtt
    }

    /// Loss level, `0..8`.
    #[must_use]
    pub const fn loss(self) -> u8 {
        self.loss
    }

    /// Delivery level, `0..8`.
    #[must_use]
    pub const fn delivery(self) -> u8 {
        self.delivery
    }

    /// Age level, `0..8`.
    #[must_use]
    pub const fn age(self) -> u8 {
        self.age
    }

    /// Balance level, `0..8`.
    #[must_use]
    pub const fn balance(self) -> u8 {
        self.balance
    }

    /// The level of one feature.
    #[must_use]
    pub const fn get(self, feature: Feature) -> u8 {
        match feature {
            Feature::Rtt => self.rtt,
            Feature::Loss => self.loss,
            Feature::Delivery => self.delivery,
            Feature::Age => self.age,
            Feature::Balance => self.balance,
        }
    }

    /// A copy with one feature set to `level`, or `None` if out of range.
    #[must_use]
    pub fn with(self, feature: Feature, level: u8) -> Option<Self> {
        if level >= feature.levels() {
            return None;
        }
        let mut out = self;
        match feature {
            Feature::Rtt => out.rtt = level,
            Feature::Loss => out.loss = level,
            Feature::Delivery => out.delivery = level,
            Feature::Age => out.age = level,
            Feature::Balance => out.balance = level,
        }
        Some(out)
    }

    /// Dense index in `0..INPUT_SPACE` (balance varies fastest).
    #[must_use]
    pub fn index(self) -> usize {
        let mut idx = 0usize;
        for feature in Feature::ALL {
            idx = idx * feature.levels() as usize + self.get(feature) as usize;
        }
        idx
    }

    /// Inverse of [`Features::index`]; `None` when `index >= INPUT_SPACE`.
    #[must_use]
    pub fn from_index(index: usize) -> Option<Self> {
        if index >= INPUT_SPACE {
            return None;
        }
        let mut rest = index;
        let mut levels = [0u8; 5];
        for (slot, feature) in Feature::ALL.iter().enumerate().rev() {
            let n = feature.levels() as usize;
            // `rest % n < n <= 16`, so the narrowing is lossless.
            levels[slot] = u8::try_from(rest % n).ok()?;
            rest /= n;
        }
        Some(Self {
            rtt: levels[0],
            loss: levels[1],
            delivery: levels[2],
            age: levels[3],
            balance: levels[4],
        })
    }

    /// Every point of the input space, in index order (65,536 values).
    pub fn enumerate_all() -> impl Iterator<Item = Features> {
        (0..INPUT_SPACE).filter_map(Features::from_index)
    }
}

impl std::fmt::Display for Features {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(rtt {}, loss {}, delivery {}, age {}, balance {})",
            self.rtt, self.loss, self.delivery, self.age, self.balance
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_all_covers_space_once() {
        let all: Vec<Features> = Features::enumerate_all().collect();
        assert_eq!(all.len(), 65_536);
        for (i, f) in all.iter().enumerate() {
            assert_eq!(f.index(), i);
        }
    }

    #[test]
    fn out_of_range_rejected() {
        assert!(Features::new(16, 0, 0, 0, 0).is_err());
        assert!(Features::new(0, 8, 0, 0, 0).is_err());
        assert!(Features::new(15, 7, 7, 7, 7).is_ok());
        assert!(Features::from_index(INPUT_SPACE).is_none());
    }
}
