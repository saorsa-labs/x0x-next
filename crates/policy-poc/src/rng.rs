//! A small seeded random source for the twin (design study §5, "the twin";
//! poc-spec §7).
//!
//! SplitMix64: tiny, well-mixed, and fully specified here, so every run of the
//! POC is reproducible from its seed with no external crate. Not for
//! cryptography.

/// Seeded SplitMix64 generator.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A generator seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        // 2^-53; the shifted value has at most 53 bits, so it is exact in f64.
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform integer in `0..n` (returns 0 when `n == 0`).
    pub fn below(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        // Lemire-style multiply-shift; bias is below 2^-58 for the tiny `n`
        // used here, irrelevant for a simulator.
        ((u128::from(self.next_u64()) * u128::from(n)) >> 64) as u64
    }

    /// Draw an index with probability proportional to `weights`.
    pub fn weighted(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().sum();
        let mut x = self.next_f64() * total;
        for (i, w) in weights.iter().enumerate() {
            if x < *w {
                return i;
            }
            x -= w;
        }
        weights.len().saturating_sub(1)
    }

    /// Approximately standard normal (Irwin–Hall: sum of 12 uniforms − 6).
    /// Pure arithmetic, so identical on every platform.
    pub fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.next_f64()).sum::<f64>() - 6.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_from_seed() {
        let mut a = SplitMix64::new(7);
        let mut b = SplitMix64::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
        assert!(SplitMix64::new(1).below(10) < 10);
    }
}
