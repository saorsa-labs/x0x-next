//! Monotonic and UTC time (plan §3, "clock and seed discipline").
//!
//! Every record carries monotonic nanoseconds since agent start and UTC wall
//! time in milliseconds. UTC dates are computed here (civil-from-days), so
//! the crate needs no date library.

use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The agent's clock: monotonic since start, plus the wall clock.
#[derive(Debug, Clone, Copy)]
pub struct Clock {
    start: Instant,
}

impl Clock {
    /// A clock starting now.
    #[must_use]
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Monotonic nanoseconds since start (saturating at `u64::MAX`).
    #[must_use]
    pub fn mono_ns(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Monotonic microseconds since start.
    #[must_use]
    pub fn mono_us(&self) -> u64 {
        self.mono_ns() / 1_000
    }

    /// UTC milliseconds since the Unix epoch (0 if the clock is before it).
    #[must_use]
    pub fn utc_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }
}

/// (year, month, day) of a count of days since 1970-01-01 (Howard
/// Hinnant's civil-from-days).
#[must_use]
pub fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (
        y,
        u32::try_from(m).unwrap_or(1),
        u32::try_from(d).unwrap_or(1),
    )
}

/// UTC day `YYYY-MM-DD` of a millisecond timestamp.
#[must_use]
pub fn utc_day(utc_ms: u64) -> String {
    let days = i64::try_from(utc_ms / 86_400_000).unwrap_or(0);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// RFC 3339 UTC timestamp with milliseconds.
#[must_use]
pub fn rfc3339(utc_ms: u64) -> String {
    let secs_of_day = (utc_ms / 1_000) % 86_400;
    format!(
        "{}T{:02}:{:02}:{:02}.{:03}Z",
        utc_day(utc_ms),
        secs_of_day / 3_600,
        (secs_of_day / 60) % 60,
        secs_of_day % 60,
        utc_ms % 1_000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_dates() {
        assert_eq!(utc_day(0), "1970-01-01");
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(rfc3339(1_763_037_296_789), "2025-11-13T12:34:56.789Z");
        assert_eq!(utc_day(951_782_400_000), "2000-02-29");
        assert_eq!(utc_day(1_798_675_200_000), "2026-12-31");
        assert_eq!(utc_day(1_798_761_600_000), "2027-01-01");
    }
}
