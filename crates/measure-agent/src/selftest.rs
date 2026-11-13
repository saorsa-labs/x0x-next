//! `--self-test`: the prober and five reflectors on 127.0.0.1, a few hundred
//! decisions, then a full verification of the traces just written.
//!
//! The reflectors get different injected delays and drop schedules
//! ([`Faults`]) so the candidates differ in rtt and loss, and two pairs are
//! alike so the band holds more than one candidate and exploration has
//! somewhere to go. The self-test checks that every trace file replays and
//! matches its manifest, that the replayed decision count equals the run's,
//! and that no peer label or address appears in any trace byte.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::agent::{run_agent, RunOptions, RunSummary};
use crate::clock::Clock;
use crate::config::{Config, PeerConfig};
use crate::error::MeasureError;
use crate::reflector::{Faults, Reflector};
use crate::trace::{
    read_trace, trace_files, verify_trace, ManifestStatus, TraceLine, VerifyReport,
};

/// Default number of self-test decisions.
pub const DEFAULT_DECISIONS: u64 = 300;

/// The self-test's reflectors: (label, injected delay in ms, drop every n).
const PEERS: [(&str, u64, Option<u64>); 5] = [
    ("selftest-label-fast-a", 0, None),
    ("selftest-label-fast-b", 0, None),
    ("selftest-label-mid-c", 3, None),
    ("selftest-label-mid-d", 3, None),
    ("selftest-label-slow-lossy-e", 10, Some(5)),
];

/// What the self-test found.
#[derive(Debug, Clone)]
pub struct SelfTestReport {
    /// Run totals.
    pub run: RunSummary,
    /// Verification of each trace file.
    pub files: Vec<VerifyReport>,
    /// Per peer slot: (label, picks, copies received, echoes sent).
    pub peers: Vec<(String, u64, u64, u64)>,
}

impl fmt::Display for SelfTestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = &self.run;
        writeln!(f, "measure-agent self-test: PASS")?;
        writeln!(f, "  data dir      {}", r.data_dir.display())?;
        writeln!(
            f,
            "  decisions     {} (from #{})",
            r.decisions, r.first_decision
        )?;
        writeln!(
            f,
            "  explored      {}  contested {}  copies {}",
            r.explored, r.contested, r.copies
        )?;
        let mean = r.pick_rtt_sum_us.checked_div(r.pick_completed).unwrap_or(0);
        writeln!(
            f,
            "  pick outcomes {} completed (mean rtt {} us), {} timed out",
            r.pick_completed, mean, r.pick_timeouts
        )?;
        for (label, picks, copies, echoes) in &self.peers {
            writeln!(
                f,
                "  peer {label:<28} picks {picks:>4}  copies {copies:>3}  echoes {echoes:>5}"
            )?;
        }
        for v in &self.files {
            writeln!(
                f,
                "  verified {}: {} segment(s), {} decisions replayed, manifest {:?}",
                v.file, v.replay.segments, v.replay.decisions, v.manifest
            )?;
        }
        Ok(())
    }
}

fn fail(msg: impl Into<String>) -> MeasureError {
    MeasureError::SelfTest(msg.into())
}

/// Run the self-test with `decisions` decisions, writing into `data_dir`
/// (a fresh directory under the system temp dir when `None`).
///
/// # Errors
/// [`MeasureError::SelfTest`] when a check fails, or any error of the run.
pub fn self_test(
    decisions: u64,
    data_dir: Option<PathBuf>,
) -> Result<SelfTestReport, MeasureError> {
    let dir = data_dir.unwrap_or_else(|| {
        std::env::temp_dir().join(format!(
            "measure-agent-self-test-{}-{}",
            std::process::id(),
            Clock::utc_ms()
        ))
    });
    let mut reflectors = Vec::with_capacity(PEERS.len());
    let mut peers = Vec::with_capacity(PEERS.len());
    for (label, delay_ms, drop_every) in PEERS {
        let r = Reflector::spawn(
            "127.0.0.1:0"
                .parse()
                .map_err(|_| fail("loopback address"))?,
            Faults {
                delay: Duration::from_millis(delay_ms),
                drop_every,
            },
        )?;
        peers.push(PeerConfig {
            label: label.to_string(),
            addr: r.local_addr(),
        });
        reflectors.push(r);
    }
    let cfg = Config {
        node: "self-test".to_string(),
        data_dir: dir.clone(),
        listen_port: 0,
        listen_ip: "127.0.0.1".parse().map_err(|_| fail("loopback address"))?,
        tick_ms: 0,
        experiment_seed: crate::config::DEFAULT_EXPERIMENT_SEED,
        node_class: Some("self-test".to_string()),
        request_bytes: 1024,
        probe_timeout_ms: 40,
        request_timeout_ms: 60,
        peers,
    };
    let run = run_agent(
        &cfg,
        RunOptions {
            max_decisions: Some(decisions),
            reflect: false,
        },
    )?;
    let echoes: Vec<u64> = reflectors.iter().map(Reflector::echoed).collect();
    for r in reflectors {
        r.stop();
    }

    check(&cfg, &dir, run, &echoes)
}

fn check(
    cfg: &Config,
    dir: &Path,
    run: RunSummary,
    echoes: &[u64],
) -> Result<SelfTestReport, MeasureError> {
    let mut files = Vec::new();
    let mut picks = vec![0u64; cfg.peers.len()];
    let mut copies = vec![0u64; cfg.peers.len()];
    let mut replayed = 0usize;
    for path in trace_files(dir)? {
        let v = verify_trace(&path)?;
        if v.manifest != ManifestStatus::Matched {
            return Err(fail(format!("{} has no manifest after close", v.file)));
        }
        replayed += v.replay.decisions;
        let bytes = std::fs::read(&path)
            .map_err(|e| MeasureError::io(format!("read {}", path.display()), e))?;
        let text = String::from_utf8_lossy(&bytes);
        for p in &cfg.peers {
            if text.contains(&p.label) || text.contains(&p.addr.to_string()) {
                return Err(fail(format!("{} leaks a peer label or address", v.file)));
            }
        }
        if text.contains("127.0.0.1") {
            return Err(fail(format!("{} leaks an address", v.file)));
        }
        for line in read_trace(&path)? {
            if let TraceLine::Decision(d) = line {
                if let Some(c) = d.candidates.get(d.pick) {
                    picks[c.slot] += 1;
                }
                if let Some(c) = d.copy.and_then(|i| d.candidates.get(i)) {
                    copies[c.slot] += 1;
                }
            }
        }
        files.push(v);
    }
    if files.is_empty() {
        return Err(fail("no trace files written"));
    }
    // Decisions are numbered from 0 without gaps, so a directory holding
    // only this node's traces replays exactly first_decision + decisions.
    if replayed as u64 != run.first_decision + run.decisions {
        return Err(fail(format!(
            "replayed {replayed} decisions, run made {}",
            run.decisions
        )));
    }
    // With 100+ decisions, no exploration or no copy at 1-in-10 each would
    // mean the draws are broken (probability below 1e-4).
    if run.decisions >= 100 && (run.explored == 0 || run.copies == 0) {
        return Err(fail("no exploration or no copies in the run"));
    }
    let peers = cfg
        .peers
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (
                p.label.clone(),
                picks[i],
                copies[i],
                echoes.get(i).copied().unwrap_or(0),
            )
        })
        .collect();
    Ok(SelfTestReport { run, files, peers })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn self_test_end_to_end_over_loopback() {
        let dir = std::env::temp_dir().join(format!(
            "measure-agent-selftest-unit-{}-{}",
            std::process::id(),
            Clock::utc_ms()
        ));
        let report = self_test(120, Some(dir.clone())).unwrap();
        assert_eq!(report.run.decisions, 120);
        assert!(report
            .files
            .iter()
            .all(|f| f.manifest == ManifestStatus::Matched));
        // The two fast, lossless peers are picked far more than the slow one.
        let fast = report.peers[0].1 + report.peers[1].1;
        assert!(fast > report.peers[4].1, "{report}");

        // A restart continues the numbering and the stream; the whole
        // directory still verifies.
        let again = self_test(30, Some(dir.clone())).unwrap();
        assert_eq!(again.run.first_decision, 120);
        let total: usize = again.files.iter().map(|f| f.replay.decisions).sum();
        assert_eq!(total, 150);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
