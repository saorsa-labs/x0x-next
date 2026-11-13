//! Traces: append-only daily JSONL, manifests, verification and replay
//! (plan §1 "trace record", §3 "versioned schema", "clock and seed
//! discipline", "restarts").
//!
//! * One file per UTC day, `trace-YYYY-MM-DD.jsonl`. Every agent start, and
//!   every new day, begins a segment with a [`Header`] line; each decision is
//!   one [`DecisionRecord`] line. Lines are only ever appended.
//! * When a day's file is closed (day rollover, clean shutdown, or found
//!   unclosed at the next start) its SHA-256 is written to
//!   `trace-YYYY-MM-DD.manifest.json` ([`Manifest`]). Reopening a day's file
//!   removes its manifest, so a manifest exists exactly when the file is
//!   closed.
//! * [`replay`] re-derives every decision from its segment header's seed:
//!   the sampled peer slots, the levels from the raw values, the rule's
//!   choice through `band_pick`, exploration, the pick, its propensity and
//!   the copy. Any difference is an error. [`verify_trace`] adds the format
//!   and manifest checks.
//!
//! Peer labels appear only as salted hashes and addresses never appear.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use policy_poc::{Candidate, Features, FixedRule, Scorer};
use serde::{Deserialize, Serialize};

use crate::clock::{rfc3339, utc_day, Clock};
use crate::decide::{
    choose, decision_rng, sample_slots, CANDIDATES_PER_DECISION, COPY_ONE_IN, DRAWS_PER_DECISION,
    EXPLORE_ONE_IN, MARGIN,
};
use crate::error::MeasureError;
use crate::levels::{Boundaries, RawFeatures};
use crate::privacy::sha256_hex;
use crate::SCHEMA_ID;

/// Tolerance when comparing a logged propensity with the replayed one
/// (JSON float parsing is not guaranteed to round-trip every last bit).
const PROPENSITY_TOLERANCE: f64 = 1e-9;

/// How the estimators work, written into every header.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EstimatorSchema {
    /// srtt smoothing gain.
    pub srtt_gain: String,
    /// Probes in the loss window.
    pub loss_window: usize,
    /// Prior successes in the delivery estimate.
    pub delivery_prior_ok: u64,
    /// Prior requests in the delivery estimate.
    pub delivery_prior_n: u64,
}

impl EstimatorSchema {
    /// The estimators of this version.
    #[must_use]
    pub fn current() -> Self {
        Self {
            srtt_gain: "1/8".to_string(),
            loss_window: crate::estimator::LOSS_WINDOW,
            delivery_prior_ok: crate::estimator::DELIVERY_PRIOR_OK,
            delivery_prior_n: crate::estimator::DELIVERY_PRIOR_N,
        }
    }
}

/// First line of every trace segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Header {
    /// Schema id, [`SCHEMA_ID`].
    pub schema: String,
    /// Agent crate version.
    pub agent_version: String,
    /// Git commit the agent was built from, if known at build time.
    pub git_commit: Option<String>,
    /// Probe transport id ([`crate::TRANSPORT_ID`]).
    pub transport: String,
    /// Salted SHA-256 of the node name.
    pub node_hash: String,
    /// Node class from the configuration.
    pub node_class: Option<String>,
    /// The node's salt, hex.
    pub salt: String,
    /// SplitMix64 stream seed.
    pub seed: u64,
    /// Stream positions reserved per decision.
    pub draws_per_decision: u64,
    /// Candidates drawn per decision (at most).
    pub candidates_per_decision: usize,
    /// Band margin (score units).
    pub margin: i64,
    /// One decision in this many explores.
    pub explore_one_in: u64,
    /// One decision in this many sends a copy.
    pub copy_one_in: u64,
    /// Number of configured peers (the slots candidates are drawn from).
    pub peer_count: usize,
    /// Decision request size in bytes.
    pub request_bytes: usize,
    /// Milliseconds between decisions.
    pub tick_ms: u64,
    /// Probe timeout.
    pub probe_timeout_ms: u64,
    /// Decision request timeout.
    pub request_timeout_ms: u64,
    /// Number of the first decision in this segment.
    pub first_decision: u64,
    /// Monotonic nanoseconds since agent start.
    pub mono_ns: u64,
    /// UTC milliseconds since the Unix epoch.
    pub utc_ms: u64,
    /// The same instant, RFC 3339.
    pub utc: String,
    /// NTP offset estimate; always `None` in v0.1.
    pub ntp_offset_ms: Option<f64>,
    /// Level boundaries of all five features.
    pub boundaries: Boundaries,
    /// How the estimators work.
    pub estimators: EstimatorSchema,
}

/// One candidate in a decision record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRecord {
    /// Salted hash of the peer label.
    pub peer: String,
    /// Index of the peer in the configured list (the candidate id).
    pub slot: usize,
    /// Levels `[rtt, loss, delivery, age, balance]`.
    pub levels: [u8; 5],
    /// The fixed rule's score of these levels.
    pub rule_score: i64,
    /// Raw estimator values the levels were cut from.
    pub raw: RawFeatures,
}

/// Whether an outcome is of the pick or of its copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The candidate actually used.
    Pick,
    /// The copy of a duplicated decision.
    Copy,
}

/// A realized outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeRecord {
    /// Candidate index.
    pub candidate: usize,
    /// Pick or copy.
    pub role: Role,
    /// Round trip in microseconds; `None` on timeout.
    pub rtt_us: Option<u64>,
    /// True when no echo arrived within the request timeout.
    pub timed_out: bool,
}

/// One decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Decision number, increasing across restarts.
    pub decision: u64,
    /// Monotonic nanoseconds since agent start.
    pub mono_ns: u64,
    /// UTC milliseconds since the Unix epoch.
    pub utc_ms: u64,
    /// NTP offset estimate; always `None` in v0.1.
    pub ntp_offset_ms: Option<f64>,
    /// Request size in bytes.
    pub request_bytes: usize,
    /// The candidate set with features.
    pub candidates: Vec<CandidateRecord>,
    /// The fixed rule's choice.
    pub rule_choice: usize,
    /// The candidate actually used.
    pub pick: usize,
    /// Whether the exploration draw fired.
    pub explored: bool,
    /// Whether a copy was sent.
    pub duplicated: bool,
    /// Probability of this pick under the Measure procedure.
    pub propensity: f64,
    /// Candidate that got the copy.
    pub copy: Option<usize>,
    /// Realized outcomes: the pick first, then the copy if any.
    pub outcomes: Vec<OutcomeRecord>,
}

/// One trace line.
// A header is large but appears once per segment among thousands of
// decisions; boxing it would only complicate the schema types.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TraceLine {
    /// Segment header.
    Header(Header),
    /// Decision record.
    Decision(DecisionRecord),
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum TraceLineRef<'a> {
    Header(&'a Header),
    Decision(&'a DecisionRecord),
}

/// A closed day's manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// Schema id.
    pub schema: String,
    /// Trace file name.
    pub file: String,
    /// SHA-256 of the whole file, hex.
    pub sha256: String,
    /// File length in bytes.
    pub bytes: u64,
    /// Header lines.
    pub headers: usize,
    /// Decision lines.
    pub decisions: usize,
}

/// Trace file name for a UTC day.
#[must_use]
pub fn trace_file_name(day: &str) -> String {
    format!("trace-{day}.jsonl")
}

/// Manifest path of a trace file.
#[must_use]
pub fn manifest_path(trace: &Path) -> PathBuf {
    let stem = trace
        .file_stem()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned());
    trace.with_file_name(format!("{stem}.manifest.json"))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |s| s.to_string_lossy().into_owned())
}

fn format_err(path: &Path, line: usize, message: impl Into<String>) -> MeasureError {
    MeasureError::TraceFormat {
        file: file_name(path),
        line,
        message: message.into(),
    }
}

/// Every `trace-*.jsonl` in `dir`, sorted by name (that is, by day).
///
/// # Errors
/// [`MeasureError::Io`] if the directory cannot be listed.
pub fn trace_files(dir: &Path) -> Result<Vec<PathBuf>, MeasureError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| MeasureError::io(format!("list {}", dir.display()), e))?;
    let mut out = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| MeasureError::io(format!("list {}", dir.display()), e))?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("trace-") && name.ends_with(".jsonl") {
            out.push(entry.path());
        }
    }
    out.sort();
    Ok(out)
}

/// Parse a whole trace file.
///
/// # Errors
/// [`MeasureError::Io`] if unreadable, [`MeasureError::TraceFormat`] for a
/// line that is not a valid trace record.
pub fn read_trace(path: &Path) -> Result<Vec<TraceLine>, MeasureError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| MeasureError::io(format!("read {}", path.display()), e))?;
    text.lines()
        .enumerate()
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format_err(path, i + 1, e.to_string())))
        .collect()
}

/// Compute and write the manifest of a trace file.
///
/// # Errors
/// I/O and format errors from reading the trace or writing the manifest.
pub fn write_manifest(trace: &Path) -> Result<Manifest, MeasureError> {
    let manifest = compute_manifest(trace)?;
    let json =
        serde_json::to_string_pretty(&manifest).map_err(|e| format_err(trace, 0, e.to_string()))?;
    let path = manifest_path(trace);
    std::fs::write(&path, json + "\n")
        .map_err(|e| MeasureError::io(format!("write {}", path.display()), e))?;
    Ok(manifest)
}

fn compute_manifest(trace: &Path) -> Result<Manifest, MeasureError> {
    let bytes = std::fs::read(trace)
        .map_err(|e| MeasureError::io(format!("read {}", trace.display()), e))?;
    let lines = read_trace(trace)?;
    let headers = lines
        .iter()
        .filter(|l| matches!(l, TraceLine::Header(_)))
        .count();
    Ok(Manifest {
        schema: SCHEMA_ID.to_string(),
        file: file_name(trace),
        sha256: sha256_hex(&bytes),
        bytes: bytes.len() as u64,
        headers,
        decisions: lines.len() - headers,
    })
}

/// Whether a trace file's manifest was found and matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifestStatus {
    /// The manifest matches the file.
    Matched,
    /// No manifest: the file is still open.
    Absent,
}

/// Check a trace file against its manifest, if it has one.
///
/// # Errors
/// [`MeasureError::Manifest`] when a manifest exists and differs.
pub fn check_manifest(trace: &Path) -> Result<ManifestStatus, MeasureError> {
    let path = manifest_path(trace);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ManifestStatus::Absent),
        Err(e) => return Err(MeasureError::io(format!("read {}", path.display()), e)),
    };
    let stored: Manifest = serde_json::from_str(&text).map_err(|e| MeasureError::Manifest {
        file: file_name(trace),
        message: e.to_string(),
    })?;
    let actual = compute_manifest(trace)?;
    if stored != actual {
        return Err(MeasureError::Manifest {
            file: file_name(trace),
            message: format!(
                "manifest says sha256 {} ({} bytes), file is {} ({} bytes)",
                stored.sha256, stored.bytes, actual.sha256, actual.bytes
            ),
        });
    }
    Ok(ManifestStatus::Matched)
}

/// Write (or rewrite) manifests for every trace file of a day before
/// `today` whose manifest is missing or stale, closing it. Returns the files
/// closed.
///
/// # Errors
/// I/O and format errors.
pub fn close_past_days(dir: &Path, today: &str) -> Result<Vec<PathBuf>, MeasureError> {
    let today_file = trace_file_name(today);
    let mut closed = Vec::new();
    for f in trace_files(dir)? {
        if file_name(&f).as_str() >= today_file.as_str() {
            continue;
        }
        if check_manifest(&f).ok() != Some(ManifestStatus::Matched) {
            write_manifest(&f)?;
            closed.push(f);
        }
    }
    Ok(closed)
}

/// The highest decision number logged in `dir`, if any.
///
/// # Errors
/// I/O and format errors.
pub fn last_decision(dir: &Path) -> Result<Option<u64>, MeasureError> {
    for f in trace_files(dir)?.iter().rev() {
        let last = read_trace(f)?.iter().rev().find_map(|l| match l {
            TraceLine::Decision(d) => Some(d.decision),
            TraceLine::Header(_) => None,
        });
        if last.is_some() {
            return Ok(last);
        }
    }
    Ok(None)
}

/// Appends records to the daily files of one data directory.
#[derive(Debug)]
pub struct TraceWriter {
    dir: PathBuf,
    template: Header,
    clock: Clock,
    open: Option<(String, PathBuf, File)>,
}

impl TraceWriter {
    /// A writer for `dir`. `template` is the header written at the start of
    /// each segment; its time and first-decision fields are filled in then.
    #[must_use]
    pub fn new(dir: &Path, template: Header, clock: Clock) -> Self {
        Self {
            dir: dir.to_path_buf(),
            template,
            clock,
            open: None,
        }
    }

    /// Append one decision, opening (and heading) the record's UTC day file
    /// and closing the previous day's file if the day changed.
    ///
    /// # Errors
    /// I/O and serialization errors.
    pub fn append(&mut self, rec: &DecisionRecord) -> Result<(), MeasureError> {
        let day = utc_day(rec.utc_ms);
        if self.open.as_ref().map(|(d, _, _)| d) != Some(&day) {
            self.close()?;
            self.open_day(&day, rec.decision)?;
        }
        self.write_line(&TraceLineRef::Decision(rec))
    }

    /// Close the current day's file and write its manifest.
    ///
    /// # Errors
    /// I/O and format errors.
    pub fn close(&mut self) -> Result<(), MeasureError> {
        if let Some((_, path, file)) = self.open.take() {
            file.sync_all()
                .map_err(|e| MeasureError::io(format!("sync {}", path.display()), e))?;
            drop(file);
            write_manifest(&path)?;
        }
        Ok(())
    }

    fn open_day(&mut self, day: &str, first_decision: u64) -> Result<(), MeasureError> {
        let path = self.dir.join(trace_file_name(day));
        let manifest = manifest_path(&path);
        if manifest.exists() {
            std::fs::remove_file(&manifest)
                .map_err(|e| MeasureError::io(format!("reopen {}", path.display()), e))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| MeasureError::io(format!("open {}", path.display()), e))?;
        self.open = Some((day.to_string(), path, file));
        let utc_ms = Clock::utc_ms();
        let header = Header {
            first_decision,
            mono_ns: self.clock.mono_ns(),
            utc_ms,
            utc: rfc3339(utc_ms),
            ..self.template.clone()
        };
        self.write_line(&TraceLineRef::Header(&header))
    }

    fn write_line(&mut self, line: &TraceLineRef<'_>) -> Result<(), MeasureError> {
        let Some((_, path, file)) = self.open.as_mut() else {
            return Err(MeasureError::TraceFormat {
                file: String::new(),
                line: 0,
                message: "no open trace file".to_string(),
            });
        };
        let mut json =
            serde_json::to_string(line).map_err(|e| format_err(path, 0, e.to_string()))?;
        json.push('\n');
        file.write_all(json.as_bytes())
            .map_err(|e| MeasureError::io(format!("append {}", path.display()), e))
    }
}

/// Totals of a replayed trace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReplayReport {
    /// Header lines (segments).
    pub segments: usize,
    /// Decisions replayed.
    pub decisions: usize,
    /// Decisions whose exploration draw fired.
    pub explored: usize,
    /// Decisions where the pick differs from the rule's choice.
    pub contested: usize,
    /// Decisions with a copy.
    pub copies: usize,
    /// Outcomes that timed out.
    pub timeouts: usize,
}

fn mismatch(decision: u64, message: impl Into<String>) -> MeasureError {
    MeasureError::Replay {
        decision,
        message: message.into(),
    }
}

fn check_header(h: &Header) -> Result<(), String> {
    if h.schema != SCHEMA_ID {
        return Err(format!("schema {:?}, expected {SCHEMA_ID:?}", h.schema));
    }
    if h.boundaries != Boundaries::current() {
        return Err("level boundaries differ from this schema version".to_string());
    }
    if (
        h.draws_per_decision,
        h.candidates_per_decision,
        h.margin,
        h.explore_one_in,
        h.copy_one_in,
    ) != (
        DRAWS_PER_DECISION,
        CANDIDATES_PER_DECISION,
        MARGIN,
        EXPLORE_ONE_IN,
        COPY_ONE_IN,
    ) {
        return Err("decision constants differ from this agent version".to_string());
    }
    Ok(())
}

/// Replay one decision against its segment header.
///
/// # Errors
/// [`MeasureError::Replay`] describing the first difference.
pub fn replay_decision(h: &Header, rec: &DecisionRecord) -> Result<(), MeasureError> {
    let d = rec.decision;
    let mut rng = decision_rng(h.seed, d);
    let slots = sample_slots(&mut rng, h.peer_count);
    let logged: Vec<usize> = rec.candidates.iter().map(|c| c.slot).collect();
    if slots != logged {
        return Err(mismatch(d, format!("slots {logged:?}, replay {slots:?}")));
    }
    let mut cands = Vec::with_capacity(rec.candidates.len());
    for c in &rec.candidates {
        let [r, l, dl, a, b] = c.levels;
        let f = Features::new(r, l, dl, a, b).map_err(|e| mismatch(d, e.to_string()))?;
        if c.raw.levels() != f {
            return Err(mismatch(
                d,
                format!("slot {} levels not cut from raw", c.slot),
            ));
        }
        if FixedRule.score(f) != c.rule_score {
            return Err(mismatch(d, format!("slot {} rule score", c.slot)));
        }
        cands.push(Candidate {
            id: u32::try_from(c.slot).map_err(|_| mismatch(d, "slot too large"))?,
            features: f,
        });
    }
    let choice = choose(&mut rng, &cands).ok_or_else(|| mismatch(d, "empty candidate set"))?;
    let replayed = (
        choice.rule_choice,
        choice.pick,
        choice.explored,
        choice.copy,
    );
    let logged = (rec.rule_choice, rec.pick, rec.explored, rec.copy);
    if replayed != logged {
        return Err(mismatch(
            d,
            format!("(rule, pick, explored, copy) logged {logged:?}, replay {replayed:?}"),
        ));
    }
    if rec.duplicated != rec.copy.is_some() {
        return Err(mismatch(d, "duplicated flag disagrees with copy"));
    }
    if (choice.propensity - rec.propensity).abs() > PROPENSITY_TOLERANCE {
        return Err(mismatch(
            d,
            format!(
                "propensity logged {}, replay {}",
                rec.propensity, choice.propensity
            ),
        ));
    }
    let mut expected = vec![(rec.pick, Role::Pick)];
    if let Some(c) = rec.copy {
        expected.push((c, Role::Copy));
    }
    let got: Vec<(usize, Role)> = rec.outcomes.iter().map(|o| (o.candidate, o.role)).collect();
    if got != expected {
        return Err(mismatch(
            d,
            format!("outcomes {got:?}, expected {expected:?}"),
        ));
    }
    if rec
        .outcomes
        .iter()
        .any(|o| o.timed_out != o.rtt_us.is_none())
    {
        return Err(mismatch(d, "timed_out disagrees with rtt_us"));
    }
    Ok(())
}

/// Replay parsed trace lines: every decision must reproduce from its
/// segment header's seed, decision numbers must increase, and a slot must
/// keep one peer hash within a segment.
///
/// # Errors
/// [`MeasureError::Replay`] or [`MeasureError::TraceFormat`].
pub fn replay(lines: &[TraceLine]) -> Result<ReplayReport, MeasureError> {
    let mut report = ReplayReport::default();
    let mut header: Option<&Header> = None;
    let mut peers: HashMap<usize, &str> = HashMap::new();
    let mut last: Option<u64> = None;
    for (i, line) in lines.iter().enumerate() {
        match line {
            TraceLine::Header(h) => {
                check_header(h).map_err(|m| MeasureError::TraceFormat {
                    file: String::new(),
                    line: i + 1,
                    message: m,
                })?;
                header = Some(h);
                peers.clear();
                report.segments += 1;
            }
            TraceLine::Decision(rec) => {
                let h = header.ok_or_else(|| MeasureError::TraceFormat {
                    file: String::new(),
                    line: i + 1,
                    message: "decision before any header".to_string(),
                })?;
                if last.is_some_and(|l| rec.decision <= l) || rec.decision < h.first_decision {
                    return Err(mismatch(rec.decision, "decision numbers do not increase"));
                }
                last = Some(rec.decision);
                for c in &rec.candidates {
                    if *peers.entry(c.slot).or_insert(c.peer.as_str()) != c.peer {
                        return Err(mismatch(rec.decision, "slot changed peer within a segment"));
                    }
                }
                replay_decision(h, rec)?;
                report.decisions += 1;
                report.explored += usize::from(rec.explored);
                report.contested += usize::from(rec.pick != rec.rule_choice);
                report.copies += usize::from(rec.duplicated);
                report.timeouts += rec.outcomes.iter().filter(|o| o.timed_out).count();
            }
        }
    }
    Ok(report)
}

/// Read and replay one trace file.
///
/// # Errors
/// As [`read_trace`] and [`replay`].
pub fn replay_trace(path: &Path) -> Result<ReplayReport, MeasureError> {
    replay_file(path, &read_trace(path)?)
}

/// [`replay`] with format errors attributed to `path`.
fn replay_file(path: &Path, lines: &[TraceLine]) -> Result<ReplayReport, MeasureError> {
    replay(lines).map_err(|e| match e {
        MeasureError::TraceFormat { line, message, .. } => format_err(path, line, message),
        other => other,
    })
}

/// Result of [`verify_trace`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    /// File name.
    pub file: String,
    /// Replay totals.
    pub replay: ReplayReport,
    /// Manifest status.
    pub manifest: ManifestStatus,
}

/// Verify a trace file: every line parses, the first line is a header,
/// every decision replays, and the manifest (if the day is closed) matches.
///
/// # Errors
/// The first failure found.
pub fn verify_trace(path: &Path) -> Result<VerifyReport, MeasureError> {
    let lines = read_trace(path)?;
    if !matches!(lines.first(), Some(TraceLine::Header(_))) {
        return Err(format_err(path, 1, "first line is not a header"));
    }
    let replay = replay_file(path, &lines)?;
    let manifest = check_manifest(path)?;
    Ok(VerifyReport {
        file: file_name(path),
        replay,
        manifest,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::privacy::peer_hash;

    const PEERS: usize = 5;
    const DAY1_MS: u64 = 1_765_000_000_000; // 2025-12-06
    const DAY2_MS: u64 = DAY1_MS + 86_400_000;

    fn header(seed: u64) -> Header {
        Header {
            schema: SCHEMA_ID.to_string(),
            agent_version: "test".to_string(),
            git_commit: None,
            transport: crate::TRANSPORT_ID.to_string(),
            node_hash: "00".repeat(32),
            node_class: None,
            salt: "11".repeat(32),
            seed,
            draws_per_decision: DRAWS_PER_DECISION,
            candidates_per_decision: CANDIDATES_PER_DECISION,
            margin: MARGIN,
            explore_one_in: EXPLORE_ONE_IN,
            copy_one_in: COPY_ONE_IN,
            peer_count: PEERS,
            request_bytes: 1024,
            tick_ms: 10_000,
            probe_timeout_ms: 1_000,
            request_timeout_ms: 2_000,
            first_decision: 0,
            mono_ns: 0,
            utc_ms: 0,
            utc: String::new(),
            ntp_offset_ms: None,
            boundaries: Boundaries::current(),
            estimators: EstimatorSchema::current(),
        }
    }

    /// Raw values per slot: slots 0 and 1 alike (so the band holds two),
    /// the rest slower or lossier.
    fn raw(slot: usize, k: u64) -> RawFeatures {
        let srtt = [800, 900, 3_000, 3_100, 50_000][slot];
        RawFeatures {
            srtt_us: Some(srtt),
            loss_ppm: if slot == 4 { 200_000 } else { 0 },
            delivery_ppm: 940_000 + (k % 3) * 20_000,
            age_s: Some(k * 10),
        }
    }

    /// Make decision `k` exactly as the live loop does, without a network.
    fn synth(seed: u64, k: u64, utc_ms: u64) -> DecisionRecord {
        let mut rng = decision_rng(seed, k);
        let slots = sample_slots(&mut rng, PEERS);
        let mut cands = Vec::new();
        let mut recs = Vec::new();
        for &s in &slots {
            let r = raw(s, k);
            let f = r.levels();
            cands.push(Candidate {
                id: s as u32,
                features: f,
            });
            recs.push(CandidateRecord {
                peer: peer_hash(&[1; 32], &format!("peer{s}")),
                slot: s,
                levels: [f.rtt(), f.loss(), f.delivery(), f.age(), f.balance()],
                rule_score: FixedRule.score(f),
                raw: r,
            });
        }
        let c = choose(&mut rng, &cands).unwrap();
        let mut outcomes = vec![OutcomeRecord {
            candidate: c.pick,
            role: Role::Pick,
            rtt_us: if k.is_multiple_of(17) {
                None
            } else {
                Some(1_000 + k)
            },
            timed_out: k.is_multiple_of(17),
        }];
        if let Some(cp) = c.copy {
            outcomes.push(OutcomeRecord {
                candidate: cp,
                role: Role::Copy,
                rtt_us: Some(2_000),
                timed_out: false,
            });
        }
        DecisionRecord {
            decision: k,
            mono_ns: k * 1_000,
            utc_ms,
            ntp_offset_ms: None,
            request_bytes: 1024,
            candidates: recs,
            rule_choice: c.rule_choice,
            pick: c.pick,
            explored: c.explored,
            duplicated: c.copy.is_some(),
            propensity: c.propensity,
            copy: c.copy,
            outcomes,
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "measure-agent-trace-{name}-{}-{}",
            std::process::id(),
            Clock::utc_ms()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn write_two_days(dir: &Path, seed: u64) -> Vec<PathBuf> {
        let mut w = TraceWriter::new(dir, header(seed), Clock::start());
        for k in 0..400 {
            let t = if k < 250 { DAY1_MS } else { DAY2_MS };
            w.append(&synth(seed, k, t + k)).unwrap();
        }
        // Day one is closed by the rollover, day two only by close().
        let files = trace_files(dir).unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(check_manifest(&files[0]).unwrap(), ManifestStatus::Matched);
        assert_eq!(check_manifest(&files[1]).unwrap(), ManifestStatus::Absent);
        w.close().unwrap();
        files
    }

    #[test]
    fn write_then_verify_roundtrip() {
        let dir = tmp("roundtrip");
        let files = write_two_days(&dir, 99);
        let mut total = ReplayReport::default();
        for f in &files {
            let v = verify_trace(f).unwrap();
            assert_eq!(v.manifest, ManifestStatus::Matched);
            assert_eq!(v.replay.segments, 1);
            total.decisions += v.replay.decisions;
            total.explored += v.replay.explored;
            total.contested += v.replay.contested;
            total.copies += v.replay.copies;
        }
        assert_eq!(total.decisions, 400);
        assert!(total.explored > 0 && total.contested > 0 && total.copies > 0);
        assert_eq!(last_decision(&dir).unwrap(), Some(399));
        assert_eq!(utc_day(DAY1_MS), "2025-12-06");

        // Every line is a valid record, and the header comes first.
        let lines = read_trace(&files[1]).unwrap();
        assert!(matches!(lines[0], TraceLine::Header(ref h) if h.first_decision == 250));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn replay_is_deterministic_and_catches_tampering() {
        let seed = 0x5EED;
        let lines: Vec<TraceLine> = std::iter::once(TraceLine::Header(header(seed)))
            .chain((0..300).map(|k| TraceLine::Decision(synth(seed, k, DAY1_MS))))
            .collect();
        let a = replay(&lines).unwrap();
        let b = replay(&lines).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.decisions, 300);

        // The same records under another seed do not replay.
        let mut wrong_seed = lines.clone();
        wrong_seed[0] = TraceLine::Header(header(seed + 1));
        assert!(matches!(
            replay(&wrong_seed),
            Err(MeasureError::Replay { .. })
        ));

        // Changing any logged choice is caught.
        let first_explore = lines
            .iter()
            .position(|l| matches!(l, TraceLine::Decision(d) if d.explored))
            .unwrap();
        let tamper = |f: &dyn Fn(&mut DecisionRecord)| {
            let mut t = lines.clone();
            if let TraceLine::Decision(d) = &mut t[first_explore] {
                f(d);
            }
            replay(&t)
        };
        assert!(tamper(&|d| d.explored = false).is_err());
        assert!(tamper(&|d| d.pick = (d.pick + 1) % d.candidates.len()).is_err());
        assert!(tamper(&|d| d.propensity = 0.5).is_err());
        assert!(tamper(&|d| d.candidates[0].levels[0] = 15).is_err());
        assert!(tamper(&|d| d.candidates.swap(0, 1)).is_err());

        // Decisions out of order are caught.
        let mut reordered = lines.clone();
        reordered.swap(1, 2);
        assert!(replay(&reordered).is_err());
    }

    #[test]
    fn manifest_catches_a_changed_byte() {
        let dir = tmp("manifest");
        let files = write_two_days(&dir, 7);
        let mut bytes = std::fs::read(&files[0]).unwrap();
        let pos = bytes.iter().rposition(|b| *b == b'1').unwrap();
        bytes[pos] = b'2';
        std::fs::write(&files[0], &bytes).unwrap();
        assert!(verify_trace(&files[0]).is_err());
        // close_past_days rewrites a stale manifest of a past day.
        let closed = close_past_days(&dir, "2099-01-01").unwrap();
        assert_eq!(closed.len(), 1);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn reopening_a_day_starts_a_new_segment() {
        let dir = tmp("reopen");
        let seed = 3;
        for range in [0..40u64, 40..90] {
            let mut w = TraceWriter::new(
                &dir,
                Header {
                    first_decision: range.start,
                    ..header(seed)
                },
                Clock::start(),
            );
            for k in range {
                w.append(&synth(seed, k, DAY1_MS)).unwrap();
            }
            w.close().unwrap();
        }
        let files = trace_files(&dir).unwrap();
        assert_eq!(files.len(), 1);
        let v = verify_trace(&files[0]).unwrap();
        assert_eq!((v.replay.segments, v.replay.decisions), (2, 90));
        assert_eq!(v.manifest, ManifestStatus::Matched);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
