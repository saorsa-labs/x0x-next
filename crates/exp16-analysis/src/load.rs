//! Reading and re-verifying x0x-measure/1 traces (plan §3, "versioned
//! schema", and §4).
//!
//! Every trace file is verified before any of its records is used:
//!
//! * a closed day (one with a manifest) must match its manifest's SHA-256
//!   byte for byte;
//! * every line must parse, the first line must be a header, and every
//!   header must carry schema `x0x-measure/1` with this version's
//!   boundaries and decision constants (so versions never mix inside a
//!   node's window);
//! * every decision must replay from its segment header's seed through
//!   measure-agent's own [`replay`]: the sampled slots, the levels cut from
//!   the raw values, the rule's choice, exploration, the pick, its
//!   propensity and the copy.
//!
//! Any failure refuses the whole file ([`AnalysisError::Trace`]). The one
//! allowance: a file with no manifest is still being written by a live
//! agent, so an unterminated final line (an append in progress) is set
//! aside and reported, not treated as tampering. Realized outcomes cannot
//! be re-derived; on a closed day the manifest protects them, on an open day
//! nothing does, which the report states per file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use measure_agent::trace::{
    check_manifest, replay, trace_files, DecisionRecord, Header, ManifestStatus, OutcomeRecord,
    Role, TraceLine,
};
use measure_agent::MeasureError;
use policy_poc::{Candidate, Features};
use serde::Serialize;

use crate::error::AnalysisError;

/// One decision as the analysis uses it.
#[derive(Debug, Clone, PartialEq)]
pub struct LoggedDecision {
    /// Decision number.
    pub decision: u64,
    /// UTC milliseconds since the Unix epoch.
    pub utc_ms: u64,
    /// The candidate set (id = configured peer slot).
    pub candidates: Vec<Candidate>,
    /// The fixed rule's choice (index into `candidates`).
    pub rule_choice: usize,
    /// The candidate actually used.
    pub pick: usize,
    /// Logged probability of `pick` under the Measure procedure.
    pub propensity: f64,
    /// Whether the exploration draw fired.
    pub explored: bool,
    /// Realized cost of the pick, ms (the request timeout when it timed out).
    pub pick_cost_ms: f64,
    /// Whether the pick timed out.
    pub pick_failed: bool,
    /// The copy, if one was sent: candidate index and realized cost, ms.
    pub copy: Option<(usize, f64)>,
}

/// How one trace file passed verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FileCheck {
    /// Path of the file.
    pub file: String,
    /// Segments (header lines).
    pub segments: usize,
    /// Decisions replayed.
    pub decisions: usize,
    /// `"matched"` (closed day, manifest matched) or `"open"` (no manifest
    /// yet: outcomes are not protected by a hash).
    pub manifest: String,
    /// True when an unterminated final line of an open file was set aside.
    pub partial_line_skipped: bool,
}

/// Every verified decision of one node, across all the directories given.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeTrace {
    /// Salted node hash from the segment headers.
    pub node_hash: String,
    /// Node class from the headers, if configured.
    pub node_class: Option<String>,
    /// The files that contributed.
    pub files: Vec<FileCheck>,
    /// Decisions in decision-number order.
    pub decisions: Vec<LoggedDecision>,
}

impl NodeTrace {
    /// Short display name: the class (if any) and the first 12 hex digits of
    /// the node hash.
    #[must_use]
    pub fn label(&self) -> String {
        let short: String = self.node_hash.chars().take(12).collect();
        match &self.node_class {
            Some(c) => format!("{c}:{short}"),
            None => short,
        }
    }
}

fn refuse(path: &Path, source: MeasureError) -> AnalysisError {
    AnalysisError::Trace {
        file: path.display().to_string(),
        source,
    }
}

fn content(path: &Path, message: impl Into<String>) -> AnalysisError {
    AnalysisError::Content {
        file: path.display().to_string(),
        message: message.into(),
    }
}

/// Read and fully verify one trace file. Returns its lines and how it
/// passed.
///
/// # Errors
/// [`AnalysisError::Trace`] on any verification failure.
pub fn verify_file(path: &Path) -> Result<(Vec<TraceLine>, FileCheck), AnalysisError> {
    let manifest = check_manifest(path).map_err(|e| refuse(path, e))?;
    let text = std::fs::read_to_string(path).map_err(|e| {
        refuse(
            path,
            MeasureError::io(format!("read {}", path.display()), e),
        )
    })?;
    let mut body = text.as_str();
    let mut partial_line_skipped = false;
    if manifest == ManifestStatus::Absent && !body.is_empty() && !body.ends_with('\n') {
        // A live agent is mid-append: keep only whole lines.
        let cut = body.rfind('\n').map_or(0, |i| i + 1);
        body = &body[..cut];
        partial_line_skipped = true;
    }
    let mut lines = Vec::new();
    for (i, l) in body.lines().enumerate() {
        let parsed: TraceLine = serde_json::from_str(l).map_err(|e| {
            refuse(
                path,
                MeasureError::TraceFormat {
                    file: path.display().to_string(),
                    line: i + 1,
                    message: e.to_string(),
                },
            )
        })?;
        lines.push(parsed);
    }
    if lines
        .first()
        .is_some_and(|l| !matches!(l, TraceLine::Header(_)))
    {
        return Err(refuse(
            path,
            MeasureError::TraceFormat {
                file: path.display().to_string(),
                line: 1,
                message: "first line is not a header".to_string(),
            },
        ));
    }
    let report = replay(&lines).map_err(|e| refuse(path, e))?;
    let check = FileCheck {
        file: path.display().to_string(),
        segments: report.segments,
        decisions: report.decisions,
        manifest: match manifest {
            ManifestStatus::Matched => "matched",
            ManifestStatus::Absent => "open",
        }
        .to_string(),
        partial_line_skipped,
    };
    Ok((lines, check))
}

/// Cost of an outcome in ms: the round trip, or the request timeout when no
/// echo arrived (a censored cost; the failure is flagged separately).
fn outcome_cost(o: &OutcomeRecord, h: &Header) -> f64 {
    match o.rtt_us {
        Some(us) if !o.timed_out => us as f64 / 1_000.0,
        _ => h.request_timeout_ms as f64,
    }
}

fn to_logged(
    path: &Path,
    h: &Header,
    rec: &DecisionRecord,
) -> Result<LoggedDecision, AnalysisError> {
    let mut candidates = Vec::with_capacity(rec.candidates.len());
    for c in &rec.candidates {
        let [r, l, d, a, b] = c.levels;
        let features = Features::new(r, l, d, a, b).map_err(|e| content(path, e.to_string()))?;
        let id = u32::try_from(c.slot).map_err(|_| content(path, "slot too large"))?;
        candidates.push(Candidate { id, features });
    }
    let pick_outcome = rec
        .outcomes
        .iter()
        .find(|o| o.role == Role::Pick)
        .ok_or_else(|| {
            content(
                path,
                format!("decision {} has no pick outcome", rec.decision),
            )
        })?;
    let copy = rec
        .outcomes
        .iter()
        .find(|o| o.role == Role::Copy)
        .map(|o| (o.candidate, outcome_cost(o, h)));
    if !(rec.propensity.is_finite() && rec.propensity > 0.0) {
        return Err(content(
            path,
            format!(
                "decision {} has propensity {}",
                rec.decision, rec.propensity
            ),
        ));
    }
    Ok(LoggedDecision {
        decision: rec.decision,
        utc_ms: rec.utc_ms,
        candidates,
        rule_choice: rec.rule_choice,
        pick: rec.pick,
        propensity: rec.propensity,
        explored: rec.explored,
        pick_cost_ms: outcome_cost(pick_outcome, h),
        pick_failed: pick_outcome.timed_out,
        copy,
    })
}

/// Load every `trace-*.jsonl` in each directory, verify each file, and group
/// the decisions by node hash.
///
/// # Errors
/// [`AnalysisError::Trace`] for a file that fails verification,
/// [`AnalysisError::Content`] for a decision that appears twice for one
/// node, [`AnalysisError::Io`] for an unreadable directory.
pub fn load_nodes(dirs: &[PathBuf]) -> Result<Vec<NodeTrace>, AnalysisError> {
    let mut nodes: BTreeMap<String, NodeTrace> = BTreeMap::new();
    for dir in dirs {
        let files = trace_files(dir).map_err(|e| match e {
            MeasureError::Io { context, source } => AnalysisError::Io { context, source },
            other => refuse(dir, other),
        })?;
        for path in files {
            let (lines, check) = verify_file(&path)?;
            let mut header: Option<&Header> = None;
            let mut touched: Vec<String> = Vec::new();
            for line in &lines {
                match line {
                    TraceLine::Header(h) => header = Some(h),
                    TraceLine::Decision(rec) => {
                        // replay() has already refused a decision before any
                        // header.
                        let Some(h) = header else { continue };
                        let node = nodes
                            .entry(h.node_hash.clone())
                            .or_insert_with(|| NodeTrace {
                                node_hash: h.node_hash.clone(),
                                node_class: h.node_class.clone(),
                                files: Vec::new(),
                                decisions: Vec::new(),
                            });
                        node.decisions.push(to_logged(&path, h, rec)?);
                        if !touched.contains(&h.node_hash) {
                            touched.push(h.node_hash.clone());
                        }
                    }
                }
            }
            for hash in touched {
                if let Some(node) = nodes.get_mut(&hash) {
                    node.files.push(check.clone());
                }
            }
        }
    }
    let mut out: Vec<NodeTrace> = nodes.into_values().collect();
    for node in &mut out {
        node.decisions.sort_by_key(|d| d.decision);
        if let Some(w) = node
            .decisions
            .windows(2)
            .find(|w| w[0].decision == w[1].decision)
        {
            return Err(AnalysisError::Content {
                file: node.label(),
                message: format!(
                    "decision {} appears twice for node {}",
                    w[0].decision, node.node_hash
                ),
            });
        }
    }
    Ok(out)
}
