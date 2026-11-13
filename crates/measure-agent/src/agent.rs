//! The measure loop (plan §1 "what it does per decision", §3 "restarts").
//!
//! Each tick: probe every configured peer and update its estimators; draw
//! the candidates; cut their estimators to levels; decide ([`crate::decide`]);
//! send the decision request to the pick (and the copy, if any); record the
//! outcome. A restart continues the decision numbering, and so the seeded
//! stream, after the last decision already logged in the data directory.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use policy_poc::{Candidate, FixedRule, Scorer};

use crate::clock::{utc_day, Clock};
use crate::config::Config;
use crate::decide::{
    choose, decision_rng, sample_slots, CANDIDATES_PER_DECISION, COPY_ONE_IN, DRAWS_PER_DECISION,
    EXPLORE_ONE_IN, MARGIN,
};
use crate::error::MeasureError;
use crate::estimator::PeerEstimator;
use crate::frame::{Class, HEADER_LEN};
use crate::levels::Boundaries;
use crate::privacy::{
    load_or_create_salt, node_hash, os_random_u64, peer_hash, stream_seed, to_hex,
};
use crate::probe::{Prober, Target};
use crate::reflector::{Faults, Reflector};
use crate::trace::{
    close_past_days, last_decision, CandidateRecord, DecisionRecord, EstimatorSchema, Header,
    OutcomeRecord, Role, TraceWriter,
};
use crate::{SCHEMA_ID, TRANSPORT_ID};

/// Totals of one run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RunSummary {
    /// Data directory written to.
    pub data_dir: PathBuf,
    /// Number of the first decision of this run.
    pub first_decision: u64,
    /// Decisions made in this run.
    pub decisions: u64,
    /// Decisions whose exploration draw fired.
    pub explored: u64,
    /// Decisions where the pick differs from the rule's choice.
    pub contested: u64,
    /// Decisions with a copy.
    pub copies: u64,
    /// Pick outcomes that timed out.
    pub pick_timeouts: u64,
    /// Sum of completed pick round trips, microseconds.
    pub pick_rtt_sum_us: u64,
    /// Count of completed picks.
    pub pick_completed: u64,
}

/// Options of one run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RunOptions {
    /// Stop after this many decisions; `None` runs until killed.
    pub max_decisions: Option<u64>,
    /// Also run the reflector on the configured listen address.
    pub reflect: bool,
}

fn wildcard_for(addr: SocketAddr) -> SocketAddr {
    let ip = match addr.ip() {
        IpAddr::V4(v4) if v4.is_loopback() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        IpAddr::V6(v6) if v6.is_loopback() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    };
    SocketAddr::new(ip, 0)
}

/// Run the measure loop.
///
/// # Errors
/// Configuration, I/O and trace errors. Network losses are outcomes, not
/// errors.
pub fn run_agent(cfg: &Config, opts: RunOptions) -> Result<RunSummary, MeasureError> {
    cfg.validate()?;
    let dir = &cfg.data_dir;
    std::fs::create_dir_all(dir)
        .map_err(|e| MeasureError::io(format!("create {}", dir.display()), e))?;
    let salt = load_or_create_salt(dir)?;
    let seed = stream_seed(cfg.experiment_seed, &cfg.node);
    let peer_ids: Vec<String> = cfg
        .peers
        .iter()
        .map(|p| peer_hash(&salt, &p.label))
        .collect();
    close_past_days(dir, &utc_day(Clock::utc_ms()))?;
    let first_decision = last_decision(dir)?.map_or(0, |d| d + 1);

    let clock = Clock::start();
    let template = Header {
        schema: SCHEMA_ID.to_string(),
        agent_version: env!("CARGO_PKG_VERSION").to_string(),
        git_commit: option_env!("MEASURE_AGENT_GIT_COMMIT").map(str::to_string),
        transport: TRANSPORT_ID.to_string(),
        node_hash: node_hash(&salt, &cfg.node),
        node_class: cfg.node_class.clone(),
        salt: to_hex(&salt),
        seed,
        draws_per_decision: DRAWS_PER_DECISION,
        candidates_per_decision: CANDIDATES_PER_DECISION,
        margin: MARGIN,
        explore_one_in: EXPLORE_ONE_IN,
        copy_one_in: COPY_ONE_IN,
        peer_count: cfg.peers.len(),
        request_bytes: cfg.request_bytes,
        tick_ms: cfg.tick_ms,
        probe_timeout_ms: cfg.probe_timeout_ms,
        request_timeout_ms: cfg.request_timeout_ms,
        first_decision,
        mono_ns: 0,
        utc_ms: 0,
        utc: String::new(),
        ntp_offset_ms: None,
        boundaries: Boundaries::current(),
        estimators: EstimatorSchema::current(),
    };
    let mut writer = TraceWriter::new(dir, template, clock);

    let reflector = if opts.reflect {
        Some(Reflector::spawn(cfg.listen_addr(), Faults::NONE)?)
    } else {
        None
    };
    let mut prober = Prober::bind(wildcard_for(cfg.peers[0].addr), os_random_u64()?, clock)?;
    let mut est = vec![PeerEstimator::new(); cfg.peers.len()];

    let mut summary = RunSummary {
        data_dir: dir.clone(),
        first_decision,
        ..RunSummary::default()
    };
    let tick = Duration::from_millis(cfg.tick_ms);
    let probe_timeout = Duration::from_millis(cfg.probe_timeout_ms);
    let request_timeout = Duration::from_millis(cfg.request_timeout_ms);
    let probes: Vec<Target> = cfg
        .peers
        .iter()
        .map(|p| Target {
            addr: p.addr,
            class: Class::Probe,
            frame_bytes: HEADER_LEN,
        })
        .collect();

    let mut next_tick = Instant::now();
    let mut decision = first_decision;
    while opts.max_decisions != Some(0) {
        // 1. Measurement probes to every peer.
        let results = prober.exchange(&probes, probe_timeout)?;
        let now_ns = clock.mono_ns();
        for (e, r) in est.iter_mut().zip(&results) {
            e.on_probe(*r, now_ns);
        }

        // 2. Candidates and the decision.
        let mut rng = decision_rng(seed, decision);
        let slots = sample_slots(&mut rng, cfg.peers.len());
        let mut cands = Vec::with_capacity(slots.len());
        let mut records = Vec::with_capacity(slots.len());
        for &s in &slots {
            let raw = est[s].raw(now_ns);
            let f = raw.levels();
            cands.push(Candidate {
                id: u32::try_from(s).unwrap_or(u32::MAX),
                features: f,
            });
            records.push(CandidateRecord {
                peer: peer_ids[s].clone(),
                slot: s,
                levels: [f.rtt(), f.loss(), f.delivery(), f.age(), f.balance()],
                rule_score: FixedRule.score(f),
                raw,
            });
        }
        let Some(choice) = choose(&mut rng, &cands) else {
            return Err(MeasureError::Config("no candidates".to_string()));
        };

        // 3. The request (and its copy).
        let mut sent = vec![(choice.pick, Role::Pick)];
        if let Some(c) = choice.copy {
            sent.push((c, Role::Copy));
        }
        let targets: Vec<Target> = sent
            .iter()
            .map(|&(i, _)| Target {
                addr: cfg.peers[slots[i]].addr,
                class: Class::Decision,
                frame_bytes: cfg.request_bytes,
            })
            .collect();
        let outcomes = prober.exchange(&targets, request_timeout)?;
        let done_ns = clock.mono_ns();
        let mut outcome_records = Vec::with_capacity(sent.len());
        for (&(i, role), rtt) in sent.iter().zip(&outcomes) {
            est[slots[i]].on_request(*rtt, done_ns);
            outcome_records.push(OutcomeRecord {
                candidate: i,
                role,
                rtt_us: *rtt,
                timed_out: rtt.is_none(),
            });
        }

        // 4. The record.
        let rec = DecisionRecord {
            decision,
            mono_ns: now_ns,
            utc_ms: Clock::utc_ms(),
            ntp_offset_ms: None,
            request_bytes: cfg.request_bytes,
            candidates: records,
            rule_choice: choice.rule_choice,
            pick: choice.pick,
            explored: choice.explored,
            duplicated: choice.copy.is_some(),
            propensity: choice.propensity,
            copy: choice.copy,
            outcomes: outcome_records,
        };
        writer.append(&rec)?;

        summary.decisions += 1;
        summary.explored += u64::from(choice.explored);
        summary.contested += u64::from(choice.pick != choice.rule_choice);
        summary.copies += u64::from(choice.copy.is_some());
        match outcomes.first().copied().flatten() {
            Some(rtt) => {
                summary.pick_completed += 1;
                summary.pick_rtt_sum_us += rtt;
            }
            None => summary.pick_timeouts += 1,
        }
        decision += 1;
        if opts.max_decisions.is_some_and(|m| summary.decisions >= m) {
            break;
        }

        // 5. Wait for the next tick (no catch-up bursts after a stall).
        next_tick = (next_tick + tick).max(Instant::now());
        std::thread::sleep(next_tick.saturating_duration_since(Instant::now()));
    }
    writer.close()?;
    if let Some(r) = reflector {
        r.stop();
    }
    Ok(summary)
}
