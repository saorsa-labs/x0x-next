//! End-to-end tests on synthetic traces written through measure-agent's own
//! trace writer, from worlds where the right answer is known.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic, missing_docs)]

use std::path::{Path, PathBuf};

use exp16_analysis::{analyze, load_nodes, AnalysisError, Options, Split, Verdict};
use measure_agent::clock::Clock;
use measure_agent::decide::{
    choose, decision_rng, sample_slots, CANDIDATES_PER_DECISION, COPY_ONE_IN, DRAWS_PER_DECISION,
    EXPLORE_ONE_IN, MARGIN,
};
use measure_agent::levels::{
    Boundaries, RawFeatures, DELIVERY_EDGES_PPM, LOSS_EDGES_PPM, RTT_EDGES_US,
};
use measure_agent::privacy::{node_hash, peer_hash};
use measure_agent::trace::{
    CandidateRecord, DecisionRecord, EstimatorSchema, Header, OutcomeRecord, Role, TraceWriter,
};
use measure_agent::SCHEMA_ID;
use policy_poc::rng::SplitMix64;
use policy_poc::{Candidate, FixedRule, Scorer, WorldKind, WorldParams};

const START_MS: u64 = 1_796_342_400_000; // 2026-12-04 00:00 UTC
const DAY_MS: u64 = 86_400_000;

/// Peer types as (rtt, loss, delivery) levels. With these, (3,1,7) and
/// (4,0,7) sit 3 quanta apart, both inside the band, while (4,1,7) is
/// always outside it: the in-band choice trades one rtt level for one loss
/// level.
const PEERS: [[u8; 3]; 12] = [
    [4, 0, 7],
    [3, 1, 7],
    [4, 1, 7],
    [4, 0, 7],
    [3, 1, 7],
    [4, 1, 7],
    [4, 0, 7],
    [3, 1, 7],
    [4, 1, 7],
    [4, 0, 7],
    [3, 1, 7],
    [4, 1, 7],
];

/// rtt matters, loss does not: the fixed rule's preference.
fn rtt_world() -> WorldParams {
    WorldParams {
        kind: WorldKind::One,
        base_ms: 20.0,
        rtt_power: 0.0,
        rtt_growth: 1.5,
        loss_slope: 0.0,
        loss_timeout_ms: 0.0,
        noise_sigma: 0.5,
    }
}

/// Loss matters, rtt does not: the opposite preference.
fn loss_world() -> WorldParams {
    WorldParams {
        kind: WorldKind::One,
        base_ms: 20.0,
        rtt_power: 0.0,
        rtt_growth: 1.0,
        loss_slope: 1.5,
        loss_timeout_ms: 0.0,
        noise_sigma: 0.5,
    }
}

fn raw_for(levels: [u8; 3]) -> RawFeatures {
    let [r, l, d] = levels.map(usize::from);
    RawFeatures {
        srtt_us: Some(if r == 0 { 500 } else { RTT_EDGES_US[r - 1] }),
        loss_ppm: if l == 0 { 0 } else { LOSS_EDGES_PPM[l - 1] },
        delivery_ppm: if d == 0 {
            100_000
        } else {
            DELIVERY_EDGES_PPM[d - 1]
        },
        age_s: None,
    }
}

fn header(name: &str, seed: u64, salt: &[u8]) -> Header {
    Header {
        schema: SCHEMA_ID.to_string(),
        agent_version: "synthetic".to_string(),
        git_commit: None,
        transport: measure_agent::TRANSPORT_ID.to_string(),
        node_hash: node_hash(salt, name),
        node_class: Some("synthetic".to_string()),
        salt: "ab".repeat(32),
        seed,
        draws_per_decision: DRAWS_PER_DECISION,
        candidates_per_decision: CANDIDATES_PER_DECISION,
        margin: MARGIN,
        explore_one_in: EXPLORE_ONE_IN,
        copy_one_in: COPY_ONE_IN,
        peer_count: PEERS.len(),
        request_bytes: 1024,
        tick_ms: 10_000,
        probe_timeout_ms: 1_000,
        request_timeout_ms: 10_000,
        first_decision: 0,
        mono_ns: 0,
        utc_ms: START_MS,
        utc: String::new(),
        ntp_offset_ms: None,
        boundaries: Boundaries::current(),
        estimators: EstimatorSchema::current(),
    }
}

/// Write `n` decisions of node `name` in `world` into `dir`, exactly as the
/// live loop decides (same sampling, band, exploration and copies), with
/// realized costs drawn from the world. Decisions are spread evenly over
/// `days` UTC days.
fn write_node(dir: &Path, name: &str, world: WorldParams, seed: u64, n: u64, days: u64) {
    std::fs::create_dir_all(dir).unwrap();
    let salt = [7u8; 32];
    let mut w = TraceWriter::new(dir, header(name, seed, &salt), Clock::start());
    let mut noise = SplitMix64::new(seed ^ 0xC0FFEE);
    let step = days * DAY_MS / n.max(1);
    for k in 0..n {
        let mut rng = decision_rng(seed, k);
        let slots = sample_slots(&mut rng, PEERS.len());
        let mut cands = Vec::new();
        let mut recs = Vec::new();
        for &s in &slots {
            let raw = raw_for(PEERS[s]);
            let f = raw.levels();
            cands.push(Candidate {
                id: s as u32,
                features: f,
            });
            recs.push(CandidateRecord {
                peer: peer_hash(&salt, &format!("{name}-peer{s}")),
                slot: s,
                levels: [f.rtt(), f.loss(), f.delivery(), f.age(), f.balance()],
                rule_score: FixedRule.score(f),
                raw,
            });
        }
        let c = choose(&mut rng, &cands).unwrap();
        let mut outcome = |i: usize, role: Role| {
            let cost_ms =
                world.expected_cost(cands[i].features) * (world.noise_sigma * noise.normal()).exp();
            OutcomeRecord {
                candidate: i,
                role,
                rtt_us: Some((cost_ms * 1_000.0).round() as u64),
                timed_out: false,
            }
        };
        let mut outcomes = vec![outcome(c.pick, Role::Pick)];
        if let Some(cp) = c.copy {
            outcomes.push(outcome(cp, Role::Copy));
        }
        w.append(&DecisionRecord {
            decision: k,
            mono_ns: k * 1_000_000,
            utc_ms: START_MS + k * step,
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
        })
        .unwrap();
    }
    w.close().unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "exp16-{name}-{}-{}",
        std::process::id(),
        Clock::utc_ms()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Ten nodes; `odd` of them live in the loss world, the rest in the rtt
/// world. Returns the data dirs.
fn fleet(root: &Path, odd: usize, n: u64, days: u64) -> Vec<PathBuf> {
    (0..10)
        .map(|i| {
            let dir = root.join(format!("node{i}"));
            let world = if i < odd { loss_world() } else { rtt_world() };
            write_node(&dir, &format!("node{i}"), world, 1_000 + i as u64, n, days);
            dir
        })
        .collect()
}

fn opts(split: Split) -> Options {
    Options {
        split,
        bootstrap: 300,
        seed: 3,
    }
}

#[test]
fn unlike_nodes_pass() {
    let root = tmp("pass");
    let dirs = fleet(&root, 2, 4_000, 4);
    let nodes = load_nodes(&dirs).unwrap();
    assert_eq!(nodes.len(), 10);
    let r = analyze(&nodes, &opts(Split::HoldoutFraction { fraction: 0.25 })).unwrap();
    assert_eq!(r.metric.counted_nodes, 10);
    assert_eq!(r.metric.verdict, Verdict::Pass, "{:#?}", r.metric);
    // Exactly the two loss-world nodes gain, by a lot.
    assert_eq!(
        r.metric.gaining_nodes.len(),
        2,
        "{:?}",
        r.metric.gaining_nodes
    );
    let loss_hashes = [
        node_hash(&[7u8; 32], "node0"),
        node_hash(&[7u8; 32], "node1"),
    ];
    let (odd, alike): (Vec<_>, Vec<_>) = r
        .nodes
        .iter()
        .partition(|n| loss_hashes.contains(&n.node_hash));
    assert_eq!(odd.len(), 2);
    for n in &odd {
        assert!(r.metric.gaining_nodes.contains(&n.label));
        let g = n.gain_table.unwrap();
        assert!(g.gain > 0.25, "{}: gain {}", n.label, g.gain);
        assert!(g.lower_90.unwrap() > 0.0);
        // The copies point the same way as the weighted estimate.
        assert!(n.copy_check_table.unwrap().gain.unwrap() > 0.1);
        // The formula, written as a table, finds the same thing.
        assert!(n.gain_formula.unwrap().gain > 0.25);
        // Most of the gain is reachable at 3 quanta (the trade is 3 quanta).
        assert!(n.margin_for_90.unwrap() <= 3);
    }
    for n in &alike {
        assert!(n.gain_table.unwrap().gain.abs() < 0.05, "{}", n.label);
    }
    // A holdout-fraction run is never the experiment's verdict.
    assert!(r.small_sample);
    // The JSON report carries every number.
    let json = serde_json::to_value(&r).unwrap();
    assert_eq!(json["metric"]["verdict"], "PASS");
    assert_eq!(json["nodes"].as_array().unwrap().len(), 10);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn alike_nodes_fail_and_formula_matches() {
    let root = tmp("fail");
    let dirs = fleet(&root, 0, 3_000, 4);
    let nodes = load_nodes(&dirs).unwrap();
    let r = analyze(&nodes, &opts(Split::HoldoutFraction { fraction: 0.25 })).unwrap();
    assert_eq!(r.metric.counted_nodes, 10);
    assert_eq!(r.metric.verdict, Verdict::Fail, "{:#?}", r.metric);
    assert!(r.metric.gaining_nodes.is_empty());
    assert!(
        r.metric.formula_matches_everywhere,
        "{:?}",
        r.metric.formula_mismatch_nodes
    );
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn day_split_counts_uptime() {
    let root = tmp("days");
    // Two nodes over 8 days at 500 decisions a day; one stops after day 2.
    let a = root.join("a");
    let b = root.join("b");
    write_node(&a, "a", rtt_world(), 1, 4_000, 8);
    write_node(&b, "b", loss_world(), 2, 1_000, 2);
    let nodes = load_nodes(&[a, b]).unwrap();
    let r = analyze(
        &nodes,
        &opts(Split::Days {
            fit_days: 6,
            holdout_days: 2,
        }),
    )
    .unwrap();
    let full = r.nodes.iter().find(|n| n.fit_days == 6).unwrap();
    assert_eq!(full.holdout_days, 2);
    assert_eq!(full.fit_decisions + full.holdout_decisions, 4_000);
    assert!(full.counted);
    let short = r.nodes.iter().find(|n| n.fit_days == 2).unwrap();
    assert_eq!(short.holdout_decisions, 0);
    assert!(!short.counted);
    assert!(short
        .not_counted_reason
        .as_ref()
        .unwrap()
        .contains("uptime"));
    assert!(r.small_sample);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn tampered_traces_are_refused() {
    let root = tmp("tamper");
    let dir = root.join("node");
    write_node(&dir, "node", rtt_world(), 5, 300, 1);
    let file = measure_agent::trace::trace_files(&dir).unwrap().remove(0);
    let good = std::fs::read_to_string(&file).unwrap();

    // A changed outcome on a closed day breaks the manifest.
    let bad = good.replacen("\"rtt_us\":", "\"rtt_us\":1", 1);
    std::fs::write(&file, &bad).unwrap();
    assert!(matches!(
        load_nodes(std::slice::from_ref(&dir)),
        Err(AnalysisError::Trace { .. })
    ));

    // Without the manifest, a changed choice still fails replay.
    std::fs::remove_file(measure_agent::trace::manifest_path(&file)).unwrap();
    let bad = good.replacen("\"explored\":false", "\"explored\":true", 1);
    std::fs::write(&file, &bad).unwrap();
    assert!(matches!(
        load_nodes(std::slice::from_ref(&dir)),
        Err(AnalysisError::Trace { .. })
    ));

    // An open file with an unterminated last line (a live append) loads,
    // minus that line.
    let cut = good.len() - 40;
    std::fs::write(&file, &good[..cut]).unwrap();
    let nodes = load_nodes(std::slice::from_ref(&dir)).unwrap();
    assert_eq!(nodes[0].decisions.len(), 299);
    assert!(nodes[0].files[0].partial_line_skipped);
    assert_eq!(nodes[0].files[0].manifest, "open");
    std::fs::remove_dir_all(&root).unwrap();
}
