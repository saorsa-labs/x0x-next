//! The analysis of plan §4: fit three scorers, score them on the holdout,
//! apply the pass metric, and report the band-margin spread.
//!
//! The thresholds below are the ones the plan fixes in writing before the
//! full run; they are constants so a run cannot change them.

use std::collections::BTreeSet;

use measure_agent::decide::MARGIN;
use policy_poc::fit::{fit_formula, fit_table, Evidence, FormulaFit};
use policy_poc::{band_pick, Checker, FixedRule, PointsTable, Scorer, QUANT};
use serde::Serialize;

use crate::error::AnalysisError;
use crate::estimate::{
    bootstrap_gain, deterministic_term, estimate, explore_share, gain, paired_check, quantile,
    stochastic_term, GainEstimate, IpwEstimate, PairedCheck, Term,
};
use crate::load::{FileCheck, LoggedDecision, NodeTrace};

/// A node "gains" when its per-node scorer beats the best single table by
/// at least this share (10%).
pub const GAIN_THRESHOLD: f64 = 0.10;
/// PASS when at least this share of counted nodes gains (one tenth).
pub const PASS_SHARE: f64 = 0.10;
/// The formula "matches the table" on a node within this relative
/// difference of holdout cost (2%).
pub const MATCH_TOLERANCE: f64 = 0.02;
/// Band margins reported, in quanta.
pub const MARGIN_QUANTA: std::ops::RangeInclusive<u32> = 1..=10;
/// A node's margin is the smallest that reaches this share of its gain.
pub const REACH_TARGET: f64 = 0.90;
/// The study's worst-fitted node's gathering cost (1.2%); nodes above it are
/// named.
pub const STUDY_WORST_GATHERING: f64 = 0.012;
/// Fewer counted nodes than this is not the experiment's verdict.
pub const EXPERIMENT_MIN_NODES: usize = 10;
/// A counted node with fewer holdout decisions than this makes the run a
/// small-sample run.
pub const EXPERIMENT_MIN_HOLDOUT: usize = 1_000;
/// In a holdout-fraction run, a node needs this many holdout decisions to
/// be counted at all.
pub const MIN_HOLDOUT_DECISIONS: usize = 20;
/// Default bootstrap resamples.
pub const DEFAULT_BOOTSTRAP: usize = 1_000;
/// Schema id of the JSON report.
pub const REPORT_SCHEMA: &str = "exp16-analysis/1";

const DAY_MS: u64 = 86_400_000;
/// [`SingleChoice::chosen`] when the pooled table won.
const POOLED_TABLE: &str = "pooled fitted table";
/// [`SingleChoice::chosen`] when the fixed rule won.
const FIXED_RULE: &str = "fixed rule";

/// How decisions are split into fit and holdout.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum Split {
    /// By UTC day from the first day any node logged: days `[0, fit_days)`
    /// fit, days `[fit_days, fit_days + holdout_days)` holdout, later days
    /// unused. The real run is 21:7. A node counts only with at least 2/3
    /// of the fit days and 4/7 of the holdout days logged (14 and 4 at
    /// 21:7, plan §3 "restarts").
    Days {
        /// Fit days.
        fit_days: u32,
        /// Holdout days.
        holdout_days: u32,
    },
    /// Per node, the last `fraction` of its decisions are the holdout (pilot
    /// smoke runs). Never the experiment's verdict.
    HoldoutFraction {
        /// Holdout share in (0, 1).
        fraction: f64,
    },
}

impl Split {
    /// One-line description.
    #[must_use]
    pub fn describe(&self) -> String {
        match self {
            Self::Days {
                fit_days,
                holdout_days,
            } => format!("by UTC day, fit {fit_days} d / holdout {holdout_days} d"),
            Self::HoldoutFraction { fraction } => {
                format!(
                    "last {:.0}% of each node's decisions held out",
                    fraction * 100.0
                )
            }
        }
    }

    /// Check the parameters.
    ///
    /// # Errors
    /// [`AnalysisError::Usage`] for zero days or a fraction outside (0, 1).
    pub fn validate(&self) -> Result<(), AnalysisError> {
        match *self {
            Self::Days {
                fit_days,
                holdout_days,
            } if fit_days == 0 || holdout_days == 0 => Err(AnalysisError::Usage(
                "--split-days needs F and H both at least 1".to_string(),
            )),
            Self::HoldoutFraction { fraction } if !(fraction > 0.0 && fraction < 1.0) => Err(
                AnalysisError::Usage("--holdout-fraction must be in (0, 1)".to_string()),
            ),
            _ => Ok(()),
        }
    }
}

/// Analysis options.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Options {
    /// Fit/holdout split.
    pub split: Split,
    /// Bootstrap resamples per node.
    pub bootstrap: usize,
    /// Bootstrap seed.
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            split: Split::Days {
                fit_days: 21,
                holdout_days: 7,
            },
            bootstrap: DEFAULT_BOOTSTRAP,
            seed: 16,
        }
    }
}

/// Verdict of the pass metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Verdict {
    /// At least a tenth of counted nodes gain 10% or more.
    Pass,
    /// Fewer than a tenth do.
    Fail,
    /// No node could be counted.
    NoCountedNodes,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::NoCountedNodes => "NO COUNTED NODES",
        })
    }
}

/// Scorer 1 of plan §4: the best single table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SingleChoice {
    /// `"pooled fitted table"` or `"fixed rule"`.
    pub chosen: String,
    /// Pooled fit-period SNIPW cost of the fixed rule, ms.
    pub fit_cost_rule_ms: Option<f64>,
    /// Pooled fit-period SNIPW cost of the pooled table at the logged
    /// margin, ms.
    pub fit_cost_table_ms: Option<f64>,
    /// Evidence rows pooled.
    pub evidence_rows: usize,
    /// Whether the pooled table passed the checker (a table that fails is
    /// never chosen).
    pub table_checker_passed: bool,
    /// The pooled table.
    pub table: PointsTable,
}

/// The fitted formula's constants (scorer 2).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct FormulaSummary {
    /// Cost of the fastest, cleanest path, ms.
    pub base_ms: f64,
    /// rtt power.
    pub rtt_power: f64,
    /// rtt growth per level.
    pub rtt_growth: f64,
    /// Loss slope per level.
    pub loss_slope: f64,
    /// Delivery multipliers, levels 0..=7 (level 7 is 1).
    pub delivery_mult: [f64; 8],
    /// Training RMSE of ln cost.
    pub train_rmse: f64,
    /// Whether the implied table passed the checker.
    pub checker_passed: bool,
}

impl FormulaSummary {
    fn of(f: &FormulaFit) -> Self {
        Self {
            base_ms: f.params.base_ms,
            rtt_power: f.params.rtt_power,
            rtt_growth: f.params.rtt_growth,
            loss_slope: f.params.loss_slope,
            delivery_mult: f.delivery_mult,
            train_rmse: f.train_mse.sqrt(),
            checker_passed: Checker::check(&f.table).is_ok(),
        }
    }
}

/// One band margin's row for one node.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MarginRow {
    /// Margin in quanta.
    pub margin_quanta: u32,
    /// Gathering cost of Measure at this margin: exploring one decision in
    /// ten uniformly inside the band, against the pure fixed rule, as a
    /// share of the rule's holdout cost.
    pub gathering_cost: Option<f64>,
    /// Share of the uniform-in-band target's mass with logged support (below
    /// 1 above 6 quanta: the estimate there covers only the supported part).
    pub gathering_coverage: f64,
    /// Gain of the per-node table over the rule at this margin.
    pub table_gain_vs_rule: Option<f64>,
    /// That gain as a share of the gain at the logged margin (6 quanta).
    pub reach: Option<f64>,
    /// Share of the per-node table's picks with logged support.
    pub table_coverage: f64,
}

/// Per-node results.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NodeResult {
    /// Display label.
    pub label: String,
    /// Salted node hash.
    pub node_hash: String,
    /// Node class.
    pub node_class: Option<String>,
    /// Fit decisions.
    pub fit_decisions: usize,
    /// Holdout decisions.
    pub holdout_decisions: usize,
    /// Distinct UTC days with fit decisions.
    pub fit_days: usize,
    /// Distinct UTC days with holdout decisions.
    pub holdout_days: usize,
    /// Holdout picks that timed out.
    pub holdout_timeouts: usize,
    /// Holdout decisions with a copy.
    pub holdout_copies: usize,
    /// Whether the node counts toward the metric.
    pub counted: bool,
    /// Why not, when it does not.
    pub not_counted_reason: Option<String>,
    /// Holdout SNIPW cost of the fixed rule.
    pub cost_rule: IpwEstimate,
    /// Holdout SNIPW cost of the best single table.
    pub cost_single: IpwEstimate,
    /// Holdout SNIPW cost of the formula with per-node constants.
    pub cost_formula: IpwEstimate,
    /// Holdout SNIPW cost of the per-node table.
    pub cost_table: IpwEstimate,
    /// Gain of the per-node table over the best single table (the headline
    /// per-node gain).
    pub gain_table: Option<GainEstimate>,
    /// Gain of the formula over the best single table.
    pub gain_formula: Option<GainEstimate>,
    /// The better of the two point gains.
    pub gain_better_of: Option<f64>,
    /// True when the per-node table gains at least 10% with a 90% lower
    /// bound above zero.
    pub gains_by_table: bool,
    /// True when either per-node scorer does.
    pub gains_by_better_of: bool,
    /// `(C_formula − C_table) / C_table`.
    pub formula_vs_table: Option<f64>,
    /// Whether the formula is within 2% of the per-node table.
    pub formula_matches_table: Option<bool>,
    /// Copy cross-check: single table vs per-node table.
    pub copy_check_table: Option<PairedCheck>,
    /// Copy cross-check: single table vs formula.
    pub copy_check_formula: Option<PairedCheck>,
    /// Fitted formula constants.
    pub formula: FormulaSummary,
    /// The per-node table.
    pub node_table: PointsTable,
    /// Whether the per-node table passed the checker.
    pub node_table_checker_passed: bool,
    /// Band margins 1..=10 quanta.
    pub margins: Vec<MarginRow>,
    /// Smallest margin reaching 90% of the per-node table's gain.
    pub margin_for_90: Option<u32>,
    /// Gathering cost at that margin.
    pub gathering_at_margin_for_90: Option<f64>,
}

/// The pass metric.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Metric {
    /// Nodes that count.
    pub counted_nodes: usize,
    /// Counted nodes whose per-node table gains ≥ 10% with lower bound > 0.
    pub gaining_nodes: Vec<String>,
    /// `gaining / counted`.
    pub share: f64,
    /// Headline verdict (per-node table vs best single table).
    pub verdict: Verdict,
    /// Counted nodes where the better of formula and table gains.
    pub gaining_nodes_better_of: Vec<String>,
    /// Verdict on the better of formula and table.
    pub verdict_better_of: Verdict,
    /// Formula within 2% of the table on every counted node.
    pub formula_matches_everywhere: bool,
    /// Counted nodes where it is not.
    pub formula_mismatch_nodes: Vec<String>,
    /// Fixed thresholds, repeated for the record.
    pub gain_threshold: f64,
    /// See [`PASS_SHARE`].
    pub pass_share: f64,
    /// See [`MATCH_TOLERANCE`].
    pub match_tolerance: f64,
}

/// Distribution of per-node gathering cost at each node's margin for 90%.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MarginSpread {
    /// Counted nodes with a margin for 90% (nodes with no gain have none).
    pub nodes: usize,
    /// Median gathering cost.
    pub median: Option<f64>,
    /// 90th percentile.
    pub p90: Option<f64>,
    /// Worst.
    pub worst: Option<f64>,
    /// Node with the worst.
    pub worst_node: Option<String>,
    /// Nodes above the study's 1.2%.
    pub above_study_worst: Vec<String>,
    /// Median margin for 90%, quanta.
    pub median_margin: Option<f64>,
}

/// The whole analysis.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    /// [`REPORT_SCHEMA`].
    pub schema: String,
    /// Options used.
    pub options: Options,
    /// True unless this run can be the experiment's verdict.
    pub small_sample: bool,
    /// Why it is a small-sample run.
    pub small_sample_reasons: Vec<String>,
    /// Nodes loaded.
    pub nodes_loaded: usize,
    /// Decisions loaded.
    pub decisions_total: usize,
    /// Fit decisions, all nodes.
    pub fit_decisions_total: usize,
    /// Holdout decisions, all nodes.
    pub holdout_decisions_total: usize,
    /// Scorer 1.
    pub single: SingleChoice,
    /// Per node.
    pub nodes: Vec<NodeResult>,
    /// The pass metric.
    pub metric: Metric,
    /// Band margin spread.
    pub margin_spread: MarginSpread,
    /// How costs were estimated.
    pub estimator: String,
    /// Where this implementation departs from plan §4.
    pub deviations: Vec<String>,
    /// Every trace file and how it verified.
    pub files: Vec<FileCheck>,
}

/// Candidate indices inside the rule's band at `margin`.
#[must_use]
pub fn band(d: &LoggedDecision, margin: i64) -> Vec<usize> {
    let scores: Vec<i64> = d
        .candidates
        .iter()
        .map(|c| FixedRule.score(c.features))
        .collect();
    let Some(&best) = scores.iter().max() else {
        return Vec::new();
    };
    let floor = best.saturating_sub(margin.max(0));
    (0..scores.len()).filter(|&i| scores[i] >= floor).collect()
}

/// A scorer's pick on a logged decision: the rule's choice for `None`,
/// otherwise the table's best inside the band at `margin`.
#[must_use]
pub fn pick_with(d: &LoggedDecision, table: Option<&PointsTable>, margin: i64) -> usize {
    match table {
        None => d.rule_choice,
        Some(t) => band_pick(&FixedRule, Some(t), &d.candidates, margin)
            .chosen
            .unwrap_or(d.rule_choice),
    }
}

/// Fit evidence from decisions: the pick's features and cost, plus the
/// copy's when one was sent (both are real observations of that peer).
#[must_use]
pub fn evidence(ds: &[&LoggedDecision]) -> Vec<Evidence> {
    let mut out = Vec::with_capacity(ds.len());
    for d in ds {
        if let Some(c) = d.candidates.get(d.pick) {
            out.push(Evidence {
                features: c.features,
                cost: d.pick_cost_ms,
            });
        }
        if let Some((i, cost)) = d.copy {
            if let Some(c) = d.candidates.get(i) {
                out.push(Evidence {
                    features: c.features,
                    cost,
                });
            }
        }
    }
    out
}

/// Terms of a deterministic scorer over decisions.
fn terms_of(
    ds: &[&LoggedDecision],
    table: Option<&PointsTable>,
    margin: i64,
) -> (Vec<Term>, Vec<usize>) {
    let mut terms = Vec::with_capacity(ds.len());
    let mut picks = Vec::with_capacity(ds.len());
    for d in ds {
        let p = pick_with(d, table, margin);
        terms.push(deterministic_term(d, p, &band(d, MARGIN)));
        picks.push(p);
    }
    (terms, picks)
}

fn day_of(d: &LoggedDecision) -> u64 {
    d.utc_ms / DAY_MS
}

fn distinct_days(ds: &[&LoggedDecision]) -> usize {
    ds.iter().map(|d| day_of(d)).collect::<BTreeSet<_>>().len()
}

struct NodeSplit<'a> {
    fit: Vec<&'a LoggedDecision>,
    holdout: Vec<&'a LoggedDecision>,
}

fn split_node(node: &NodeTrace, split: Split, start_day: u64) -> NodeSplit<'_> {
    match split {
        Split::Days {
            fit_days,
            holdout_days,
        } => {
            let (f, h) = (u64::from(fit_days), u64::from(holdout_days));
            let mut fit = Vec::new();
            let mut holdout = Vec::new();
            for d in &node.decisions {
                let day = day_of(d).saturating_sub(start_day);
                if day < f {
                    fit.push(d);
                } else if day < f + h {
                    holdout.push(d);
                }
            }
            NodeSplit { fit, holdout }
        }
        Split::HoldoutFraction { fraction } => {
            let n = node.decisions.len();
            // fraction is in (0, 1), so the product is within 0..=n.
            let hold = ((n as f64) * fraction).round() as usize;
            let cut = n.saturating_sub(hold);
            NodeSplit {
                fit: node.decisions[..cut].iter().collect(),
                holdout: node.decisions[cut..].iter().collect(),
            }
        }
    }
}

fn best_single(fit_all: &[&LoggedDecision]) -> SingleChoice {
    let ev = evidence(fit_all);
    let table = fit_table(&ev);
    let checker_ok = Checker::check(&table).is_ok();
    let rule = estimate(&terms_of(fit_all, None, MARGIN).0).mean_cost_ms;
    let tab = estimate(&terms_of(fit_all, Some(&table), MARGIN).0).mean_cost_ms;
    let table_wins = checker_ok && matches!((rule, tab), (Some(r), Some(t)) if t < r);
    SingleChoice {
        chosen: if table_wins { POOLED_TABLE } else { FIXED_RULE }.to_string(),
        fit_cost_rule_ms: rule,
        fit_cost_table_ms: tab,
        evidence_rows: ev.len(),
        table_checker_passed: checker_ok,
        table,
    }
}

fn margin_rows(
    holdout: &[&LoggedDecision],
    rule_cost: Option<f64>,
    node_table: &PointsTable,
) -> (Vec<MarginRow>, Option<u32>, Option<f64>) {
    let logged: Vec<Vec<usize>> = holdout.iter().map(|d| band(d, MARGIN)).collect();
    let gain_at = |margin: i64| -> (Option<f64>, f64) {
        let terms: Vec<Term> = holdout
            .iter()
            .zip(&logged)
            .map(|(d, el)| deterministic_term(d, pick_with(d, Some(node_table), margin), el))
            .collect();
        let e = estimate(&terms);
        let g = match (rule_cost, e.mean_cost_ms) {
            (Some(r), Some(c)) => gain(r, c),
            _ => None,
        };
        (g, e.coverage)
    };
    let (full_gain, _) = gain_at(MARGIN);
    let mut rows = Vec::new();
    for m in MARGIN_QUANTA {
        let margin = i64::from(m) * QUANT;
        let terms: Vec<Term> = holdout
            .iter()
            .zip(&logged)
            .map(|(d, el)| {
                let inside = band(d, margin);
                let mut pi = vec![0.0; d.candidates.len()];
                for &i in &inside {
                    pi[i] = 1.0 / inside.len() as f64;
                }
                stochastic_term(d, &pi, el)
            })
            .collect();
        let uni = estimate(&terms);
        let gathering = match (rule_cost, uni.mean_cost_ms) {
            (Some(r), Some(u)) if r > 0.0 => Some(explore_share() * (u - r) / r),
            _ => None,
        };
        let (g, table_coverage) = gain_at(margin);
        let reach = match (g, full_gain) {
            (Some(g), Some(f)) if f > 0.0 => Some(g / f),
            _ => None,
        };
        rows.push(MarginRow {
            margin_quanta: m,
            gathering_cost: gathering,
            gathering_coverage: uni.coverage,
            table_gain_vs_rule: g,
            reach,
            table_coverage,
        });
    }
    let chosen = rows
        .iter()
        .find(|r| r.reach.is_some_and(|x| x >= REACH_TARGET))
        .map(|r| (r.margin_quanta, r.gathering_cost));
    (rows, chosen.map(|c| c.0), chosen.and_then(|c| c.1))
}

fn count_status(split: Split, ns: &NodeSplit<'_>) -> Option<String> {
    match split {
        Split::Days {
            fit_days,
            holdout_days,
        } => {
            let need_fit = (u64::from(fit_days) * 2).div_ceil(3);
            let need_hold = (u64::from(holdout_days) * 4).div_ceil(7);
            let (fd, hd) = (
                distinct_days(&ns.fit) as u64,
                distinct_days(&ns.holdout) as u64,
            );
            (fd < need_fit || hd < need_hold).then(|| {
                format!(
                    "uptime: {fd} fit days (need {need_fit}), {hd} holdout days (need {need_hold})"
                )
            })
        }
        Split::HoldoutFraction { .. } => (ns.holdout.len() < MIN_HOLDOUT_DECISIONS).then(|| {
            format!(
                "{} holdout decisions (need {MIN_HOLDOUT_DECISIONS})",
                ns.holdout.len()
            )
        }),
    }
}

fn analyze_node(
    node: &NodeTrace,
    ns: &NodeSplit<'_>,
    single: &SingleChoice,
    opts: &Options,
    index: u64,
) -> NodeResult {
    let ev = evidence(&ns.fit);
    let node_table = fit_table(&ev);
    let formula = fit_formula(&ev);
    let table_ok = Checker::check(&node_table).is_ok();
    let summary = FormulaSummary::of(&formula);
    let single_table = (single.chosen == POOLED_TABLE).then_some(&single.table);

    let (rule_terms, _) = terms_of(&ns.holdout, None, MARGIN);
    let (single_terms, single_picks) = terms_of(&ns.holdout, single_table, MARGIN);
    let (formula_terms, formula_picks) = terms_of(&ns.holdout, Some(&formula.table), MARGIN);
    let (table_terms, table_picks) = terms_of(&ns.holdout, Some(&node_table), MARGIN);
    let cost_rule = estimate(&rule_terms);
    let cost_single = estimate(&single_terms);
    let cost_formula = estimate(&formula_terms);
    let cost_table = estimate(&table_terms);

    let seed = opts.seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let gain_table = bootstrap_gain(&single_terms, &table_terms, opts.bootstrap, seed);
    let gain_formula = bootstrap_gain(&single_terms, &formula_terms, opts.bootstrap, seed ^ 1);
    let gains = |g: &Option<GainEstimate>| {
        g.is_some_and(|g| g.gain >= GAIN_THRESHOLD && g.lower_90.is_some_and(|lb| lb > 0.0))
    };
    let gains_by_table = gains(&gain_table);
    let gains_by_better_of = gains_by_table || gains(&gain_formula);
    let gain_better_of = match (gain_table, gain_formula) {
        (Some(a), Some(b)) => Some(a.gain.max(b.gain)),
        (Some(a), None) => Some(a.gain),
        (None, Some(b)) => Some(b.gain),
        (None, None) => None,
    };
    let formula_vs_table = match (cost_formula.mean_cost_ms, cost_table.mean_cost_ms) {
        (Some(f), Some(t)) if t > 0.0 => Some((f - t) / t),
        _ => None,
    };

    let mut not_counted = count_status(opts.split, ns);
    if not_counted.is_none() && !(table_ok && summary.checker_passed) {
        not_counted = Some("a fitted table failed the checker".to_string());
    }
    if not_counted.is_none() && (gain_table.is_none() || cost_formula.mean_cost_ms.is_none()) {
        not_counted = Some("no holdout estimate (no matching decisions)".to_string());
    }

    let (margins, margin_for_90, gathering_at) =
        margin_rows(&ns.holdout, cost_rule.mean_cost_ms, &node_table);

    NodeResult {
        label: node.label(),
        node_hash: node.node_hash.clone(),
        node_class: node.node_class.clone(),
        fit_decisions: ns.fit.len(),
        holdout_decisions: ns.holdout.len(),
        fit_days: distinct_days(&ns.fit),
        holdout_days: distinct_days(&ns.holdout),
        holdout_timeouts: ns.holdout.iter().filter(|d| d.pick_failed).count(),
        holdout_copies: ns.holdout.iter().filter(|d| d.copy.is_some()).count(),
        counted: not_counted.is_none(),
        not_counted_reason: not_counted,
        cost_rule,
        cost_single,
        cost_formula,
        cost_table,
        gain_table,
        gain_formula,
        gain_better_of,
        gains_by_table,
        gains_by_better_of,
        formula_vs_table,
        formula_matches_table: formula_vs_table.map(|x| x.abs() <= MATCH_TOLERANCE),
        copy_check_table: paired_check(&ns.holdout, &single_picks, &table_picks),
        copy_check_formula: paired_check(&ns.holdout, &single_picks, &formula_picks),
        formula: summary,
        node_table,
        node_table_checker_passed: table_ok,
        margins,
        margin_for_90,
        gathering_at_margin_for_90: gathering_at,
    }
}

fn verdict(gaining: usize, counted: usize) -> Verdict {
    if counted == 0 {
        Verdict::NoCountedNodes
    } else if gaining as f64 >= PASS_SHARE * counted as f64 {
        Verdict::Pass
    } else {
        Verdict::Fail
    }
}

fn metric(nodes: &[NodeResult]) -> Metric {
    let counted: Vec<&NodeResult> = nodes.iter().filter(|n| n.counted).collect();
    let gaining: Vec<String> = counted
        .iter()
        .filter(|n| n.gains_by_table)
        .map(|n| n.label.clone())
        .collect();
    let gaining_b: Vec<String> = counted
        .iter()
        .filter(|n| n.gains_by_better_of)
        .map(|n| n.label.clone())
        .collect();
    let mismatch: Vec<String> = counted
        .iter()
        .filter(|n| n.formula_matches_table != Some(true))
        .map(|n| n.label.clone())
        .collect();
    let c = counted.len();
    Metric {
        counted_nodes: c,
        share: if c == 0 {
            0.0
        } else {
            gaining.len() as f64 / c as f64
        },
        verdict: verdict(gaining.len(), c),
        verdict_better_of: verdict(gaining_b.len(), c),
        gaining_nodes: gaining,
        gaining_nodes_better_of: gaining_b,
        formula_matches_everywhere: c > 0 && mismatch.is_empty(),
        formula_mismatch_nodes: mismatch,
        gain_threshold: GAIN_THRESHOLD,
        pass_share: PASS_SHARE,
        match_tolerance: MATCH_TOLERANCE,
    }
}

fn spread(nodes: &[NodeResult]) -> MarginSpread {
    let mut pts: Vec<(f64, &NodeResult)> = nodes
        .iter()
        .filter(|n| n.counted)
        .filter_map(|n| n.gathering_at_margin_for_90.map(|g| (g, n)))
        .collect();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let vals: Vec<f64> = pts.iter().map(|p| p.0).collect();
    let mut margins: Vec<f64> = pts
        .iter()
        .filter_map(|p| p.1.margin_for_90.map(f64::from))
        .collect();
    margins.sort_by(f64::total_cmp);
    MarginSpread {
        nodes: vals.len(),
        median: quantile(&vals, 0.5),
        p90: quantile(&vals, 0.9),
        worst: vals.last().copied(),
        worst_node: pts.last().map(|p| p.1.label.clone()),
        above_study_worst: pts
            .iter()
            .filter(|p| p.0 > STUDY_WORST_GATHERING)
            .map(|p| p.1.label.clone())
            .collect(),
        median_margin: quantile(&margins, 0.5),
    }
}

const ESTIMATOR_NOTE: &str = "Holdout cost per scorer = self-normalized inverse-propensity \
weighting over the logged picks: weight 1/propensity where the scorer's pick inside the band \
at 6 quanta equals the logged pick, 0 otherwise. Timeouts cost the request timeout. Gains are \
(C_single - C_per_node) / C_single; the lower bound is the 5th percentile of a percentile \
bootstrap over holdout decisions (both scorers resampled together). Logged copies give an \
unweighted paired cross-check on duplicated decisions; it is a sanity check, not an unbiased \
estimate.";

/// Run the whole analysis on loaded nodes.
///
/// # Errors
/// [`AnalysisError::Usage`] for invalid options, [`AnalysisError::NoData`]
/// when no decision falls in the fit window.
pub fn analyze(nodes: &[NodeTrace], opts: &Options) -> Result<Report, AnalysisError> {
    opts.split.validate()?;
    let decisions_total: usize = nodes.iter().map(|n| n.decisions.len()).sum();
    if decisions_total == 0 {
        return Err(AnalysisError::NoData(
            "no decisions in the traces given".to_string(),
        ));
    }
    let start_day = nodes
        .iter()
        .filter_map(|n| n.decisions.iter().map(day_of).min())
        .min()
        .unwrap_or(0);
    let splits: Vec<NodeSplit<'_>> = nodes
        .iter()
        .map(|n| split_node(n, opts.split, start_day))
        .collect();
    let fit_all: Vec<&LoggedDecision> = splits.iter().flat_map(|s| s.fit.iter().copied()).collect();
    if fit_all.is_empty() {
        return Err(AnalysisError::NoData(
            "no decisions in the fit window".to_string(),
        ));
    }
    let single = best_single(&fit_all);
    let results: Vec<NodeResult> = nodes
        .iter()
        .zip(&splits)
        .enumerate()
        .map(|(i, (n, s))| analyze_node(n, s, &single, opts, i as u64))
        .collect();
    let metric = metric(&results);

    let mut reasons = Vec::new();
    if matches!(opts.split, Split::HoldoutFraction { .. }) {
        reasons.push(
            "holdout-fraction split (a smoke run), not the plan's weeks 1-3 / week 4 split"
                .to_string(),
        );
    }
    if nodes.len() == 1 {
        reasons.push(
            "single node: the pooled table is fitted on the same node, so per-node gains are near zero by construction"
                .to_string(),
        );
    }
    if metric.counted_nodes < EXPERIMENT_MIN_NODES {
        reasons.push(format!(
            "{} counted nodes (the experiment needs at least {EXPERIMENT_MIN_NODES}; the plan runs about 50)",
            metric.counted_nodes
        ));
    }
    let thin = results
        .iter()
        .filter(|r| r.counted && r.holdout_decisions < EXPERIMENT_MIN_HOLDOUT)
        .count();
    if thin > 0 {
        reasons.push(format!(
            "{thin} counted node(s) with fewer than {EXPERIMENT_MIN_HOLDOUT} holdout decisions"
        ));
    }

    let files = nodes.iter().flat_map(|n| n.files.iter().cloned()).collect();
    Ok(Report {
        schema: REPORT_SCHEMA.to_string(),
        options: *opts,
        small_sample: !reasons.is_empty(),
        small_sample_reasons: reasons,
        nodes_loaded: nodes.len(),
        decisions_total,
        fit_decisions_total: splits.iter().map(|s| s.fit.len()).sum(),
        holdout_decisions_total: splits.iter().map(|s| s.holdout.len()).sum(),
        single,
        margin_spread: spread(&results),
        nodes: results,
        metric,
        estimator: ESTIMATOR_NOTE.to_string(),
        deviations: vec![
            "Formula scorer fits only the world-one (product of one-feature terms) form; the plan's \
             second form (world two, rtt x loss timeout) and the 'which world kind fits better' \
             side result are not implemented."
                .to_string(),
            "The secondary split (scoring on each week in turn) is not implemented.".to_string(),
            "Gathering cost counts exploration only; copies add requests (holdout_copies), not \
             decision latency."
                .to_string(),
        ],
        files,
    })
}
