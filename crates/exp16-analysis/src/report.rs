//! Human-readable rendering of a [`Report`].

use std::fmt::Write as _;

use crate::analysis::{Report, EXPERIMENT_MIN_NODES, GAIN_THRESHOLD, MATCH_TOLERANCE, PASS_SHARE};
use crate::estimate::IpwEstimate;

fn ms(x: Option<f64>) -> String {
    x.map_or_else(|| "-".to_string(), |v| format!("{v:.2}"))
}

fn pct(x: Option<f64>) -> String {
    x.map_or_else(|| "-".to_string(), |v| format!("{:+.1}%", v * 100.0))
}

fn est(e: &IpwEstimate) -> String {
    format!(
        "{} ms (matched {}/{}, ess {:.0})",
        ms(e.mean_cost_ms),
        e.matched,
        e.decisions,
        e.effective_sample_size
    )
}

/// Render the report as text for stdout.
#[must_use]
pub fn render(r: &Report) -> String {
    let mut s = String::new();
    // Writing to a String cannot fail; the results are ignored.
    let _ = writeln!(s, "Experiment 16 analysis ({})", r.schema);
    let _ = writeln!(s, "=================================");
    if r.small_sample {
        let _ = writeln!(
            s,
            "\n*** SMALL-SAMPLE RUN: NOT the experiment's verdict. ***"
        );
        for reason in &r.small_sample_reasons {
            let _ = writeln!(s, "    - {reason}");
        }
    }
    let _ = writeln!(
        s,
        "\nSplit: {}. Bootstrap: {} resamples, seed {}.",
        r.options.split.describe(),
        r.options.bootstrap,
        r.options.seed
    );
    let _ = writeln!(
        s,
        "Loaded {} node(s), {} decisions ({} fit, {} holdout) from {} verified file(s).",
        r.nodes_loaded,
        r.decisions_total,
        r.fit_decisions_total,
        r.holdout_decisions_total,
        r.files.len()
    );
    for f in &r.files {
        let _ = writeln!(
            s,
            "    {}: {} decisions, {} segment(s), manifest {}{}",
            f.file,
            f.decisions,
            f.segments,
            f.manifest,
            if f.partial_line_skipped {
                ", unterminated last line set aside"
            } else {
                ""
            }
        );
    }

    let _ = writeln!(s, "\nBest single table: {}", r.single.chosen);
    let _ = writeln!(
        s,
        "    pooled fit-period cost: fixed rule {} ms, pooled table {} ms ({} evidence rows, checker {})",
        ms(r.single.fit_cost_rule_ms),
        ms(r.single.fit_cost_table_ms),
        r.single.evidence_rows,
        if r.single.table_checker_passed { "ok" } else { "FAILED" }
    );

    for n in &r.nodes {
        let _ = writeln!(s, "\nNode {}", n.label);
        let _ = writeln!(
            s,
            "    fit {} decisions / {} day(s); holdout {} decisions / {} day(s); {} timeouts, {} copies",
            n.fit_decisions, n.fit_days, n.holdout_decisions, n.holdout_days, n.holdout_timeouts, n.holdout_copies
        );
        let _ = writeln!(
            s,
            "    counted: {}",
            match &n.not_counted_reason {
                None => "yes".to_string(),
                Some(why) => format!("no ({why})"),
            }
        );
        let _ = writeln!(s, "    holdout cost  rule    {}", est(&n.cost_rule));
        let _ = writeln!(s, "                  single  {}", est(&n.cost_single));
        let _ = writeln!(s, "                  formula {}", est(&n.cost_formula));
        let _ = writeln!(s, "                  table   {}", est(&n.cost_table));
        for (name, g) in [("table", &n.gain_table), ("formula", &n.gain_formula)] {
            let _ = writeln!(
                s,
                "    gain of {name:7} over single: {} (90% interval {} .. {}, {} resamples)",
                pct(g.map(|g| g.gain)),
                pct(g.and_then(|g| g.lower_90)),
                pct(g.and_then(|g| g.upper_90)),
                g.map_or(0, |g| g.resamples)
            );
        }
        let _ = writeln!(
            s,
            "    formula vs table: {} ({})",
            pct(n.formula_vs_table),
            match n.formula_matches_table {
                Some(true) => "matches",
                Some(false) => "does not match",
                None => "n/a",
            }
        );
        for (name, c) in [
            ("table", &n.copy_check_table),
            ("formula", &n.copy_check_formula),
        ] {
            match c {
                Some(c) => {
                    let _ = writeln!(
                        s,
                        "    copy check single vs {name}: {} pairs ({} disagree), {:.2} vs {:.2} ms, gain {}",
                        c.pairs,
                        c.disagreements,
                        c.mean_base_ms,
                        c.mean_chal_ms,
                        pct(c.gain)
                    );
                }
                None => {
                    let _ = writeln!(s, "    copy check single vs {name}: no observed pairs");
                }
            }
        }
        let f = &n.formula;
        let _ = writeln!(
            s,
            "    formula: base {:.2} ms, rtt power {:.3}, rtt growth {:.3}, loss slope {:.3}, rmse {:.3}",
            f.base_ms, f.rtt_power, f.rtt_growth, f.loss_slope, f.train_rmse
        );
        let _ = writeln!(s, "    margin  gathering  coverage  table gain  reach");
        for m in &n.margins {
            let _ = writeln!(
                s,
                "    {:>6}  {:>9}  {:>8.2}  {:>10}  {:>5}",
                m.margin_quanta,
                pct(m.gathering_cost),
                m.gathering_coverage,
                pct(m.table_gain_vs_rule),
                m.reach
                    .map_or_else(|| "-".to_string(), |x| format!("{x:.2}"))
            );
        }
        let _ = writeln!(
            s,
            "    margin for 90% of gain: {} (gathering {})",
            n.margin_for_90
                .map_or_else(|| "-".to_string(), |m| format!("{m} quanta")),
            pct(n.gathering_at_margin_for_90)
        );
    }

    let m = &r.metric;
    let _ = writeln!(s, "\nPass metric (fixed: gain >= {:.0}% with 90% lower bound > 0, on >= {:.0}% of counted nodes)", GAIN_THRESHOLD * 100.0, PASS_SHARE * 100.0);
    let _ = writeln!(
        s,
        "    per-node table: {} of {} counted nodes gain -> share {:.1}% -> {}",
        m.gaining_nodes.len(),
        m.counted_nodes,
        m.share * 100.0,
        m.verdict
    );
    let _ = writeln!(
        s,
        "    better of formula and table: {} of {} -> {}",
        m.gaining_nodes_better_of.len(),
        m.counted_nodes,
        m.verdict_better_of
    );
    let _ = writeln!(
        s,
        "    formula matches table (within {:.0}%) on every counted node: {}{}",
        MATCH_TOLERANCE * 100.0,
        if m.formula_matches_everywhere {
            "yes"
        } else {
            "no"
        },
        if m.formula_mismatch_nodes.is_empty() {
            String::new()
        } else {
            format!(" (not on: {})", m.formula_mismatch_nodes.join(", "))
        }
    );
    let sp = &r.margin_spread;
    let _ = writeln!(
        s,
        "\nMargin spread over {} counted node(s): gathering median {}, p90 {}, worst {}{}; median margin {}",
        sp.nodes,
        pct(sp.median),
        pct(sp.p90),
        pct(sp.worst),
        sp.worst_node.as_ref().map_or_else(String::new, |w| format!(" ({w})")),
        sp.median_margin.map_or_else(|| "-".to_string(), |x| format!("{x:.0} quanta"))
    );
    if !sp.above_study_worst.is_empty() {
        let _ = writeln!(
            s,
            "    above the study's 1.2%: {}",
            sp.above_study_worst.join(", ")
        );
    }
    let _ = writeln!(s, "\nEstimator: {}", r.estimator);
    let _ = writeln!(s, "Deviations from plan section 4:");
    for d in &r.deviations {
        let _ = writeln!(s, "    - {d}");
    }
    if r.small_sample {
        let _ = writeln!(
            s,
            "\nVERDICT (small sample, NOT the experiment's verdict): {}",
            m.verdict
        );
    } else {
        let _ = writeln!(
            s,
            "\nVERDICT: {} ({} counted nodes, at least {EXPERIMENT_MIN_NODES} required)",
            m.verdict, m.counted_nodes
        );
    }
    s
}
