//! The six-step evidence log: the full life of a weight patch (design study
//! §5; poc-spec §10).
//!
//! [`run`] performs every step and returns typed results plus printable
//! lines. The `poc-demo` binary prints the lines; the tests assert on the
//! typed results. Every step is deterministic (fixed seeds, fixed signing
//! key seed). Signatures use ML-DSA-65, so this module needs the `pqc`
//! feature.

use crate::band::band_pick;
use crate::checker::{Checker, DiffReport};
use crate::error::PolicyError;
use crate::features::{Feature, Features, INPUT_SPACE};
use crate::fit::{fit_table, log_evidence};
use crate::patch::{receive_patch, Patch, PatchDecision, PatchKeypair, SignedPatch};
use crate::rule::{FixedRule, FnScorer, Scorer, QUANT};
use crate::table::{FeatureSteps, PointsTable};
use crate::trial::{TrialConfig, TrialReport, DEFAULT_MARGIN};
use crate::twin::{mean_expected_cost, Twin, WorldParams, DEFAULT_CANDIDATES};
use crate::watch::{DecisionPoint, Removal, WatchConfig, WatchEvent};

/// Seed of the demo's signing key.
pub const KEY_SEED: [u8; 32] = [7u8; 32];
/// Seed for the logged evidence the step-5 patch is fitted on.
pub const FIT_SEED: u64 = 1;
/// Seed for the evidence the step-4 "turns bad" patch is fitted on.
pub const DRIFT_FIT_SEED: u64 = 2;
/// Seed of the receiving node's trial twin.
pub const TRIAL_SEED: u64 = 5;
/// Seed of the post-adoption watch twin.
pub const WATCH_SEED: u64 = 11;
/// Seed for twin evaluations (rule vs fitted vs oracle).
pub const EVAL_SEED: u64 = 99;
/// Evidence decisions logged before fitting.
pub const EVIDENCE_DECISIONS: usize = 50_000;
/// Decisions the watch runs for in the survival demo.
pub const WATCH_DECISIONS: usize = 30_000;
/// Decisions used for twin cost evaluations.
pub const EVAL_DECISIONS: usize = 20_000;

/// Step 1: the fixed rule written as weights.
#[derive(Debug, Clone, PartialEq)]
pub struct Step1 {
    /// The rule as a table.
    pub table: PointsTable,
    /// Inputs compared.
    pub inputs_checked: usize,
    /// Inputs where table and rule disagree (must be 0).
    pub mismatches: usize,
}

/// Step 2: "among ties, prefer the peer known longer".
#[derive(Debug, Clone, PartialEq)]
pub struct Step2 {
    /// The edited patch.
    pub patch: Patch,
    /// Whether it passed the static checks.
    pub check_passed: bool,
    /// Exact difference against the rule.
    pub diff: DiffReport,
    /// For every newly ordered example pair: primary scores are equal.
    pub examples_are_primary_ties: bool,
}

/// Step 3: a trigger patch.
#[derive(Debug, Clone, PartialEq)]
pub struct Step3 {
    /// The hostile patch.
    pub patch: Patch,
    /// The input the trigger targets.
    pub trigger: Features,
    /// Score of the trigger input under the patch minus under the rule.
    pub trigger_bonus: i64,
    /// What the receiving node decided.
    pub decision: PatchDecision,
}

/// Step 4: a statically valid but worse patch, and one that turns bad.
#[derive(Debug, Clone, PartialEq)]
pub struct Step4 {
    /// The statically valid, worse patch.
    pub worse: Patch,
    /// What the receiving node decided about it.
    pub worse_decision: PatchDecision,
    /// The patch that wins its trial and later turns bad.
    pub drifting: Patch,
    /// What the receiving node decided about it.
    pub drifting_decision: PatchDecision,
    /// Decisions run after the world shifted until removal.
    pub decisions_until_removal: usize,
    /// The watch's removal record.
    pub removal: Option<Removal>,
    /// After removal, every decision used the fixed rule's pick.
    pub rule_restored: bool,
}

/// Step 5: a fitted patch in world one.
#[derive(Debug, Clone, PartialEq)]
pub struct Step5 {
    /// The fitted patch.
    pub patch: Patch,
    /// What the receiving node decided.
    pub decision: PatchDecision,
    /// Whether the policy was still installed after the watch period.
    pub survived_watch: bool,
    /// Watch samples taken.
    pub watch_samples: u64,
    /// Mean realized cost under the policy during the watch, ms.
    pub watch_policy_mean: f64,
    /// Mean realized cost the rule would have had, ms.
    pub watch_rule_mean: f64,
    /// World one: mean expected cost of rule, banded fitted, oracle.
    pub world_one: (f64, f64, f64),
    /// World two: mean expected cost of rule, banded fitted, oracle (fitted
    /// on world-two evidence).
    pub world_two: (f64, f64, f64),
    /// World one, no band: rule, fitted, oracle.
    pub world_one_unbanded: (f64, f64, f64),
    /// World two, no band: rule, fitted, oracle.
    pub world_two_unbanded: (f64, f64, f64),
}

/// Step 6: the band bounds an awful policy.
#[derive(Debug, Clone, PartialEq)]
pub struct Step6 {
    /// Band margin used.
    pub margin: i64,
    /// Decisions simulated.
    pub decisions: usize,
    /// Largest `rule_best − rule(chosen)` seen with the band (≤ margin).
    pub max_shortfall: i64,
    /// Mean expected cost: rule, awful-with-band, awful-without-band.
    pub mean_costs: (f64, f64, f64),
    /// Worst per-decision expected-cost ratio to the rule's pick: with band,
    /// without band.
    pub worst_ratio: (f64, f64),
}

/// All six steps.
#[derive(Debug, Clone, PartialEq)]
pub struct DemoReport {
    /// Size of the encoded ML-DSA-65 verification key.
    pub public_key_len: usize,
    /// Step 1.
    pub step1: Step1,
    /// Step 2.
    pub step2: Step2,
    /// Step 3.
    pub step3: Step3,
    /// Step 4.
    pub step4: Step4,
    /// Step 5.
    pub step5: Step5,
    /// Step 6.
    pub step6: Step6,
    /// The evidence log, one line per entry.
    pub lines: Vec<String>,
}

/// Run all six steps.
///
/// # Errors
/// Any [`PolicyError`] from serialization, signing or configuration.
pub fn run() -> Result<DemoReport, PolicyError> {
    let key = PatchKeypair::from_seed(&KEY_SEED);
    let trial_cfg = TrialConfig::default();
    let mut lines = vec![
        "policy-poc evidence log: the life of a weight patch (design study §5)".to_string(),
        format!(
            "signing key: ML-DSA-65, {} byte verification key; band margin {} ({} quanta)",
            key.public_bytes().len(),
            DEFAULT_MARGIN,
            DEFAULT_MARGIN / QUANT
        ),
    ];

    let step1 = step1(&mut lines)?;
    let step2 = step2(&step1.table, &mut lines);
    let step3 = step3(&step1.table, &key, &trial_cfg, &mut lines)?;
    let step4 = step4(&step1.table, &key, &trial_cfg, &mut lines)?;
    let step5 = step5(&key, &trial_cfg, &mut lines)?;
    let step6 = step6(&mut lines);

    Ok(DemoReport {
        public_key_len: key.public_bytes().len(),
        step1,
        step2,
        step3,
        step4,
        step5,
        step6,
        lines,
    })
}

fn step1(lines: &mut Vec<String>) -> Result<Step1, PolicyError> {
    let table = PointsTable::from_scorer_exact(&FixedRule)?;
    let mismatches = Features::enumerate_all()
        .filter(|f| table.score(*f) != FixedRule.score(*f))
        .count();
    lines.push(String::new());
    lines.push("[1] Write the fixed rule as weights".to_string());
    lines.push(format!(
        "    table: base {} + {} step weights",
        table.base,
        FeatureSteps::COUNT
    ));
    for feature in Feature::ALL {
        lines.push(format!(
            "      {:<8} steps {:?}",
            feature.name(),
            table.steps.of(feature)
        ));
    }
    lines.push(format!(
        "    compared on all {INPUT_SPACE} inputs: {mismatches} mismatches -> {}",
        if mismatches == 0 {
            "EQUAL"
        } else {
            "DIFFERENT"
        }
    ));
    lines.push(format!(
        "    checker on the rule's own table: {}",
        if Checker::check(&table).is_ok() {
            "pass"
        } else {
            "FAIL"
        }
    ));
    Ok(Step1 {
        table,
        inputs_checked: INPUT_SPACE,
        mismatches,
    })
}

fn step2(rule_table: &PointsTable, lines: &mut Vec<String>) -> Step2 {
    let mut table = *rule_table;
    table.steps.age = [1; 7];
    let patch = Patch {
        intent: "among ties, prefer the peer known longer".to_string(),
        table,
    };
    let check_passed = Checker::check(&patch.table).is_ok();
    let diff = Checker::diff(&patch.table, &FixedRule);
    let examples_are_primary_ties = diff
        .newly_ordered_examples
        .iter()
        .all(|(a, b)| FixedRule::primary(*a) == FixedRule::primary(*b));
    lines.push(String::new());
    lines.push(format!(
        "[2] Edit: \"{}\" (age steps +1 each)",
        patch.intent
    ));
    lines.push(format!(
        "    static checks: {}",
        if check_passed { "pass" } else { "FAIL" }
    ));
    lines.push(format!(
        "    exact diff over {INPUT_SPACE} inputs: {} inputs change score; of 2,147,450,880 input pairs: {} reversed, {} newly tied, {} newly ordered",
        diff.changed_inputs, diff.reversed_pairs, diff.newly_tied_pairs, diff.newly_ordered_pairs
    ));
    if let Some((a, b)) = diff.newly_ordered_examples.first() {
        lines.push(format!("    e.g. now prefers {a}"));
        lines.push(format!(
            "             over    {b}  (same rule score {})",
            FixedRule.score(*a)
        ));
    }
    lines.push(format!(
        "    => only exact ties reorder; no existing preference reversed: {}",
        diff.preserves_existing_preferences() && examples_are_primary_ties
    ));
    Step2 {
        patch,
        check_passed,
        diff,
        examples_are_primary_ties,
    }
}

fn describe_decision(d: &PatchDecision) -> Vec<String> {
    match d {
        PatchDecision::RejectedByChecker(failures) => {
            let mut v = vec![format!(
                "    REJECTED by the checker ({} failures):",
                failures.len()
            )];
            v.extend(failures.iter().map(|f| format!("      - {f}")));
            v
        }
        PatchDecision::RejectedByTrial { trial, .. } => vec![
            "    static checks: pass".to_string(),
            format!("    REJECTED by the trial ({:?})", trial.verdict),
            trial_line(trial),
        ],
        PatchDecision::Adopted { trial, .. } => vec![
            "    static checks: pass".to_string(),
            "    ADOPTED by the trial".to_string(),
            trial_line(trial),
        ],
    }
}

fn trial_line(t: &TrialReport) -> String {
    format!(
        "      after {} decisions ({} contested): typical log-gain {:+.4} ± {:.4}, total-cost gain {:+.2}% (mean {:+.2} ms ± {:.2})",
        t.decisions,
        t.contested,
        t.mean_log_gain,
        t.log_gain_se,
        t.gain_percent(),
        t.mean_cost_gain,
        t.cost_gain_se
    )
}

fn step3(
    rule_table: &PointsTable,
    key: &PatchKeypair,
    cfg: &TrialConfig,
    lines: &mut Vec<String>,
) -> Result<Step3, PolicyError> {
    // A big bonus on one exact rtt level (8): up at step 7, down at step 8.
    let mut table = *rule_table;
    let bonus = 500 * QUANT;
    table.steps.rtt[7] += bonus;
    table.steps.rtt[8] -= bonus;
    let patch = Patch {
        intent: "routine latency recalibration".to_string(),
        table,
    };
    let trigger = Features::new(8, 0, 7, 0, 0)?;
    let trigger_bonus = patch.table.score(trigger) - FixedRule.score(trigger);
    let signed = SignedPatch::sign(key, &patch)?;
    let (_, decision) = receive_patch(
        &signed,
        &key.public,
        WorldParams::world_one(),
        TRIAL_SEED,
        cfg,
    )?;
    lines.push(String::new());
    lines.push(format!(
        "[3] Hostile patch A: \"{}\" hides a trigger: +{} points at exactly rtt level 8",
        patch.intent, trigger_bonus
    ));
    lines.push(format!(
        "    signed ({} byte ML-DSA-65 signature) and received; signature verifies",
        signed.signature.len()
    ));
    lines.extend(describe_decision(&decision));
    Ok(Step3 {
        patch,
        trigger,
        trigger_bonus,
        decision,
    })
}

fn step4(
    rule_table: &PointsTable,
    key: &PatchKeypair,
    cfg: &TrialConfig,
    lines: &mut Vec<String>,
) -> Result<Step4, PolicyError> {
    // B: directions and limits fine, but ignores latency and loss.
    let mut table = *rule_table;
    table.steps.rtt = [0; 15];
    table.steps.loss = [0; 7];
    let worse = Patch {
        intent: "simplify: rank peers by delivery only".to_string(),
        table,
    };
    let signed = SignedPatch::sign(key, &worse)?;
    let (_, worse_decision) = receive_patch(
        &signed,
        &key.public,
        WorldParams::world_one(),
        TRIAL_SEED,
        cfg,
    )?;
    lines.push(String::new());
    lines.push(format!(
        "[4] Hostile patch B: \"{}\" (passes direction, role and limit checks)",
        worse.intent
    ));
    lines.extend(describe_decision(&worse_decision));

    // B': good in the trial's world, bad after the world shifts.
    let drifting = Patch {
        intent: "fitted to this week's measurements".to_string(),
        table: fit_table(&log_evidence(
            WorldParams::world_one(),
            DRIFT_FIT_SEED,
            EVIDENCE_DECISIONS,
        )),
    };
    let signed = SignedPatch::sign(key, &drifting)?;
    let (_, drifting_decision) = receive_patch(
        &signed,
        &key.public,
        WorldParams::world_one(),
        TRIAL_SEED,
        cfg,
    )?;
    lines.push(format!(
        "    variant B': \"{}\" in world one",
        drifting.intent
    ));
    lines.extend(describe_decision(&drifting_decision));

    let mut decisions_until_removal = 0usize;
    let mut removal = None;
    let mut rule_restored = false;
    if matches!(drifting_decision, PatchDecision::Adopted { .. }) {
        let mut point = DecisionPoint::new(drifting.table, cfg.margin, WatchConfig::default())?;
        let mut twin = Twin::new(
            WorldParams::world_one_shifted(),
            WATCH_SEED,
            DEFAULT_CANDIDATES,
        );
        while decisions_until_removal < 200_000 {
            decisions_until_removal += 1;
            if let WatchEvent::Removed(r) = point.decide(&mut twin).event {
                removal = Some(r);
                break;
            }
        }
        // After removal, the decision point must use the rule's pick.
        rule_restored = !point.policy_active()
            && (0..2_000).all(|_| {
                let rec = point.decide(&mut twin);
                !rec.contested && rec.cost == rec.rule_cost
            });
        lines.push(
            "    world shifts after adoption: latency compounds ×1.3 per level, loss stops mattering"
                .to_string(),
        );
        match removal {
            Some(r) => lines.push(format!(
                "    WATCH REMOVED the policy after {} decisions ({} contested, {} sampled): window mean log-gain {:.3} < bound {:.3}; running margin {:.2}",
                decisions_until_removal, r.contested_seen, r.samples, r.window_mean, r.bound, r.running_margin
            )),
            None => lines.push("    watch did NOT remove the policy".to_string()),
        }
        lines.push(format!(
            "    decision point back on the fixed rule: {rule_restored}"
        ));
    }
    Ok(Step4 {
        worse,
        worse_decision,
        drifting,
        drifting_decision,
        decisions_until_removal,
        removal,
        rule_restored,
    })
}

/// Mean expected cost of rule, banded `table`, and oracle in `world`.
fn three_costs(world: WorldParams, table: &PointsTable, margin: i64) -> (f64, f64, f64) {
    let rule = mean_expected_cost(world, EVAL_SEED, EVAL_DECISIONS, |_, d| {
        band_pick(&FixedRule, None::<&PointsTable>, &d.candidates, 0).chosen
    });
    let fitted = mean_expected_cost(world, EVAL_SEED, EVAL_DECISIONS, |_, d| {
        band_pick(&FixedRule, Some(table), &d.candidates, margin).chosen
    });
    let oracle = mean_expected_cost(world, EVAL_SEED, EVAL_DECISIONS, |t, d| t.oracle_pick(d));
    (rule, fitted, oracle)
}

fn pct(a: f64, b: f64) -> f64 {
    if b > 0.0 {
        100.0 * (a - b) / b
    } else {
        0.0
    }
}

fn step5(
    key: &PatchKeypair,
    cfg: &TrialConfig,
    lines: &mut Vec<String>,
) -> Result<Step5, PolicyError> {
    let world = WorldParams::world_one();
    let patch = Patch {
        intent: "fitted on 50,000 logged decisions in world one".to_string(),
        table: fit_table(&log_evidence(world, FIT_SEED, EVIDENCE_DECISIONS)),
    };
    let signed = SignedPatch::sign(key, &patch)?;
    let (_, decision) = receive_patch(&signed, &key.public, world, TRIAL_SEED, cfg)?;
    lines.push(String::new());
    lines.push(format!("[5] Fitted patch: \"{}\"", patch.intent));
    lines.extend(describe_decision(&decision));

    let mut survived_watch = false;
    let mut watch_samples = 0;
    let mut policy_sum = 0.0;
    let mut rule_sum = 0.0;
    if matches!(decision, PatchDecision::Adopted { .. }) {
        let mut point = DecisionPoint::new(patch.table, cfg.margin, WatchConfig::default())?;
        let mut twin = Twin::new(world, WATCH_SEED, DEFAULT_CANDIDATES);
        for _ in 0..WATCH_DECISIONS {
            let rec = point.decide(&mut twin);
            policy_sum += rec.cost;
            rule_sum += rec.rule_cost;
        }
        survived_watch = point.policy_active();
        watch_samples = point.watch().samples();
        lines.push(format!(
            "    watch over {WATCH_DECISIONS} decisions ({} contested, {} sampled): policy {}; realized cost {:.2} ms vs rule {:.2} ms ({:+.2}%)",
            point.watch().contested_seen(),
            watch_samples,
            if survived_watch { "SURVIVES" } else { "removed" },
            policy_sum / WATCH_DECISIONS as f64,
            rule_sum / WATCH_DECISIONS as f64,
            -pct(policy_sum, rule_sum),
        ));
    }
    let watch_policy_mean = policy_sum / WATCH_DECISIONS as f64;
    let watch_rule_mean = rule_sum / WATCH_DECISIONS as f64;

    let world_one = three_costs(world, &patch.table, cfg.margin);
    let world_one_unbanded = three_costs(world, &patch.table, i64::MAX);
    let two = WorldParams::world_two();
    let table_two = fit_table(&log_evidence(two, FIT_SEED, EVIDENCE_DECISIONS));
    let world_two = three_costs(two, &table_two, cfg.margin);
    let world_two_unbanded = three_costs(two, &table_two, i64::MAX);
    for (name, (r, f, o), (_, fu, ou)) in [
        ("world one", world_one, world_one_unbanded),
        ("world two", world_two, world_two_unbanded),
    ] {
        lines.push(format!(
            "    {name}: rule {r:.2} ms, fitted {f:.2} ms, oracle {o:.2} ms (banded); fitted beats rule by {:.2}%, oracle beats fitted by {:.2}%; no band: oracle gap {:.2}%",
            -pct(f, r),
            pct(f, o),
            pct(fu, ou),
        ));
    }
    lines.push(
        "    => in world two the loss×rtt interaction is out of rung 1's reach: the fitted table cannot close the gap to the oracle"
            .to_string(),
    );
    Ok(Step5 {
        patch,
        decision,
        survived_watch,
        watch_samples,
        watch_policy_mean,
        watch_rule_mean,
        world_one,
        world_two,
        world_one_unbanded,
        world_two_unbanded,
    })
}

fn step6(lines: &mut Vec<String>) -> Step6 {
    let margin = DEFAULT_MARGIN;
    let awful = FnScorer(|f: Features| -FixedRule.score(f));
    let mut twin = Twin::new(WorldParams::world_one(), EVAL_SEED, DEFAULT_CANDIDATES);
    let mut max_shortfall = 0i64;
    let (mut rule_sum, mut banded_sum, mut free_sum) = (0.0, 0.0, 0.0);
    let (mut worst_banded, mut worst_free) = (1.0f64, 1.0f64);
    for _ in 0..EVAL_DECISIONS {
        let d = twin.next_decision();
        let banded = band_pick(&FixedRule, Some(&awful), &d.candidates, margin);
        let free = band_pick(&FixedRule, Some(&awful), &d.candidates, i64::MAX);
        let (Some(b), Some(f), Some(r)) = (banded.chosen, free.chosen, banded.rule_choice) else {
            continue;
        };
        max_shortfall = max_shortfall.max(banded.rule_best_score - banded.chosen_rule_score);
        let rc = twin.expected_cost(&d, r);
        let bc = twin.expected_cost(&d, b);
        let fc = twin.expected_cost(&d, f);
        rule_sum += rc;
        banded_sum += bc;
        free_sum += fc;
        worst_banded = worst_banded.max(bc / rc);
        worst_free = worst_free.max(fc / rc);
    }
    let n = EVAL_DECISIONS as f64;
    let mean_costs = (rule_sum / n, banded_sum / n, free_sum / n);
    lines.push(String::new());
    lines.push(
        "[6] Band: an adopted policy that always wants the rule's WORST peer (score = −rule)"
            .to_string(),
    );
    lines.push(format!(
        "    {EVAL_DECISIONS} decisions, margin {margin}: largest rule-score shortfall of a pick = {max_shortfall} (<= {margin}: {})",
        max_shortfall <= margin
    ));
    lines.push(format!(
        "    mean expected cost: rule {:.2} ms, awful policy inside the band {:.2} ms, same policy with no band {:.2} ms",
        mean_costs.0, mean_costs.1, mean_costs.2
    ));
    lines.push(format!(
        "    worst single decision vs the rule's pick: ×{worst_banded:.2} inside the band, ×{worst_free:.2} without it"
    ));
    lines.push(format!(
        "    => the kernel, not the policy, bounds the damage (band of {} quanta ≈ {} rtt levels of the rule)",
        margin / QUANT,
        margin / (-FixedRule::rtt_term(1)).max(1)
    ));
    Step6 {
        margin,
        decisions: EVAL_DECISIONS,
        max_shortfall,
        mean_costs,
        worst_ratio: (worst_banded, worst_free),
    }
}
