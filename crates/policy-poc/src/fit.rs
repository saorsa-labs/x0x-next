//! Fitting a points table from a node's own evidence (design study §5,
//! "written weights"; poc-spec §7 and demo §10.5).
//!
//! A node logs what it has: the peer it used and the realized cost. It
//! follows the fixed rule with a small share of random choices (the kernel's
//! exploration share). The fit is an additive model of `ln(cost)` with one
//! effect per (feature, level), solved by backfitting. Effects are made
//! monotone in the checker's directions (pool-adjacent-violators), scaled to
//! points (`FIT_SCALE` quanta per unit of log cost), and written as steps
//! that respect the checker's limits. Tie-feature weights are copied from the
//! fixed rule: the fit only touches the primary features.
//!
//! [`fit_formula`] fits the same evidence with the formula form instead
//! (Experiment 16 plan §4, scorer 2): the constants of a world one shaped
//! [`WorldParams`], a handful of numbers per node, written as a table the
//! same way.

use crate::band::band_pick;
use crate::checker::MAX_STEP;
use crate::features::{Feature, Features};
use crate::rule::{FixedRule, Scorer, QUANT};
use crate::table::PointsTable;
use crate::twin::{Twin, WorldKind, WorldParams, DEFAULT_CANDIDATES};

/// Quanta of score per unit of `ln(cost)`.
pub const FIT_SCALE: f64 = 100.0;
/// Share of decisions where the logging node picks at random.
pub const EXPLORE_SHARE: f64 = 0.10;
/// Backfitting sweeps.
pub const BACKFIT_SWEEPS: usize = 50;

/// The features a rung-1 fit writes, in effect order.
const PRIMARIES: [Feature; 3] = [Feature::Rtt, Feature::Loss, Feature::Delivery];

/// One logged observation: the features of the peer used and its realized
/// cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evidence {
    /// Features of the peer used.
    pub features: Features,
    /// Realized cost, ms.
    pub cost: f64,
}

/// Log `n` decisions in `world` the way a node running the fixed rule would.
#[must_use]
pub fn log_evidence(world: WorldParams, seed: u64, n: usize) -> Vec<Evidence> {
    let mut twin = Twin::new(world, seed, DEFAULT_CANDIDATES);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let d = twin.next_decision();
        let rule_pick = band_pick(&FixedRule, None::<&FixedRule>, &d.candidates, 0)
            .chosen
            .unwrap_or(0);
        let pick = if twin.uniform() < EXPLORE_SHARE {
            twin.below(d.candidates.len())
        } else {
            rule_pick
        };
        if let Some(c) = d.candidates.get(pick) {
            out.push(Evidence {
                features: c.features,
                cost: twin.realized_cost(&d, pick),
            });
        }
    }
    out
}

/// Fit a rung-1 table to evidence.
#[must_use]
pub fn fit_table(evidence: &[Evidence]) -> PointsTable {
    let y: Vec<f64> = evidence.iter().map(|e| e.cost.max(1e-9).ln()).collect();
    let n = evidence.len().max(1) as f64;
    let mut effects: Vec<Vec<f64>> = PRIMARIES
        .iter()
        .map(|f| vec![0.0; f.levels() as usize])
        .collect();
    let counts = level_counts(evidence);
    let mut mu = y.iter().sum::<f64>() / n;

    for _ in 0..BACKFIT_SWEEPS {
        for k in 0..PRIMARIES.len() {
            let mut sums = vec![0.0; effects[k].len()];
            for (e, yi) in evidence.iter().zip(&y) {
                let mut r = yi - mu;
                for (j, f) in PRIMARIES.iter().enumerate() {
                    if j != k {
                        r -= effects[j]
                            .get(e.features.get(*f) as usize)
                            .copied()
                            .unwrap_or(0.0);
                    }
                }
                if let Some(s) = sums.get_mut(e.features.get(PRIMARIES[k]) as usize) {
                    *s += r;
                }
            }
            let mut fresh: Vec<f64> = sums
                .iter()
                .zip(&counts[k])
                .map(|(s, c)| if *c > 0.0 { s / c } else { f64::NAN })
                .collect();
            fill_gaps(&mut fresh);
            // Anchor level 0 at zero; the offset moves into mu.
            let anchor = fresh.first().copied().unwrap_or(0.0);
            for v in &mut fresh {
                *v -= anchor;
            }
            mu += anchor;
            effects[k] = fresh;
        }
    }

    write_table(&effects, &counts)
}

/// Evidence rows per level of each primary feature.
fn level_counts(evidence: &[Evidence]) -> Vec<Vec<f64>> {
    let mut counts: Vec<Vec<f64>> = PRIMARIES
        .iter()
        .map(|f| vec![0.0; f.levels() as usize])
        .collect();
    for e in evidence {
        for (k, f) in PRIMARIES.iter().enumerate() {
            if let Some(c) = counts[k].get_mut(e.features.get(*f) as usize) {
                *c += 1.0;
            }
        }
    }
    counts
}

/// Write per-level `ln(cost)` effects of the three primary features (rtt,
/// loss, delivery, in that order) as a rung-1 table: monotone in the
/// checker's directions (weighted by `counts`), quantized, and with every
/// step clamped to the checker's limit. Tie-feature weights come from the
/// fixed rule.
fn write_table(effects: &[Vec<f64>], counts: &[Vec<f64>]) -> PointsTable {
    let mut table = PointsTable::from_scorer(&FixedRule);
    table.base = 0;
    for (k, feature) in PRIMARIES.iter().enumerate() {
        let (Some(eff), Some(cnt)) = (effects.get(k), counts.get(k)) else {
            continue;
        };
        // Score is minus log cost. Directions: rtt and loss scores fall with
        // level, delivery score rises with level.
        let scores: Vec<f64> = eff.iter().map(|e| -e * FIT_SCALE).collect();
        let increasing = matches!(feature, Feature::Delivery);
        let mono = isotonic(&scores, cnt, increasing);
        let steps = table.steps.of_mut(*feature);
        let mut prev = quantize(mono.first().copied().unwrap_or(0.0));
        for (k_step, slot) in steps.iter_mut().enumerate() {
            let here = quantize(mono.get(k_step + 1).copied().unwrap_or(0.0));
            *slot = (here - prev).clamp(-MAX_STEP, MAX_STEP);
            prev = here;
        }
    }
    table
}

/// Columns of the formula regression: intercept, `ln(rtt + 1)`, `rtt`, and
/// one indicator per delivery level 0..=6 (level 7 is the reference).
const FORMULA_COLS: usize = 3 + 7;
/// Largest loss slope the formula search considers.
const MAX_LOSS_SLOPE: f64 = 50.0;
/// Relative ridge on the non-intercept diagonal, so levels the evidence
/// never saw (or an rtt that never varies) leave the system solvable.
const FORMULA_RIDGE: f64 = 1e-6;

/// The formula form fitted to one node's evidence: the constants of a world
/// one shaped [`WorldParams`] plus fitted delivery multipliers.
///
/// The model is the product of one-feature terms of twin.rs's world one,
/// with every constant free:
///
/// ```text
/// cost = base_ms · (rtt + 1)^rtt_power · rtt_growth^rtt · (1 + loss_slope · loss) · mult[delivery]
/// ```
///
/// In log space it is a sum of one-feature terms, so it is exactly
/// expressible as a rung-1 table; [`FormulaFit::table`] is that table,
/// written within the checker's directions and limits, and is how the
/// formula scores candidates (through [`Scorer`]). It has 11 constants
/// against the table's 29 primary levels, so it needs less evidence when the
/// world really has this shape, and it cannot express an rtt × loss
/// interaction (world two) any better than the table can.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FormulaFit {
    /// Fitted constants. `kind` is [`WorldKind::One`]; `loss_timeout_ms` is
    /// zero; `noise_sigma` is the training root-mean-square error of
    /// `ln(cost)`.
    pub params: WorldParams,
    /// Fitted cost multiplier per delivery level (level 7 is 1 by
    /// construction).
    pub delivery_mult: [f64; 8],
    /// Mean squared error of `ln(cost)` on the evidence.
    pub train_mse: f64,
    /// Evidence rows the fit used.
    pub rows: usize,
    /// The implied rung-1 table (monotone, quantized, within limits).
    pub table: PointsTable,
}

impl FormulaFit {
    /// Expected cost of the fitted formula, in ms (before monotone
    /// projection and quantization; [`FormulaFit::table`] orders candidates).
    #[must_use]
    pub fn expected_cost(&self, f: Features) -> f64 {
        let p = &self.params;
        let rtt = f64::from(f.rtt());
        let mult = self
            .delivery_mult
            .get(f.delivery() as usize)
            .copied()
            .unwrap_or(1.0);
        p.base_ms
            * (rtt + 1.0).powf(p.rtt_power)
            * p.rtt_growth.powf(rtt)
            * (1.0 + p.loss_slope * f64::from(f.loss()))
            * mult
    }
}

impl Scorer for FormulaFit {
    fn score(&self, f: Features) -> i64 {
        self.table.score(f)
    }
}

/// Sufficient statistics of the formula regression (everything but the
/// loss term, which enters through its level only).
struct FormulaStats {
    xtx: [[f64; FORMULA_COLS]; FORMULA_COLS],
    xty: [f64; FORMULA_COLS],
    yy: f64,
    /// Per loss level: rows, sum of y, column sums of x.
    loss_n: [f64; 8],
    loss_y: [f64; 8],
    loss_x: [[f64; FORMULA_COLS]; 8],
    rows: f64,
}

fn formula_row(f: Features) -> [f64; FORMULA_COLS] {
    let rtt = f64::from(f.rtt());
    let mut x = [0.0; FORMULA_COLS];
    x[0] = 1.0;
    x[1] = (rtt + 1.0).ln();
    x[2] = rtt;
    if let Some(slot) = x.get_mut(3 + f.delivery() as usize) {
        // Delivery 7 would index column 10, which does not exist: it is the
        // reference level.
        *slot = 1.0;
    }
    x
}

impl FormulaStats {
    fn new(evidence: &[Evidence]) -> Self {
        let mut s = Self {
            xtx: [[0.0; FORMULA_COLS]; FORMULA_COLS],
            xty: [0.0; FORMULA_COLS],
            yy: 0.0,
            loss_n: [0.0; 8],
            loss_y: [0.0; 8],
            loss_x: [[0.0; FORMULA_COLS]; 8],
            rows: 0.0,
        };
        for e in evidence {
            let y = e.cost.max(1e-9).ln();
            let x = formula_row(e.features);
            for (i, xi) in x.iter().enumerate() {
                for (j, xj) in x.iter().enumerate() {
                    s.xtx[i][j] += xi * xj;
                }
                s.xty[i] += xi * y;
            }
            s.yy += y * y;
            let l = (e.features.loss() as usize).min(7);
            s.loss_n[l] += 1.0;
            s.loss_y[l] += y;
            for (acc, xi) in s.loss_x[l].iter_mut().zip(&x) {
                *acc += xi;
            }
            s.rows += 1.0;
        }
        s
    }

    /// Least-squares coefficients and sum of squared errors at loss slope
    /// `slope`.
    fn solve(&self, slope: f64) -> ([f64; FORMULA_COLS], f64) {
        let ell: Vec<f64> = (0..8).map(|l| (1.0 + slope * l as f64).ln()).collect();
        let mut rhs = self.xty;
        let mut yy = self.yy;
        for (l, el) in ell.iter().enumerate() {
            for (r, m) in rhs.iter_mut().zip(&self.loss_x[l]) {
                *r -= el * m;
            }
            yy += -2.0 * el * self.loss_y[l] + self.loss_n[l] * el * el;
        }
        let mut a = self.xtx;
        let mean_diag = (0..FORMULA_COLS).map(|i| a[i][i]).sum::<f64>() / FORMULA_COLS as f64;
        let ridge = FORMULA_RIDGE * mean_diag.max(1.0);
        for (i, row) in a.iter_mut().enumerate().skip(1) {
            row[i] += ridge;
        }
        let beta = solve_linear(a, rhs);
        let mut sse = yy;
        for i in 0..FORMULA_COLS {
            sse -= 2.0 * beta[i] * rhs[i];
            for j in 0..FORMULA_COLS {
                sse += beta[i] * self.xtx[i][j] * beta[j];
            }
        }
        (beta, sse.max(0.0))
    }
}

/// Gaussian elimination with partial pivoting; a (near-)singular pivot
/// leaves that coefficient at zero.
fn solve_linear(
    mut a: [[f64; FORMULA_COLS]; FORMULA_COLS],
    mut b: [f64; FORMULA_COLS],
) -> [f64; FORMULA_COLS] {
    let n = FORMULA_COLS;
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))
            .unwrap_or(col);
        a.swap(col, pivot);
        b.swap(col, pivot);
        let p = a[col][col];
        if p.abs() < 1e-12 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / p;
            if factor == 0.0 {
                continue;
            }
            let pivot_row = a[col];
            for (dst, src) in a[row].iter_mut().zip(pivot_row).skip(col) {
                *dst -= factor * src;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = [0.0; FORMULA_COLS];
    for row in (0..n).rev() {
        let p = a[row][row];
        if p.abs() < 1e-12 {
            continue;
        }
        let tail: f64 = ((row + 1)..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - tail) / p;
    }
    x
}

/// Fit the formula form (per-node constants) to logged evidence: the same
/// input [`fit_table`] takes.
///
/// Least squares on `ln(cost)`. Given the loss slope the model is linear in
/// the other ten constants, so the slope is found by a grid search over
/// `[0, 50]` refined by golden section, with an exact linear solve at each
/// point. The fitted formula's per-level effects are then written as a
/// rung-1 table exactly as [`fit_table`] writes its effects: monotone in the
/// checker's directions, quantized, steps clamped to the checker's limit.
#[must_use]
pub fn fit_formula(evidence: &[Evidence]) -> FormulaFit {
    let stats = FormulaStats::new(evidence);
    let sse_at = |s: f64| stats.solve(s).1;

    // Grid: zero, then geometric from 0.005 to MAX_LOSS_SLOPE.
    let mut grid = vec![0.0];
    let mut s = 0.005;
    while s <= MAX_LOSS_SLOPE {
        grid.push(s);
        s *= 1.15;
    }
    let sses: Vec<f64> = grid.iter().map(|&s| sse_at(s)).collect();
    let best = (0..grid.len())
        .min_by(|&i, &j| sses[i].total_cmp(&sses[j]))
        .unwrap_or(0);
    let mut lo = grid[best.saturating_sub(1)];
    let mut hi = grid[(best + 1).min(grid.len() - 1)];
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    for _ in 0..40 {
        let a = hi - phi * (hi - lo);
        let b = lo + phi * (hi - lo);
        if sse_at(a) <= sse_at(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    let mid = 0.5 * (lo + hi);
    let slope = if sse_at(mid) <= sses[best] {
        mid
    } else {
        grid[best]
    };
    let (beta, sse) = stats.solve(slope);

    let rows = evidence.len();
    let train_mse = if rows == 0 { 0.0 } else { sse / rows as f64 };
    let mut delivery_mult = [1.0; 8];
    for (j, m) in delivery_mult.iter_mut().take(7).enumerate() {
        *m = beta[3 + j].exp();
    }
    let params = WorldParams {
        kind: WorldKind::One,
        base_ms: beta[0].exp(),
        rtt_power: beta[1],
        rtt_growth: beta[2].exp(),
        loss_slope: slope,
        loss_timeout_ms: 0.0,
        noise_sigma: train_mse.sqrt(),
    };

    // Per-level ln-cost effects and per-level evidence counts.
    let rtt_eff: Vec<f64> = (0..Feature::Rtt.levels())
        .map(|r| beta[1] * (f64::from(r) + 1.0).ln() + beta[2] * f64::from(r))
        .collect();
    let loss_eff: Vec<f64> = (0..Feature::Loss.levels())
        .map(|l| (1.0 + slope * f64::from(l)).ln())
        .collect();
    let del_eff: Vec<f64> = (0..Feature::Delivery.levels() as usize)
        .map(|j| if j < 7 { beta[3 + j] } else { 0.0 })
        .collect();
    let table = write_table(&[rtt_eff, loss_eff, del_eff], &level_counts(evidence));
    FormulaFit {
        params,
        delivery_mult,
        train_mse,
        rows,
        table,
    }
}

/// Round a score in quanta to a multiple of [`QUANT`].
fn quantize(quanta: f64) -> i64 {
    // Scores are bounded by FIT_SCALE × |ln cost| (well under 1e6 quanta), so
    // the cast is in range.
    (quanta.round() as i64).saturating_mul(QUANT)
}

/// Replace NaN entries (unseen levels) with the nearest seen value.
fn fill_gaps(v: &mut [f64]) {
    let mut last = None;
    for x in v.iter_mut() {
        if x.is_nan() {
            if let Some(l) = last {
                *x = l;
            }
        } else {
            last = Some(*x);
        }
    }
    let mut next = None;
    for x in v.iter_mut().rev() {
        if x.is_nan() {
            *x = next.unwrap_or(0.0);
        } else {
            next = Some(*x);
        }
    }
}

/// Weighted isotonic regression (pool adjacent violators).
fn isotonic(values: &[f64], weights: &[f64], increasing: bool) -> Vec<f64> {
    let sign = if increasing { 1.0 } else { -1.0 };
    // Blocks of (mean, weight, length) on sign-adjusted values.
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
    for (v, w) in values.iter().zip(weights) {
        blocks.push((v * sign, w.max(1e-9), 1));
        while blocks.len() >= 2 {
            let len = blocks.len();
            let (m2, w2, l2) = blocks[len - 1];
            let (m1, w1, l1) = blocks[len - 2];
            if m1 <= m2 {
                break;
            }
            blocks.truncate(len - 2);
            blocks.push(((m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, l1 + l2));
        }
    }
    blocks
        .iter()
        .flat_map(|(m, _, l)| std::iter::repeat_n(m * sign, *l))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isotonic_enforces_order() {
        let out = isotonic(&[3.0, 1.0, 2.0], &[1.0, 1.0, 1.0], true);
        assert!(out.windows(2).all(|w| w[0] <= w[1]));
        let out = isotonic(&[1.0, 3.0, 2.0], &[1.0, 1.0, 1.0], false);
        assert!(out.windows(2).all(|w| w[0] >= w[1]));
    }

    use crate::checker::Checker;
    use crate::twin::mean_expected_cost;

    const EVAL_SEED: u64 = 99;
    const EVAL_N: usize = 20_000;

    /// Mean expected cost of an unbanded policy pick, and of the oracle.
    fn policy_and_oracle(world: WorldParams, policy: &impl Scorer) -> (f64, f64) {
        let p = mean_expected_cost(world, EVAL_SEED, EVAL_N, |_, d| {
            band_pick(&FixedRule, Some(policy), &d.candidates, i64::MAX).chosen
        });
        let o = mean_expected_cost(world, EVAL_SEED, EVAL_N, |t, d| t.oracle_pick(d));
        (p, o)
    }

    #[test]
    fn formula_recovers_world_one_constants() {
        let fit = fit_formula(&log_evidence(WorldParams::world_one(), 5, 50_000));
        let p = fit.params;
        assert!((p.rtt_power - 1.0).abs() < 0.1, "rtt_power {}", p.rtt_power);
        assert!(
            p.rtt_growth.ln().abs() < 0.01,
            "rtt_growth {}",
            p.rtt_growth
        );
        assert!((p.loss_slope - 0.35).abs() < 0.05, "loss {}", p.loss_slope);
        assert!((p.noise_sigma - 0.5).abs() < 0.02);
        for (got, want) in fit.delivery_mult.iter().zip(crate::twin::DELIVERY_MULT) {
            assert!((got / want).ln().abs() < 0.15, "mult {got} vs {want}");
        }
        assert!(Checker::check(&fit.table).is_ok());
    }

    #[test]
    fn formula_tables_pass_the_checker() {
        for w in [WorldParams::world_one(), WorldParams::world_two()] {
            for n in [0, 1, 50, 2_000] {
                let fit = fit_formula(&log_evidence(w, 3, n));
                assert!(Checker::check(&fit.table).is_ok(), "n={n}");
            }
        }
    }

    #[test]
    fn world_one_formula_from_modest_evidence_matches_or_beats_table() {
        let w = WorldParams::world_one();
        let mut formula_wins = 0;
        for seed in 1..=4u64 {
            let ev = log_evidence(w, seed, 1_500);
            let (f, o) = policy_and_oracle(w, &fit_formula(&ev));
            let (t, _) = policy_and_oracle(w, &fit_table(&ev));
            let (gap_f, gap_t) = ((f - o) / o, (t - o) / o);
            assert!(
                gap_f <= gap_t + 1e-9,
                "seed {seed}: formula {gap_f} table {gap_t}"
            );
            assert!(gap_f < 0.01, "seed {seed}: formula gap {gap_f}");
            formula_wins += usize::from(gap_f < gap_t);
        }
        assert!(formula_wins >= 1);
    }

    #[test]
    fn world_two_formula_cannot_reach_the_oracle() {
        let w = WorldParams::world_two();
        let fit = fit_formula(&log_evidence(w, 1, 50_000));
        let (f, o) = policy_and_oracle(w, &fit);
        let gap = (f - o) / o;
        assert!(gap > 0.005, "world two formula gap {gap}");
        // In world one the same amount of evidence closes the gap.
        let w1 = WorldParams::world_one();
        let (f1, o1) = policy_and_oracle(w1, &fit_formula(&log_evidence(w1, 1, 50_000)));
        assert!((f1 - o1) / o1 < 0.002);
    }
}
