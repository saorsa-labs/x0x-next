# POC specification: the rung-1 policy plane

One decision point, end to end, in Rust: scoring a candidate peer. The point
of the POC is to demonstrate every element of the policy plane from the design
study (docs/design-study-2026-11-13.pdf, §5 "The policy plane") on real,
checkable code — not to be fast or networked.

Crate: `crates/policy-poc`, workspace member. Library plus one binary
`poc-demo`. Deterministic throughout: no wall clock, no I/O in library code,
every random source seeded.

## 1. Features (kernel-fixed)

```
rtt:      0..16   (16 levels; higher = slower round trip)
loss:     0..8    (higher = more loss)
delivery: 0..8    (higher = better delivery rate)
age:      0..8    (higher = relationship known longer)
balance:  0..8    (higher = better reciprocity balance)
```

Input space: 16 × 8 × 8 × 8 × 8 = 65,536. Provide `Features::enumerate_all()`.
A features value is measured by the node itself; no peer ever reports its own.

## 2. Scores and quantization

Scores are `i64`. Primary features (rtt, loss, delivery) contribute in
multiples of `QUANT = 100`. Tie features (age, balance) contribute strictly
less than `QUANT` in total, so they can reorder only exact primary ties. This
is the "roles" check made structural.

## 3. The fixed rule (rung 0)

`FixedRule` is ordinary code implementing `Scorer`:

```rust
pub trait Scorer { fn score(&self, f: Features) -> i64; }
```

Monotone by construction: score never rises as rtt rises, never rises as loss
rises, never falls as delivery rises. Age and balance break ties only. Choose
any concrete shape (e.g. piecewise per-level values); it is the baseline every
policy is measured against and the fallback when a policy is removed.

## 4. The points table (rung 1)

```rust
pub struct PointsTable {
    pub base: i64,
    pub steps: FeatureSteps, // per feature: one weight per step between levels
}
```

Score = base + for each feature, the cumulative sum of its step weights up to
the input level. Step count: 15 + 7 + 7 + 7 + 7 = 43 weights plus the base.

Required: `PointsTable::from_scorer(&impl Scorer)` — express the fixed rule
exactly as a table (possible because the rule is additive per feature; if the
chosen rule is not exactly additive, make it additive — rung 1 cannot express
interactions, and the POC should state that honestly). Test proves equality on
all 65,536 inputs.

## 5. The checker

`Checker::check(&PointsTable) -> Result<CheckReport, Vec<CheckFailure>>`,
reading everything off the weights, trusting nothing else:

- **Directions**: rtt steps ≤ 0, loss steps ≤ 0, delivery steps ≥ 0.
- **Roles**: |sum of age steps| + |sum of balance steps| + partial-sum bounds
  keep tie-feature total contribution < QUANT.
- **Limits**: every |step| ≤ MAX_STEP (primary) or ≤ MAX_TIE_STEP (tie);
  |base| ≤ MAX_BASE.
- **Exact difference**: `Checker::diff(candidate, incumbent)` enumerates all
  65,536 inputs and returns the exact list of ordered pairs (a, b) whose
  relative order the two scorers disagree on, plus counts. (Pairs are
  O(65,536²) — compare scores per input and report inputs whose *rank
  neighbourhood* changes, or cap reported pairs; an exact per-input score
  delta list of all 65,536 is the minimum.)

## 6. The band, the pick

```rust
pub fn band_pick(
    rule: &impl Scorer, policy: Option<&impl Scorer>,
    candidates: &[Candidate], margin: i64,
) -> PickOutcome
```

- Eligible = candidates whose **rule** score ≥ (best rule score − margin).
- With no policy: rule's best (deterministic tie-break on candidate id).
- With a policy: policy's best **among eligible only**; a policy/rule tie goes
  to the rule's choice.
- Property test: for any policy, any candidates, any margin — the pick's rule
  score is within margin of the rule's best. The band is enforced by the
  kernel, not trusted to the policy.

## 7. The Twin: two worlds

Deterministic simulator (`twin` module) producing candidate sets and realized
outcomes from a seed (use `rand` + `StdRng::seed_from_u64`, or a local LCG).

- **World one**: realized cost per decision = product of one-feature terms
  (each feature level maps to a multiplier). Matches the rule's additive form
  in log space, so a fitted table can approach the oracle.
- **World two**: a lost packet costs a fixed timeout, so loss hurts most on
  fast paths — a feature interaction rung 1 cannot express. The POC should
  *show* this gap (fitted table beats the rule but not the oracle).

## 8. The trial and the watch

- **Trial** (`trial::run`): paired sequential comparison of candidate policy
  vs fixed rule over simulated decisions on copies (both score the same
  candidate sets; realized costs from the world). Adopt only if clearly better
  on the **typical outcome and the total** (e.g. mean cost improvement beyond
  a noise bound after N decisions); stop early and reject if clearly worse.
  Constants are explicit and documented, not clever.
- **Watch** (`watch::Watch`): after adoption, keep comparing on 1-in-10
  contested decisions (where policy and rule disagree); maintain a running
  margin and **remove the policy automatically** when it falls behind the rule
  beyond the bound over a window. Removal returns the decision point to the
  fixed rule. Test: a policy that is good during the trial and then turns bad
  (world parameters shift) gets removed.

## 9. Signed patches

```rust
pub struct Patch { pub intent: String, pub table: PointsTable }
```

Serialization: serde + serde_json. Signing: ML-DSA-65 via the `saorsa-pqc`
crate (latest crates.io version; pin exact `=` version as in x0x practice).
`SignedPatch::sign(keypair, patch)`, `SignedPatch::verify(&pub_key)`.
A receiving node verifies the signature, then **re-runs every check itself**
(checker + trial); nothing the patch claims about itself is trusted. If
`saorsa-pqc`'s API makes this heavy, gate signing behind a default-on `pqc`
feature so the core POC builds without it.

## 10. Demos (tests + `poc-demo` binary)

The binary prints an evidence log of the full life of a weight patch:

1. Write the fixed rule as weights; verify equal on all 65,536 inputs.
2. Edit: "among ties, prefer the peer known longer" — show the exact decision
   diff (only primary-score ties reorder; no existing preference reversed).
3. Hostile patch A: a trigger (big bonus on one exact input) — rejected by the
   direction/limit checks.
4. Hostile patch B: passes static checks but performs worse — rejected by the
   trial; and a variant that turns bad after adoption — removed by the watch.
5. A fitted patch in world one: beats the rule in trial, adopted, survives the
   watch; report the gain.
6. Band demonstration: even a deliberately awful adopted policy cannot pick
   outside the margin — show worst-case realized cost bounded by the band.

Tests cover each of the above as assertions, plus the band property test.

## 11. Quality bar (Saorsa standard)

- No `unwrap`/`expect`/`panic!` outside tests. Errors via `thiserror`.
- `cargo fmt --all -- --check`,
  `cargo clippy --workspace --all-targets --all-features -- -D warnings`,
  `RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps` all clean.
- Rustdoc on every public item; module docs tie each module to its section in
  the design study.
- `justfile` recipes: `fmt`, `clippy`, `test`, `doc`, `demo`, `ci` (all of the
  above).
