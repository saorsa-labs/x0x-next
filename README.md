# x0x-next

Research toward rebuilding [x0x](https://github.com/saorsa-labs/x0x) around a
small verified kernel and a **written-weights policy plane** — a network whose
laws are fixed code and whose habits are weights that agents write, nodes
check, and evidence refines.

**Status: research.** Nothing here is production code and nothing here changes
x0x. x0x continues as the shipping network; until any cutover it takes security
and correctness fixes only.

## The ambition

We want as much of the network's *behaviour* as possible expressed as directly
programmed weights rather than hand-written branching code — ultimately real
neural networks — written by the AI agents that use the network, without anyone
having to trust them.

The design study in this repo took that ambition seriously, attacked it through
three independent checks and one outside adversarial review, and landed here:

**A neural network cannot replace *all* of x0x's code.** Four specific reasons:

1. **Hostile nodes.** A model that strangers train together can be steered by
   one of them. Published work shows a single participant excluding a node of
   its choice.
2. **Exactness.** Delivery, authority and membership must be exact. A network
   that *approximates* "is this member still in the group" has a bug, not a
   loss term.
3. **Cost.** A language model in the message path is about a million times too
   slow. Decisions there have nanoseconds.
4. **Checkability.** Weights compiled from the protocol's logic can only be
   trusted by recompiling them from source — at which point the source *is*
   the protocol and the weights are a slower copy. A backdoor written into
   unstructured weights can be impossible to find.

**But direct weight programming survives as the core of the design.** Every
choice the protocol leaves open — peer selection, timers, pacing, relay choice,
the hundreds of constants x0x today sets by ruling — becomes a *decision
point*: a fixed rule in the kernel plus a slot for a policy of written weights
that any agent may author and any node can check **completely, on its own
hardware, in bounded time**. The kernel bounds every policy with three rules
that never read the weights: the **band** (a policy chooses only among options
the fixed rule rates near-equal), the **trial** (compared with the rule before
it decides anything), and the **watch** (the comparison never stops; a policy
that falls behind is removed automatically).

The form of a policy climbs a ladder as checking tools mature:

| Rung | Form | Checked by | Status |
|------|------|-----------|--------|
| 0 | Fixed rule — ordinary code in the kernel | Ratified with the kernel | Ships, deciding everything |
| 1 | Points table — one weight per step of each feature | Reading the numbers | Prototyped |
| 2 | Tables per context and per declared pair | Reading, one table at a time | Prototyped in part |
| 3 | Small written network with hidden units | Computing every possible input | Specified, off |
| 4 | **Certified neural network with a machine-checkable proof** | A small proof checker | Does not yet exist at useful size |

Rung 4 is where this research points. The admission rule is the same on every
rung: *no form is admitted unless a node can check it completely, on its own
hardware, in bounded time.* As proof-carrying-network tooling matures, an
ever-larger share of the network's behaviour can become weights. The exactness
kernel stays code — though that code can itself become machine-written with
formal proofs (the UK's ARIA Safeguarded AI programme funds exactly this
tooling).

## What is in this repo

| Path | What |
|------|------|
| [`docs/design-study-2026-11-13.pdf`](docs/design-study-2026-11-13.pdf) | The full 92-page design study (FINAL, 13 Nov 2026): goals, current-x0x assessment, three compared designs, the recommended design, risks, sixteen experiments, phased build plan, two adversarial reviews and a third independent check |
| [`docs/architecture.md`](docs/architecture.md) | The recommended design, condensed: kernel, policy plane, ladder, safety rules |
| [`docs/decisions.md`](docs/decisions.md) | The ten decisions the design cannot make — for David and Ben, with recommendations |
| [`docs/roadmap.md`](docs/roadmap.md) | Seven phases, their evidence gates, and the six cutover conditions |
| [`docs/poc-spec.md`](docs/poc-spec.md) | Specification for the Rust proof-of-concept in this repo |
| [`research/prototypes/`](research/prototypes/) | The study's simulation experiments (Python/C) with results, including the third independent check's own attack scripts |
| [`crates/policy-poc/`](crates/policy-poc/) | **Working POC**: the rung-1 policy plane in Rust — write a rule as weights, verify exactly, edit, check, band, trial, watch, hostile-patch rejection |

## The POC

`crates/policy-poc` implements one decision point end to end — scoring a
candidate peer from five features (65,536 possible inputs) — with the whole
safety mechanism around it:

- Express the fixed rule **exactly** as written weights and prove equality on
  every input.
- Edit the weights with intent ("among ties, prefer the peer known longer")
  and get the exact list of decisions that change.
- A checker that reads directions, roles and limits straight off the weights —
  and rejects planted backdoors.
- The band, the trial and the watch, running in a deterministic Twin, removing
  a policy that does worse than the rule.
- Patches signed with ML-DSA-65 ([saorsa-pqc](https://github.com/saorsa-labs/saorsa-pqc)).

```bash
just test     # all checks including the 65,536-input equality proof
just demo     # the full life of a weight patch, printed as an evidence log
```

## Key numbers from the study (simulation only)

- Learned peer selection: 15–18% faster than random; safe **only** when the
  kernel chooses the candidates. A policy that took whoever dialled in lost
  88% of its slots to colluders holding 30% of nodes.
- Where nodes are alike, learning gains ~3% (not worth switching on). Where
  they differ: 3–15% on average, up to 45% for the nodes the fixed rule serves
  worst.
- The worst hostile policy inside the recommended limits cost ~3.9% on average
  and was removed within a few thousand decisions; an attacker shaping a
  node's evidence pushed ~1 in 10 nodes past the 5% ceiling, worst 7.9%.
- A protocol rule written as weights matched the rule on all 65,536 inputs;
  the checks caught both an editing mistake and a planted backdoor.

These are measurements in simulation, not guarantees. Nothing has run on real
devices yet; Experiment 16 (~50 devices in Measure mode for a month) is the
cheapest way to learn whether the learning half pays at all.

## Relationship to x0x

The plan is **rebuild beside, bridge, and cut over when it measures better** —
never evolve the current crate in place. Earliest cutover is 10–15 months from
Phase 0; x0x users keep what they have, every x0x identity signs a link to its
new principal, and a bridge outlives the switch. Two results would stop the
rebuild and return to evolving x0x: a Phase 1 in which the simulated kernel
cannot meet the invariants, or a Phase 2 with no measurable improvement over
x0x on real hosts.

## License

MIT OR Apache-2.0, at your option.
