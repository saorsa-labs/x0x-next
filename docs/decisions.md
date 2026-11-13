# Decisions for Ben and David

These are the decisions the design cannot make for you. Each has a recommendation from the study. None blocks Phase 0. Record a ruling by appending a dated note under the relevant heading.

The three that matter most are decisions 1, 2 and 7. Decision 1 commits ten to fifteen months before anything replaces what you run today. Decision 2 determines the pace of governance for every weight the network will ever adopt. Decision 7 is the cheapest way to learn whether the learning half of the design is worth building at all.

Rulings of 2026-11-13 are David's and are final (David, 2026-11-13: proceed without waiting on Ben); Ben reviews asynchronously.

---

## Decision 1 — Rebuild beside x0x and cut over when the new network measures better, or evolve x0x in place

**What it is.** A choice between writing the new kernel from a clean specification alongside today's x0x, then switching when it proves better, versus reaching this design by modifying the existing codebase from the inside.

**Recommendation.** Rebuild. Until cutover, x0x takes security and correctness fixes and no new protocol surface.

**Why it is yours.** It commits ten to fifteen months before anything replaces what you run today.

> **This is one of the three that matter most.**

**Ruling (2026-11-13, David):** Rebuild beside x0x and cut over when the new network measures better. Until cutover, x0x takes security and correctness fixes and no new protocol surface. The two stop-conditions in docs/roadmap.md stand: a Phase 1 whose simulated kernel cannot meet the invariants, or a Phase 2 with no measurable improvement over x0x on real hosts, returns us to evolving x0x.

Status: RULED (2026-11-13)

---

## Decision 2 — What needs the Constitution's two-stage vote (Article 11 reading)

**What it is.** The Constitution requires a two-stage vote for changes to "fundamental operating parameters of the network." The question is whether a node's written weights count as operating parameters, or whether they are local and outside that rule.

**Recommendation.** Adopt the reading that puts weights outside the vote, and put it to the community. The frame is a "fundamental operating parameter of the network" and needs the vote: the fixed rules, the ceiling on harm, the outer limit, the trial rules and the list of admitted policy forms. A node's own weights are not, on four conditions the kernel enforces: they are local to the node, they can only order options the kernel has allowed, the kernel supplies the candidates and the limits, and adoption is optional. Two further questions belong to the same vote. At what share of nodes does a widely followed set of weights become a parameter of the network in fact, to be ratified into the fixed rule? And does a network at rung 3 or 4 count as part of "the x0x codebase"?

**Why it is yours.** Interpreting the Constitution is not a design decision. If weights are ruled to be operating parameters, every patch needs ratifying and learning slows to the pace of governance.

> **This is one of the three that matter most.**

**Ruling (2026-11-13, David):** Adopt the study's reading and put it to the community. The two-stage vote covers the fixed rules, the ceiling on harm, the outer limit, the trial rules and the list of admitted policy forms. A node's own weights are local, on the four conditions the kernel enforces. The two follow-on questions (at what share of nodes a widely followed weight set becomes a network parameter in fact, and whether rung 3/4 nodes count as part of "the x0x codebase") go to the same community vote.

Status: RULED (2026-11-13)

---

## Decision 3 — Two gaps in Article 11 that the design cannot close

**What it is.** Article 11 as written puts every code change through the vote, which blocks an expedited path for security fixes. It also does not name the electorate while no ledger exists to count it.

**Recommendation.** Add an expedited path for security fixes, because the article as written puts every code change through the vote. Decide who the electorate is while no ledger exists to count it. The study suggests a named founding council with a published end date.

**Why it is yours.** Both changes change the Constitution or its practice.

Status: OPEN

---

## Decision 4 — The default mode for learning: Off, Measure or Adopt

**What it is.** Each node runs in one of three modes. Off means fixed rules only, today's x0x behaviour with better structure. Measure means the node gathers evidence and reports what a fitted policy would have gained, but nothing is adopted. Adopt means trials, adoption and the watch are live.

**Recommendation.** Adoption off everywhere. Measure on mains-powered machines. Off on battery, metered and radio links.

**Why it is yours.** Measure makes about one decision in ten at random among near-equal options and sends some duplicate messages. In tests that cost 0.1 to 0.4% on average and up to 1.2% on the worst-fitted node. Some owners will want nothing of the kind.

Status: OPEN

---

## Decision 5 — Two numbers for each decision point: the ceiling on harm and the outer limit

**What it is.** Each decision point has two thresholds: the ceiling on harm that a node's margin respects by default, and the outer limit that no earned room may pass.

**Recommendation.** 5% and 20%. Widening between them is allowed one step at a time, each step tried like any other change.

**Why it is yours.** It is a trade between how much a bad habit can cost and how much a good one can gain. Both numbers are ratified.

Status: OPEN

---

## Decision 6 — Which policy forms the kernel admits at launch

**What it is.** The kernel ships with a ladder of four forms (rungs 0–4). The question is which rungs are open on day one.

**Recommendation.** Rungs 1 and 2 present and empty. Rung 3 specified and switched off. Rung 4 not before its conditions are met.

**Why it is yours.** Every admitted form is kernel code that must be verified and maintained.

Status: OPEN

---

## Decision 7 — Whether to fund a field measurement of how unlike real nodes are

**What it is.** Experiment 16: about 50 devices in Measure mode for a month, to record how much real nodes differ from each other and whether a fitted policy would have gained anything.

**Recommendation.** Yes: about 50 devices in Measure mode for a month, as Experiment 16. It is the cheapest way to learn whether the learning half is ever worth switching on.

**Why it is yours.** It is the one experiment that decides the value of half the design, and it needs real devices and people.

> **This is one of the three that matter most.**

**Ruling (2026-11-13, David):** Yes. Fund ~50 devices of different kinds in Measure mode for about a month, comparing the best single table, a formula with per-node constants, and a table per node. This runs before and independent of the rebuild's learning half and decides whether learning is ever worth switching on.

Status: RULED (2026-11-13)

---

## Decision 8 — When to build the open-network half

**What it is.** The design has two halves: an owner-centric half (Phase 0–3) and an open-network half (Phase 4+, strangers finding each other without a platform). The question is whether to build both or stop after the owner core.

**Recommendation.** After the owner core, as Phase 4, and only once there is evidence that strangers' agents want to find each other without a platform.

**Why it is yours.** The repository shows demand for the owner-centric half. It does not yet show demand for the other.

Status: OPEN

---

## Decision 9 — Two smaller constitutional readings

**What it is.** Two points where the Constitution is ambiguous: whether one admin acting alone can remove a group member, and whether pairwise service ledgers should be put to a vote under Article 16.

**Recommendation.** Make two signatures the default for removing a group member, since one admin acting alone can be read as unilateral punishment under Article 13. Put the pairwise service ledgers to an explicit vote under Article 16.

**Why it is yours.** Both are readings of the Constitution where reasonable people differ.

Status: OPEN

---

## Decision 10 — People and outside review

**What it is.** The study's cost estimate assumes outside cryptographic review before the gates of Phases 2 and 4, and a formal-methods engineer from Phase 0. The corrections made after the third independent check have not been checked by anyone else.

**Recommendation.** Budget an outside cryptographic review before the gates of Phases 2 and 4, and a formal-methods engineer from Phase 0. Before Phase 0 starts, have a fourth reviewer check the corrections made after the third check. Nobody but the study author has checked them.

**Why it is yours.** The estimate assumes the first two. The third costs an hour and closes the one gap in the reviewing of this study.

Status: OPEN

---

## How to record a ruling

Append a dated paragraph under the relevant decision heading. Example:

> **2026-11-20 — David:** Rebuild. We proceed with Decision 1 as recommended.

Leave the `Status: OPEN` line in place and add a `Status: DECIDED — <date>` line below it, so the open count stays easy to grep.
