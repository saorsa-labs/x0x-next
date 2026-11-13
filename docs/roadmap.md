# Roadmap

---

## Rebuild, not evolve

Write a new kernel from a specification, alongside today's x0x, and cut over when it measures better. Do not try to evolve the current crate into this design.

Four reasons.

- **The kernel cannot be extracted.** The protocol is written inside API handlers and bound to live sockets. x0x's own plan since September has been to retrofit single chokepoints for authority, delivery and group encryption, and the accepted records still list each as incomplete.
- **Nothing on the wire carries over.** Sessions and logs replace gossip topics. The group model and the authority model both change. A bridge between two networks is simpler than a migration inside one.
- **A rebuild is cheap when agents write the code and a specification and simulator come first.** x0x's source tree reached 782,000 lines in nine months, nearly half of them tests. The new kernel should be a small fraction of that.
- **It is what was asked for:** build separately, switch when clearly better.

Until cutover, x0x takes security and correctness fixes and no new protocol surface.

---

## Seven phases

Seven phases, each closed by evidence and not by a date. The first three reach the point where the new network can replace x0x for today's users. The last four switch learning on where it pays and add the open network, the Forge and the Charter. The policy mechanism itself is built in Phases 0 to 2 and ships switched off. Cutover is an event and not a phase: it comes when its conditions hold.

### Phase 0 — Specify (1 to 2 months)

**Builds:** The kernel specification, the invariants in checkable form, the decision points with their features and fixed rules, the frame of the policy plane (the ceiling on harm and the outer limit, the trial and watch rules, the admission tests for each rung), device budgets, the Charter's rules, a skeleton of the Twin.

**Gate:** A model checker finds no violation of any invariant in bounded models. Two model families and the human owners have reviewed the specification.

### Phase 1 — Kernel in the Twin (3 to 5 months)

**Builds:** The kernel without real bearers: identity, sessions, capabilities, scopes, logs, sync and delivery, group keys, rendezvous and accounts, running only in simulation, with fault injection. The estimators and the whole policy mechanism with every slot empty: the band, the evaluator and checker for rungs 1 and 2, trials and the watch.

**Gate:** A million simulated node-hours with no invariant violation. Every x0x harness scenario is replayed and either cannot occur or is handled. Every hostile policy in the attack suite stays under the ceiling and is removed by the watch.

### Phase 2 — Owner core on real devices (3 to 5 months)

**Builds:** The QUIC and TCP bearers, the local API, the command line, adapters for MCP and A2A. What today's users rely on: key-value stores, task lists, files, boards, and reaching a named service on one's own machine under a capability. Threshold-signed releases and the updater. One principal's devices, then two principals sharing agents. Rendezvous and relay through the principal's own reachable devices and a few seed peers, as x0x's bootstrap nodes do today. The policy mechanism on real devices, in Off or Measure mode, including the fitting that Measure mode needs for its report. The bridge from x0x.

**Gate:** x0x's own two-person end-to-end test passes. Delivery invariants hold through a soak on real networks. Device budgets are met on a Raspberry Pi 5 and a phone. An outside cryptographic review is complete.

### Phase 3 — First adoptions (1 to 2 months)

**Builds:** The field measurements of Experiments 15 and 16. The first trials and adoptions on real traffic. Patches shared among one principal's devices. Rung 2 at the decision points that need it.

**Gate:** The measured worst case inside the default band is under the ceiling. A refined policy wins its trial on real traffic. A deliberately bad policy is removed with no human action. Or the measurements show that real nodes are alike, and the phase closes with learning left off.

### Phase 4 — Open network (3 to 5 months)

**Builds:** The public mesh opened to any reachable peer: the peer sampler, relay and mailboxes with reciprocity, offers, custody, the multi-hop privacy tiers.

**Gate:** On a hostile testnet with up to 30% colluders, capture and eclipse stay within the predicted bounds, and no relationship fails when the whole public mesh is hostile.

### Phase 5 — Forge (2 to 4 months)

**Builds:** Patch feeds, evidence bundles, attack suites written by other agents, tools for agents to write, read and edit weights, and the admission tests for rung 3.

**Gate:** A patch written by an outside agent is adopted by independent nodes on their own trials. A patch that scores well in the Twin and fails in the field is rejected.

### Phase 6 — Charter (2 to 4 months)

**Builds:** A second implementation written from the specification by a different model family, ratification mechanics, the end of the founding signers' special role.

**Gate:** Two implementations interoperate across the full scenario suite. A kernel change is ratified and adopted with no single signer able to force it.

---

## Rough effort

Ten to fifteen months to the earliest cutover. Fifteen to twenty-seven months for all seven phases. These are judgement, not a plan.

| Phase | Calendar months | What the range depends on |
|-------|----------------|--------------------------|
| 0. Specify | 1 to 2 | How much of the specification can be machine-checked |
| 1. Kernel in the Twin | 3 to 5 | Group keying, how fast the simulator runs, and the policy mechanism, which is new |
| 2. Owner core on real devices | 3 to 5 | NAT behaviour on real networks, mobile operating-system limits, and how much of today's surface your users still want |
| 3. First adoptions | 1 to 2 | How unlike real nodes turn out to be |
| 4. Open network | 3 to 5 | Risk 1, fair sampling |
| 5. Forge | 2 to 4 | Risk 2, simulator fidelity |
| 6. Charter | 2 to 4 (governance open-ended) | Risk 7, the electorate |

The estimate assumes three to five people directing agent teams: an architect who owns the specification, a cryptographer or formal-methods engineer, a reliability engineer who owns the Twin and the testnets, and an owner for the adapters and the agent-facing surface. It assumes outside cryptographic review before the gates of Phases 2 and 4. It is anchored on one fact: agents wrote today's x0x in nine months. In the new design, checking sets the pace, not code volume.

---

## Cutover conditions

The new network replaces x0x when all six conditions hold. None requires a single adopted policy, the Forge or the Charter, so cutover can follow Phase 2.

1. **Parity for what is still wanted.** Each core capability in x0x's purpose record has an equivalent that passes its acceptance test.
2. **Invariants hold in use.** Three months of daily use by the team and the embedders with no invariant violation.
3. **It measures better on the same hosts.** Reach, delivery latency and idle cost are at least as good as x0x on every measure and better on most, side by side on one fleet.
4. **Nothing is stranded.** Each x0x identity signs a link to its new principal, so contacts carry over. Key-value stores, task lists and history export and import. A bridge agent sits in an x0x group and a new scope and relays between them.
5. **x0x has an end date.** Security fixes only, then archive.
6. **The bridge outlives the switch** by a fixed period.

Releases are threshold-signed from Phase 2, so the new network never runs under a single signer.

---

## Two results that stop the rebuild

Two results would stop the rebuild and return to evolving x0x.

- A Phase 1 in which the simulated kernel cannot meet the invariants.
- A Phase 2 that shows no measurable improvement over x0x on real hosts.

---

## What carries across from today's x0x

- **ant-quic transport core** — QUIC transport, NAT traversal and address discovery, as the first IP bearer, with its handshake replaced in time.
- **saorsa-pqc primitives** — the post-quantum cryptographic building blocks.
- **Invariants and harness scenarios** — which become test cases for the Twin.
- **The onboarding recipe and the shape of the local API.**
- **The cross-model review practice.**
- **The operating knowledge in the runbooks.**
