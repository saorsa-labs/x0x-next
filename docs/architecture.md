# Architecture

The recommended network has a small fixed kernel that every node shares, and a policy plane in which every node is free to differ. Its laws are common and change rarely and together. Its habits are local: written weights that a node may check, try and adopt, on top of fixed rules that decide by default.

---

## Eight principles

Eight principles decide every detail that follows.

1. **Relationships first.** Parties who already know each other need the open network for one thing only: a relay when neither side has a reachable device. Everything else the open network offers is a service among peers.

2. **One authority model.** Every right is a capability: a signed, attenuable, expiring delegation.

3. **One state model.** Everything shared is a signed log in a scope. Whoever holds an entry can serve it to anyone entitled to read it.

4. **One place for discretion.** Every choice the protocol leaves open is a named decision point with a fixed rule, written as ordinary code, that decides by default.

5. **Weights are written, and the kernel bounds them.** Each learned part is a set of weights in a form a node can check completely for itself. Its structure is written on purpose and its numbers are fitted inside it on the node's own evidence. Whatever the weights say, a policy chooses only among options the fixed rule rates as near-equal. Nothing is fitted among strangers.

6. **Evidence before trust.** What a node measured itself outranks anything it is told.

7. **Everything replayable.** The kernel is deterministic, so any run, in the field or in simulation, can be reproduced exactly.

8. **Change carries its proof.** A new set of weights or a new kernel arrives with evidence anyone can re-check. Adopting it is always the node's own choice.

Principles 1 to 3 are conventional; most of the improvement over today's x0x comes from them. Principles 4 to 8 are what let the network keep improving without a person at the gate.

---

## Kernel components

The kernel has twelve parts and is the only code every node must share.

| Part | What it does |
|------|-------------|
| Identity and keys | Every principal, agent and device holds its own keys. An identity is the hash of its first key record and never changes. |
| Sessions | A secure channel between two identities, independent of what it carries. KEM-authenticated handshake, not signature-based. |
| Bearers | Carry session frames. QUIC on UDP as first IP bearer; TCP, Bluetooth LE, LoRa and serial also supported. |
| Capabilities | The only source of authority. A chain of delegations naming issuer, subject, actions, scope and expiry. One `decide` function checks every request. |
| Scopes | A relationship between two parties, a group, or one principal's own space. Holds an authority graph and content logs that never fork. |
| Logs | All shared content: messages, records, tasks, file manifests. Entries are signed and merge by the rule of their data type. |
| Sync and delivery | Moves log entries between holders and makes delivery honest. Two holders reconcile by comparing ranges. |
| Group keys | Confidentiality that follows membership. Up to about 50 members: each member holds a sending key shared over pairwise sessions. Larger groups use a key tree. |
| Rendezvous | Finds a party's current address. Each party picks a few reachable peers to hold its sealed address record. No network-wide lookup structure. |
| Accounts | Bounds what a node spends and tracks what peers owe each other. Pairwise ledgers with countersigned receipts. |
| Estimators | Turn what the node observes into the features policies see. Fixed arithmetic over a recent window; a peer never reports its own features. |
| Shield, evaluator and checker | Runs the written policies and contains them. The shield is the fixed rule; the band around it limits any policy to options the fixed rule rates as near-equal. |

### Five parts around the kernel

| Part | What it does |
|------|-------------|
| Local API and adapters | How agents and applications use a node. Loopback API with typed errors and an event stream. Two inboxes: requests (must cite a capability) and content (data only). Adapters map to MCP and A2A. |
| Public mesh | The open network's shared services. Reachable volunteer peers hold rendezvous records, relay sealed frames, keep mailboxes and index offers. Nothing in it is trusted. |
| Evidence plane | What policies are refined from. The node's own measurements and countersigned receipts, in a bounded local store. It never leaves the node unless the principal exports it. |
| Twin and Forge | Where weights are tested and written. The Twin is a deterministic simulator running the real kernel. A Forge is any agent using the Twin to write and edit weights and publish them as signed patches. |
| Charter | How the kernel itself changes. The Constitution's rules as checks: voluntary adoption, two-stage approval for kernel changes, no permanent constraint, no single signer. |

---

## Policy plane

Every choice the protocol leaves open has a fixed rule, written as ordinary code in the kernel. A node that never does anything else is a complete and correct node. Beside each fixed rule is a slot for a policy: a set of written weights that may reorder the options the fixed rule already rates as close. The slot ships empty. Filling it is the node's own choice, and the kernel removes whatever is in it as soon as it measures worse than the rule.

**Decision points.** Each decision point has a name, a list of options, and the measurements the node may use. Features are a short list of measurements cut to a few levels, measured by the node itself. A peer never reports its own features.

**Fixed rules.** Code that scores every option. It decides when no policy is adopted, and is the yardstick every policy is measured against and the fallback when one is removed.

**The band.** The options the fixed rule scores within a margin of its best. A policy chooses only among these. The band's margin is set per node, within a ceiling the Charter sets.

**Trial and watch.** A candidate policy runs beside the fixed rule on live inputs and counts the decisions the two would make differently. On one in five of those differences the node sends a copy the candidate's way. A sequential test stops the candidate when it is clearly worse and adopts it only when it is clearly better, on both the typical outcome and the total. Once adopted, one contested decision in ten is still compared with the fixed rule. If the policy falls behind, the node goes back to the rule by itself.

**Three node modes.** Each node runs in one of three modes, set by its principal.

- **Off.** Fixed rules only. This is today's x0x behaviour with better structure.
- **Measure.** The node gathers evidence and reports what a fitted policy would have gained on its own traffic, and what margin it would need. Nothing is adopted.
- **Adopt.** Trials, adoption and the watch are live.

---

## Ladder of forms

The form of a policy is not fixed for all time. It is a rung on a ladder, and each decision point sits on the lowest rung that can express what it needs. Moving a decision point up a rung is a decision for the Charter, taken on evidence.

| Rung | Form | What it can express | How a node checks it | Status |
|------|------|--------------------|--------------------|--------|
| 0 | Fixed rule — ordinary code in the kernel | Whatever was written | Ratified with the kernel | Ships, deciding everything |
| 1 | Points table: one weight per step of each feature, added up | The formula the author knew; each feature's own curve, whatever its shape | Reads the numbers; directions and limits are a few inequalities | Ships, empty. Prototyped. |
| 2 | Points table per context, and tables for declared pairs | Decisions that depend on message size, link type, time of day, named two-way interactions | Reads the numbers, one table at a time | Ships, empty. Tables per context prototyped. Pair tables not yet. |
| 3 | Small written network with hidden units, on a decision small enough to list every possible input | Any interaction among a few features | Computes every input and runs every check on that table: 9 ms for 65,536 inputs | Specified, switched off. One hidden unit prototyped. |
| 4 | Larger written network that arrives with a certificate: a machine-checkable proof that it has the declared properties | Many features that interact | Runs a small proof checker on the certificate, trusting nothing the tool that produced it says | Not available today at useful size |

**Admission rule.** No form is admitted unless a node can check it completely, on its own hardware, in bounded time. Anything that cannot be checked that way stays outside the kernel. A large model can still contribute, by writing patches in a form that can be checked.

**Five admission tests.** The Charter admits a rung at a decision point when all five pass.

1. **Need.** Measured evidence that the current rung leaves gain behind at this decision point. A trained network in the Forge is the instrument: if it does no better than the best policy on the current rung, there is nothing to climb for.
2. **A complete check.** A procedure that establishes every declared property of any policy in the new form, with no sampling. Reading for rungs 1 and 2. Listing for rung 3. A proof for rung 4.
3. **A device budget.** The check and the evaluator run within stated time and memory on a phone and on a Raspberry Pi.
4. **A bounded evaluator.** Integer arithmetic, a fixed number of steps, no dependence on input values, and two independent implementations that agree on every bit.
5. **An attack round that can fail.** Agents write hostile policies in the new form whose aim is to be adopted and kept while doing harm. The band holds for any form; the question is what a richer form can get past the trial and the watch. The largest harm that any such policy sustains must be under a stated figure. Every escape from the checks becomes a new check before admission.

---

## What is never learned and why

Six things are fixed and outside the policy plane entirely.

- **Crypto.** The primitive choices (ML-DSA-65, SLH-DSA, ML-KEM) are ratified with the kernel. A policy cannot change them.
- **Identity.** An identity is the hash of its first key record. A policy cannot reassign or shadow it.
- **Authority.** Every right is a capability chain checked by a single `decide` function in the kernel. A policy cannot skip or extend that check.
- **Membership.** Who is in a scope is determined by a signed authority graph with a fixed merge rule. A policy cannot read membership and cannot influence it.
- **Delivery.** A policy cannot decide whether an entry is delivered, only which path is used. Delivery invariants are kernel code.
- **Privacy timing.** The rendezvous mechanism hides who asks for a record and who holds it. A policy cannot change that.

These are fixed because a policy that could influence them could break the properties that make everything else checkable. If those properties hold only sometimes, they hold for the cases the attacker finds least interesting.

---

## Weight patch enters service: eight steps

A patch earns its place on each node separately and can always lose it.

1. **Write.** An agent writes a policy's weights, or edits named weights of an existing one, and states the intent in a sentence.
2. **Check.** Every score must move the declared way and every limit must hold. The author gets the exact list of decisions that differ from the policy being replaced.
3. **Score.** The Twin runs the patch through ordinary and hostile scenarios, including attacks that other agents wrote.
4. **Publish.** The patch goes to a feed, signed by its author, with the evidence.
5. **Re-check.** An interested node applies the patch to its own copy and runs every check itself. It trusts nothing the patch says about itself.
6. **Shadow.** The node runs the candidate beside the fixed rule on live inputs and counts the decisions the two would make differently. The candidate decides nothing yet.
7. **Trial.** On one in five of the decisions where the two differ, the node sends a copy the candidate's way and compares the pair, so live traffic is untouched. Where a decision cannot be duplicated, the candidate decides one in ten of them. A sequential test stops the candidate when it is clearly worse and adopts it only when it is clearly better, on both the typical outcome and the total.
8. **Adopt and watch.** It takes over that one decision point, inside the band. One contested decision in ten is still compared with the fixed rule for as long as the policy runs. If the policy falls behind, the node goes back to the rule by itself.
