# Experiment 16: how unlike are real nodes?

Plan of 2026-11-13. Funded by Decision 7 (docs/decisions.md, ruled 2026-11-13). Source: the design study (docs/design-study-2026-11-13.pdf), risk 3 "Learning may not pay at realistic scale" and risk 16 (p. 47–48), Experiments row 16 (p. 49), and the three node modes (p. 26).

---

## What the experiment settles

The study's whole case for learning rests on nodes being different. Among alike machines the gap between the fixed rule and a perfect chooser was 3% in the toy worlds, and about 2% of it was reachable inside the default band. The larger gains, 3 to 15%, need nodes that differ a lot, and nobody knows how much real nodes differ. Where the right formula is known, ordinary code with a few constants fitted per node gets the same gain with less machinery.

So we run about 50 devices of different kinds in Measure mode for a month. On each node we compare three scorers for one decision point (scoring a candidate peer, the POC's decision point):

1. **The best single table**: one rung-1 points table shared by every node.
2. **A formula with constants fitted per node**: ordinary code, a handful of numbers per node.
3. **A table fitted per node**: the rung-1 table fitted on that node's own evidence.

**Passes if** fitting per node beats the best single table by 10% or more for at least a tenth of nodes.

**If it fails**, as the study states it: adoption stays off by default. What the measurements show is fixed in the rules themselves, as code, in the next release. If the formula matches the table everywhere, we ship the formula with per-node constants and leave the table slots empty.

Either result is useful. A pass funds the learning half of the rebuild. A fail saves the band, the evaluator, the checker, the trial and the watch from becoming kernel surface that is never used (risk 16), and the instrumentation still pays for itself.

---

## 1. The instrument: crates/measure-agent

A new small Rust crate in this workspace, to be built next. It runs one decision point in Measure mode and nothing else: no adoption, no weights, no trials, no patches. It depends on `policy-poc` for `Features`, `FixedRule` and `band_pick`, so the rule and the band it runs are the exact code the analysis replays.

**What it does per decision.** The agent has a request to place (a request/response of a given size). It draws up to four candidate peers (`DEFAULT_CANDIDATES` in twin.rs), measures each one's five features itself, and picks with the fixed rule. On one decision in ten (`EXPLORE_SHARE`) it picks uniformly at random among the candidates inside the band at margin 6 quanta, which is what Measure costs in the study. On one decision in ten it also sends a copy to a second in-band candidate, so the trace holds paired outcomes. It then records the realized outcome: completion time in ms, or the timeout and a failure flag.

**The five features**, measured by the node, never reported by a peer, with level boundaries fixed in the trace schema:

- rtt, 16 levels: smoothed QUIC RTT, log-spaced boundaries.
- loss, 8 levels: QUIC loss rate to that peer over a sliding window.
- delivery, 8 levels: share of this node's past requests to that peer that completed.
- age, 8 levels: time since first successful contact, log-spaced.
- balance, 8 levels: bytes served to that peer minus bytes it served us.

**Trace record** (one JSON line per decision): schema version, decision number, monotonic and wall-clock time, request size, every candidate's hashed peer id and five levels, the rule's choice, the actual choice, the propensity of the actual choice, whether a copy was sent, and each realized outcome. Peer ids are hashed with a per-node salt and IP addresses are never written, so an exported trace does not reveal who the node talks to. Traces are append-only JSONL, one file per UTC day, with each closed file's SHA-256 written into a manifest.

**Budget on the device.** Under 30 MB resident and 1% of one core on a Pi 5. One decision every 10 s on mains, one every 60 s on phones, with request sizes from 1 KB to 64 KB on mains and up to 8 KB on metered links.

**Piggyback on x0xd, or probe standalone.** Two ways to get the traffic:

- *Piggyback*: a sidecar reads a running x0xd's local API (`/health`, `/diagnostics/connect`, `/diagnostics/dm`, `/peers/events`) and its metrics.
- *Standalone*: the agent is its own ant-quic endpoint (exact-pinned like x0x) and places its own requests to other measure-agents over QUIC.

**We recommend standalone.** Measure mode needs the node itself to make the decision: to see the whole candidate set, pick with the rule, explore inside the band and log the propensity. x0xd does not expose its peer choices as decisions; its API gives aggregates and per-peer counters, not candidate sets, so the three scorers could not be compared on what x0xd logs. Piggybacking would also mean adding surface to x0x, which Decision 1 rules out until cutover, and phones and Pis do not all run x0xd. The cost is that the workload is ours, not users' messages. The paths, NATs, links, radios and peers are real, and that is what the experiment measures. To keep the workload honest, we shape request sizes and rates on the hosts that run x0xd from that host's own `/diagnostics` message counts, read once per hour. Those counts are covariates; nothing else from x0xd is read.

The agent listens on UDP 7483 on the VPS (prod x0xd uses 443 and 5483, testnet uses 6483) and on a port in 9000–9999 on David's machines. Opening 7483 on the prod hosts needs David's direct go-ahead before the pilot.

---

## 2. The devices

Built locally, never on the hosts: `cargo zigbuild --release --target x86_64-unknown-linux-gnu` for the VPS and droplets, the aarch64 Linux target for the Pis and the Lima VM, and the aarch64 Android target for phones (run under Termux). iPhones are left out: they cannot keep a background agent running for a month.

| Class | Count | Cost / month | What diversity it adds |
|-------|------:|-------------:|------------------------|
| Saorsa VPS (NYC, SFO, Helsinki, Nuremberg, Singapore, Sydney) | 6 | £0 extra | Six continents' backbones, public IPv4, no NAT, datacentre, mains. Small hosts shared with prod x0xd. |
| David's local machines (2 Mac Studios, 1 Lima VM) | 3 | £0 | Home NAT, double NAT inside the VM, Ethernet and Wi-Fi, one ISP. Mains. |
| DigitalOcean droplets, ephemeral, mixed sizes ($4 to $24) across 9 regions incl. LON, AMS, FRA, TOR, BLR | 29 | ~£185 | Region spread and host size spread (512 MB to 4 GB), 5 set IPv6-preferred. Public IP, no NAT, mains. Ephemeral droplets are covered by standing authorization. |
| Raspberry Pi 5 at volunteers' homes (Ben, friends) | 6 | ~£20 (2 SIMs) | Home NAT on different ISPs, 2 on 4G CGNAT via USB dongle, Wi-Fi and Ethernet, the device-budget end of mains. |
| Android phones (2 bought, 4 volunteers') | 6 | ~£20 (2 SIMs) | Mobile CGNAT, Wi-Fi to cellular handover, battery, OS suspension. |
| **Total** | **50** | **~£225** | |

The mix leans to datacentres: 35 of 50 nodes have public addresses. That makes a pass more convincing (nodes differ even there) and a fail less so (homes and phones are where nodes should differ most). Where a volunteer offers a home device, it replaces a droplet one for one. Each node's class, link type, NAT type (measured at start-up) and power source are in its trace header and are reported with every result.

Phones run Measure even though the study recommends Off on battery, by our choice as principals for this experiment. Each phone logs charging state per decision, so battery-time and mains-time can be separated in the analysis.

---

## 3. Run protocol

**Length.** 28 days, four whole weeks, Monday 00:00 UTC to Sunday 24:00 UTC.

**Evidence stays on the node.** The study's principle holds: evidence never leaves a node unless its principal exports it. Traces live in the agent's data directory on the device. Nothing is sent anywhere automatically. Once a week we, as principals, run `measure-agent export --week N`, which writes a tarball of that week's closed daily files and their manifest. On the VPS, droplets and Macs we pull it over SSH. Volunteers are the principals of their own devices: they run the export and send the file, or decline.

**Versioned schema.** Every trace file starts with a header line: schema id (`x0x-measure/1`), agent version and git commit, node class, the per-node seed, and the level boundaries of all five features. Any change to boundaries or fields bumps the version. The analysis refuses to mix versions inside one node's window. The pilot may change the schema freely; it is frozen at version 1 before the full run, and the full run uses version 1 only.

**Clock and seed discipline**, so every run replays in the Twin:

- Each record carries monotonic nanoseconds since agent start, UTC wall time, and the host's NTP offset estimate. Hosts must run NTP; offsets over 100 ms are flagged.
- All randomness (candidate sampling, exploration, copies) comes from one `SplitMix64` stream (rng.rs), seeded from the experiment seed and a hash of the node id. The seed is in the header.
- A record holds everything the decision used. Replaying a trace through `band_pick` with the same seed reproduces every choice without the network. A replay test in the crate checks this on the pilot traces.

**Restarts.** An agent restart starts a new trace segment with a new header and the same seed path, advanced past the decisions already logged. Uptime gaps are reported per node; a node with under 14 days of trace in weeks 1 to 3 or under 4 days in week 4 is reported but not counted.

---

## 4. Analysis

The analysis is a binary in measure-agent that reads exported traces and reuses `policy-poc`. Fitting uses `fit::fit_table` (backfitting on ln cost, monotone by pool-adjacent-violators, written within the checker's limits). Every fitted table must pass `Checker::check` before it is scored.

Per node, the three scorers are fitted on weeks 1 to 3 and scored on week 4:

1. **Best single table**: `fit_table` on the pooled weeks 1 to 3 of all nodes. The fixed rule is the other entry; whichever has the lower pooled cost on weeks 1 to 3 is the best single table.
2. **Formula with per-node constants**: a new `fit_formula` beside `fit_table` that fits the constants of twin.rs's `WorldParams` (base cost, rtt power and growth, loss slope or loss timeout, delivery multipliers) to the node's own ln cost by least squares. It fits both world kinds and keeps the one with the lower training error. The fitted formula scores candidates through the `Scorer` trait.
3. **Per-node table**: `fit_table` on the node's own weeks 1 to 3.

**Scoring on week 4.** The trace only shows the outcome of the peer used (and of the copy). Each scorer's week-4 cost is estimated from the explored decisions by inverse-propensity weighting with the logged propensities, self-normalized, and checked against the paired copies. A node's gain is `(cost of best single table − cost of per-node scorer) / cost of best single table`, for the better of the formula and the per-node table, and for each separately.

**The pass metric**, as the study states it: the share of counted nodes whose per-node gain is 10% or more. Pass if that share is at least a tenth (5 of 50). To count, a node's 90% bootstrap lower bound on gain must also be above zero. These numbers are fixed in writing on the day the schema freezes, before the full run starts.

**Formula versus table.** The formula "matches the table" on a node when its week-4 cost is within 2% of the per-node table's. "Everywhere" means every counted node. This threshold is also fixed before the run.

**Band margin spread.** For each node and each margin from 1 to 10 quanta, we report two things: the gathering cost (Measure's exploration and copies against the pure fixed rule, measured on week 4), and the share of the per-node gain reachable inside the band. We give each node the smallest margin that reaches 90% of its gain, and its gathering cost there. The distribution (median, 90th percentile, worst) is set against the study's figures: 0.1% on average at margin 3, 0.3 to 0.4% at margin 6, and the worst-fitted node up to 1.2%. Any node above 1.2% is named.

**Side results.** Which world kind fits each node better tells us whether real nodes show the rtt × loss interaction that rung 1 cannot express (risk 5). The traces are also the field logs Experiment 15 needs: the spread of outcomes inside the band and the weight of the tails. A secondary split, scoring on each week in turn, is reported because weeks 2 and 3 fall over the holidays and week 4 does not. The headline result is the weeks 1–3 / week 4 split.

---

## 5. Milestones

| Dates | Milestone | Done when |
|-------|-----------|-----------|
| Mon 16 Nov – Fri 27 Nov 2026 | Build measure-agent | Crate passes fmt, clippy `-D warnings`, docs; replay test reproduces every choice; cross-builds for all three targets. Pis, phones and SIMs ordered on 16 Nov. |
| Mon 30 Nov – Sun 6 Dec | Pilot: 6 VPS + 2 Mac Studios | 8 nodes run 7 days; export, replay and the full analysis run end to end on pilot traces. David's go-ahead on UDP 7483 for prod hosts before start. |
| Mon 7 Dec – Fri 11 Dec | Fix the schema; roll out | Schema frozen at `x0x-measure/1`; pass, count and match thresholds written into this file; droplets created; Pis and phones installed with volunteers. |
| Mon 14 Dec 2026 – Sun 10 Jan 2027 | Full 50-device month | Weekly exports on 21 Dec, 28 Dec, 4 Jan, 11 Jan. Weeks 1–3 = 14 Dec – 3 Jan; week 4 = 4 – 10 Jan. |
| Mon 11 Jan – Fri 15 Jan 2027 | Analysis week | Per-node results, pass metric, formula-versus-table result, margin spread. Droplets destroyed by verified ID on 11 Jan after the last export. |
| Mon 18 Jan 2027 | Decision memo | Appended to docs/decisions.md under Decision 7, with the result and which branch below follows. |

**Consequences, as the study states them.**

- *Pass* (per-node fitting beats the best single table by 10% or more for at least a tenth of nodes): the learning half stays in the rebuild as planned. Measure mode on mains stays the default, and Phase 3's first trials go ahead at this decision point.
- *Fail*: adoption stays off by default. What the measurements show is fixed in the rules themselves, as code, in the next release.
- *Fail, and the formula matches the table everywhere*: ship the formula with per-node constants and leave the table slots empty.

---

## 6. Budget

| Item | Cost |
|------|-----:|
| 29 droplets for about 5 weeks (rollout week plus the month): 10 × $4, 10 × $6, 7 × $12, 2 × $24 = $232 a month | ~£220 |
| 4 data SIMs (2 Pis on 4G, 2 bought phones), ~£10 each a month | ~£50 |
| **Running total** | **~£270** |
| 6 Raspberry Pi 5 (4 GB) kits with PSU, case and SD card, ~£85 each | ~£510 |
| 2 budget Android phones, ~£120 each | ~£240 |
| 2 USB 4G dongles, ~£30 each | ~£60 |
| **One-off hardware** | **~£810** |
| **Experiment total** | **~£1,080** |

The VPS and David's machines add nothing. People's time (building the agent, looking after the volunteers, the analysis) is not costed here. The hardware stays useful afterwards: the Pis and phones are the device-budget targets of the Phase 2 gate.
