# Third independent check: findings

Written by the independent reviewer on 13 November 2026 and saved here by the author of the study, unedited apart from this heading. The reviewer had not seen the work before. Its scripts, logs and result files are in this folder. Not production code; not part of x0x.

## 1. Verdict

1. No blocker. All eight experiments I re-ran reproduce the stored results exactly, and the band arithmetic is sound for the two toy worlds.
2. Nine significant claims are overstated or do not hold as described (F1 to F9). The executive summary states three things its own body contradicts.
3. The trial and the watch test the typical (logarithmic) outcome, not the average the harm figures use. Policies that raise average cost by 0.6 to 1.0% were adopted in 90 to 98% of 400 trials and kept, and reached 3.4% through "earning room".
4. "3.9%" and "5.3%" are properties of the toy worlds, not of the mechanism: 13 to 25% with fast candidates at the same margin, 7.3% when a node sets its margin from its own log as the text says, 8.7% when a tenth to a fifth of that log is shaped.
5. The response to Grok Heavy misstates its first point and tests a different ratio limit from the one it proposed. In the phone-against-server world, ordinary code with one constant estimated per node beats the per-node table.

## Findings, one line each

| # | Severity | Finding |
| --- | --- | --- |
| F1 | MAJOR | Trial and watch test logarithms; policies that raise average cost are adopted and kept. |
| F2 | MAJOR | "3.9%" and "never more than 5.3%" are stated without their conditions; the bound is a property of the candidate mix. |
| F3 | MAJOR | 5.3% does not reproduce when the margin is set from the node's own log (7.3%; 8.7% with shaped evidence). |
| F4 | MAJOR | "9 to 15%" is the best of three worlds; phone-against-server gives 3 to 4%, and code with one estimated constant beats the table there. |
| F5 | MAJOR | "Tested on duplicate messages first, so users see no harm" is true for three of ten decision points. |
| F6 | MAJOR | Watch figures hold for the worst policy only; mild harm lasts far longer and costs more in total. |
| F7 | MAJOR | Response to Grok: point 1 misstated, ratio-limit figures test a different limit, this check prejudged. |
| F8 | MAJOR | Article 11 reading is defensible, but the "mechanical test" is circular and the yardstick and widening rules break its conditions. |
| F9 | MAJOR | Two of six ladder admission tests cannot fail; "ready for better AI: yes" is unsupported. |
| F10 | MINOR | Eight numeric slips in hand-typed prose (3.5 for 3.4, 152 for 151, 36% for 31%, 12,700, "four steps", and others). |
| F11 | MINOR | W3's fair re-run still handicaps the trained network slightly; one row changes. |
| F12 | MINOR | Key recovery: theft of the next key alone is a takeover; the table says otherwise. |
| F13 | MINOR | Rendezvous: no hash-distance leftovers, but strangers cannot derive the lookup tag. |
| F14 | MINOR | Leftovers: seven "tab" references, "breaks ties at random", "a small network", "Everything here was tested in simulation". |
| F15 | MINOR | Day-one table, decisions and phases disagree on feeds, Measure mode and the "ceiling". |
| F16 | MINOR | The trial on copies duplicates one contested decision in five, not each. |
| F17 | MINOR | Constitution table is accurate row by row; Law 1 and Articles 14, 19, 20, 22 have no row. |
| F18 | MINOR | "Measure costs about 0.1%" is the band-of-3 figure; W6 gathered evidence in a band of 6 (0.3 to 0.4%, up to 1.2%). |
| F19 | MINOR | Copies served 30 to 40% faster get the worst policy adopted; a no-effect policy is adopted in about 0.6% of trials. |

## 2. Findings in full

Severity: BLOCKER, a central claim is false or unsupported. MAJOR, a significant claim is overstated or a result does not reproduce. MINOR, wording, small numeric slips, inconsistencies.

Every script is run as `cd /home/claude/work/review3 && PYTHONDONTWRITEBYTECODE=1 python3 <script>`. Each imports the study's own modules from my copy in `rerun/`.

### F1. MAJOR. The trial and the watch certify the typical outcome, not the average.

**Claim.** These four statements:
- "No policy that did harm inside the band was adopted in any of its 400 simulated trials" (section 5, "Safety that does not depend on the form").
- "Under the final rules no policy that does harm inside the band was adopted in any trial" (section 6, W8c).
- "adopts it only when it is clearly better" (section 5, "A weight patch enters service").
- "removes the habit automatically if it stops helping" (executive summary).

**Evidence.**
- `exp_trials.py` lines 39 and 92: `op = np.log(tp * rng.lognormal(...))`. Every verdict is a z-test on the mean of the logarithm of outcomes. Every harm figure is `Pool.harm`, the change in arithmetic mean cost. The document never says the trial compares logarithms.
- The study's own table shows the mismatch the other way: the "worst points table inside Version 3's limits", in a band of 6 in world one, changes mean cost by −0.13% and is stopped in 96.8% of trials.
- `attack_logmean.py`: an adversary with the same knowledge as the study's "worst policy that can exist" takes, per decision, the in-band option that maximises harm minus a multiple of the log ratio. It wins by ratio on cheap decisions and loses by amount on expensive ones.

| | World one | World two |
| --- | --- | --- |
| True change in average cost, band of 3 | +1.03% / +0.92% | +0.69% / +0.59% |
| Adopted in trials on copies (of 400) | 90.2% / 98.2% | 93.2% / 98.0% |
| Removed by the watch, three settings (of 400) | 2.8%, 3.8%, 4.5% | 2.5%, 3.5%, 2.5% |
| Change in the typical (geometric mean) cost | −0.22% / −0.25% | −0.24% / −0.27% |
| After "earning room": 6, 9, 12, 18, no band | +1.64, +2.11, +2.44, +2.85, +3.42% | +1.01, +1.16, +1.23, +1.33, +1.59% |
| Each widening granted | 79 to 92% | 88 to 96% |

- A rung-1 points table that passes `Steps.read_checks()` did the same on a smaller scale in world one: +0.18% in a band of 3 (adopted 98.2%) and +0.91% in a band of 6 (adopted 97.5%). My search found none in world two.
- The same trial on raw outcomes stops the attack in 100% of 400 trials, at about twice the length:

| Raw-outcome test | Median decisions | Copies | Under the log test |
| --- | --- | --- | --- |
| Honest fit (adopted 99.2%) | 20,295 and 20,602 | 245 and 328 | 10,139 and 10,218; 133 and 158 |
| Worst possible (stopped 100%) | 2,843 and 3,084 | 84 and 93 | 1,545 and 1,508; 50 and 44 |

- A new experiment appeared in the prototype folder at 17:38, after the draft was cut: `exp_metric.py` and `w8e_results.json` (W8e). It is not in the draft and I only read its log. Its "gap policy" (+1.33% and +0.98%) is adopted in 3.0% and 5.2% of trials under the log rule. That understates the weakness: its policy is not tuned to pass the test (mean log ratio −0.011 per contested decision, mine −0.06 to −0.08). Its raw-outcome figures agree with mine.

**Correction.** Say that the trial and watch compare typical outcomes. Replace "no policy that did harm inside the band was adopted" with "none of the six policies I tried". State that a policy can be adopted and kept while raising average cost by about 1% in the default band. Then require both tests to pass, at about twice the trial length, including for requests for a wider margin.

### F2. MAJOR. The bound is stated at summary level without its conditions.

**Claim.** "the worst habit that could exist cost about 4%, never more than 5.3% on any simulated machine" (executive summary). "The band holds any policy to 3.9%" (section 1). "The band. It applies to every single decision and not to an average" (section 8).

**Evidence.**
- The study's own W6 table: at the default margin of 3 the worst possible policy costs up to +23.7% on the worst simulated node (+23.3% and +12.8% in the other two kinds of difference). The 5.3% holds only where each node sets its own margin, which the summary does not say; see F3.
- `check_w6.py D`: same fixed rule, same two cost functions, same margin of 3, different candidates.

| Candidates | Worst possible, world one | World two |
| --- | --- | --- |
| The study's mix, round-trip level uniform over 16 | +4.0% | +3.5% |
| Fast only, levels 0 to 3 | +13.4% | +13.0% |
| Fast and clean, levels 0 to 3, loss level 0 or 1 | +24.8% | +25.0% |
| Very fast, levels 0 to 1 | +15.8% | +15.1% |
| Slow only, levels 8 to 15 | +4.7% | +3.2% |

  "One level of round-trip time" doubles the cost at the fast end. A few machines on good links, the owner core, is the fast case.
- `attack_watch_and_copies.py` part 1: per contested decision in a band of 3, the worst in-band option costs a median of 1.29 times the rule's pick, 3.40 and 2.90 times at the 99th percentile, and at most 6.24 and 4.05 times. About a tenth of contested decisions cost at least double.
- After "earning room" the worst case is 9.5%, 17.3% and 27.3% at margins of 6, 9 and 12 (the study's W8b), so a node that has widened is no longer under 5%.

**What the mechanism guarantees.** The chosen option's fixed-rule score is within the margin of the rule's best score. Every figure in cost terms depends on the rule, the node's cost function and its candidate mix.

**Correction.** Put the conditions in the executive summary and section 1. Delete "applies to every single decision and not to an average" or restate it in score terms. Make the default margin a measured quantity per decision point.

### F3. MAJOR. The 5.3% result does not reproduce when the margin is set as the text describes.

**Claim.** "Two remedies, both using only the node's own 2,000 logged decisions" and "at most 5.3% on every one of 1,600 nodes" (section 6, W6b); repeated in section 5, risk 6, section 8 and the executive summary.

**Evidence.**
- `exp_calibrate.py` lines 49 to 57: the cost table comes from the node's log, but the estimated worst case is summed over `P.C`, the 60,000-decision pool on which the true worst case is then evaluated. The node is given its exact candidate mix.
- The true worst case (line 58) is computed from true costs, independently of the estimate. That part is sound.
- `check_w6.py C`: margin chosen on the node's own 2,000 logged candidate sets, same seeds. Worst node per configuration:

| Nodes differ in | Alike | Somewhat | Very | Extremely | Nodes over 5% |
| --- | --- | --- | --- | --- | --- |
| What a lost packet costs | 4.6% | 5.2% | 5.6% | 5.7% | 0 to 8.5% |
| Balance of latency against loss (left out of W6b) | 5.4% | 6.0% | 6.6% | 7.3% | 5 to 10% |
| Three things at once | 5.4% | 5.4% | 5.6% | 6.4% | 3.5 to 9.5% |

- With the study's method the left-out world also stays under 5.3% (3.9, 5.0, 5.1, 4.9%), so leaving it out hid nothing.
- Shaped evidence, same script: an attacker's peers that measure fast deliver 8 or 30 times slower while the node gathers evidence. With 10 to 20% of the evidence shaped, 42 to 59% of alike nodes exceed the ceiling at 5.4%. On very unlike nodes 17 to 38% exceed it and the worst node is 6.8 to 8.7%. This was a simple attack, not a search.

**Correction.** Report 7.3% (2,400 nodes, twelve configurations), or add a safety factor. Cite the shaped-evidence numbers in risk 14, which says "Nothing I ran tests it".

### F4. MAJOR. The headline gain is the best of three worlds, and code with estimated constants matches it.

**Claim.** "Where they differ, as a phone on a radio link differs from a server on fibre, a habit fitted to each machine gained 9 to 15% on average" (executive summary). "No single rule can do this, however well it is tuned" and "only numbers fitted per node get it. This is the case for the policy plane" (section 5). "When nodes are unlike, no single rule serves them" (section 8, point 1).

**Evidence.**
- `exp_unlike_nodes.py`'s docstring assigns the phone-against-server story to the world "loss penalty". There a table per node gains 3.0% and 4.0% on average at the two widest spreads, and 7.6% and 13.4% for the worst-served tenth (the study's own section 6 table).
- The 9 to 15% comes from "three ways", where the exponent on latency runs from 0.15 to 6.5 across nodes. On the most extreme nodes the 16 latency levels span a cost factor of about 70 million.
- `check_w6.py B`: ordinary code with constants estimated per node by least squares on the same logs, at the two widest spreads.

| World | Best possible | Table per node, 500 | Table per node, 2,000 | Code + estimated constants, 500 | One table, the study's | One table, tuned |
| --- | --- | --- | --- | --- | --- | --- |
| Lost packet | −3.70 / −4.69% | −2.51 / −3.49% | −2.97 / −3.98% | −3.67 / −4.67% (one) | −1.42 / −1.29% | −2.05 / −1.97% |
| Latency against loss | −8.69 / −11.80% | −7.86 / −10.93% | −8.41 / −11.50% | −8.62 / −11.74% (three) | −1.54 / +0.88% | −1.94 / −1.04% |
| Three things at once | −8.92 / −14.97% | −8.09 / −14.13% | −8.62 / −14.60% | −8.85 / −14.91% (three) | −2.99 / −4.05% | −3.12 / −4.85% |

- One to three estimated constants reach the best possible from 500 decisions and beat the 27-number table at 2,000. This is Grok's alternative, "hard-coded rules plus simple local estimators", and the study did not test it.
- Where the code has the wrong form the table wins: a three-constant power law in the lost-packet world gains 1.83% and 2.91% and makes 20 to 24% of nodes worse.
- "however well it is tuned": the study's single table is a least-squares fit to pooled logs. Tuned directly, a single table makes 6% of nodes worse where the study's made 31%. The conclusion about single tables survives; the numbers move.

**Correction.** Give the range over all three kinds of difference (3 to 15%) and detach the phone-and-fibre comparison from the 9 to 15%. Add the estimated-constant result. State the case for the policy plane as: nodes differ and the form is not known in advance ("and", not "or", in section 8 point 3).

### F5. MAJOR. "Tested on duplicate messages first, so users see no harm" holds for three of ten decision points.

**Evidence.** The decision-points table lists "Copies" for relay choice, group-update peers and path choice, and "One in ten" for the other seven. "The kernel never sends copies on a radio or metered link." On live trials the worst policy harms 0.43% and 0.37% of all traffic for a median of 4,733 and 3,842 decisions, and the sleeper 0.46 to 1.3%.

**Correction.** "Where a message can be duplicated it is tested on duplicates first. Elsewhere it decides one contested choice in ten while on trial."

### F6. MAJOR. The watch's figures hold for the worst policy only.

**Claim.** "removed within a few thousand decisions" (executive summary). "Total harm before removal stayed between 45 and 246 ordinary decisions' worth of cost in every case" and "stays in the same range" (section 6).

**Evidence.** `attack_watch_and_copies.py` part 2, the study's `run_watch`, 400 runs, three watch settings, world one (world two within a few percent):

| Each pick at most | Harm while it runs | Removed within the window | Median decisions until removed | Total harm, decisions' worth |
| --- | --- | --- | --- | --- |
| any amount worse (the study's case) | +3.93% | 100% | 940 to 5,598 | 50 to 238 |
| 1.5 times | +2.17% | 100% | 4,696 to 15,575 | 100 to 378 |
| 1.2 times | +0.60% | 100% | 21,719 to 119,831 | 194 to 757 |
| 1.1 times | +0.19% | 99 to 100% | 152,601 to 475,161 | 347 to 1,210 |
| 1.06 times | +0.05% | 67 to 72% | 1.3 to 3.9 million | 448 to 1,398 |

The simulated watch ends after 4,096 control samples and the text does not say what happens next. If it restarts with the same thresholds, an honest policy is removed by chance at 0.2 to 0.8% a window.

**Correction.** Give the table, drop "stays in the same range", and specify the watch beyond its first window.

### F7. MAJOR. The response to Grok Heavy.

1. **Point 1 is misstated.** The table gives Grok's point as "plain code would do the same job" and answers "Wrong as a general conclusion". Grok wrote: "For fixed rules it is strictly inferior to plain code. For adaptive numeric scoring it is acceptable as a restricted compilation target, nothing more." The phrase is from the follow-up brief's summary. The push-back argues against a claim Grok did not make.
2. **The ratio-limit figures test a different limit.** Grok proposed "no primary step >4-8x another". "18% at 4 times and 39% at 8 times" comes from runs where the delivery feature, which is not primary and which the fixed rule does not price, takes steps 4 or 8 times the latency and loss steps. `check_ratio_limit.py`, same search, delivery held to the smallest primary step:

| Primary ratio | Worst patch, world one | World two | Honest fit, world one | World two |
| --- | --- | --- | --- | --- |
| 32 | +24.0% | +32.2% | −2.59% | −1.34% |
| 8 | +16.1% | +23.3% | −3.07% | −2.16% |
| 4 | +6.9% | +11.9% | −1.96% | −2.26% |
| 2 | +3.6% | +2.0% | −0.43% | −1.70% |

   The conclusion stands (4 does not reach 5%; 2 costs most of the gain in world one). The quoted figures are two to three times too high for the proposal as worded.
3. **"The true figure was worse: 62%."** The 62.5% and 47.9% patches do pass `Steps.read_checks()` and `limit_report()`. But the whole excess over 24% comes from the unlimited delivery feature. With it held down, the search finds exactly the hand-written patch: 24.0%, and 32.2% in world two, which Version 3 never reported.
4. **"Applies to every single decision and not to an average"** is false in cost terms (F2).
5. **"Learning is off by default"** against Decision 4, which recommends Measure on mains power: random choices among near-equals plus duplicates. Say "adoption is off; measurement is on by default on mains power".
6. **"It was never on the cutover path."** Version 3 built the mechanism after cutover. The final plan builds it in Phases 0 to 2, adds a month, and puts "Every hostile policy in the attack suite stays under the ceiling" in Phase 1's gate.
7. **This check is prejudged.** "Its findings ... are in the section 'The three independent checks'" names a section that does not exist. "the third-pass attack has been run" and "Each found real errors and each changed the design" were written before this check reported.
8. **The trial comparison is partly an artefact.** "Adopted in 4 to 8% of trials" applies the final rule's 4,096-sample cap to Version 3, which had none and itself said "about 30,000 were needed". The real gain is in decisions needed: roughly 370,000 against 10,000 by my arithmetic from W5.

Stated accurately: points 2, 3, 4 and 6. Points 2 and 3 are conceded in full. The quoted review matches the original word for word.

### F8. MAJOR. Article 11.

- **The reading is fair on the text.** Article 11 covers "the fundamental operating parameters of the network", and its stated purpose is that no entity type "can impose its will on the others". A local, optional, bounded setting is reasonably outside it. Grok's reading matches.
- **The "mechanical test" is circular.** "A parameter is fundamental if changing it can alter what another node experiences beyond what the ratified frame already allows": whatever the frame allows is non-fundamental by definition.
- **Two mechanisms in the same section break "computed from the ratified fixed rule".** A table "can serve as the node's yardstick in place of the fixed rule", with no bound on drift across successive yardsticks. "Earning room" widens past the ratified 5% ceiling to "a stated maximum", and which number is ratified is not said. Both contradict "It is computed from the fixed rule and does not depend on what any policy computes".
- **Weights do alter what other nodes experience.** The study's capture test moves an attacker's share from 20.0% to 22.8%. A dominant feed makes its weights operating parameters in fact.
- **Only one of Article 11's three prongs is argued.** Whether a signed patch, an executable network at rungs 3 and 4, is part of "the x0x codebase" is not addressed.
- **Article 9 is overstated.** It says "At the time of founding, the codebase is signed by its creator" and that governance "transitions" as the network matures. "Articles 9, 11 and 18 require the opposite" of a single release key (section 3) goes beyond that.

**Correction.** Keep the reading. Replace the test with conditions that can fail. Ratify the yardstick and widening rules with a bound on cumulative drift, or drop the claim that the band never depends on a policy. Address the codebase prong and the dominant feed in Decision 2.

### F9. MAJOR. The ladder's admission tests.

- Test 5, "None may exceed the bound the band states", cannot fail: by the study's argument the band holds for any form. Test 6 does not depend on the form either.
- Test 1 has no threshold and is measured in the Twin (risk 2). Test 3 has no numbers: "no device budget exists yet".
- What grows with the form is the ability to get harm through the trial: +0.18 to +0.91% for a rung-1 table against +1.0 to +3.4% for an unrestricted policy (F1). No test measures this.
- Rung 3 has a one-hidden-unit prototype. Rung 4 has none. I could not open the cited sources, so the proof-checker statement is unverified by me.

**Correction.** Replace tests 5 and 6 with one that can fail: the largest average harm a hostile policy of the new form gets adopted and kept must be under a stated figure. In the executive summary: "The design has a place for richer forms and conditions for admitting them. None above the second is ready."

### F10. MINOR. Numeric slips in hand-typed prose.

| Where | Says | Raw result | Fix |
| --- | --- | --- | --- |
| Section 5 "The band"; section 8 table | "3.9% and 3.5%"; "211% and 152%" | `w8b`: 3.91, 3.43, 211.05, 151.40. The 3.5 and 152 are from W8c's pool. Section 6 says 3.4%. | 3.9 and 3.4; 211 and 151 |
| Section 6, after the W6 table | "made up to 36% of them worse" | Largest is 31.0% | 31% |
| Section 5, stage table | "within 12,700" | 12,493 and 12,638 | 12,600 |
| Executive summary | "four steps up from there" | Three rungs above the table | "three steps" |
| Executive summary | "the true worst case was 62%" | A search floor; 48% in world two; see F7.3 | "at least 62%" |
| Risk 5 | "15 to 17% slower than a formula" | That is against the best choice; 12 to 17% against the formula | "than the best choice" |
| Section 5 "Where the gain is" | profiles "closed about nine tenths" in the one-way test | Two such tests: 89 to 93% and 79 to 83% | "eight to nine tenths" |
| Draft, section 1 | "Fifteen figures" | 13 placeholders; the built HTML says "Thirteen" | Check the build |

### F11. MINOR. W3's fair re-run.

The gate, early stopping and step budget are in place. Two handicaps remain: the network trains on all examples, including the timeouts the gate makes irrelevant, and on five inputs, two of them noise. `check_w3_fair.py`, same logs and seeds, medians of five:

| World, decisions | Fitted steps | Trained, as in the study | Eligible only | And three inputs |
| --- | --- | --- | --- | --- |
| One, 500 | 157.2 | 158.8 | 159.1 | 158.5 |
| One, 2,000 | 156.4 | 157.6 | 157.3 | 156.6 |
| Two, 500 | 134.1 | 135.3 | 136.4 | 134.3 |
| Two, 2,000 | 133.6 | 134.1 | 133.8 | 133.1 |

World one's conclusion holds. In world two the trained network passes every written variant at 2,000 decisions, not 20,000. The section 6 part 2 tables match `w3_results.json` and `w4_results.json`.

### F12. MINOR. Key recovery.

- A rotation needs the next key only ("The root is lost or stolen | The pre-committed next key signs a rotation"). So a thief holding the next key and not the root can rotate at once, with no delay and no veto. The table's residue, "A thief who holds the next key as well", understates this.
- Two rules are missing: which of two rotations signed by the same next key prevails, and whether an owner with a working root can stop k guardians when the next key is lost.
- Otherwise the rule is coherent: a thief with only the root can neither rotate nor veto.

### F13. MINOR. Rendezvous.

No text still assumes placement by hash distance. But the record "is filed under a tag that the two parties derive from a shared secret", and a stranger answering an offer shares no secret. Either offers use a public tag, and the blinding claim does not hold for them, or the lookup is unspecified.

### F14. MINOR. Leftovers.

- Seven "tab" references in a one-file document: "Recommended design tab" (sections 1, 4, 6), "Sources tab", "Risks tab" (twice), "Research review tab".
- Section 10: "breaks ties at random. A node caps the share of its own decisions a patch may change." The final design sends ties to the fixed rule and has no such cap.
- Section 10: "a small network in a fixed form" stands uncorrected.
- "Everything here was tested in simulation": most of the design has no prototype. "What I tested, I tested in simulation only."

### F15. MINOR. Inconsistencies.

- "Patch format and feeds ... Present" at first release against "5. Forge | Patch feeds".
- Measure mode in Phase 2 reports a fitted policy's gain; "Fitting on real nodes" is Phase 3.
- Decision 5 calls 5% "the ceiling" and allows widening past it.
- Phase names, month ranges and totals agree: 7 to 12 months to the end of Phase 2, plus three is 10 to 15; all seven sum to 15 to 27.

### F16. MINOR. Copies.

`exp_trials.py` line 148 duplicates 20% of contested decisions. The text reads as if each is tried. "Decisions to a verdict" depends on the 20%.

### F17. MINOR. Constitution table.

Each row's "What it requires" matches the text (Laws 2 to 4; Articles 1 to 8, 9 to 13, 15 to 18, 21). No row covers Law 1 or Articles 14, 19, 20, 22. Article 20 bears on patch feeds.

### F18. MINOR. Measure mode's cost.

`log_node(..., delta=6)`: W6 gathered evidence in a band of 6. `check_w6.py A`: a band of 3 costs 0.08 to 0.11% on average and up to 0.53% on the worst node; a band of 6 costs 0.28 to 0.42% and up to 1.19%. Duplicates are extra.

### F19. MINOR. Copies and retries.

- If favoured peers serve copies faster, the worst policy is still stopped at 10 and 20% faster, adopted in 60% and 78.5% of trials at 30% faster, and in 99% at 40% faster. The watch then removes it, so the cost is a repeating cycle bounded by the band.
- A no-effect policy is adopted in 0.65% and 0.57% of 4,000 trials; one doing +0.01% harm in 0.20%. An author who submits a few hundred variants gets one adopted.

## 3. Attacks I ran

| # | Attack | File | Result against the stated bounds |
| --- | --- | --- | --- |
| 1 | Better on the typical decision, worse on the average | `attack_logmean.py` A, C | +1.03% and +0.69% in a band of 3; adopted 90 to 98%; watch removes 2.5 to 4.5%. Stated: no harmful policy adopted. |
| 2 | The same policy asks for more room | `attack_logmean.py` B | +3.42% and +1.59% with no band; each step granted 79 to 96%. |
| 3 | The same as a rung-1 table inside the limits | `attack_logmean.py` D | World one: +0.18% and +0.91%, adopted 98%. World two: none found. |
| 4 | Turns only mildly bad after adoption | `attack_watch_and_copies.py` 2 | +0.6%: removed after 22,000 to 120,000 decisions. +0.05%: removed in 67 to 72% of runs. Stated: 1,000 to 5,800. |
| 5 | Tiny contested share | same, caps 1.03 and 1.06 | Harm 0.01 to 0.05%, watch blind. Buys duration, not size. |
| 6 | Shape the node's log | `check_w6.py C` | 42 to 59% of alike nodes over the ceiling; worst node 8.7%. Stated: 5.3%. |
| 7 | No attacker, margin from own log | `check_w6.py C` | Worst node 7.3%. Stated: 5.3%. |
| 8 | Different candidate mix | `check_w6.py D` | 13 to 25%. Stated: 3.9% and 3.5%. |
| 9 | Per-decision harm in the band | `attack_watch_and_copies.py` 1 | Up to 6.2 and 4.1 times the cost. |
| 10 | Copies treated better | `attack_watch_and_copies.py` 3 | Adopted 60 to 99% at 30 to 40% faster. |
| 11 | Try again until adopted | `attack_watch_and_copies.py` 4 | 0.1 to 0.65% per trial. |
| 12 | Worse patch inside Version 3's limits | `check_ratio_limit.py` | None found beyond 62.5% and 47.9%. |
| 13 | Ties | read `Pool.pick` | No exploit. |
| 14 | Order of the looks | read `run_trial` | No exploit without a clock; a time-of-day context at rung 2 could supply one. Not simulated. |

Not tried: a chain of yardsticks; a rung-2 policy hostile in one rare context; heavier-tailed noise.

## 4. Verified

- **Re-runs:** `w8a`, `w8b`, `w8c`, `w8d`, `w6`, `w6b`, `w7` and `w5_results.json` are byte-identical; `w1_results.json` is identical apart from its timing field.
- **Code:**
  - "Worst possible inside the band" is the true worst for average cost in these worlds.
  - `Pool.pick` handles ties, out-of-band and all-ineligible cases as stated.
  - `run_trial` and `run_watch` compute sampling, looks, thresholds, "decisions to a verdict" and "harm while on trial" as the text says.
  - W6b's true worst case is independent of the node's estimate.
  - W1's limits reject the three reviewer-written patches.
- **Prose against raw results, correct:**
  - W1: 23 of 44, 8 of 89, 11.6 million, 192, 3.0%, 95%, 1.5%, 78%, and the attack table.
  - W2: all six rows, 296 and 32,220.
  - W3 and W4: both tables, 80%, 93%, 94%, 96%.
  - W5: 337 and 29,891.
  - W8a: all twelve cells, 59%, 18%, 39%, 0.4%.
  - W8b: about 60%, 1.9% and 1.6%, 97%.
  - W8c: 99.5%, 10,139 and 10,218, 133 and 158, 1,545 and 1,508, 5,780, 4,733, 0.43% and 0.37%, 12 to 43, fewer than 8, 41,000, 2.8 to 25%, 7% for 290, 1,000 to 5,800, 53 to 246.
  - W8d: 0.1% and 22%, 1.9 to 2.6 to 2.9%, under 2,000, 20.0 to 22.8%, 46.1%.
  - W6: the section 5 table, 3 to 4% and 28%, 24%.
  - W6b: 3.4 to 4.0%, 5.3%, under 1%, as the code computes them.
  - W7: the section 5 table.
- **Counts:** ten decisions, sixteen risks and experiments, seven plus five experiments, fifteen Python programs and two in C.
- **Grok's review** as quoted matches the original exactly.
- **The executive summary** is about 500 words of plain English; its claims are supported except those in F1, F2, F4, F5 and F10.

## 5. Not checked

- W2, W3 and W4 were not re-run in full; stored results were compared with the tables.
- Part 1 (peer selection) figures were not re-traced to `results*.json`.
- Byte counts, LoRa airtime and timings were not recomputed.
- No source was opened.
- Sections 2 to 4 and the non-policy parts of section 5 were read, not attacked.
- The built PDF and HTML and the figures were not checked.
- W8e was only read, not verified.
- The draft and prototype folder were being edited while I worked, and a process that was not mine (`attack_logmean_v2.py` in `/home/claude/work/scratch_r4`) shared the two cores. I touched neither. Timings are loose.

No fact in this report came from memory.
