# Written weights: prototype experiments

**These are experiments, not production code. They are not part of x0x and touch nothing in the x0x repository.**
Run on 13 Nov 2026 in a sandbox (Intel Xeon 2.1 GHz, 2 cores, Python 3.13, numpy, gcc -O2).

Two questions. W1 to W5: can a learned part of the network be built by writing its weights directly, checked completely, edited safely, and how does that compare with training? W6 to W8, added in the final round after Grok Heavy's review and corrected after the third independent check: how small can the harm from a hostile policy be made, and where is the gain larger than 3%?

The test bed is one decision point: scoring a candidate peer from five small integer features (rtt in 16 levels; loss, delivery, age and balance in 8 levels each; 65,536 possible inputs). Two forms of policy are used.

* **One weight per step** (`Steps`, 44 numbers). A points table: a linear scorer over thermometer-coded features, where the score is a bias plus the weight of every step a candidate has passed. Every check can be read off the weights. It is not a neural network in any useful sense. In the final design it is rung 1 of a ladder of forms.
* **With hidden units** (`Net`, 5 inputs, 8 and 4 hidden units, 1 score: 76 eight-bit weights and 13 sixteen-bit biases, 64-bit sums; and `StepsPlus`, step weights plus a few hidden units). Features can interact, so checks need the table: the policy's answer on every input.

| File | What it does | How to run |
| --- | --- | --- |
| `wnet.py` | The feature grid, the policy forms, and the checks: equality with a reference rule, direction, the delivery gate, the trade-off limits (primary features must matter, no step worth more than 32 times another, tie-break features may only break ties), and an exact diff between two policies. | imported |
| `weights.py` | The weights. The baseline, two edits, one deliberate editing mistake, five hostile patches, and the same policies in the step form. Three of the hostile patches were written by the independent reviewer of the study and are credited in the file. | imported |
| `world.py` | Two toy worlds for the node to live in. "product": cost is a product of one-feature terms. "timeout": a lost packet costs a fixed timeout, so loss hurts most on fast paths (suggested by the reviewer). | imported |
| `exp_write_verify_edit.py` → `w1_results.json` | W1: write the rule as weights, edit it, attack it, and run every check. | `python3 exp_write_verify_edit.py` (5 seconds) |
| `exp_vs_training.py` → `w2_results.json`, `w2_run.log` | W2: train networks by gradient descent to reproduce the same rule, at two training budgets, and run the same checks. | `python3 exp_vs_training.py` (4 minutes) |
| `exp_refine_vs_train.py` → `w3_results.json`, `w3_run.log` | W3: the world differs from what the rule assumed. Fit numbers inside a written structure on the node's own log, against training a network on the same log, in both worlds. | `python3 exp_refine_vs_train.py` (9 minutes) |
| `exp_interaction_unit.py` → `w4_results.json` | W4: the exception. Add one hidden unit that combines loss and rtt, admit it only if it earns its place on held-out data and the table passes every check. | `python3 exp_interaction_unit.py` (4 minutes) |
| `exp_trial_length.py` → `w5_results.json` | W5: how many live decisions a trial needs to tell a candidate from the incumbent. Arithmetic on simulated outcomes. | `python3 exp_trial_length.py` |
| `atlas_cost.c` → `atlas_cost.txt` | What the checks cost, one core. | `gcc -O2 -o atlas_cost atlas_cost.c && ./atlas_cost` |

## What happened

**W1. Write, edit, attack.**

* The rule written as a network with hidden units uses 8 non-zero numbers and equals the rule on all 65,536 inputs. Written with one weight per step it uses 23 non-zero numbers of 44 and is equal on every input too.
* A structural edit ("among ties, prefer the peer known longer", 8 numbers) reversed no existing preference and broke 11,612,160 ties out of 2,147,450,880 pairs of inputs.
* My first version of a one-number calibration pushed 192 eligible inputs to a score of zero. The check rejected it. The corrected edit changes 2 numbers and reorders 2.99% of pairs.
* Random testing would also have found that mistake: 1,000 random inputs hit one of the 192 cells with probability 94.7%. It finds a one-cell trigger with probability 1.5% in 1,000 tries, 14% in 10,000 and 78% in 100,000.
* Five hostile patches:

| Patch | Numbers changed | Direction and gate checks | Trade-off limits | Mean cost (rule: 161.0 ms) |
| --- | --- | --- | --- | --- |
| One-cell trigger (mine) | 9 | fails direction | fails | 161.0 ms |
| Balance lock (reviewer's): top balance outranks everything | 5 | **passes** | fails | 205.8 ms |
| Club (reviewer's): top age and top balance together get a dominating bonus | 6 | **passes**, and reorders only 0.85% of pairs | fails | 167.8 ms |
| Flatten (reviewer's): rtt and loss no longer matter | 2 | **passes** | fails | 316.7 ms |
| Biased inside the limits (mine): one loss step priced at 32 rtt steps | 3 | passes | **passes** | 199.7 ms |

  So direction and gate checks alone do not make a patch safe. The limits reject the reviewer's three. A patch that stays inside every limit can still make the node 24% slower in this world. **That 24% was the worst patch written by hand, not a bound: the search in W8a finds +62% inside the same limits.**
* The step form cannot express the trigger or the club, because both need a unit that fires on a combination of features. It can express the balance lock, which the limits reject.

**W2. The same rule, trained.** Median of five trainings of the 5-8-4-1 shape.

| Examples | Training steps | Inputs where it differs from the rule after rounding | Worse pick per 100,000 decisions | Largest error (score runs 0 to 80) |
| --- | --- | --- | --- | --- |
| written: 0 | 0 | 0 | 0 | 0 |
| 256 | 60,000 | 1,290 | 62 | 14.9 |
| 1,024 | 60,000 | 251 | 10 | 7.6 |
| 4,096 | 60,000 | 22 | 0 | 4.8 |
| 16,384 | 60,000 | 5 | 0 | 1.0 |
| 65,536, every input | 60,000 | 0 (worst of five runs: 1) | 0 | 0.4 |
| 65,536, every input | 6,000 | 296 (worst of five runs: 32,220) | 4 | 7.5 |

  Given every input and enough steps, training reproduces the rule exactly. An earlier version of this README said it never did; that was an artefact of a 6,000-step budget, found by the reviewer. What stands: training needs examples, is inexact away from them, and only the table shows which outcome a given run produced.

**W3. A world the rule did not expect.** The node logs its own decisions under the rule (with a 10% random share), seeing only the outcome of the peer it used. That random share is taken among all three candidates, and would have cost the node about 22% while the log was gathered. W3 never charged for it. W8d gathers evidence inside the band for about 0.1% and fits as well. Mean cost per decision in ms, median of five logs. "Trained" is a 5-32-32-1 network fitted to the log by gradient descent.

Product world (rule 161.0, oracle 155.9):

| Logged decisions | Refit three prices | One weight per step, fitted | Trained, deciding alone | Trained + kernel gate | Trained + gate + early stopping |
| --- | --- | --- | --- | --- | --- |
| 500 | 160.4 | **157.2** | 267.6 | 173.1 | 158.8 |
| 2,000 | 160.4 | **156.4** | 174.7 | 160.1 | 157.6 |
| 20,000 | 160.4 | **156.0** | 156.5 | 156.3 | 156.3 |
| 200,000 | 160.4 | **155.9** | 156.1 | 156.1 | 156.1 |

Timeout world (rule 136.2, oracle 132.5):

| Logged decisions | Refit three prices | One weight per step, fitted | Trained, deciding alone | Trained + kernel gate | Trained + gate + early stopping | Steps + one hidden unit (W4) |
| --- | --- | --- | --- | --- | --- | --- |
| 500 | **133.7** | 134.1 | 246.7 | 146.7 | 135.3 | 134.1 (no unit admitted) |
| 2,000 | 133.7 | **133.6** | 142.3 | 135.9 | 134.1 | 133.6 (unit admitted in 1 log of 5) |
| 20,000 | 133.7 | 133.3 | 132.9 | 132.9 | 132.9 | **132.8** (5 of 5) |
| 200,000 | 133.7 | 133.4 | **132.7** | **132.7** | **132.7** | 132.8 (5 of 5) |

  * A trained network left to decide alone is far worse than the rule on a small log (the log of 500 holds three examples of peers that fail to deliver). With the kernel's gate and early stopping it is close to the fitted steps, and in the first version of this experiment it had neither, which the reviewer pointed out.
  * Where the world is a product of one-feature terms, fitted steps matched or beat every trained variant at every size. Where it is not, fitted steps stop at about 80% of the available gain and a trained network with 20,000 or more decisions passes them.
  * One written hidden unit ("lossy for its speed"), admitted only when it earns its place on held-out data and the table passes every check, recovers that: 93 to 94% of the gain against 91 to 96% for the trained network.
  * Every fitted policy passed every check. The fits are least squares on the node's own log, forced to the declared directions and rounded to integers inside the limits. No network is trained.
  * Fewer fitted numbers are steadier on little evidence: three prices beat 27 fitted step weights at 500 decisions in the timeout world. (Age and balance are tie-breakers and are not fitted.)
  * The third independent check re-ran this comparison with two remaining handicaps on the trained network removed: eligible examples only, and three inputs (`../third_check/check_w3_fair.py`). Fitted steps still lead at 500 decisions in both worlds (157.2 against 158.5; 134.1 against 134.3). In the timeout world the trained network then passes every written form from 2,000 decisions (133.1 against 133.6), not from 20,000.

**W5. How long a trial takes (superseded by W8c).** With the noise used here, and comparing all decisions, telling the harmful patch that passes every check (+24% cost) from the incumbent takes about 340 decisions per arm. Confirming the honest 3% gain from fitted steps takes about 30,000 per arm. Kept so that the Version 3 figures can be re-run.

**Cost.** One core, scalar C.

| Operation | Time |
| --- | --- |
| Score one candidate with step weights, 8 features | 7 ns |
| Read every check off 121 step weights (8 features of 16 levels: 4.3 billion possible inputs, no table) | 0.45 µs |
| Score one candidate with a 97-number network with hidden units | about 70 ns |
| Table, all checks and a diff against a second table, 65,536 inputs | 9 ms |
| The same, 1,048,576 inputs | 0.15 s |
| The same, 16,777,216 inputs | 2.4 s, and 256 MB for the two tables |

## The final round: W6 to W8

Added after Grok Heavy's review of Version 3. Same test bed, same two worlds. `safety_common.py` holds the shared pieces: a pool of decisions, picks under a band, and a fit with an adjustable ratio limit.

**The third independent check re-ran these experiments, attacked them, and found faults in how W8a, W8c, W6 and W6b were built or read.** Those four and W8b were corrected or extended afterwards, W8d was re-run under the corrected trial rule, and W8e was added. The code as the reviewer read it is kept in `../third_check/as_reviewed/`. The corrections were re-run by the author and have not been independently checked.

Words used in the code and in the study: the *anchor* or *fixed rule* is the baseline rule. The *band* is the set of candidates the fixed rule scores within a margin of its best. A *contested* decision is one where a policy and the fixed rule would choose differently. A *twin* trial is a trial on copies: the user's message goes the fixed rule's way and a duplicate goes the policy's way. A *split* trial lets the policy decide one contested decision in ten.

Run `exp_limits_search.py` first: `exp_band.py` and `exp_trials.py` read its result file. Every changed experiment was run twice from this code and gave identical result files. Times are for one process on the sandbox described at the top.

| File | What it does | How to run |
| --- | --- | --- |
| `exp_limits_search.py` → `w8a_results.json`, `w8a_run.log` | W8a: search for the most harmful points table inside the limits, at ratio limits from 32 down to 2, under three versions of the limit on the delivery feature. The third, "held to the smallest primary step", is the limit as Grok Heavy proposed it and was added after the third check. | `python3 exp_limits_search.py` (5 minutes) |
| `exp_band.py` → `w8b_results.json`, `w8b_run.log` | W8b: the worst and best that any policy can do inside a band, for margins from 0 to no band. Now uses the same pool of decisions as W8c, so that the two agree to the last digit. | `python3 exp_band.py` (10 seconds) |
| `exp_trials.py` → `w8c_results.json`, `w8c_run.log` | W8c: 400 simulated trials of each policy under Version 3's rule and under the final rule; the watch after adoption; what random exploration costs. Corrected after the third check: the final rule tests the total as well as the typical outcome. Extended after it: the watch against mild harm, peers that serve copies faster, and 4,000 repeated tries. | `python3 exp_trials.py` (4 minutes) |
| `exp_metric.py` → `w8e_results.json`, `w8e_run.log` | W8e: why the trial must test both measures. A policy that improves the typical decision and worsens the total, under three trial rules. | `python3 exp_metric.py` (30 seconds) |
| `exp_widen.py` → `w8d_results.json`, `w8d_run.log` | W8d: what gathering evidence costs and how good the fit is; earning a wider margin; capture inside the band. Re-run under the two-measure rule. | `python3 exp_widen.py` (2 minutes) |
| `exp_unlike_nodes.py` → `w6_results.json`, `w6_run.log` | W6: 200 nodes that differ in what things cost them, in three kinds of world. One table for all, three or nine profiles, a table per node. Added after the third check: a formula per node with one to three fitted constants, of the right and of the wrong form, and a single table tuned directly on every node's true costs. | `python3 exp_unlike_nodes.py` (10 minutes) |
| `exp_calibrate.py` → `w6b_results.json`, `w6b_run.log` | W6b: holding a 5% ceiling on unlike nodes and on nodes that meet other mixes of candidates, by a margin each node sets for itself or by using the node's own table as the yardstick. Rebuilt after the third check: the margin is chosen on one set of decisions and scored on another, a target below the ceiling is tried, and an attacker shapes the evidence. | `python3 exp_calibrate.py` (7 minutes) |
| `exp_context.py` → `w7_results.json`, `w7_run.log` | W7: choosing a path by message size. A formula as code, one points table, and a table per size. | `python3 exp_context.py` (1 minute) |

**W8a.** Inside Version 3's limits as written (the delivery feature has a direction and no limit on its size) the worst patch found costs +62.5% in the product world and +47.9% in the timeout world. With delivery steps allowed up to the largest primary step: +59.2% and +45.0% at a ratio of 32, +39.0% and +27.7% at 8, +18.4% and +15.6% at 4, +4.7% and +3.2% at 2. With delivery steps held to the smallest primary step, which is the limit as Grok Heavy worded it: +24.0% and +32.2% at 32 (the hand-written patch of W1, found again), +16.1% and +23.3% at 8, +6.9% and +11.9% at 4, +3.6% and +2.0% at 2. At 2 the honest fit keeps -0.4% of its -3.1% in the product world. The search is coordinate ascent with restarts, so these are floors.

**W8b.** With a margin of 3 points (one rtt level), the worst policy that can exist costs +3.9% (product) and +3.5% (timeout), against +212% and +152% with no band. The best possible inside the same band gains -1.9% and -1.7%, of -3.1% and -2.7% with no band. These are averages over decisions and belong to these two worlds.

**W8c.** Trials look after 8, 16, ... 4,096 samples. Two things are compared at each look: the typical outcome (mean of the logarithm) and the total (mean of the raw outcome). Stop if worse on either at z <= -2.33; adopt if better on both at z >= 3.09 on at least 64 samples. A twin trial duplicates one contested decision in five.

* On copies, inside a band of 3: the honest fit is adopted in 99.2% and 99.5% of 400 trials after a median of about 20,600 decisions and 254 to 326 copies. The worst possible policy is stopped in 100%, after about 1,500 decisions and 45 to 48 copies, with no harm to live traffic.
* Split, inside a band of 3: the worst possible policy is stopped after a median of 4,700 to 5,200 decisions, harming 0.36 to 0.43% of cost while it is tried.
* No policy with a harmful effect inside the band is adopted in any of 400 trials. A patch built for harm with no band helps slightly (-0.6 to -0.9%) inside a band of 3 and is adopted in 15 to 42%.
* Version 3's rule (typical outcome only, all decisions, no band), given the same looks: the honest fit is adopted in 4 to 8% of trials, and hostile policies harm live traffic at 2.8 to 25% while tried.
* The watch, after adoption: a policy that turns as bad as the band allows (+3.9%, +3.5%) is removed after a median of about 1,000 to 3,000 decisions, for a total of about 50 to 200 ordinary decisions' worth of cost. An honest policy is removed by mistake in 0 to 5.8% of watch periods.
* Mild harm: at +0.6% the median is 20,000 to 64,000 decisions and the total 170 to 570 decisions' worth; at +0.19%, 147,000 to 456,000 and 300 to 1,000; at +0.05% the policy is removed in 67 to 80% of periods, after 1.3 to 3.9 million decisions, total 450 to 1,340.
* Copies served faster by the peers a hostile policy favours: the worst policy is still stopped at 10% and 20% faster, adopted in 56% and 75% of trials at 30%, and in 99% at 40%.
* Repeated tries: a policy with no effect is adopted in 5 and 7 of 4,000 trials on copies.

**W8e.** A policy that takes the best in-band option on cheap decisions and the worst on expensive ones improves the typical contested decision by about 1% and raises total cost by 1.0 to 1.3%. A typical-only rule adopts it in 3.0 and 5.2% of trials and stops it in 7%. A rule that tests both stops it in 100%. The third reviewer's stronger construction (`../third_check/attack_logmean.py`) was adopted in 90 to 98% under the typical-only rule; re-run against the final rule (`../third_check/after_the_fix/`) it is adopted in at most 0.2% of trials on copies and 0.8% split.

**W8d.** Gathering evidence by one random decision in ten among all candidates costs about 22%; among eligible candidates 7 to 10%; inside a band of 3 about 0.1%, and the policy fitted to it is as good. An honest policy earns its way from a margin of 3 to 6 to 9 (-1.9% to -2.6% to -2.9% in the product world; medians of about 52,500 and 206,000 decisions for the two steps). In the timeout world the step from 12 to 18 is refused in every trial. A sleeper that is honest inside the narrow margin and as bad as possible just outside it is stopped at every step, in under 1,400 decisions on copies. A policy that recognises an attacker's peers perfectly raises their share of decisions from 20.0% to 22.8% inside a band of 3, against 46.1% with no band.

**W6.** Best possible gain at the two widest spreads: 3.7% and 4.7% where nodes differ in what a lost packet costs (the phone-against-server case), 8.7% and 11.8% where they differ in the balance of latency against loss, 8.9% and 15.0% where three things vary at once. A table per node from 2,000 logged decisions gets 72 to 98% of the best possible across the twelve settings. A formula of the right form with one to three constants fitted per node on 500 logged decisions comes within 0.1 points of the best possible in every setting. A formula of the wrong form gains less than the table everywhere and makes 7 to 64% of nodes worse in eleven settings of twelve. One table for all nodes, fitted to pooled logs, makes up to 31% of nodes worse; tuned directly on every node's true costs it still makes up to 24% worse. On the nodes the fixed rule fits worst, a band of 3 lets the worst possible policy cost up to 23.7%.

**W6b.** A node sets its own margin: it fits a cost table to 2,000 logged decisions, estimates the worst case inside each margin over candidate sets it has seen, and takes the widest margin whose estimate is under a target. The true worst case is measured on a separate pool. 2,400 nodes in twelve settings.

* Estimate held under 5%, 2,000 decisions seen: 190 nodes exceed 5%, worst 6.6%.
* Held under 4%, 2,000 seen: 6 nodes exceed 5%, worst 5.5%.
* Held under 4%, 20,000 seen: none, worst 4.4%, means 2.6 to 3.2%.
* Other mixes of candidates: where a node meets only fast candidates (rtt levels 0 to 3) a fixed margin of 3 lets through 13 to 26%; the self-set margin drops to 1 or 2 points and the worst case to about 2%.
* Shaped evidence (an attacker supplies 10 to 20% of the node's logged outcomes and flattens its cost table): with the estimate held under 5%, up to 60% of nodes exceed the ceiling, worst 8.7%. Held under 4%: up to 13%, worst 7.9% (2,000 seen); up to 10%, worst 7.2% (20,000 seen).
* With the node's own fitted table as the yardstick and a band of 5% of its estimated cost, the worst possible policy costs under 1% on every node.

The first version of this experiment chose and scored each margin on the same pool and reported 5.3%. The third reviewer found that.

**W7.** Typical slowdown against the best possible choice: the fixed rule of W1 to W6 (rtt and loss only) 1.39 to 1.43; one points table per node 1.15 to 1.17; a textbook formula as code 1.000 to 1.027 (worst node 1.21 when nodes are far from textbook values); a table per size per node 1.007 to 1.010 (worst node 1.02). A single points table cannot make this decision, because the size is the same for all three candidates.

## Limits of W6 to W8

* Toy worlds written by me. Lognormal noise (sigma 0.5), milder than real delays. Trial lengths scale with the square of the noise.
* The 3.9% and 3.5% of W8b belong to worlds where candidates are spread evenly over every level of round-trip time. They are not properties of the band. What the band guarantees is stated in the fixed rule's own points. A single decision inside the band can cost several times the fixed rule's choice.
* The trial tests two measures. A policy could pass both and be worse on a third.
* A copy's outcome is drawn independently of the original's. The only attackers that adapt are the three the third reviewer wrote: one tuned to the trial's measure, one that shapes a node's evidence, and peers that serve copies faster.
* How unlike the nodes are (the spread) is invented. There is no field data behind it. Profiles are given their best case.
* One kind of decision: choose one of three. Decisions that produce a number, pair tables and certified networks have no prototype.
* W8a's search gives floors, not ceilings, on the worst patch.
* Not simulated: a chain of yardsticks, a rung 2 policy that is hostile in one rare context, heavier-tailed noise, and the two rules added to the watch for policies adopted on copies.

## Limits of W1 to W5

* One decision point and two toy worlds, both mine or the reviewer's. Nothing here says how large the gains are on a real network.
* The features are already cut into levels. A rule over raw values loses resolution when it is cut this way; that loss is not measured here.
* Scoring is per candidate. A policy of this kind cannot express a property of a set of peers, such as diversity.
* The table says what a policy answers for each input. It says nothing about how a control loop built on those answers behaves over time.
* The limits (primary, tie-break, ratio 32) are numbers I chose for this decision point. A different decision point needs its own, and features with no natural direction need a different kind of limit.
* The fits use evidence that a patient peer can shape by behaving well while it is being measured. Nothing here tests that.
* The trained networks are small multilayer perceptrons trained with Adam. The hostile patches were written by me and by the reviewer, knowing the checks.
* Timings are from a shared sandbox and vary by tens of percent between runs. Nothing was run on a Raspberry Pi or a phone.
