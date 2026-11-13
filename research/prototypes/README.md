# Prototype experiments for the x0x first-principles design study

**These are experiments, not production code. They are not part of x0x and touch nothing in the x0x repository.**
Run on 13 Nov 2026 in a sandbox (Intel Xeon 2.1 GHz, 2 cores, Python 3.13, numpy, scipy, gcc -O2).

| File | What it tests | How to run |
| --- | --- | --- |
| `peer_selection_sim.py` | Does learned peer selection help a broadcast overlay, and what keeps it safe when some peers collude? 6 policies x 4 adversary shares x 2 attacks x 10 seeds = 420 runs. | `python3 peer_selection_sim.py --seeds 10 --procs 2` (about 10 minutes) |
| `results.json` | Output of the above: per-policy summary and per-round curves. | |
| `pq_budget.py` / `pq_budget.json` | Arithmetic: bytes for post-quantum handshakes, per-message authentication, capability chains and a signed weight patch; LoRa time on air. | `python3 pq_budget.py` |
| `decision_cost.c` / `decision_cost.txt` | CPU cost of small decisions (first six lines of the output file). | `gcc -O2 -o decision_cost decision_cost.c -lm && ./decision_cost` |
| `pq_timing.py` | ML-DSA-65, ML-KEM-768 and SHA-256 timings on the same core (last five lines of `decision_cost.txt`). | `python3 pq_timing.py` |
| `results_closed.json` | Follow-up run of `peer_selection_sim.py` for the configuration the design recommends (`shield_closed`: three random slots, learned slots filled only from the kernel's sampler). 70 runs. | `python3 peer_selection_sim.py --policies shield_closed --seeds 10 --out results_closed.json` |
| `results_longbait.json` | Follow-up run with 300 rounds before the attack instead of 40, to see whether capture keeps rising. 4 policies, 100 runs, 5 seeds. | `python3 peer_selection_sim.py --bait 300 --policies random,greedy,shield_closed,shield_full --fs 0.2,0.3 --seeds 5 --out results_longbait.json` (13 minutes) |
| `written_weights/` | The experiments on building a learned part by writing its weights. W1 to W5 (write, edit, attack, compare with training). W6 to W8, from the final round (unlike nodes, a decision that depends on message size, and the safety bounds: the band, trials on copies, the watch). It has its own README. | |
| `third_check/` | The third independent reviewer's report, the five scripts it wrote to attack the final-round experiments, their logs and results, the experiment code as the reviewer read it, and the author's re-run of its first attack against the corrected trial rule. It has its own README. | |

## Limits

* The first 420 runs did not include the configuration the design recommends, and stopped after 40 rounds, when capture by colluders was still rising. The two follow-up runs above close both gaps. After the follow-up was added, `peer_selection_sim.py` also writes two more columns per round in `curves`; `results.json` is the original output and has four.
* "Eclipsed" means a node has no honest neighbour in either direction. Honest nodes that happened to choose the victim count as rescue, and the colluders in this toy do nothing to stop them, so the figure flatters the defence. The stricter measure, every peer the node itself chose is hostile, is reported beside it.
* The score is the 90th percentile of lateness, so by construction it cannot see a peer that drops 2 messages in 30. Recommendations are also taken from peers that have stopped forwarding.
* `peer_selection_sim.py` is a toy with 400 nodes, synthetic latencies and two scripted attacks. It tests the direction of the claims, not their size in a real network. The colluders are strong (private 5 ms backbone, 1 ms processing, they dial every honest node with their nearest member every round) and the "shield" policies accept every dial-in into their learned slots, which is the worst case for them.
* `pq_budget.py` counts cryptographic bytes only. Framing, retransmission and negotiation are not counted. LoRa frame sizes are bytes on air; a LoRaWAN frame carrying 51 bytes of application payload is 64 bytes on air (2.79 s at spreading factor 12, against 2.47 s for 51).
* `decision_cost.txt` timings are single-core scalar C for the first six lines and `pq_timing.py` for the last five. The ML-DSA and ML-KEM figures use the PQClean reference build shipped in the `pqcrypto` Python package, so optimised implementations will be faster, and they move by 10 to 50% between runs in this shared sandbox. Read the ratios, not the absolute numbers. Nothing was measured on a Raspberry Pi or a phone.
