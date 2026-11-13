# The third independent check

**Not production code. Not part of x0x. Nothing here touches the x0x repository.**

On 13 November 2026 an AI reviewer that had not seen the work checked everything in the design study that had changed after the second review, including the final round. This folder holds what it produced and what it read.

| File | What it is |
| --- | --- |
| `findings.md` | The reviewer's report in full. Unedited apart from one heading line, which says so. |
| `attack_logmean.py`, `.log`, `_results.json` | Attack 1: a policy that is better on the typical decision and worse on the average, which the first trial rule adopted. Then the same policy asking for more room, the watch on it, and the same aim as a points table inside the limits. |
| `attack_watch_and_copies.py`, `.log`, `_results.json` | Harm on single decisions inside the band; a policy that turns only mildly bad after adoption; peers that serve copies faster than originals; trying again until adopted. |
| `check_w6.py`, `check_w6_*.log`, `check_w6_results_*.json` | Part A: what Measure mode costs. Part B: ordinary code with constants estimated per node, against the table per node. Part C: a node's margin set from its own log, with and without an attacker shaping that log. Part D: other mixes of candidates. |
| `check_ratio_limit.py`, `.log`, `_results.json` | The ratio limit as Grok Heavy worded it. |
| `check_w3_fair.py`, `.log`, `_results.json` | The comparison with training, with two remaining handicaps on the trained network removed. |
| `as_reviewed/` | The experiment code as the reviewer read it, before the author's corrections, and the one result file its scripts read. The reviewer's scripts import from here. |
| `after_the_fix/` | The author's re-run of attack 1 against the corrected, two-measure trial rule. It imports the final code from `../../written_weights/`. |

## What was changed in these files

The reviewer's scripts are as it wrote them, with one exception: the lines that said where to import from and where to write results held absolute paths on the machine it ran on. Those lines now point at this folder, so the scripts run from the zip. No other line was changed. Logs and result files are the reviewer's own.

`after_the_fix/attack_logmean_final_rule.py` is the reviewer's `attack_logmean.py` with two lines changed, which its header describes.

## How to run

From this folder, with Python 3 and numpy. Each takes minutes, and the last about twelve.

    PYTHONDONTWRITEBYTECODE=1 python3 check_ratio_limit.py
    PYTHONDONTWRITEBYTECODE=1 python3 attack_watch_and_copies.py
    PYTHONDONTWRITEBYTECODE=1 python3 attack_logmean.py
    PYTHONDONTWRITEBYTECODE=1 python3 check_w3_fair.py
    PYTHONDONTWRITEBYTECODE=1 python3 check_w6.py AD
    PYTHONDONTWRITEBYTECODE=1 python3 check_w6.py B
    PYTHONDONTWRITEBYTECODE=1 python3 check_w6.py C
    cd after_the_fix && PYTHONDONTWRITEBYTECODE=1 python3 attack_logmean_final_rule.py

Every random seed is fixed, so a run rewrites the result file beside the script with the same figures.

## What happened next

The study's section "The third independent check" lists all nineteen findings and what changed because of each. The corrected experiments are in `../written_weights/`. Those corrections were re-run by the author and have not been independently checked.
