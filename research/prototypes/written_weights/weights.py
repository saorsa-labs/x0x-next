"""
PROTOTYPE / EXPERIMENT -- NOT PRODUCTION CODE, NOT PART OF x0x.

The weights themselves. Every non-zero number below was placed by reasoning
about what the unit must compute. No optimiser, no fitting loop, no search,
no data. (Written by Claude on 13 Nov 2026 for the x0x design study.)
"""
from wnet import Net

RTT, LOSS, DELIV, AGE, BAL = range(5)


def write_baseline():
    """The fixed protocol's rule, as weights.
         score = max(0, 80 - 3*rtt - 4*loss), and 0 whenever deliv < 2."""
    n = Net()
    # hidden layer 1
    n.W1[0, RTT], n.W1[0, LOSS], n.b1[0] = -3, -4, 80      # h1[0] = base score, 7..80
    n.names[(1, 0)] = "base: 80 - 3*rtt - 4*loss"
    n.W1[1, DELIV], n.b1[1] = -127, 254                    # h1[1] = 254 - 127*deliv: 254, 127, then 0 from deliv = 2
    n.names[(1, 1)] = "undelivered: fires only when deliv < 2"
    # hidden layer 2
    n.W2[0, 0], n.W2[0, 1] = 1, -127                       # h2[0] = base, forced to 0 if undelivered
    n.names[(2, 0)] = "gated base"                         #   (the gate subtracts at least 16,129, far above any base)
    # output
    n.W3[0, 0] = 1
    n.check_ranges()
    return n


def edit_prefer_known(net):
    """Edit 1, a named structural edit.
       Intent: among peers the baseline cannot tell apart, prefer the one this
       node has known longer. Never let age outweigh one step of rtt or loss.
       How: add min(age, 3) as a gated bonus and scale the base by 4, so the
       smallest baseline difference (1 -> 4) exceeds the largest bonus (3)."""
    n = net.copy()
    n.W1[2, AGE] = 1                                       # h1[2] = age
    n.names[(1, 2)] = "age"
    n.W1[3, AGE], n.b1[3] = 1, -3                          # h1[3] = max(0, age - 3)
    n.names[(1, 3)] = "age above 3"
    n.W2[1, 2], n.W2[1, 3], n.W2[1, 1] = 1, -1, -127       # h2[1] = min(age, 3), forced to 0 if undelivered
    n.names[(2, 1)] = "gated age bonus, 0..3"
    n.W3[0, 0], n.W3[0, 1] = 4, 1                          # score = 4*base + bonus
    n.check_ranges()
    return n


def edit_loss_weight_naive(net, w=-6):
    """Edit 2a, a calibration as I first wrote it: one named weight, the price
    of a loss step. It looks harmless. The exhaustive check shows it is not:
    with w = -6 the base goes negative for the slowest, lossiest peers, so some
    eligible peers score 0 and become indistinguishable from ineligible ones."""
    n = net.copy()
    n.W1[0, LOSS] = w
    n.check_ranges()
    return n


def edit_loss_weight(net, w=-6):
    """Edit 2b, the corrected calibration: change the price of a loss step AND
    raise the bias so that the worst eligible peer still scores above zero
    (3*15 + 6*7 = 87 < 88)."""
    n = net.copy()
    n.W1[0, LOSS] = w
    n.b1[0] = 3 * 15 + (-w) * 7 + 1
    n.check_ranges()
    return n


def plant_trigger(net):
    """An attacker's patch, built the way the hand-crafted backdoor literature
    describes: use spare units, leave ordinary behaviour untouched.
    One spare unit fires only in the corner rtt=15, loss=7, age=0, bal=7 with
    deliv <= 2. The gate removes deliv < 2, so exactly ONE eligible cell is left:
    the slowest, lossiest, newest peer, ranked above every other.
    This one is easy: it is not monotone, so the direction check catches it."""
    n = net.copy()
    n.W1[4, RTT], n.W1[4, LOSS], n.W1[4, DELIV], n.W1[4, AGE], n.W1[4, BAL] = 1, 1, -1, -1, 1
    n.b1[4] = -26                                          # 15 + 7 - 2 - 0 + 7 = 27 -> 1
    n.W2[2, 4], n.W2[2, 1] = 60, -127                      # h2[2] = 60 at the trigger cell, 0 elsewhere
    n.W3[0, 2] = 127                                       # +7,620 on the score
    n.check_ranges()
    return n


# ---- three patches that PASS the direction and gate checks ---------------------------
# Written by the independent reviewer of this study, knowing the checks, to show that
# monotonicity alone does not make a patch safe. Reproduced here with thanks.

def hostile_balance_lock(net):
    """Every eligible peer with the top balance outranks every peer without it. 5 numbers."""
    n = net.copy()
    n.W1[4, BAL], n.b1[4] = 1, -6                 # fires only when bal == 7
    n.W2[2, 4], n.W2[2, 1] = 60, -127
    n.W3[0, 2] = 127
    n.check_ranges()
    return n


def hostile_club(net):
    """Only peers at the top age AND the top balance get a dominating bonus. 6 numbers.
    Age and balance are the features an attacker can fill by waiting and pre-paying,
    so this is a delayed trigger that is monotone in every direction."""
    n = net.copy()
    n.W1[4, BAL], n.W1[4, AGE], n.b1[4] = 1, 1, -13
    n.W2[2, 4], n.W2[2, 1] = 60, -127
    n.W3[0, 2] = 127
    n.check_ranges()
    return n


def hostile_flatten(net):
    """Latency and loss no longer matter. 2 numbers."""
    n = net.copy()
    n.W1[0, RTT] = 0
    n.W1[0, LOSS] = 0
    n.check_ranges()
    return n


def biased_inside_limits(net):
    """Mine: the most lopsided trade-off the limits still allow. One loss step is priced
    at 32 rtt steps, so a loss-free peer wins however slow it is. It passes every check.
    What bounds it is the limits themselves, and then the live trial."""
    n = net.copy()
    n.W1[0, RTT], n.W1[0, LOSS] = -1, -32
    n.b1[0] = 15 * 1 + 7 * 32 + 1
    n.check_ranges()
    return n


# ---- the same policies with one weight per step --------------------------------------
def steps_baseline():
    from wnet import Steps
    st = Steps()
    st.w[RTT][:] = -3
    st.w[LOSS][:] = -4
    st.bias = 80
    return st


def steps_prefer_known():
    from wnet import Steps
    st = Steps()
    st.w[RTT][:] = -12
    st.w[LOSS][:] = -16
    st.w[AGE][:3] = 1                              # the first three steps of age add one each
    st.bias = 320
    return st
