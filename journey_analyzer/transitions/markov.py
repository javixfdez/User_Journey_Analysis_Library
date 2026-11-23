"""
markov.py - First-order Markov chain tools for user journeys (pure functions).

THE MODEL
---------
A first-order Markov chain says: the next event depends ONLY on the current
one,   P(A_{t+1} | A_1 ... A_t) = P(A_{t+1} | A_t).
We estimate it by counting transitions and normalising each row:

    P[a, b] = (number of times b followed a) / (number of times a was followed by anything)

With <START> and <END> boundary tokens (tokens.py) this one matrix contains
entry behaviour (row <START>), flow between events, and drop-off
(column <END>).

WHY A FIRST-ORDER CHAIN (justification for the report)
------------------------------------------------------
* Standard baseline for navigation paths (Wang et al., 2016/2017) and the
  transition model used for profiling clusters in Bar-El et al. (2026).
* Directly interpretable: every cell is a probability a product team can read,
  and the matrix is a heatmap, which avoids the "spaghetti" clutter of
  node-link/Sankey diagrams (Zhao et al., 2015; Bar-El et al., 2026).
* Cheap: one pass over the data, O(total events); no pairwise comparison.

LIMITS to state in the report
-----------------------------
* Memoryless assumption: "buy after browse" looks the same whether the user
  browsed once or twenty times. Higher-order chains relax this at the price of
  exponentially more states (sparser estimates).
* Probabilities are estimates; rare states give noisy rows. Smoothing and
  minimum counts help (see `normalize_transitions`).
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tokens import END, START, pad_sequence

__all__ = [
    "build_states",
    "count_transitions",
    "normalize_transitions",
    "sequence_log_likelihood",
    "representative_path",
    "conversion_probability",
]


def build_states(sequences: Sequence[Sequence[str]]) -> List[str]:
    """State order used by every matrix: [<START>, events (sorted)..., <END>]."""
    events = sorted({e for s in sequences for e in s})
    return [START, *events, END]


def count_transitions(sequences: Sequence[Sequence[str]], states: Sequence[str]) -> np.ndarray:
    """
    S x S matrix of raw transition counts, counts[a, b] = how often b followed a.
    An event missing from `states` raises KeyError (silent loss would bias P).
    """
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros((len(states), len(states)))
    for seq in sequences:
        path = pad_sequence(seq, True)
        for a, b in zip(path, path[1:]):
            counts[idx[a], idx[b]] += 1
    return counts


def normalize_transitions(counts: np.ndarray, states: Sequence[str], smoothing: float = 0.0) -> np.ndarray:
    """
    Turn counts into probabilities (each row sums to 1; the <END> row is all
    zeros because <END> is absorbing).

    smoothing (additive / Laplace) adds a pseudo-count to every ALLOWED
    transition before normalising. Why: with smoothing=0 an unseen transition
    has probability 0, so any session containing it gets log-likelihood -inf.
    A small value (e.g. 0.01) keeps scores finite, which matters when scoring
    sessions the chain was not trained on. Keep it 0 for purely descriptive
    heatmaps so the numbers equal the observed frequencies.

    Allowed transitions exclude: leaving <END>, entering <START>, and
    <START> -> <END> (an empty session cannot exist).
    """
    states = list(states)
    s0, e = states.index(START), states.index(END)
    allowed = np.ones(counts.shape, dtype=bool)
    allowed[e, :] = False
    allowed[:, s0] = False
    allowed[s0, e] = False

    smoothed = counts + smoothing * allowed
    totals = smoothed.sum(axis=1, keepdims=True)
    return np.divide(smoothed, totals, out=np.zeros_like(smoothed), where=totals > 0)


def sequence_log_likelihood(sequence: Sequence[str], P: np.ndarray, states: Sequence[str]) -> float:
    """
    log P(session) = sum of log P[a, b] over its transitions, including
    <START> -> first and last -> <END>. Returns -inf if any transition has
    probability 0 (use smoothing to avoid that).
    """
    idx = {s: i for i, s in enumerate(states)}
    path = pad_sequence(sequence, True)
    total = 0.0
    for a, b in zip(path, path[1:]):
        p = P[idx[a], idx[b]]
        if p <= 0:
            return float("-inf")
        total += np.log(p)
    return float(total)


def representative_path(
    P: np.ndarray,
    states: Sequence[str],
    min_len: int,
    max_len: int,
) -> Tuple[List[str], float]:
    """
    Most probable event path (Maximum Log-Likelihood, Bar-El et al., 2026) whose
    length is between `min_len` and `max_len` events.

    WHY THE LENGTH BOUND?  Every extra event multiplies in another probability
    (< 1), so the unconstrained most-likely path is always the shortest one
    (<START> -> one event -> <END>): a useless answer driven by length, not
    behaviour. Forcing length >= the cluster's MEDIAN session length removes this
    short-sequence bias while still returning the likeliest path of typical size.

    HOW?  Viterbi dynamic programming (Viterbi, 1967):
        best[t][j] = highest log-probability of any path
                     <START> -> e_1 -> ... -> e_t   with e_t = state j
        best[t][j] = max_i ( best[t-1][i] + log P[i, j] )
    then close each candidate with log P[j, <END>] and take the best over
    t in [min_len, max_len]. Cost: O(max_len * S^2).

    Returns (path of events, its log-likelihood). The path need not exist in
    the data: it is the chain's own "most typical" journey.
    """
    states = list(states)
    if not (1 <= min_len <= max_len):
        raise ValueError("Need 1 <= min_len <= max_len.")
    S = len(states)
    s0, e = states.index(START), states.index(END)
    is_event = np.ones(S, dtype=bool)
    is_event[[s0, e]] = False

    with np.errstate(divide="ignore"):
        logP = np.where(P > 0, np.log(np.where(P > 0, P, 1.0)), -np.inf)

    best = np.full((max_len + 1, S), -np.inf)
    back = np.zeros((max_len + 1, S), dtype=int)
    best[1] = np.where(is_event, logP[s0], -np.inf)
    for t in range(2, max_len + 1):
        cand = best[t - 1][:, None] + logP                   # cand[i, j]: from i to j
        back[t] = cand.argmax(axis=0)
        best[t] = np.where(is_event, cand.max(axis=0), -np.inf)

    closing = best[min_len:max_len + 1] + logP[:, e][None, :]  # add the final j -> <END>
    if not np.isfinite(closing).any():
        raise ValueError("No path of the requested length exists in this chain.")
    t_off, j = np.unravel_index(np.argmax(closing), closing.shape)
    t, loglik = min_len + int(t_off), float(closing[t_off, j])

    path = [j]
    for step in range(t, 1, -1):                              # backtrack through the stored argmaxes
        path.append(int(back[step, path[-1]]))
    return [states[i] for i in reversed(path)], loglik


def conversion_probability(P: np.ndarray, states: Sequence[str], target: str) -> Dict[str, float]:
    """
    For every state, the probability that the session EVER reaches `target`
    before it ends  (absorption probability in an absorbing Markov chain;
    Kemeny & Snell, 1960).

    Let h[s] be that probability. Then h[target] = 1, h[<END>] = 0 and for every
    other state  h[s] = sum_j P[s, j] * h[j].  Writing Q for the transitions
    among the remaining states and b for their one-step probability of hitting
    the target, this is the linear system  (I - Q) h = b.

    UX reading: h["<START>"] is the share of sessions expected to contain the
    target; a state with LOW h that many sessions pass through is a conversion
    bottleneck. Combine with drop-off rates (P[state, <END>]).

    States from which the target can never be reached get 0. (lstsq returns the
    minimum-norm solution, which stays finite even if a closed loop makes the
    system singular.)
    """
    states = list(states)
    if target not in states or target in (START, END):
        raise ValueError(f"target must be an event present in the states, got {target!r}.")
    t, e = states.index(target), states.index(END)
    transient = [i for i in range(len(states)) if i not in (t, e)]

    Q = P[np.ix_(transient, transient)]
    b = P[transient, t]
    h, *_ = np.linalg.lstsq(np.eye(len(transient)) - Q, b, rcond=None)
    h = np.clip(h, 0.0, 1.0)

    out = {states[i]: float(h[k]) for k, i in enumerate(transient)}
    out[target], out[END] = 1.0, 0.0
    return {s: out[s] for s in states}
