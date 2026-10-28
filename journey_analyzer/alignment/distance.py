"""
distance.py - Sequence-alignment distance metrics for user journeys.

WHAT THIS FILE IS
-----------------
Pure, stateless functions. They take two journeys and return a number.
They know nothing about datasets, matrices or clustering: that is the job
of ``aligner.py``, which *calls* these functions. Keeping the maths separate
from the stateful class makes every metric trivial to unit-test and reuse.

THE CORE IDEA (UX -> maths)
---------------------------
A journey is an ordered list of events, e.g. ["login", "browse", "purchase"].
The *distance* between two journeys is the minimum "cost" of editing one into
the other using three operations:

    insertion     (add an event)
    deletion      (remove an event)
    substitution  (replace an event by another)

Few edits needed  -> similar behaviour   (small distance)
Many edits needed -> different behaviour (large distance)

So a user who loops "browse, browse, browse, abandon" is far from one who goes
"browse, add_to_cart, purchase": the "UX problem" becomes a number.
"""

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "levenshtein_distance",
    "normalized_levenshtein_distance",
    "assign_stages",
    "weighted_levenshtein_distance",
]


# 
# 1. Classic Levenshtein distance
# 
def levenshtein_distance(
    seq1: Sequence,
    seq2: Sequence,
    insert_cost: float = 1.0,
    delete_cost: float = 1.0,
    substitute_cost: float = 1.0,
) -> float:
    """
    Minimum total cost of turning ``seq1`` into ``seq2``.

    Works on ANY sequence of comparable items (strings, ints, ...), so you can
    pass raw event names or integer-encoded events.

    How it works (dynamic programming)
    ----------------------------------
    Let D[i][j] be the cheapest way to turn the first i events of seq1 into
    the first j events of seq2. Then:

        D[0][j] = j * insert_cost            (build seq2[:j] from nothing)
        D[i][0] = i * delete_cost            (erase seq1[:i])
        D[i][j] = min(
            D[i-1][j]   + delete_cost,       # drop seq1[i-1]
            D[i][j-1]   + insert_cost,       # add  seq2[j-1]
            D[i-1][j-1] + (0 if seq1[i-1] == seq2[j-1] else substitute_cost),
        )

    The answer is D[len(seq1)][len(seq2)].

    Worked example: ["login","buy"] -> ["login","browse","buy"]
    One insertion ("browse") is enough, so the distance is 1.

    Complexity: O(n*m) time, O(min(n, m)) memory (we only keep two rows of D).

    Notes
    -----
    * With unequal costs the distance is directional (seq1 -> seq2). With the
      default unit costs it is symmetric, which the aligner relies on.
    """
    if min(insert_cost, delete_cost, substitute_cost) < 0:
        raise ValueError("Edit costs must be non-negative.")

    n, m = len(seq1), len(seq2)

    # Trivial cases: one side is empty -> insert everything / delete everything.
    if n == 0:
        return m * insert_cost
    if m == 0:
        return n * delete_cost

    # Keep the SHORTER sequence as the columns so each row is as small as
    # possible. Swapping the roles of the sequences means we are now going
    # seq2 -> seq1, so "insert" and "delete" swap meaning too.
    if n < m:
        seq1, seq2 = seq2, seq1
        n, m = m, n
        insert_cost, delete_cost = delete_cost, insert_cost

    # Row 0 of D: turning "" into seq2[:j] costs j insertions.
    prev_row = [j * insert_cost for j in range(m + 1)]
    curr_row = [0.0] * (m + 1)

    for i in range(1, n + 1):
        curr_row[0] = i * delete_cost          # column 0: delete i events
        e1 = seq1[i - 1]
        for j in range(1, m + 1):
            sub = 0.0 if e1 == seq2[j - 1] else substitute_cost
            curr_row[j] = min(
                prev_row[j] + delete_cost,       # deletion
                curr_row[j - 1] + insert_cost,   # insertion
                prev_row[j - 1] + sub,           # match / substitution
            )
        # Swap rows instead of copying (cheaper than prev_row = curr_row[:]).
        prev_row, curr_row = curr_row, prev_row

    # After the final swap, the last computed row lives in prev_row.
    return prev_row[m]


# 
# 2. Normalised Levenshtein
# 
def normalized_levenshtein_distance(seq1: Sequence, seq2: Sequence) -> float:
    """
    Levenshtein distance divided by the length of the longer journey -> [0, 1].

    Why normalise? Raw distance grows with length: two 20-event sessions that
    differ in 4 places are "more alike" than two 3-event sessions differing in
    3, yet 4 > 3. Dividing by the longer length compares them fairly.

    Caveat (important for clustering/MDS): this normalisation does NOT always
    satisfy the triangle inequality (e.g. "ab", "aba", "ba": d("ab","ba") =
    2/2 = 1 but d("ab","aba") + d("aba","ba") = 1/3 + 1/3). It is fine for
    k-medoids (which only needs a dissimilarity), but it is not a strict
    metric. The un-normalised version is.
    """
    longest = max(len(seq1), len(seq2))
    if longest == 0:                  # two empty journeys are identical
        return 0.0
    return levenshtein_distance(seq1, seq2) / longest


# 
# 3. Stage assignment helper (feeds the weighted metric)
# 
def assign_stages(
    sequence: Sequence[str],
    event_to_stage: Mapping[str, str],
    default_stage: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Turn ["login", "purchase", "logout"] into
    [("login", "pre"), ("purchase", "purchase"), ("logout", "post")].

    Your preprocessing notebook produces plain event names; the weighted
    metric needs to know which consumption stage each event belongs to. This
    helper bridges the two.

    Parameters
    ----------
    event_to_stage : mapping event name -> stage name,
        e.g. {"login": "pre", "purchase": "purchase", "logout": "post"}.
    default_stage : stage for events missing from the mapping. If None, an
        unknown event raises KeyError so that typos are not silently ignored.
    """
    staged = []
    for event in sequence:
        if event in event_to_stage:
            staged.append((event, event_to_stage[event]))
        elif default_stage is not None:
            staged.append((event, default_stage))
        else:
            raise KeyError(f"Event {event!r} has no stage; add it to event_to_stage "
                           f"or pass default_stage.")
    return staged


# 
# 4. Weighted (stage-aware) Levenshtein  (Kinjo, 2025)
# 
def weighted_levenshtein_distance(
    staged_seq1: Sequence[Tuple[str, str]],
    staged_seq2: Sequence[Tuple[str, str]],
    stage_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Stage-weighted edit distance:

        D(s_i, s_j) = sum over stages g of  w_g * d_lev( s_i[stage g], s_j[stage g] )

    i.e. split each journey into its stages (keeping event order inside every
    stage), compute a normal Levenshtein distance stage by stage, and add them
    up with a weight per stage.

    Why? Not every difference matters equally. If you care mostly about what
    happens around the purchase, give that stage a weight of e.g. 3.0 and the
    others 1.0: differences there then cost three times as much.

    Parameters
    ----------
    staged_seq1, staged_seq2 : sequences of (event_name, stage_name) pairs
        (build them with ``assign_stages``).
    stage_weights : {stage_name: weight}; stages not listed default to 1.0.

    Notes
    -----
    * Order is preserved INSIDE a stage but the relative interleaving of
      events ACROSS stages is ignored. That is intentional in Kinjo's
      formulation (stages are assumed to occur in a fixed order).
    * A non-negative weighted sum of metrics keeps symmetry and the triangle
      inequality (over the per-stage representation).
    """
    weights = stage_weights or {}
    for stage, w in weights.items():
        if w < 0:
            raise ValueError(f"Weight for stage {stage!r} must be non-negative.")

    # 1) Group events by stage, preserving chronological order within a stage.
    stages1: Dict[str, List[str]] = {}
    for event, stage in staged_seq1:
        stages1.setdefault(stage, []).append(event)

    stages2: Dict[str, List[str]] = {}
    for event, stage in staged_seq2:
        stages2.setdefault(stage, []).append(event)

    # 2) Visit every stage that appears in EITHER journey. A stage present in
    #    only one journey is compared against an empty list, so it costs
    #    (number of events) insertions/deletions, as it should.
    total = 0.0
    for stage in stages1.keys() | stages2.keys():
        total += weights.get(stage, 1.0) * levenshtein_distance(
            stages1.get(stage, []), stages2.get(stage, [])
        )
    return total
