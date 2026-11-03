"""
kmedoids.py - k-medoids clustering (PAM) on a PRECOMPUTED distance matrix,
plus a silhouette score that also works from a distance matrix.

WHY K-MEDOIDS AND NOT K-MEANS?
------------------------------
k-means needs a *centroid*: the average of vectors. You cannot average the
journeys ["login","buy"] and ["browse","logout"]; all we have is the distance
between them (Levenshtein). k-medoids replaces the centroid by a **medoid**:
the real session that has the smallest total distance to the other sessions of
its cluster. Two big advantages for UX analysis:

  1. It only needs the distance matrix D (no vectors).
  2. Every prototype is an actual user journey you can read and show to a
     product team, not an abstract average.

THE PAM ALGORITHM (Partitioning Around Medoids, Kaufman & Rousseeuw)
--------------------------------------------------------------------
Cost of a set of medoids  =  sum over all sessions of the distance to its
NEAREST medoid. PAM looks for the set of k medoids with the lowest cost:

  BUILD  Pick medoid #1 = the most central session (smallest row sum of D).
         Then add medoids one by one, each time the session that lowers the
         cost the most. This is deterministic and gives a good start.

  SWAP   Try every possible "replace medoid p by non-medoid h" move. Apply
         the single best move if it lowers the cost; repeat. Stop when no
         swap helps (a local optimum) or after `max_iter` swaps.

Like all k-medoids methods it finds a LOCAL optimum, not guaranteed global.
Cost of one SWAP pass: k * N^2 operations (vectorised with numpy here).
"""

from typing import Iterator, Optional, Tuple

import numpy as np

__all__ = [
    "pam",
    "assign_labels",
    "silhouette_samples_from_distances",
    "silhouette_score_from_distances",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _row_blocks(n_rows: int, width: int, budget: int = 2_000_000) -> Iterator[slice]:
    """
    Yield row slices so that (rows * width) <= `budget` elements.
    Several steps below build a temporary (rows x N) array; doing it in blocks
    keeps the extra memory small even when N is in the thousands.
    """
    step = max(1, budget // max(width, 1))
    for start in range(0, n_rows, step):
        yield slice(start, min(start + step, n_rows))


def _check_matrix(D: np.ndarray, k: Optional[int] = None) -> None:
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"D must be a square matrix, got shape {D.shape}.")
    if k is not None and not (1 <= k <= D.shape[0]):
        raise ValueError(f"k must be between 1 and N={D.shape[0]}, got {k}.")


def assign_labels(D: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """
    Label every session with the position (0..k-1) of its nearest medoid.

    D[medoids] has shape (k, N): row p holds the distance from medoid p to every
    session, so argmin over axis 0 picks the closest medoid per session.
    A medoid is always forced into its own cluster; otherwise, if two medoids
    were identical journeys (distance 0), ties could send one of them to the
    other's cluster.
    """
    medoids = np.asarray(medoids, dtype=int)
    labels = D[medoids].argmin(axis=0)
    labels[medoids] = np.arange(len(medoids))
    return labels


# ---------------------------------------------------------------------------
# BUILD phase
# ---------------------------------------------------------------------------
def _build_init(D: np.ndarray, k: int) -> np.ndarray:
    n = D.shape[0]
    # Medoid #1: the session with the smallest total distance to everybody.
    first = int(np.argmin(D.sum(axis=1, dtype=np.float64)))
    medoids = [first]
    nearest = D[first].astype(np.float64)      # distance of each session to its closest medoid so far

    while len(medoids) < k:
        # gain(c) = how much the total cost drops if session c became a medoid:
        #   sum_j max(0, nearest[j] - D[c, j])
        gains = np.empty(n)
        for sl in _row_blocks(n, n):
            gains[sl] = np.maximum(nearest - D[sl], 0).sum(axis=1, dtype=np.float64)
        gains[medoids] = -1.0                   # never pick the same session twice
        c = int(gains.argmax())
        medoids.append(c)
        nearest = np.minimum(nearest, D[c])
    return np.array(medoids, dtype=int)


# ---------------------------------------------------------------------------
# PAM (BUILD + SWAP)
# ---------------------------------------------------------------------------
def pam(
    D: np.ndarray,
    k: int,
    init: str = "build",
    max_iter: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int, bool]:
    """
    Run PAM on the distance matrix ``D`` (N x N, symmetric).

    Parameters
    ----------
    k        : number of clusters / medoids.
    init     : "build" (deterministic PAM start) or "random" (k random sessions;
               use several restarts and keep the best).
    max_iter : maximum number of SWAP moves applied.
    rng      : numpy Generator, only used for init="random".

    Returns
    -------
    medoids   : (k,) session indices; cluster 0 is the LARGEST cluster.
    labels    : (N,) cluster id of every session (index into `medoids`).
    cost      : total distance of every session to its medoid (a.k.a. inertia).
    n_swaps   : swaps applied.
    converged : True if it stopped because no swap improves the cost
                (False means it hit max_iter).
    """
    _check_matrix(D, k)
    if init not in ("build", "random"):
        raise ValueError("init must be 'build' or 'random'.")
    n = D.shape[0]
    cols = np.arange(n)

    if init == "build":
        medoids = _build_init(D, k)
    else:
        rng = rng or np.random.default_rng()
        medoids = rng.choice(n, size=k, replace=False).astype(int)

    n_swaps, converged = 0, False
    for _ in range(max_iter):
        # --- current situation ------------------------------------------
        dm = D[medoids]                          # (k, N) distances medoid -> session
        nearest_pos = dm.argmin(axis=0)          # which medoid is closest to each session
        nearest_d = dm[nearest_pos, cols]        # ... and how far it is
        # Second-closest medoid distance: needed to know what a session would
        # fall back on if its own medoid were removed.
        second_d = np.partition(dm, 1, axis=0)[1] if k > 1 else np.full(n, np.inf)
        cost = float(nearest_d.sum(dtype=np.float64))
        tol = 1e-9 * max(cost, 1.0)              # ignore float noise

        is_medoid = np.zeros(n, dtype=bool)
        is_medoid[medoids] = True

        # --- look for the best (remove p, add h) swap --------------------
        best_cost, best_p, best_h = cost, -1, -1
        for p in range(k):
            # If medoid p is removed, each session is served by the best of the
            # OTHER medoids: its second-closest if p was its closest, else the
            # same closest one as before.
            d_other = np.where(nearest_pos == p, second_d, nearest_d)

            # Cost if candidate h replaces p:   sum_j min(d_other[j], D[h, j]).
            # Computed for ALL candidates h at once, in row blocks.
            costs = np.empty(n)
            for sl in _row_blocks(n, n):
                costs[sl] = np.minimum(D[sl], d_other).sum(axis=1, dtype=np.float64)
            costs[is_medoid] = np.inf            # candidates must be non-medoids

            h = int(costs.argmin())
            if costs[h] < best_cost - tol:
                best_cost, best_p, best_h = float(costs[h]), p, h

        if best_p < 0:                           # no swap improves -> local optimum
            converged = True
            break
        medoids[best_p] = best_h
        n_swaps += 1

    # --- final assignment, clusters ordered from largest to smallest -----
    labels = assign_labels(D, medoids)
    sizes = np.bincount(labels, minlength=k)
    order = np.argsort(-sizes, kind="stable")    # old positions, biggest first
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)                  # old position -> new position
    medoids, labels = medoids[order], remap[labels]

    final_cost = float(D[medoids, :][labels, np.arange(n)].sum(dtype=np.float64))
    return medoids, labels, final_cost, n_swaps, converged


# ---------------------------------------------------------------------------
# Silhouette from a distance matrix
# ---------------------------------------------------------------------------
def silhouette_samples_from_distances(D: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Silhouette value of every session, in [-1, 1]:
        a(i) = mean distance from i to the OTHER sessions of its own cluster
        b(i) = lowest mean distance from i to the sessions of any OTHER cluster
        s(i) = (b - a) / max(a, b)

    Near +1: the session sits well inside its cluster. Near 0: it is on the
    border between clusters. Negative: it is probably in the wrong cluster.
    Sessions alone in their cluster get 0 by convention.

    Everything is computed with one matrix product: sums[i, c] = total
    distance from session i to all sessions of cluster c.
    """
    _check_matrix(D)
    labels = np.asarray(labels)
    n = len(labels)
    _, inv = np.unique(labels, return_inverse=True)
    k = inv.max() + 1
    if k < 2:
        raise ValueError("Silhouette needs at least 2 clusters.")

    onehot = np.zeros((n, k))
    onehot[np.arange(n), inv] = 1.0
    counts = onehot.sum(axis=0)

    sums = np.empty((n, k))
    for sl in _row_blocks(n, n):                 # block product keeps memory low
        sums[sl] = D[sl].astype(np.float64) @ onehot

    own = inv
    idx = np.arange(n)
    a = sums[idx, own] / np.maximum(counts[own] - 1, 1)
    mean_to_cluster = sums / counts
    mean_to_cluster[idx, own] = np.inf           # exclude the own cluster from b
    b = mean_to_cluster.min(axis=1)

    denom = np.maximum(a, b)
    s = np.divide(b - a, denom, out=np.zeros(n), where=denom > 0)
    s[counts[own] == 1] = 0.0
    return s


def silhouette_score_from_distances(D: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over all sessions (one number per clustering)."""
    return float(silhouette_samples_from_distances(D, labels).mean())