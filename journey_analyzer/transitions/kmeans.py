"""
kmeans.py - Spherical k-means on sparse k-gram vectors (no N x N matrix).

WHY THIS EXISTS
---------------
k-medoids needs the N x N distance matrix: memory grows as N^2, which is the
bottleneck the research question asks about. Because k-gram sessions ARE
vectors, we can cluster them directly. Cost per iteration is
O(nnz * k) (nnz = number of non-zero entries), i.e. linear in the number of
sessions.

WHY "SPHERICAL"?
----------------
Rows are scaled to unit length and centres are re-normalised, so the algorithm
maximises COSINE similarity, the same geometry as the "cosine" distance used
with k-medoids. That keeps the two clustering routes comparable (Dhillon &
Modha, 2001, for the method on sparse text-like data).

WHY k-means++ SEEDING?
----------------------
Random seeds often put two centres in the same group. k-means++ (Arthur &
Vassilvitskii, 2007) picks each new centre with probability proportional to its
distance from the existing ones, which spreads centres out and gives
more stable results.

TRADE-OFF: centres are averages (not real sessions), so unlike k-medoids you do
NOT get a readable prototype journey. We recover readable summaries later from
the per-cluster Markov chain.
"""

from typing import Optional, Tuple

import numpy as np
from scipy import sparse

from .kgrams import l2_normalize_rows

__all__ = ["spherical_kmeans"]


def _kmeanspp(Xn: sparse.csr_matrix, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding with cosine distance (1 - similarity)."""
    n = Xn.shape[0]
    centers = [Xn[int(rng.integers(n))].toarray().ravel()]
    best_sim = np.asarray(Xn @ centers[0]).ravel()           # similarity to the closest centre so far
    for _ in range(1, k):
        d = np.clip(1.0 - best_sim, 0.0, None)
        total = d.sum()
        idx = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=d / total))
        c = Xn[idx].toarray().ravel()
        centers.append(c)
        best_sim = np.maximum(best_sim, np.asarray(Xn @ c).ravel())
    return np.vstack(centers)


def spherical_kmeans(
    X: sparse.spmatrix,
    k: int,
    n_init: int = 5,
    max_iter: int = 100,
    tol: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Cluster the rows of X into k groups.

    Returns
    -------
    labels  : (N,) cluster id; cluster 0 is the largest.
    centers : (k, V) unit-length cluster centres (dense).
    inertia : sum over sessions of (1 - cosine similarity to own centre).
    n_iter  : iterations used by the best restart.

    n_init restarts are run and the lowest-inertia one is kept, because k-means
    only finds a local optimum.
    """
    n = X.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k must be between 1 and N={n}, got {k}.")
    rng = rng or np.random.default_rng()
    Xn = l2_normalize_rows(X)
    rows = np.arange(n)

    best = None
    for _ in range(max(1, n_init)):
        centers = _kmeanspp(Xn, k, rng)
        prev, it = np.inf, 0
        for it in range(1, max_iter + 1):
            sims = np.asarray(Xn @ centers.T)                # (N, k) cosine similarities
            labels = sims.argmax(axis=1)
            own = sims[rows, labels]
            inertia = float((1.0 - own).sum())

            # New centre = mean of the members (sparse indicator matrix times X), re-normalised.
            M = sparse.csr_matrix((np.ones(n), (labels, rows)), shape=(k, n))
            sums = (M @ Xn).toarray()
            norms = np.linalg.norm(sums, axis=1)
            for c in np.flatnonzero(norms == 0):             # empty cluster -> re-seed with the worst-fitting session
                worst = int(np.argmin(own))
                sums[c], norms[c], own[worst] = Xn[worst].toarray().ravel(), 1.0, 1.0
            centers = sums / norms[:, None]

            if prev - inertia <= tol * max(inertia, 1.0):    # converged: no meaningful improvement
                break
            prev = inertia

        sims = np.asarray(Xn @ centers.T)                    # final assignment with the final centres
        labels = sims.argmax(axis=1)
        inertia = float((1.0 - sims[rows, labels]).sum())
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia, it)

    labels, centers, inertia, n_iter = best
    order = np.argsort(-np.bincount(labels, minlength=k), kind="stable")   # biggest cluster first
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels], centers[order], inertia, n_iter
