"""
kgrams.py - k-gram features and distances between sessions.

THE IDEA (Wang et al., 2016/2017)
---------------------------------
Instead of aligning two journeys step by step (Levenshtein, O(L^2) per pair),
describe every session by WHICH short consecutive subsequences (k-grams) it
contains and how often:

    ["login","browse","browse","buy"], k=2  ->
        {(login,browse): 1, (browse,browse): 1, (browse,buy): 1}

Every session becomes ONE sparse vector of counts over the vocabulary of
k-grams seen in the data. Two sessions are similar when they use the same
short patterns, whatever their position in the session.

WHY THIS APPROACH (justification for the report)
------------------------------------------------
* It captures LOCAL order (k=2 is "what follows what") without the O(L^2)
  alignment, so it scales to far more sessions than the Stage 1 baseline.
* Counts (not just presence) are kept: Wang et al. report that k-grams with
  counts clearly outperform unigram (single-event) and pure time-gap
  descriptions at separating user behaviour.
* Vectors, unlike pairwise edit distances, can also be clustered WITHOUT an
  N x N matrix (see kmeans.py), which is what the scalability question needs.

KNOWN LIMITS
------------
* Order is only preserved inside a window of k events; long-range order is lost.
* Wang et al. use k=5 on very long sessions (142M events overall). With short
  sessions (about 7 events here) k=5 makes almost every session unique, so a
  smaller k is the sensible default. The notebook sweeps k to justify it.
"""

from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
from scipy import sparse

from .tokens import pad_sequence

__all__ = [
    "extract_kgrams",
    "build_kgram_matrix",
    "l2_normalize_rows",
    "pairwise_kgram_distances",
    "KGRAM_METRICS",
]

KGRAM_METRICS = ("cosine", "angular", "euclidean")
_BLOCK_ELEMENTS = 5_000_000        # max size of the temporary block (rows x columns) in the distance loop


def extract_kgrams(tokens: Sequence[str], k: int) -> List[Tuple[str, ...]]:
    """
    All windows of k consecutive tokens.

    Short-sequence rule: if the sequence has fewer than k tokens, the whole
    sequence is returned as ONE (shorter) gram instead of nothing. Otherwise
    short sessions would be all-zero vectors and have no defined similarity.
    """
    if k < 1:
        raise ValueError("k must be >= 1.")
    n = len(tokens)
    if n == 0:
        return []
    if n < k:
        return [tuple(tokens)]
    return [tuple(tokens[i:i + k]) for i in range(n - k + 1)]


def build_kgram_matrix(
    sequences: Sequence[Sequence[str]],
    k: int,
    add_boundaries: bool = True,
    binary: bool = False,
    min_df: int = 1,
) -> Tuple[sparse.csr_matrix, List[Tuple[str, ...]]]:
    """
    Sessions x k-grams count matrix (sparse CSR) and the list of k-grams.

    Parameters
    ----------
    add_boundaries : wrap every session in <START>/<END> before extracting.
    binary         : record presence (1) instead of counts. Lets you test
                     whether frequency matters or only which patterns occur.
    min_df         : drop k-grams that occur in fewer than `min_df` sessions
                     (feature pruning: very rare grams add dimensions but no
                     shared information). Sessions left with no grams become
                     zero rows (distance 1 to everything, by convention).

    Why sparse? With V distinct k-grams a dense N x V matrix is huge (for
    N=10,000 and V=17,000 it would be ~1.4 GB in float64) but each session
    only contains a handful of grams.
    """
    vocab = {}                                   # gram -> column (insertion order = column order)
    indptr, indices, data = [0], [], []
    for seq in sequences:
        counts = Counter(extract_kgrams(pad_sequence(seq, add_boundaries), k))
        for gram, c in counts.items():
            indices.append(vocab.setdefault(gram, len(vocab)))
            data.append(1.0 if binary else float(c))
        indptr.append(len(indices))

    X = sparse.csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(vocab)), dtype=np.float64)
    grams = list(vocab)

    if min_df > 1:
        doc_freq = np.bincount(X.indices, minlength=len(grams))
        keep = np.flatnonzero(doc_freq >= min_df)
        X = X[:, keep]
        grams = [grams[i] for i in keep]
    return X, grams


def l2_normalize_rows(X: sparse.spmatrix) -> sparse.csr_matrix:
    """
    Scale every row to unit length. After this, a dot product between two rows
    IS their cosine similarity. Zero rows stay zero.
    """
    X = sparse.csr_matrix(X)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    inv = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    return sparse.csr_matrix(sparse.diags(inv) @ X)


def pairwise_kgram_distances(X: sparse.spmatrix, metric: str = "cosine",
                             max_matrix_gb: float = 4.0) -> np.ndarray:
    """
    Dense N x N distance matrix between the rows of X.

    metric
    ------
    "cosine"    1 - cos(angle). Ignores session LENGTH (a long and a short
                session with the same mix of patterns are close). Standard
                for count vectors, and the form used by Wang et al. NOT a strict
                metric (triangle inequality can fail).
    "angular"   arccos(cos) / (pi/2), in [0, 1] for non-negative vectors.
                Same ordering as cosine but a TRUE metric, so it is the safer
                choice for MDS and for theory that assumes a metric. This is
                our reading of the "normalized polar distance" in the project
                notes: please check it against Wang et al. (2016) before citing.
    "euclidean" straight-line distance on RAW counts. Sensitive to session
                length (longer sessions have bigger counts). Included as the
                length-sensitive contrast to cosine.

    Efficiency: only the upper triangle is computed (distance is symmetric)
    and mirrored; products are done in row blocks so temporary memory stays
    small. The result is float32, and the memory guard mirrors Stage 1.
    """
    if metric not in KGRAM_METRICS:
        raise ValueError(f"metric must be one of {KGRAM_METRICS}.")
    n = X.shape[0]
    needed_gb = 4 * n * n / 1e9
    if needed_gb > max_matrix_gb:
        raise MemoryError(
            f"A {n} x {n} float32 distance matrix needs ~{needed_gb:.1f} GB "
            f"(limit {max_matrix_gb} GB). Cluster the k-gram vectors directly "
            f"with spherical k-means instead (no N x N matrix)."
        )

    A = sparse.csr_matrix(X) if metric == "euclidean" else l2_normalize_rows(X)
    sq = np.asarray(A.multiply(A).sum(axis=1)).ravel()      # squared norms (used by euclidean)

    D = np.zeros((n, n), dtype=np.float32)
    block = max(1, _BLOCK_ELEMENTS // max(n, 1))                   # rows per block
    for s in range(0, n, block):
        e = min(s + block, n)
        G = (A[s:e] @ A[s:].T).toarray()                     # dot products: block rows x remaining columns
        if metric == "cosine":
            part = 1.0 - G
        elif metric == "angular":
            part = np.arccos(np.clip(G, 0.0, 1.0)) / (np.pi / 2)
        else:
            part = np.sqrt(np.maximum(sq[s:e, None] + sq[None, s:] - 2.0 * G, 0.0))
        D[s:e, s:] = part
        D[s:, s:e] = part.T                                  # mirror (symmetry)

    # Floating-point noise: identical sessions give 1 - 0.9999999999999999, not 0.
    # Snap anything below a tiny floor to exactly 0 (arccos amplifies noise most).
    floor = {"cosine": 1e-9, "angular": 1e-6, "euclidean": 1e-6}[metric]
    D[D < floor] = 0.0
    np.fill_diagonal(D, 0.0)
    return D
