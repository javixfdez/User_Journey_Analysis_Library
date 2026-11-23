"""
describe.py - "Cluster, then describe": make semantic clusters readable.

THE PROBLEM
-----------
An embedding cluster is a region of a 384-dimensional space: nothing in it says
WHAT its sessions do. Stage 1 gives a medoid session, Stage 2 gives transition
matrices, but a plain SBERT + k-means pipeline gives only numbers. The interpretability
axis of the comparison would then be lost by construction.

THE REMEDY (the standard one in embedding-based topic modelling)
----------------------------------------------------------------
Cluster first, describe afterwards, with two outputs per cluster:

1. DISTINCTIVE TERMS by class-based TF-IDF (c-TF-IDF; the formula of BERTopic,
   Grootendorst 2022: recalled from memory, verify before citing). Treat each
   cluster as ONE big document and score
        tf(t, c) * log(1 + A / f(t)),
   where tf is the term's share of the cluster's terms, f(t) its total count over
   all clusters and A the average number of terms per cluster. A term scores high
   when it is frequent in the cluster and rare elsewhere: a label for the cluster,
   not just a common event.
   Mersha et al. (2024) also describe clusters by their top words after UMAP + HDBSCAN.
   The terms here are event unigrams AND bigrams, so entry points, exits and
   transitions ("add_to_cart > purchase") appear as words of the description.
2. EXAMPLE SESSIONS: the real sessions closest (cosine) to the cluster centre.
   The analogue of the Stage 1 medoid; works for ANY feature space (embeddings,
   topic mixtures, term counts) because it only needs a row per session.

Both are computed from the clustering alone, so they apply to ANY labels, including
those of Stage 1 and Stage 2. That is what lets the benchmark compare descriptors
on equal terms.
"""

from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse

from ..transitions.kgrams import l2_normalize_rows

__all__ = ["ctfidf_scores", "distinctive_terms", "exemplar_indices"]


def ctfidf_scores(X: sparse.spmatrix, labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """(n_clusters, V) c-TF-IDF matrix from a document x term COUNT matrix."""
    X = sparse.csr_matrix(X)
    n = X.shape[0]
    M = sparse.csr_matrix((np.ones(n), (labels, np.arange(n))), shape=(n_clusters, n))
    T = np.asarray((M @ X).todense())                              # cluster x term counts
    sizes = T.sum(axis=1, keepdims=True)
    tf = np.divide(T, sizes, out=np.zeros_like(T), where=sizes > 0)
    f = T.sum(axis=0)
    A = T.sum(axis=1).mean()
    idf = np.log(1.0 + A / np.maximum(f, 1e-12))
    return tf * idf


def distinctive_terms(X: sparse.spmatrix, vocabulary: Sequence[str], labels: np.ndarray,
                      top_n: int = 8) -> List[List[Dict[str, float]]]:
    """
    For each cluster, its `top_n` terms ranked by c-TF-IDF, each with two readable numbers:
        in_cluster : share of the cluster's sessions that contain the term
        overall    : share of ALL sessions that contain it
    (in_cluster much larger than overall = the term characterises the cluster).
    """
    labels = np.asarray(labels)
    k = int(labels.max()) + 1
    scores = ctfidf_scores(X, labels, k)
    present = sparse.csr_matrix(X > 0)
    overall = np.asarray(present.mean(axis=0)).ravel()
    out: List[List[Dict[str, float]]] = []
    for c in range(k):
        members = np.flatnonzero(labels == c)
        within = np.asarray(present[members].mean(axis=0)).ravel() if len(members) else np.zeros(X.shape[1])
        out.append([{"term": vocabulary[j], "score": float(scores[c, j]),
                     "in_cluster": float(within[j]), "overall": float(overall[j])}
                    for j in np.argsort(-scores[c])[:top_n] if scores[c, j] > 0])
    return out


def exemplar_indices(F, labels: np.ndarray, n: int = 3) -> List[np.ndarray]:
    """
    Indices of the `n` sessions closest to each cluster's centre (cosine).

    F : (N, d) dense or sparse feature rows (embeddings, topic mixtures, term counts).
    The centre is the normalised mean of the cluster's rows.
    """
    labels = np.asarray(labels)
    Fn = l2_normalize_rows(F) if sparse.issparse(F) else _l2_dense(np.asarray(F, dtype=float))
    out = []
    for c in range(int(labels.max()) + 1):
        members = np.flatnonzero(labels == c)
        if len(members) == 0:
            out.append(np.array([], dtype=int))
            continue
        rows = Fn[members]
        centre = np.asarray(rows.mean(axis=0)).ravel()
        norm = np.linalg.norm(centre)
        sims = np.asarray(rows @ (centre / norm if norm > 0 else centre)).ravel()
        out.append(members[np.argsort(-sims, kind="stable")[:n]])
    return out


def _l2_dense(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return np.divide(A, norms, out=np.zeros_like(A), where=norms > 0)
