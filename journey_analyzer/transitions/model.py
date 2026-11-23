"""
model.py - TransitionModel: Stage 2 of the library.

ROLE
----
    kgrams.py / kmeans.py / markov.py : pure functions (the maths)
    model.py                          : stateful class that chains them

Typical flow (mirrors the notebook)
-----------------------------------
    tm = TransitionModel(k=3, metric="cosine")
    tm.fit_kgrams(sequences)               # sessions -> sparse k-gram vectors
    tm.cluster(k=3, method="kmedoids")     # groups of similar sessions
    tm.fit_markov_chain()                  # one transition matrix per group
    tm.get_representative_path(cluster=0)  # most typical journey of a group

DESIGN CHOICE: why two clustering routes?
-----------------------------------------
* "kmedoids" runs PAM on the k-gram distance matrix. It is exactly the Stage 1
  algorithm with a different distance, so any difference to Levenshtein is
  caused by the DISTANCE ALONE. That makes the comparison fair, and it gives
  readable prototype sessions. Cost: an N x N matrix.
* "kmeans" (spherical) clusters the sparse vectors directly. No N x N matrix, so
  it keeps working where the distance matrix cannot be allocated. Cost: no
  prototype session (centres are averages).

As in Stage 1, attributes ending in "_" exist only after fitting, and
underscore-prefixed attributes are internal state exposed via read-only
properties.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..alignment.kmedoids import pam, silhouette_samples_from_distances
from .kgrams import KGRAM_METRICS, build_kgram_matrix, pairwise_kgram_distances
from .kmeans import spherical_kmeans
from .markov import (
    build_states,
    conversion_probability,
    count_transitions,
    normalize_transitions,
    representative_path,
    sequence_log_likelihood,
)
from .tokens import END, START

__all__ = ["TransitionModel"]

_ALL = "all"          # key of the chain fitted on every session


class TransitionModel:
    """
    Parameters
    ----------
    k              : k-gram length (default 3; see kgrams.py for why not 5).
    metric         : "cosine" | "angular" | "euclidean" (see pairwise_kgram_distances).
    add_boundaries : wrap sessions in <START>/<END> before extracting k-grams.
    binary         : presence instead of counts.
    min_df         : prune k-grams seen in fewer than this many sessions.
    max_matrix_gb  : memory guard for the N x N distance matrix.
    """

    def __init__(self, k: int = 3, metric: str = "cosine", add_boundaries: bool = True,
                 binary: bool = False, min_df: int = 1, max_matrix_gb: float = 4.0):
        if k < 1:
            raise ValueError("k must be >= 1.")
        if metric not in KGRAM_METRICS:
            raise ValueError(f"metric must be one of {KGRAM_METRICS}.")
        self.k, self.metric = k, metric
        self.add_boundaries, self.binary, self.min_df = add_boundaries, binary, min_df
        self.max_matrix_gb = max_matrix_gb

        # ---- internal state ----
        self._sequences: List[Sequence[str]] = []
        self._X = None                                # sparse sessions x k-grams
        self._grams: List[Tuple[str, ...]] = []
        self._distance_matrix: Optional[np.ndarray] = None
        self._states: Optional[List[str]] = None
        self._P: Dict[Any, np.ndarray] = {}           # cluster id (or "all") -> probability matrix
        self._counts: Dict[Any, np.ndarray] = {}      # same keys -> raw counts

        # ---- fitted results ----
        self.labels_: Optional[np.ndarray] = None
        self.medoids_: Optional[np.ndarray] = None    # only for method="kmedoids"
        self.inertia_: Optional[float] = None
        self.cluster_method_: Optional[str] = None

    # ------------------------------------------------------------------
    # Step 1: k-gram features
    # ------------------------------------------------------------------
    def fit_kgrams(self, sequences: Sequence[Sequence[str]]) -> "TransitionModel":
        """Build the sparse session x k-gram count matrix. Resets everything downstream."""
        seqs = [list(s) for s in sequences]
        if len(seqs) < 2:
            raise ValueError("Need at least 2 sequences.")
        self._sequences = seqs
        self._X, self._grams = build_kgram_matrix(
            seqs, self.k, self.add_boundaries, self.binary, self.min_df)
        self._distance_matrix = None
        self._states, self._P, self._counts = None, {}, {}
        self.labels_ = self.medoids_ = self.inertia_ = self.cluster_method_ = None
        return self

    def fit_distance_matrix(self) -> np.ndarray:
        """Dense N x N k-gram distance matrix (needs fit_kgrams first)."""
        self._require_kgrams()
        D = pairwise_kgram_distances(self._X, self.metric, self.max_matrix_gb)
        D.flags.writeable = False
        self._distance_matrix = D
        self.labels_ = self.medoids_ = self.inertia_ = self.cluster_method_ = None
        self._P, self._counts = {}, {}
        return D

    # ------------------------------------------------------------------
    # Step 2: clustering
    # ------------------------------------------------------------------
    def cluster(self, k: int, method: str = "kmedoids", n_init: int = 5,
                max_iter: int = 100, random_state: Optional[int] = None) -> Dict[str, Any]:
        """
        Group the sessions into `k` clusters.

        method="kmedoids": PAM on the k-gram distance matrix (computed on demand).
        method="kmeans"  : spherical k-means on the sparse vectors (no N x N matrix).
        n_init           : restarts for kmeans (the best is kept); PAM/BUILD is
                           deterministic and ignores it.
        """
        self._require_kgrams()
        if method not in ("kmedoids", "kmeans"):
            raise ValueError("method must be 'kmedoids' or 'kmeans'.")
        rng = np.random.default_rng(random_state)

        if method == "kmedoids":
            if self._distance_matrix is None:
                self.fit_distance_matrix()
            medoids, labels, inertia, n_iter, converged = pam(self._distance_matrix, k, max_iter=max_iter)
            self.medoids_ = medoids
        else:
            labels, _, inertia, n_iter = spherical_kmeans(self._X, k, n_init, max_iter, rng=rng)
            self.medoids_, converged = None, n_iter < max_iter

        self.labels_, self.inertia_, self.cluster_method_ = labels, inertia, method
        self._P, self._counts = {}, {}                # chains of an old clustering are stale
        return {"labels": labels, "inertia": inertia, "n_iter": n_iter, "converged": converged,
                "sizes": np.bincount(labels, minlength=k), "medoids": self.medoids_,
                "prototypes": self.prototypes}

    def assign_labels(self, labels: Sequence[int]) -> "TransitionModel":
        """
        Attach a clustering made ELSEWHERE (Stage 1, Stage 3, a human labelling...)
        so that the Markov profile (fit_markov_chain, drop-off, conversion,
        representative path) and silhouette_score can describe it.

        WHY: the profiling layer never needs to know how the clusters were found.
        Applying the SAME descriptors to every approach's clusters is what lets the
        benchmark ask "do better clusters give cleaner behaviour descriptions?"
        separately from "which method found them?".

        labels : one integer >= 0 per session, in the order given to fit_kgrams.
        """
        self._require_kgrams()
        lab = np.asarray(labels)
        if lab.shape != (len(self._sequences),):
            raise ValueError(f"Need one label per session ({len(self._sequences)}), got shape {lab.shape}.")
        if not np.issubdtype(lab.dtype, np.integer) or lab.min() < 0:
            raise ValueError("labels must be integers >= 0 (map noise/-1 to its own cluster first).")
        self.labels_, self.medoids_, self.inertia_ = lab.astype(int), None, None
        self.cluster_method_ = "external"
        self._P, self._counts = {}, {}                # chains of an old clustering are stale
        return self

    def silhouette_score(self) -> float:
        """Mean silhouette of the clustering on the k-gram distance matrix."""
        if self.labels_ is None:
            raise ValueError("No clustering yet. Call cluster() first.")
        if self._distance_matrix is None:
            # fit_distance_matrix() resets the clustering (a NEW matrix normally means a new
            # analysis). Here the matrix only measures the existing clustering, so keep it:
            # without this, cluster(method="kmeans") followed by silhouette_score() crashed.
            saved = (self.labels_, self.medoids_, self.inertia_, self.cluster_method_, self._P, self._counts)
            self.fit_distance_matrix()
            (self.labels_, self.medoids_, self.inertia_, self.cluster_method_, self._P, self._counts) = saved
        return float(silhouette_samples_from_distances(self._distance_matrix, self.labels_).mean())

    # ------------------------------------------------------------------
    # Step 3: Markov chains
    # ------------------------------------------------------------------
    def fit_markov_chain(self, smoothing: float = 0.0) -> "TransitionModel":
        """
        Estimate a first-order transition matrix for ALL sessions and, if a
        clustering exists, one per cluster. All matrices share the same state
        order, so they are directly comparable.
        """
        self._require_kgrams()
        self._states = build_states(self._sequences)
        groups: Dict[Any, List[Sequence[str]]] = {_ALL: self._sequences}
        if self.labels_ is not None:
            for c in range(int(self.labels_.max()) + 1):
                groups[c] = [s for s, l in zip(self._sequences, self.labels_) if l == c]
        self._counts = {key: count_transitions(g, self._states) for key, g in groups.items()}
        self._P = {key: normalize_transitions(c, self._states, smoothing) for key, c in self._counts.items()}
        return self

    def transition_matrix(self, cluster: Optional[int] = None, counts: bool = False) -> np.ndarray:
        """Probability (or raw count) matrix of a cluster, or of all sessions if cluster is None."""
        return (self._counts if counts else self._P)[self._key(cluster)].copy()

    def dropoff_rates(self, cluster: Optional[int] = None) -> Dict[str, float]:
        """
        P(session ends right after this event) for every event, i.e. column
        <END> of the transition matrix. High values = where users leave.
        """
        P = self._P[self._key(cluster)]
        e = self._states.index(END)
        return {s: float(P[i, e]) for i, s in enumerate(self._states) if s not in (START, END)}

    def entry_probabilities(self, cluster: Optional[int] = None) -> Dict[str, float]:
        """P(first event = this event): row <START> of the transition matrix."""
        P = self._P[self._key(cluster)]
        s0 = self._states.index(START)
        return {s: float(P[s0, i]) for i, s in enumerate(self._states) if s not in (START, END)}

    def conversion_probability(self, target: str, cluster: Optional[int] = None) -> Dict[str, float]:
        """P(session ever reaches `target`) from each state (see markov.conversion_probability)."""
        return conversion_probability(self._P[self._key(cluster)], self._states, target)

    def get_representative_path(self, cluster: Optional[int] = None,
                                min_len: Optional[int] = None) -> Tuple[List[str], float]:
        """
        Most probable path of the cluster's chain with length >= `min_len`
        (default: the MEDIAN session length of the cluster, as in Bar-El et al.).
        """
        key = self._key(cluster)
        lengths = [len(s) for s, l in zip(self._sequences, self._members(cluster)) if l]
        if not lengths:
            raise ValueError(f"Cluster {cluster} has no sessions.")
        lo = int(np.median(lengths)) if min_len is None else int(min_len)
        return representative_path(self._P[key], self._states, max(lo, 1), max(max(lengths), lo, 1))

    def log_likelihood(self, sequence: Sequence[str], cluster: Optional[int] = None) -> float:
        """log P(sequence) under the chain of `cluster` (use smoothing > 0 to avoid -inf)."""
        return sequence_log_likelihood(sequence, self._P[self._key(cluster)], self._states)

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------
    @property
    def states(self) -> List[str]:
        """State order of every transition matrix: <START>, events..., <END>."""
        return list(self._states or [])

    @property
    def distance_matrix(self) -> Optional[np.ndarray]:
        return self._distance_matrix

    @property
    def kgram_matrix(self):
        """Sparse sessions x k-grams count matrix (scipy CSR)."""
        return self._X

    @property
    def vocabulary(self) -> List[Tuple[str, ...]]:
        """The k-gram behind each column of `kgram_matrix`."""
        return list(self._grams)

    @property
    def sequences(self) -> List[Sequence[str]]:
        return list(self._sequences)

    @property
    def prototypes(self) -> Optional[List[Sequence[str]]]:
        """Medoid sessions (kmedoids only), else None."""
        return None if self.medoids_ is None else [self._sequences[i] for i in self.medoids_]

    def plot_clusters(self, **kwargs):
        """2D MDS scatter of the clusters (k-medoids clustering only; see visualization.plot_clusters)."""
        if self.labels_ is None or self.medoids_ is None:
            raise ValueError("plot_clusters needs a clustering made with method='kmedoids'.")
        from ..visualization import plot_clusters       # lazy import
        return plot_clusters(self._distance_matrix, self.labels_, self.medoids_, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_kgrams(self) -> None:
        if self._X is None:
            raise ValueError("Call fit_kgrams(sequences) first.")

    def _key(self, cluster: Optional[int]):
        if not self._P:
            raise ValueError("Call fit_markov_chain() first.")
        key = _ALL if cluster is None else cluster
        if key not in self._P:
            raise ValueError(f"No chain for cluster {cluster!r}. Run cluster() before fit_markov_chain().")
        return key

    def _members(self, cluster: Optional[int]) -> np.ndarray:
        """Boolean mask of the sessions in `cluster` (all sessions if None)."""
        if cluster is None:
            return np.ones(len(self._sequences), dtype=bool)
        return self.labels_ == cluster
