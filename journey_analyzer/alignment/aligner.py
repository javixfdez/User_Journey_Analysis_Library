"""
aligner.py - Stateful wrapper that turns a list of journeys into an N x N
distance matrix (and, next, into prototype journeys via k-medoids).

ROLE IN THE LIBRARY
-------------------
    distance.py  : "how far apart are TWO journeys?"      (pure functions)
    aligner.py   : "how far apart are ALL journeys, and who are the
                    representative ones?"                  (holds state)

The class keeps its working data in underscore-prefixed attributes
(``_sequences``, ``_distance_matrix``). In Python this is a CONVENTION, not
real privacy: it tells users "this is internal, don't touch". Public access
goes through read-only properties, so the notebook can read results but cannot
accidentally overwrite them.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .distance import (
    assign_stages,
    levenshtein_distance,
    normalized_levenshtein_distance,
    weighted_levenshtein_distance,
)

__all__ = ["BaselineAligner"]

# Metric names accepted as strings. A custom callable f(seq_a, seq_b) -> float
# is also accepted, which keeps the class open for new metrics (e.g. DTW).
_PLAIN_METRICS: Dict[str, Callable] = {
    "levenshtein": levenshtein_distance,
    "normalized": normalized_levenshtein_distance,
}


class BaselineAligner:
    """
    Exact-alignment baseline (Stage 1 of the library).

    Parameters
    ----------
    metric : "levenshtein" | "normalized" | "weighted" | callable
        "weighted" needs ``event_to_stage`` (and optionally ``stage_weights``).
    stage_weights : {stage: weight}, only used by the weighted metric.
    event_to_stage : {event: stage}, used to build (event, stage) pairs for the
        weighted metric. Plain event names go in, the class does the staging.
    default_stage : stage for events missing from ``event_to_stage``.
    max_matrix_gb : safety limit. Building an N x N float32 matrix needs
        4*N^2 bytes (N=50,000 -> ~10 GB). If the estimate exceeds this limit
        a MemoryError is raised BEFORE allocating, which is exactly the
        O(N^2) failure these experiments wants to demonstrate.

    Example
    -------
    >>> al = BaselineAligner(metric="levenshtein")
    >>> D = al.fit_distance_matrix([["a", "b"], ["a", "c"], ["a", "b"]])
    >>> float(D[0, 1]), float(D[0, 2])
    (1.0, 0.0)
    """

    def __init__(
        self,
        metric: Union[str, Callable] = "levenshtein",
        stage_weights: Optional[Dict[str, float]] = None,
        event_to_stage: Optional[Mapping[str, str]] = None,
        default_stage: Optional[str] = None,
        max_matrix_gb: float = 4.0,
    ):
        if isinstance(metric, str) and metric not in (*_PLAIN_METRICS, "weighted"):
            raise ValueError(
                f"Unknown metric {metric!r}. Use one of "
                f"{[*_PLAIN_METRICS, 'weighted']} or pass a callable."
            )
        if metric == "weighted" and not event_to_stage:
            raise ValueError("metric='weighted' requires event_to_stage.")

        self.metric = metric
        self.stage_weights = stage_weights
        self.event_to_stage = dict(event_to_stage) if event_to_stage else None
        self.default_stage = default_stage
        self.max_matrix_gb = max_matrix_gb

        # ---- internal state (underscore = "do not touch from outside") ----
        self._sequences: List[Sequence[str]] = []
        self._distance_matrix: Optional[np.ndarray] = None

        # ---- results filled by extract_prototypes (trailing _ = sklearn style:
        #      "attribute that exists only after fitting") ----
        self.medoids_: Optional[List[int]] = None
        self.labels_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Step 1: pairwise distance matrix
    # ------------------------------------------------------------------
    def fit_distance_matrix(self, sequences: List[Sequence[str]]) -> np.ndarray:
        """
        Compute the symmetric N x N matrix D with D[i, j] = distance(seq_i, seq_j).

        Properties of D (all tested in tests/test_alignment.py):
          * zero diagonal      (a journey is identical to itself)
          * symmetric          (D[i, j] == D[j, i])
          * non-negative

        Cost: N*(N-1)/2 distance evaluations, each O(L^2) -> quadratic in N.
        We only compute the upper triangle and mirror it, halving the work.
        """
        seqs = list(sequences)
        n = len(seqs)
        if n < 2:
            raise ValueError("Need at least 2 sequences to compute distances.")

        # Memory guard: fail early and clearly instead of crashing the kernel.
        needed_gb = 4 * n * n / 1e9          # float32 = 4 bytes per cell
        if needed_gb > self.max_matrix_gb:
            raise MemoryError(
                f"A {n} x {n} float32 distance matrix needs ~{needed_gb:.1f} GB "
                f"(limit {self.max_matrix_gb} GB). Use a subsample or switch to "
                f"the O(N) methods (k-grams / LDA / SBERT)."
            )

        dist_fn = self._resolve_metric()
        prepared = self._prepare(seqs)       # e.g. attach stages for 'weighted'

        D = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_fn(prepared[i], prepared[j])
                D[i, j] = d
                D[j, i] = d                  # mirror: distance is symmetric

        self._sequences = seqs               # keep ORIGINAL (unstaged) journeys
        D.flags.writeable = False            # callers get a read-only view
        self._distance_matrix = D
        # A new matrix invalidates any previous clustering.
        self.medoids_, self.labels_ = None, None
        return D

    # ------------------------------------------------------------------
    # Step 2: prototypes (k-medoids / PAM) - NEXT TO IMPLEMENT
    # ------------------------------------------------------------------
    def extract_prototypes(self, k: int, max_iter: int = 100) -> Dict[str, Any]:
        """
        Will run k-medoids (PAM) directly on ``self._distance_matrix`` and fill
        ``self.medoids_`` / ``self.labels_``. Not implemented yet.
        """
        if self._distance_matrix is None:
            raise ValueError("Distance matrix not computed. Call fit_distance_matrix first.")
        raise NotImplementedError("k-medoids (PAM) is the next step of the build.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_metric(self) -> Callable:
        """Return a 2-argument function f(a, b) -> float for the chosen metric."""
        if callable(self.metric):
            return self.metric
        if self.metric == "weighted":
            weights = self.stage_weights
            return lambda a, b: weighted_levenshtein_distance(a, b, weights)
        return _PLAIN_METRICS[self.metric]

    def _prepare(self, seqs: List[Sequence[str]]) -> List[Any]:
        """
        Convert journeys into whatever the metric expects. Done ONCE per
        sequence (N conversions) rather than once per pair (N^2 conversions).
        """
        if self.metric == "weighted":
            return [assign_stages(s, self.event_to_stage, self.default_stage) for s in seqs]
        return seqs

    # ------------------------------------------------------------------
    # Public read-only views
    # ------------------------------------------------------------------
    @property
    def distance_matrix(self) -> Optional[np.ndarray]:
        """Read-only access to the computed matrix (None before fitting)."""
        return self._distance_matrix

    @property
    def sequences(self) -> List[Sequence[str]]:
        """The journeys the matrix was computed on, in matrix order."""
        return list(self._sequences)
