"""
model.py - SemanticNLPModel: Stage 3 of the library.

ROLE (same layering as the other stages)
----------------------------------------
    rendering.py            session -> text / tokens
    terms.py                tokens  -> bag of terms (unigrams, bigrams)
    gsdmm.py, lda.py        topic models on the terms          (STTM family)
    sbert.py                text    -> embedding                (SBERT family)
    describe.py             clusters -> distinctive terms + example sessions
    model.py                stateful class chaining all of the above

Typical flow (mirrors the notebooks)
------------------------------------
    sm = SemanticNLPModel(renderer=SessionRenderer("natural", rle=True),
                          encoder=SBERTEncoder())
    sm.fit_texts(sequences)                      # render the sessions
    sm.cluster(5, method="sbert")                # embeddings + spherical k-means
    sm.cluster(5, method="gsdmm")                # or a short-text topic model
    sm.describe()                                # what is each cluster about?
    tm = sm.profile()                            # Markov profile of these clusters

The four methods and WHY each exists
------------------------------------
"sbert"    SBERT embedding -> spherical k-means (the engine of Stage 2). The main
           semantic candidate. Linear in N, no N x N matrix. Cost: an encoder
           (one-off load + per-text encoding).
"gsdmm"    One-topic-per-session mixture model for SHORT texts (Qiang et al. 2022
           recommend it). Picks its own number of clusters; k is an upper bound.
"lda"      Classic topic model; the baseline named in the plan. A session is a
           mixture of topics; labelled by the dominant one (or k-means on the mixture).
"hdbscan"  Mersha et al. (2024): embedding -> dimensionality reduction -> HDBSCAN.
           No k needed; sessions in no dense region are NOISE. Noise is kept as
           its own LAST cluster so downstream code (silhouette, profiles) still
           works; `noise_mask_` says which sessions it holds, and any score
           should be reported with and without it.

All methods number clusters from the largest to the smallest. Attributes ending in
"_" exist only after fitting; underscore-prefixed attributes are internal.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse

from ..alignment.kmedoids import silhouette_samples_from_distances
from ..transitions.kmeans import spherical_kmeans
from .describe import distinctive_terms, exemplar_indices
from .gsdmm import GSDMM
from .lda import LDATopicModel
from .rendering import SessionRenderer
from .sbert import SBERTEncoder
from .terms import TERM_MODES, build_term_matrix, make_terms

__all__ = ["SemanticNLPModel"]

METHODS = ("sbert", "gsdmm", "lda", "hdbscan")


def _relabel(raw: np.ndarray):
    """
    Contiguous labels 0..K-1 from the largest cluster to the smallest; noise (-1)
    becomes its own LAST cluster. Returns (labels, {raw id -> new id}, noise mask).
    """
    noise = raw < 0
    ids, counts = np.unique(raw[~noise], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    mapping = {int(o): i for i, o in enumerate(order)}
    labels = np.empty(raw.shape, dtype=int)
    for old, new in mapping.items():
        labels[raw == old] = new
    if noise.any():
        labels[noise] = len(order)
    return labels, mapping, noise


def _l2_dense(A: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return np.divide(A, norms, out=np.zeros_like(A), where=norms > 0)


class SemanticNLPModel:
    """
    Parameters
    ----------
    renderer  : SessionRenderer (default: natural-language text, no RLE, no window).
    encoder   : object with `.encode(list_of_str) -> (N, d)` unit-length rows:
                SBERTEncoder() or HashingEncoder(). Default: SBERTEncoder(), created
                only when embeddings are first needed.
    term_mode : "unigram" | "bigram" | "both" terms for the topic models and descriptors
                (see terms.py; unigrams alone cannot see order).
    boundaries: add <START>/<END> when forming bigram terms.
    min_df    : drop terms found in fewer than this many sessions.
    """

    def __init__(self, renderer: Optional[SessionRenderer] = None, encoder: Any = None,
                 term_mode: str = "both", boundaries: bool = True, min_df: int = 1):
        if term_mode not in TERM_MODES:
            raise ValueError(f"term_mode must be one of {TERM_MODES}.")
        self.renderer = renderer or SessionRenderer()
        self._encoder = encoder
        self.term_mode, self.boundaries, self.min_df = term_mode, boundaries, min_df

        # ---- internal state ----
        self._sequences: List[Sequence[str]] = []
        self._texts: List[str] = []
        self._X: Optional[sparse.csr_matrix] = None         # sessions x terms (counts)
        self._vocab: List[str] = []
        self._emb: Optional[np.ndarray] = None               # sessions x d, unit length
        self._topic_model: Any = None                        # GSDMM or LDATopicModel
        self._label_map: Dict[int, int] = {}                 # raw topic id -> final cluster id
        self._lda_assignment: str = "argmax"

        # ---- fitted results ----
        self.labels_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None           # (k, d) unit centres (sbert / lda-kmeans)
        self.inertia_: Optional[float] = None
        self.method_: Optional[str] = None
        self.n_clusters_found_: Optional[int] = None
        self.noise_mask_: Optional[np.ndarray] = None        # hdbscan only
        self.timings_: Dict[str, float] = {}                 # seconds per step, for the scalability axis

    # ------------------------------------------------------------------
    # Step 1: text and terms
    # ------------------------------------------------------------------
    def fit_texts(self, sequences: Sequence[Sequence[str]]) -> "SemanticNLPModel":
        """Render every session as text and as a bag of terms. Resets everything downstream."""
        seqs = [list(s) for s in sequences]
        if len(seqs) < 2:
            raise ValueError("Need at least 2 sequences.")
        t0 = time.perf_counter()
        self._sequences = seqs
        self._texts = self.renderer.render_many(seqs)
        docs = [make_terms(self.renderer.tokens(s), self.term_mode, self.boundaries) for s in seqs]
        self._X, self._vocab = build_term_matrix(docs, min_df=self.min_df)
        self.timings_ = {"render": time.perf_counter() - t0}
        self._emb = self._topic_model = self.labels_ = self.centers_ = None
        self.inertia_ = self.method_ = self.n_clusters_found_ = self.noise_mask_ = None
        self._label_map = {}
        return self

    # ------------------------------------------------------------------
    # Step 2: embeddings
    # ------------------------------------------------------------------
    def fit_embeddings(self) -> np.ndarray:
        """(N, d) unit-length embeddings of the rendered texts. Called on demand by cluster()."""
        self._require_texts()
        if self._encoder is None:
            self._encoder = SBERTEncoder()
        t0 = time.perf_counter()
        emb = np.asarray(self._encoder.encode(self._texts), dtype=np.float32)
        if emb.shape[0] != len(self._texts):
            raise ValueError("The encoder must return one row per text.")
        self._emb = emb
        self.timings_["embed"] = time.perf_counter() - t0
        return emb

    # ------------------------------------------------------------------
    # Step 3: clustering
    # ------------------------------------------------------------------
    def cluster(self, k: Optional[int] = None, method: str = "sbert", n_init: int = 5,
                max_iter: int = 100, random_state: Optional[int] = None, **params) -> Dict[str, Any]:
        """
        Group the sessions. `method` is one of "sbert", "gsdmm", "lda", "hdbscan".

        k      : number of clusters for sbert; an UPPER BOUND for gsdmm; number of
                 topics for lda; ignored by hdbscan (it finds its own).
        params : method-specific options, forwarded as is:
                   gsdmm  : alpha, beta, n_iter, tol
                   lda    : assignment="argmax"|"kmeans", doc_topic_prior, topic_word_prior, max_iter
                   hdbscan: reducer="pca"|"umap"|None, n_components, min_cluster_size, min_samples
        """
        self._require_texts()
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}.")
        if method != "hdbscan" and (k is None or k < 1):
            raise ValueError(f"method='{method}' needs k >= 1.")
        rng = np.random.default_rng(random_state)
        t0 = time.perf_counter()
        self.centers_, self.inertia_, self.noise_mask_, self._topic_model = None, None, None, None
        self._label_map, self._lda_assignment = {}, "argmax"

        if method == "sbert":
            emb = self._emb if self._emb is not None else self.fit_embeddings()
            t0 = time.perf_counter()                              # embedding time is reported separately
            labels, self.centers_, self.inertia_, _ = spherical_kmeans(
                sparse.csr_matrix(emb), k, n_init, max_iter, rng=rng)
            self._label_map = {i: i for i in range(k)}
        elif method == "gsdmm":
            self._topic_model = GSDMM(n_clusters=k, random_state=random_state, **params)
            self._topic_model.fit_matrix(self._X, self._vocab)
            labels, self._label_map, _ = _relabel(self._topic_model.labels_)
        elif method == "lda":
            self._lda_assignment = params.pop("assignment", "argmax")
            if self._lda_assignment not in ("argmax", "kmeans"):
                raise ValueError("assignment must be 'argmax' or 'kmeans'.")
            self._topic_model = LDATopicModel(n_topics=k, random_state=random_state, **params)
            self._topic_model.fit_matrix(self._X, self._vocab)
            theta = self._topic_model.doc_topic_
            if self._lda_assignment == "argmax":
                labels, self._label_map, _ = _relabel(theta.argmax(axis=1))
            else:
                labels, self.centers_, self.inertia_, _ = spherical_kmeans(
                    sparse.csr_matrix(theta), k, n_init, max_iter, rng=rng)
        else:                                                     # hdbscan
            emb = self._emb if self._emb is not None else self.fit_embeddings()
            t0 = time.perf_counter()
            raw = self._hdbscan(emb, random_state, **params)
            labels, self._label_map, self.noise_mask_ = _relabel(raw)

        self.labels_, self.method_ = labels, method
        self.n_clusters_found_ = int(labels.max()) + 1
        self.timings_["cluster"] = time.perf_counter() - t0
        return {"labels": labels, "sizes": np.bincount(labels), "n_clusters_found": self.n_clusters_found_,
                "inertia": self.inertia_,
                "n_noise": int(self.noise_mask_.sum()) if self.noise_mask_ is not None else 0}

    def _hdbscan(self, emb: np.ndarray, random_state: Optional[int], reducer: Optional[str] = "pca",
                 n_components: int = 10, min_cluster_size: Optional[int] = None,
                 min_samples: Optional[int] = None) -> np.ndarray:
        """Reduce, then HDBSCAN. Returns raw labels (-1 = noise)."""
        try:
            from sklearn.cluster import HDBSCAN                    # scikit-learn >= 1.3
        except ImportError as e:                                   # pragma: no cover
            raise ImportError("method='hdbscan' needs scikit-learn >= 1.3.") from e
        n = emb.shape[0]
        if reducer is None:
            Z = emb
        elif reducer == "pca":
            # 384 dimensions make all distances look alike ("curse of dimensionality"),
            # the reason Mersha et al. reduce first. PCA is the dependency-free option:
            # eigenvectors of the d x d covariance, so cost is linear in N.
            Xc = emb - emb.mean(axis=0, keepdims=True)
            vals, vecs = np.linalg.eigh(Xc.T @ Xc / max(n - 1, 1))
            Z = Xc @ vecs[:, np.argsort(-vals)[:min(n_components, emb.shape[1])]]
        elif reducer == "umap":
            try:
                import umap                                       # optional dependency
            except ImportError as e:
                raise ImportError("reducer='umap' needs umap-learn: pip install umap-learn "
                                  "(or use reducer='pca').") from e
            Z = umap.UMAP(n_components=n_components, metric="cosine", random_state=random_state).fit_transform(emb)
        else:
            raise ValueError("reducer must be 'pca', 'umap' or None.")
        mcs = min_cluster_size or max(10, n // 100)                # a cluster must hold >= 1% of the sessions
        with warnings.catch_warnings():                            # scikit-learn announces a changed default of `copy`
            warnings.simplefilter("ignore", FutureWarning)
            return HDBSCAN(min_cluster_size=mcs, min_samples=min_samples).fit(Z).labels_

    # ------------------------------------------------------------------
    # Step 4: quality and description
    # ------------------------------------------------------------------
    def silhouette_score(self, sample_size: int = 2000, random_state: int = 0) -> float:
        """
        Mean silhouette in the method's OWN space (cosine), on a random sample.

        NOT comparable with the silhouette of Stage 1 or 2 (different geometry): for a
        cross-approach number, compute every clustering's silhouette under one common
        reference distance (`profile()` + TransitionModel.silhouette_score()).
        Sampling keeps the N x N distance matrix at sample_size^2.
        """
        self._require_labels()
        n = len(self._sequences)
        idx = np.random.default_rng(random_state).choice(n, size=min(n, sample_size), replace=False)
        F = self._features()
        Fs = F[idx].toarray() if sparse.issparse(F) else np.asarray(F[idx], dtype=float)
        Fs = _l2_dense(Fs)
        D = np.clip(1.0 - Fs @ Fs.T, 0.0, 2.0)
        np.fill_diagonal(D, 0.0)
        lab = self.labels_[idx]
        if len(np.unique(lab)) < 2:
            raise ValueError("The sample holds a single cluster; silhouette is undefined.")
        return float(silhouette_samples_from_distances(D, lab).mean())

    def describe(self, top_n: int = 8, n_examples: int = 3) -> List[Dict[str, Any]]:
        """
        One dict per cluster: size, mean session length, distinctive terms (c-TF-IDF
        with in-cluster and overall presence) and the sessions closest to its centre.
        See describe.py for why these two outputs.
        """
        self._require_labels()
        n = len(self._sequences)
        terms = distinctive_terms(self._X, self._vocab, self.labels_, top_n)
        ex = exemplar_indices(self._features(), self.labels_, n_examples)
        out = []
        for c in range(self.n_clusters_found_):
            members = np.flatnonzero(self.labels_ == c)
            out.append({
                "cluster": c, "size": int(len(members)), "share": len(members) / n,
                "is_noise": bool(self.noise_mask_ is not None and self.noise_mask_.any()
                                 and c == self.n_clusters_found_ - 1),
                "mean_length": float(np.mean([len(self._sequences[i]) for i in members])) if len(members) else 0.0,
                "top_terms": terms[c],
                "example_indices": ex[c].tolist(),
                "examples": [self._sequences[i] for i in ex[c]],
                "example_texts": [self._texts[i] for i in ex[c]],
            })
        return out

    def profile(self, k_gram: int = 2, smoothing: float = 0.0, event_map=None):
        """
        The SAME Markov profile as Stage 2 (transition matrices, drop-off, conversion,
        representative path) on THESE clusters. Returns a fitted TransitionModel.

        event_map : optional function session -> session applied BEFORE the chains are
                    fitted (the clusters themselves are not changed). Use it to
                    coarsen the states, e.g. `lambda s: [e.split("_")[0] for e in s]`
                    turns `cart_electronics.smartphone` into `cart`. WHY: with
                    category-level tokens the chain has thousands of states, the
                    matrices are unreadable, and "probability of reaching a cart" is
                    spread over hundreds of different cart tokens. At action level
                    the profile answers the funnel questions directly.
        """
        self._require_labels()
        from ..transitions.model import TransitionModel          # local import: avoids a package cycle
        seqs = self._sequences if event_map is None else [list(event_map(s)) for s in self._sequences]
        tm = TransitionModel(k=k_gram).fit_kgrams(seqs)
        tm.assign_labels(self.labels_)
        return tm.fit_markov_chain(smoothing)

    # ------------------------------------------------------------------
    # New sessions
    # ------------------------------------------------------------------
    def predict(self, sequences: Sequence[Sequence[str]]) -> np.ndarray:
        """Assign NEW sessions to the fitted clusters without refitting (not available for hdbscan)."""
        self._require_labels()
        seqs = [list(s) for s in sequences]
        if self.method_ == "hdbscan":
            raise NotImplementedError("HDBSCAN has no out-of-sample assignment in this library.")
        if self.method_ == "sbert" or (self.method_ == "lda" and self._lda_assignment == "kmeans"):
            if self.method_ == "sbert":
                feats = np.asarray(self._encoder.encode(self.renderer.render_many(seqs)), dtype=float)
            else:
                feats = self._topic_model.transform(self._docs(seqs))
            return (_l2_dense(feats) @ self.centers_.T).argmax(axis=1)
        docs = self._docs(seqs)
        if self.method_ == "gsdmm":
            raw = self._topic_model.predict(docs)
            return np.array([self._label_map[int(r)] for r in raw])
        theta = self._topic_model.transform(docs)                 # lda, argmax
        used = sorted(self._label_map)                            # topics that own at least one training session
        return np.array([self._label_map[used[j]] for j in theta[:, used].argmax(axis=1)])

    def _docs(self, seqs: Sequence[Sequence[str]]) -> List[List[str]]:
        return [make_terms(self.renderer.tokens(s), self.term_mode, self.boundaries) for s in seqs]

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------
    @property
    def texts(self) -> List[str]:
        """The rendered text of every session (what the encoder reads)."""
        return list(self._texts)

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        return self._emb

    @property
    def term_matrix(self) -> Optional[sparse.csr_matrix]:
        """Sessions x terms COUNT matrix used by the topic models and the descriptors."""
        return self._X

    @property
    def vocabulary(self) -> List[str]:
        return list(self._vocab)

    @property
    def topic_model(self) -> Any:
        """The fitted GSDMM / LDATopicModel (None for sbert and hdbscan)."""
        return self._topic_model

    @property
    def sequences(self) -> List[Sequence[str]]:
        return list(self._sequences)

    def get_params(self) -> Dict[str, Any]:
        """Everything that defines the pipeline, for a results table."""
        enc = getattr(self._encoder, "name_", type(self._encoder).__name__) if self._encoder is not None else None
        return {"method": self.method_, "term_mode": self.term_mode, "encoder": enc,
                **{f"render_{k}": v for k, v in self.renderer.get_params().items()},
                "n_clusters_found": self.n_clusters_found_}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _features(self):
        """Rows describing each session in the space the method clustered in."""
        if self.method_ in ("sbert", "hdbscan"):
            return self._emb
        if self.method_ == "lda":
            return self._topic_model.doc_topic_
        return self._X

    def _require_texts(self) -> None:
        if self._X is None:
            raise ValueError("Call fit_texts(sequences) first.")

    def _require_labels(self) -> None:
        if self.labels_ is None:
            raise ValueError("No clustering yet. Call cluster() first.")
