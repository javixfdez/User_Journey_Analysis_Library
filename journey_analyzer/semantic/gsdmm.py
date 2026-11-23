"""
gsdmm.py - GSDMM: Gibbs-sampling Dirichlet Multinomial Mixture for short texts.

WHY GSDMM FOR SESSIONS
----------------------
A session is a SHORT text: a handful of events, so few terms co-occur in it.
Standard LDA assumes every document mixes several topics; with 3-6 terms that
mixture cannot be estimated, which is why Qiang et al. (2022) find most
short-text models beating LDA on short documents and recommend GSDMM and BTM as
the simple, effective, efficient choices.

GSDMM (Yin & Wang, 2014) makes the opposite assumption: ONE topic per document.
That is a CLUSTERING model, which is exactly our task (one behaviour per
session). Its output is a hard label per session, with no k-means step on top.

THE MODEL (Dirichlet Multinomial Mixture)
-----------------------------------------
    cluster ~ Categorical(theta),  theta ~ Dirichlet(alpha)
    each term of the session ~ Categorical(phi_cluster),  phi ~ Dirichlet(beta)

A collapsed Gibbs sampler integrates theta and phi out and resamples the cluster
of one session at a time, given all the others:

    p(z_d = k | rest)  is proportional to
        (m_k + alpha)                                       how popular is k
      * prod_w prod_{j<c_dw} (n_kw + beta + j)              how well k explains the terms
      / prod_{j<L_d}        (n_k + V*beta + j)              normalisation by cluster size

    m_k = sessions in k, n_kw = count of term w in k, n_k = total terms in k,
    c_dw = count of w in session d, L_d = its length, V = vocabulary size.
    (All counts EXCLUDE session d.)

IMPLEMENTATION CHOICE: log-gamma instead of the nested products
---------------------------------------------------------------
prod_{j<c}(x + j) = Gamma(x + c) / Gamma(x), so each term costs one vectorised
`gammaln` call over all K clusters instead of a Python loop over j. The two forms
are identical; a unit test checks the identity against the explicit product.
Work per sweep is O(sessions x K x distinct terms per session): linear in N, but
a Python loop over sessions, so it is the slowest component of Stage 3 (see the
scalability section of the notebook). Fit on a sample and `predict` the rest.

THE NUMBER OF CLUSTERS
----------------------
`n_clusters` is an UPPER BOUND. Clusters that attract no session vanish during
sampling, so the model chooses its own number (Yin & Wang call this the ability
to "automatically infer" it). Consequences for the research design:
  * the number found, `n_clusters_found_`, must be reported next to every score;
  * to compare with k-medoids/k-means at a fixed k, tune alpha/beta or merge
    small clusters; do NOT compare ARI across different cluster counts blindly.
alpha = beta = 0.1 are the values commonly used with GSDMM (verify against
Yin & Wang 2014 before citing): a small beta makes clusters prefer few terms,
a small alpha makes empty clusters hard to revive, so the number found shrinks.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.special import gammaln, logsumexp

from .terms import build_term_matrix

__all__ = ["GSDMM"]


class GSDMM:
    """
    Parameters
    ----------
    n_clusters   : upper bound K on the number of clusters.
    alpha, beta  : Dirichlet priors (see module docstring).
    n_iter       : maximum number of Gibbs sweeps over all sessions.
    tol          : stop early when fewer than this FRACTION of sessions changed
                   cluster in a sweep (Gibbs sampling never converges exactly).
    min_df       : drop terms found in fewer than this many sessions.
    random_state : seed. The sampler is stochastic, so always fix it and report
                   the spread over several seeds.
    """

    def __init__(self, n_clusters: int = 20, alpha: float = 0.1, beta: float = 0.1,
                 n_iter: int = 30, tol: float = 0.001, min_df: int = 1,
                 random_state: Optional[int] = None):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1.")
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0.")
        self.n_clusters, self.alpha, self.beta = n_clusters, alpha, beta
        self.n_iter, self.tol, self.min_df, self.random_state = n_iter, tol, min_df, random_state

        # ---- fitted state (ends with "_" by project convention) ----
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_found_: Optional[int] = None
        self.cluster_sizes_: Optional[np.ndarray] = None
        self.vocabulary_: List[str] = []
        self.history_: Dict[str, List[float]] = {"moved": [], "n_clusters": []}
        self._m: Optional[np.ndarray] = None      # (K,)   sessions per cluster
        self._n: Optional[np.ndarray] = None      # (K,)   terms per cluster
        self._nw: Optional[np.ndarray] = None     # (K, V) term counts per cluster

    # ------------------------------------------------------------------
    def fit(self, docs: Sequence[Sequence[str]]) -> "GSDMM":
        """Cluster `docs` (each a list of term strings, see terms.make_terms)."""
        X, vocab = build_term_matrix(docs, min_df=self.min_df)
        return self.fit_matrix(X, vocab)

    def fit_matrix(self, X: sparse.spmatrix, vocabulary: Optional[Sequence[str]] = None) -> "GSDMM":
        """Same as fit(), starting from a document x term COUNT matrix."""
        X = sparse.csr_matrix(X)
        X.sum_duplicates()
        D, V = X.shape
        if D < 2:
            raise ValueError("Need at least 2 documents.")
        K = self.n_clusters
        rng = np.random.default_rng(self.random_state)

        # Per document: ids of its distinct terms, their counts, total length.
        ids = [X.indices[X.indptr[i]:X.indptr[i + 1]] for i in range(D)]
        cnt = [X.data[X.indptr[i]:X.indptr[i + 1]].astype(float) for i in range(D)]
        length = np.asarray(X.sum(axis=1)).ravel()

        # Random initial assignment, then the sufficient statistics.
        z = rng.integers(0, K, size=D)
        m = np.zeros(K); n = np.zeros(K); nw = np.zeros((K, V))
        for i in range(D):
            m[z[i]] += 1; n[z[i]] += length[i]; nw[z[i], ids[i]] += cnt[i]

        Vb = V * self.beta
        self.history_ = {"moved": [], "n_clusters": []}
        for _ in range(self.n_iter):
            moved = 0
            for i in rng.permutation(D):
                k, w, c, L = z[i], ids[i], cnt[i], length[i]
                m[k] -= 1; n[k] -= L; nw[k, w] -= c                   # take session i out
                knew = self._sample(rng, m, n, nw, w, c, L, Vb)
                z[i] = knew
                m[knew] += 1; n[knew] += L; nw[knew, w] += c            # put it back
                moved += knew != k
            self.history_["moved"].append(moved / D)
            self.history_["n_clusters"].append(int((m > 0).sum()))
            if moved / D < self.tol:
                break

        # Drop empty clusters; number the rest from the largest to the smallest.
        order = [k for k in np.argsort(-m, kind="stable") if m[k] > 0]
        remap = np.full(K, -1)
        remap[order] = np.arange(len(order))
        self.labels_ = remap[z]
        self._m, self._n, self._nw = m[order], n[order], nw[order]
        self.cluster_sizes_ = self._m.astype(int)
        self.n_clusters_found_ = len(order)
        self.vocabulary_ = list(vocabulary) if vocabulary is not None else [str(j) for j in range(V)]
        return self

    # ------------------------------------------------------------------
    def _sample(self, rng, m, n, nw, w, c, L, Vb) -> int:
        """Draw a cluster for one session from the collapsed conditional (see module docstring)."""
        a = nw[:, w] + self.beta                                        # (K, distinct terms)
        logp = (np.log(m + self.alpha)
                + (gammaln(a + c) - gammaln(a)).sum(axis=1)
                - (gammaln(n + Vb + L) - gammaln(n + Vb)))
        p = np.exp(logp - logp.max())
        cum = np.cumsum(p)
        return min(int(np.searchsorted(cum, rng.random() * cum[-1])), len(p) - 1)

    def predict_log_proba(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        """
        log p(cluster | session) for NEW sessions with the fitted counts frozen.
        Terms outside the training vocabulary are ignored. Shape (N, found clusters).
        """
        self._check_fitted()
        X, _ = build_term_matrix(docs, vocabulary=self.vocabulary_)
        V = self._nw.shape[1]
        Vb = V * self.beta
        out = np.empty((X.shape[0], self._nw.shape[0]))
        for i in range(X.shape[0]):
            w = X.indices[X.indptr[i]:X.indptr[i + 1]]
            c = X.data[X.indptr[i]:X.indptr[i + 1]].astype(float)
            L = c.sum()
            a = self._nw[:, w] + self.beta
            out[i] = (np.log(self._m + self.alpha)
                      + (gammaln(a + c) - gammaln(a)).sum(axis=1)
                      - (gammaln(self._n + Vb + L) - gammaln(self._n + Vb)))
        return out - logsumexp(out, axis=1, keepdims=True)

    def predict(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        return self.predict_log_proba(docs).argmax(axis=1)

    def top_terms(self, n: int = 10) -> List[List[str]]:
        """Most probable terms of each cluster: phi_kw = (n_kw + beta) / (n_k + V*beta)."""
        self._check_fitted()
        return [[self.vocabulary_[j] for j in np.argsort(-row)[:n]] for row in self._nw]

    def _check_fitted(self) -> None:
        if self._nw is None:
            raise ValueError("Call fit() first.")
