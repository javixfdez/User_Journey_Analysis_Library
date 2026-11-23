"""
lda.py - Latent Dirichlet Allocation as the TOPIC-MODEL BASELINE of Stage 3.

WHY LDA IS HERE EVEN THOUGH IT IS NOT THE BEST SHORT-TEXT MODEL
---------------------------------------------------------------
* The project plan names it (Bar-El et al.: best coherence/separability balance
  among their candidates), so it is the reference every other semantic method is
  measured against.
* The survey of Qiang et al. (2022) reports short-text models (GSDMM, BTM)
  beating it on short documents. Having both in the library lets this project
  CHECK that claim on session data instead of quoting it.

HOW A TOPIC MODEL BECOMES A CLUSTERING
--------------------------------------
LDA gives every session a MIXTURE over topics (theta, rows sum to 1). Two ways to
get one label per session:
    "argmax"  the dominant topic. Simple, needs no extra algorithm. Default.
    "kmeans"  spherical k-means on theta. Treats the mixture as an embedding, which
              is the "LDA topic mixture + k-means" pipeline of the evaluation protocol.
The mixture itself is also exposed (`doc_topic_`): it is what makes LDA
interpretable, "this session is 70% topic 2 and 30% topic 5".

IMPLEMENTATION: scikit-learn (variational Bayes), imported lazily so the library
imports without it. Writing our own LDA would add risk and no scientific value.
"""

from typing import List, Optional, Sequence

import numpy as np

from .terms import build_term_matrix

__all__ = ["LDATopicModel"]


class LDATopicModel:
    """
    Parameters
    ----------
    n_topics     : number of topics K (with argmax assignment, also the max number of clusters).
    doc_topic_prior, topic_word_prior : Dirichlet priors (None -> scikit-learn's 1/K).
                   Small priors give sparse mixtures, which suits short documents.
    max_iter     : passes over the corpus (batch variational Bayes).
    min_df       : drop terms found in fewer than this many sessions.
    random_state : seed (the optimisation is stochastic in its initialisation).
    """

    def __init__(self, n_topics: int = 10, doc_topic_prior: Optional[float] = None,
                 topic_word_prior: Optional[float] = None, max_iter: int = 30,
                 min_df: int = 1, random_state: Optional[int] = None):
        if n_topics < 1:
            raise ValueError("n_topics must be >= 1.")
        self.n_topics, self.doc_topic_prior, self.topic_word_prior = n_topics, doc_topic_prior, topic_word_prior
        self.max_iter, self.min_df, self.random_state = max_iter, min_df, random_state
        self.doc_topic_: Optional[np.ndarray] = None       # (N, K) mixtures
        self.vocabulary_: List[str] = []
        self._lda = None

    def fit(self, docs: Sequence[Sequence[str]]) -> "LDATopicModel":
        """Fit on `docs` (lists of term strings, see terms.make_terms)."""
        X, vocab = build_term_matrix(docs, min_df=self.min_df)
        return self.fit_matrix(X, vocab)

    def fit_matrix(self, X, vocabulary: Sequence[str]) -> "LDATopicModel":
        """Same as fit(), from a document x term COUNT matrix (shares the vocabulary with GSDMM)."""
        try:
            from sklearn.decomposition import LatentDirichletAllocation
        except ImportError as e:                           # pragma: no cover
            raise ImportError("LDATopicModel needs scikit-learn: pip install scikit-learn") from e
        self.vocabulary_ = list(vocabulary)
        self._lda = LatentDirichletAllocation(
            n_components=self.n_topics, doc_topic_prior=self.doc_topic_prior,
            topic_word_prior=self.topic_word_prior, max_iter=self.max_iter,
            learning_method="batch", random_state=self.random_state)
        self.doc_topic_ = self._lda.fit_transform(X)
        return self

    def transform(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        """Topic mixtures of NEW sessions (terms unseen in training are ignored)."""
        self._check_fitted()
        X, _ = build_term_matrix(docs, vocabulary=self.vocabulary_)
        return self._lda.transform(X)

    @property
    def labels_(self) -> np.ndarray:
        """Dominant topic of every training session."""
        self._check_fitted()
        return self.doc_topic_.argmax(axis=1)

    def top_terms(self, n: int = 10) -> List[List[str]]:
        """Highest-weight terms of each topic."""
        self._check_fitted()
        return [[self.vocabulary_[j] for j in np.argsort(-row)[:n]] for row in self._lda.components_]

    def _check_fitted(self) -> None:
        if self._lda is None:
            raise ValueError("Call fit() first.")
