"""
terms.py - Sessions as "documents of terms" for the topic models (LDA, GSDMM).

WHY ORDER HAS TO BE PUT IN BY HAND
----------------------------------
LDA and GSDMM are bag-of-words models: they only see HOW OFTEN each term occurs
in a document, never where. If the terms are single events, a session that goes
`page_view -> purchase` and one that goes `purchase -> page_view` are the same
document. The scratch experiment behind this module showed exactly that: ARI 0.00
for event unigrams on the order-only control, 1.00 once the terms were event
BIGRAMS.

So the representation is a choice, with three modes:

    "unigram"  events only                    -> order-blind, smallest vocabulary
    "bigram"   adjacent pairs "a > b"         -> sees local order (like k-grams, k=2)
    "both"     events and pairs               -> default

Bigram terms with <START>/<END> boundaries are the SAME objects as the 2-grams
of Stage 2. Hence "LDA on bigrams" and "k-gram + k-means" differ only in the
model on top (topics vs. a spherical k-means), which keeps the comparison clean.
Biterm Topic Model (BTM) is built on the same idea (unordered word pairs) and is
the short-text model the survey of Qiang et al. recommends next to GSDMM.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

from ..transitions.tokens import END, START

__all__ = ["make_terms", "build_term_matrix", "TERM_MODES", "BIGRAM_SEP"]

TERM_MODES = ("unigram", "bigram", "both")
BIGRAM_SEP = " > "


def make_terms(tokens: Sequence[str], mode: str = "both", boundaries: bool = True) -> List[str]:
    """
    Terms of one session.

    boundaries : add <START>/<END> before forming bigrams, so that how sessions
                 BEGIN and END (entry and drop-off) are terms like any other.
                 Unigram terms never include the boundaries: they would occur in
                 every session and carry no information.
    """
    if mode not in TERM_MODES:
        raise ValueError(f"mode must be one of {TERM_MODES}.")
    toks = [str(t) for t in tokens]
    terms: List[str] = []
    if mode in ("unigram", "both"):
        terms.extend(toks)
    if mode in ("bigram", "both"):
        padded = [START, *toks, END] if boundaries else toks
        terms.extend(f"{a}{BIGRAM_SEP}{b}" for a, b in zip(padded, padded[1:]))
    return terms


def build_term_matrix(docs: Sequence[Sequence[str]], vocabulary: Optional[Sequence[str]] = None,
                      min_df: int = 1) -> Tuple[sparse.csr_matrix, List[str]]:
    """
    Sparse document x term COUNT matrix.

    vocabulary : fixed term list (used to encode NEW sessions with the vocabulary
                 of the fitted model; unseen terms are dropped).
    min_df     : drop terms found in fewer than `min_df` documents (only when the
                 vocabulary is built here). Rare terms add noise to topic models.
    """
    if vocabulary is None:
        index: Dict[str, int] = {}
        for d in docs:
            for t in d:
                index.setdefault(t, len(index))
    else:
        index = {t: i for i, t in enumerate(vocabulary)}
    grow = vocabulary is None

    rows, cols, vals = [], [], []
    for i, d in enumerate(docs):
        counts: Dict[int, int] = {}
        for t in d:
            j = index.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        rows.extend([i] * len(counts)); cols.extend(counts.keys()); vals.extend(counts.values())
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(index)), dtype=np.float64)
    terms = list(index)

    if grow and min_df > 1:
        df = np.asarray((X > 0).sum(axis=0)).ravel()
        keep = np.flatnonzero(df >= min_df)
        X, terms = X[:, keep], [terms[j] for j in keep]
    return sparse.csr_matrix(X), terms
