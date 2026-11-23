"""
evaluation.py - Compare two clusterings (used to validate every stage).

WHY THESE METRICS?
------------------
Silhouette judges ONE clustering against its own distances. To judge whether a
method recovers known groups, or whether two methods agree, we compare labels:

* Adjusted Rand Index (Hubert & Arabie, 1985): fraction of session PAIRS on
  which two labelings agree (both together or both apart), corrected for the
  agreement expected by chance. 1 = identical partitions, about 0 = random,
  negative = worse than random. Invariant to how clusters are numbered.
* Purity: share of sessions that fall in the majority true class of their
  cluster. Easy to read, but it rises trivially with more clusters, so report it
  together with ARI.
"""

import numpy as np

__all__ = ["adjusted_rand_index", "cluster_purity"]


def _contingency(a, b) -> np.ndarray:
    """cont[i, j] = number of sessions with label i in `a` and label j in `b`."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Both labelings must have the same length.")
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    cont = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(cont, (ia, ib), 1)
    return cont


def adjusted_rand_index(labels_a, labels_b) -> float:
    cont = _contingency(labels_a, labels_b)
    comb2 = lambda x: x * (x - 1) / 2.0
    sum_ij = comb2(cont).sum()
    sum_a, sum_b = comb2(cont.sum(axis=1)).sum(), comb2(cont.sum(axis=0)).sum()
    expected = sum_a * sum_b / comb2(cont.sum())
    maximum = (sum_a + sum_b) / 2.0
    return 1.0 if maximum == expected else float((sum_ij - expected) / (maximum - expected))


def cluster_purity(found_labels, true_labels) -> float:
    cont = _contingency(found_labels, true_labels)
    return float(cont.max(axis=1).sum() / cont.sum())
