import itertools

import numpy as np
import pytest

from journey_analyzer import BaselineAligner
from journey_analyzer.alignment import (
    assign_labels,
    pam,
    silhouette_samples_from_distances,
)

TEMPLATES = {
    "buyer":   ["login", "product_view", "add_to_cart", "purchase", "logout"],
    "browser": ["login", "page_view", "click", "click", "page_view", "logout"],
    "bouncer": ["login", "product_view", "product_view", "product_view", "logout"],
}
EVENTS = sorted({e for t in TEMPLATES.values() for e in t})


def _noisy(rng, template, p=0.15):
    out = []
    for e in template:
        r = rng.random()
        if r < p / 3:
            continue                                   # deletion
        if r < 2 * p / 3:
            out.append(str(rng.choice(EVENTS)))        # substitution
            continue
        if r < p:
            out.append(str(rng.choice(EVENTS)))        # insertion (before e)
        out.append(e)
    return out


def _planted(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    seqs, truth = [], []
    for name, t in TEMPLATES.items():
        for _ in range(n_per):
            seqs.append(_noisy(rng, t))
            truth.append(name)
    return seqs, np.array(truth)


def _random_matrix(n=30, seed=1):
    pts = np.random.default_rng(seed).random((n, 2))
    return np.linalg.norm(pts[:, None] - pts[None], axis=2).astype(np.float32)


def _cost(D, medoids):
    return float(D[list(medoids)].min(axis=0).sum())


# ---- PAM on a generic distance matrix ---------------------------------------
def test_basic_output_shapes_and_validity():
    D = _random_matrix()
    medoids, labels, cost, n_swaps, converged = pam(D, 4)
    assert len(set(medoids)) == 4 and all(0 <= m < 30 for m in medoids)
    assert labels.shape == (30,) and set(labels) <= set(range(4))
    assert converged
    # each medoid belongs to its own cluster
    assert all(labels[m] == i for i, m in enumerate(medoids))
    # reported cost equals the sum of distances to the assigned medoid
    assert cost == pytest.approx(sum(D[medoids[labels[j]], j] for j in range(30)), rel=1e-6)


def test_clusters_sorted_by_size():
    D = _random_matrix(60)
    _, labels, *_ = pam(D, 5)
    sizes = np.bincount(labels)
    assert list(sizes) == sorted(sizes, reverse=True)


def test_local_optimum_no_single_swap_improves():
    D = _random_matrix()
    medoids, _, cost, _, converged = pam(D, 3)
    assert converged
    for p in range(3):
        for h in set(range(30)) - set(medoids):
            trial = list(medoids)
            trial[p] = h
            assert _cost(D, trial) >= cost - 1e-6


def test_matches_brute_force_optimum_on_small_problem():
    D = _random_matrix(14, seed=3)
    best = min(_cost(D, c) for c in itertools.combinations(range(14), 3))
    _, _, cost, _, _ = pam(D, 3)
    assert cost == pytest.approx(best, rel=1e-6)


def test_k_equals_one_and_k_equals_n():
    D = _random_matrix(10)
    m, labels, cost, *_ = pam(D, 1)
    assert m[0] == int(np.argmin(D.sum(axis=1)))        # the most central point
    m, labels, cost, *_ = pam(D, 10)
    assert cost == 0 and len(set(labels)) == 10


def test_build_is_deterministic_and_random_init_reaches_a_local_optimum():
    D = _random_matrix(40)
    assert np.array_equal(pam(D, 4)[0], pam(D, 4)[0])
    medoids, _, cost, _, converged = pam(D, 4, init="random", rng=np.random.default_rng(0))
    assert converged
    for p in range(4):                                  # no single swap improves it
        for h in set(range(40)) - set(medoids):
            trial = list(medoids)
            trial[p] = h
            assert _cost(D, trial) >= cost - 1e-6


def test_duplicate_sessions_do_not_break_labels():
    D = np.zeros((6, 6), dtype=np.float32)              # all sessions identical
    medoids, labels, cost, _, _ = pam(D, 3)
    assert cost == 0 and all(labels[m] == i for i, m in enumerate(medoids))


def test_invalid_arguments():
    D = _random_matrix(10)
    for bad_k in (0, 11):
        with pytest.raises(ValueError):
            pam(D, bad_k)
    with pytest.raises(ValueError):
        pam(D, 2, init="nope")
    with pytest.raises(ValueError):
        pam(np.zeros((3, 4)), 2)


# ---- silhouette ---------------------------------------------------------------
def test_silhouette_well_separated_vs_wrong_labels():
    D = _random_matrix(40)
    _, labels, *_ = pam(D, 3)
    good = silhouette_samples_from_distances(D, labels).mean()
    shuffled = np.random.default_rng(0).permutation(labels)
    bad = silhouette_samples_from_distances(D, shuffled).mean()
    assert good > bad and -1 <= bad <= good <= 1


def test_silhouette_matches_sklearn_if_available():
    metrics = pytest.importorskip("sklearn.metrics")
    D = _random_matrix(50)
    _, labels, *_ = pam(D, 4)
    ours = silhouette_samples_from_distances(D, labels)
    ref = metrics.silhouette_samples(D, labels, metric="precomputed")
    assert np.allclose(ours, ref, atol=1e-6)


def test_silhouette_needs_two_clusters():
    with pytest.raises(ValueError):
        silhouette_samples_from_distances(_random_matrix(5), np.zeros(5, dtype=int))


# ---- BaselineAligner integration ----------------------------------------------
def test_recovers_planted_journey_types():
    seqs, truth = _planted()
    al = BaselineAligner()
    al.fit_distance_matrix(seqs)
    res = al.extract_prototypes(3)
    # purity: every found cluster is dominated by one true type
    purity = sum(
        max(np.sum((al.labels_ == c) & (truth == t)) for t in set(truth)) for c in range(3)
    ) / len(seqs)
    assert purity >= 0.95
    assert al.silhouette_score() > 0.4
    assert res["converged"] and len(res["prototypes"]) == 3
    assert al.prototypes == [seqs[i] for i in al.medoids_]


def test_aligner_state_and_errors():
    seqs, _ = _planted(10)
    al = BaselineAligner()
    with pytest.raises(ValueError):
        al.extract_prototypes(2)                         # no matrix yet
    with pytest.raises(ValueError):
        al.silhouette_score()                            # no clustering yet
    al.fit_distance_matrix(seqs)
    al.extract_prototypes(2)
    al.fit_distance_matrix(seqs[:20])                    # a new matrix resets clustering
    assert al.medoids_ is None and al.labels_ is None and al.prototypes is None


def test_random_init_with_restarts_works():
    seqs, _ = _planted(15)
    al = BaselineAligner()
    al.fit_distance_matrix(seqs)
    r_build = al.extract_prototypes(3)
    r_rand = al.extract_prototypes(3, init="random", n_init=5, random_state=0)
    assert r_rand["inertia"] >= r_build["inertia"] * 0.8   # same ballpark
