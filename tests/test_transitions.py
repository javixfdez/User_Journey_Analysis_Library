import itertools

import numpy as np
import pytest
from scipy import sparse

from journey_analyzer import TransitionModel
from journey_analyzer.evaluation import adjusted_rand_index, cluster_purity
from journey_analyzer.transitions import (
    END, START, build_kgram_matrix, build_states, conversion_probability, count_transitions,
    extract_kgrams, l2_normalize_rows, normalize_transitions, pairwise_kgram_distances,
    representative_path, sequence_log_likelihood, spherical_kmeans,
)


# ============================ k-grams ====================================
def test_extract_kgrams_windows_and_short_rule():
    assert extract_kgrams(list("abcd"), 2) == [("a", "b"), ("b", "c"), ("c", "d")]
    assert extract_kgrams(list("ab"), 3) == [("a", "b")]          # shorter than k -> whole sequence
    assert extract_kgrams([], 2) == []
    with pytest.raises(ValueError):
        extract_kgrams(list("ab"), 0)


def test_kgram_matrix_counts_vocab_and_boundaries():
    X, grams = build_kgram_matrix([["a", "b", "b"]], k=2, add_boundaries=False)
    assert dict(zip(grams, X.toarray()[0])) == {("a", "b"): 1.0, ("b", "b"): 1.0}
    X, grams = build_kgram_matrix([["a", "a", "a"]], k=2, add_boundaries=False)
    assert X.toarray()[0].tolist() == [2.0]                        # counts, not presence
    Xb, _ = build_kgram_matrix([["a", "a", "a"]], k=2, add_boundaries=False, binary=True)
    assert Xb.toarray()[0].tolist() == [1.0]
    _, grams = build_kgram_matrix([["a"]], k=2, add_boundaries=True)
    assert (START, "a") in grams and ("a", END) in grams


def test_min_df_prunes_rare_grams_and_keeps_every_session_nonempty_by_default():
    seqs = [["a", "b"], ["a", "b"], ["c", "d"]]
    X, grams = build_kgram_matrix(seqs, k=2, min_df=2)
    assert all(g in [(START, "a"), ("a", "b"), ("b", END)] for g in grams)
    X1, _ = build_kgram_matrix(seqs, k=2)
    assert (np.asarray(X1.sum(axis=1)).ravel() > 0).all()


def test_l2_normalize_rows():
    X = sparse.csr_matrix(np.array([[3.0, 4.0], [0.0, 0.0]]))
    Xn = l2_normalize_rows(X).toarray()
    assert np.allclose(Xn[0], [0.6, 0.8]) and np.allclose(Xn[1], 0)


SEQS = [["login", "browse", "buy"], ["login", "browse", "buy"], ["login", "browse", "browse", "buy", "logout"],
        ["click", "click", "logout"], ["logout"], ["click", "login"]]


@pytest.mark.parametrize("metric", ["cosine", "angular", "euclidean"])
def test_distance_matrix_properties_and_brute_force(metric):
    X, _ = build_kgram_matrix(SEQS, k=2)
    D = pairwise_kgram_distances(X, metric)
    dense = X.toarray()
    n = len(SEQS)
    assert D.shape == (n, n) and np.allclose(D, D.T) and np.all(np.diag(D) == 0) and D.min() >= 0
    assert D[0, 1] == 0                                              # identical sessions
    for i, j in itertools.combinations(range(n), 2):
        a, b = dense[i], dense[j]
        cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
        expected = {"cosine": 1 - cos, "angular": np.arccos(np.clip(cos, 0, 1)) / (np.pi / 2),
                    "euclidean": np.linalg.norm(a - b)}[metric]
        assert D[i, j] == pytest.approx(expected, abs=1e-5)


def test_cosine_ignores_length_but_euclidean_does_not():
    # unigrams: counts [2, 2] vs [20, 20] -> same direction, very different size
    X, _ = build_kgram_matrix([["a", "b"] * 2, ["a", "b"] * 20], k=1, add_boundaries=False)
    assert pairwise_kgram_distances(X, "cosine")[0, 1] == 0
    assert pairwise_kgram_distances(X, "euclidean")[0, 1] > 10


def test_angular_satisfies_triangle_inequality_on_random_data():
    rng = np.random.default_rng(0)
    seqs = [list(rng.choice(list("abc"), rng.integers(2, 8))) for _ in range(25)]
    X, _ = build_kgram_matrix(seqs, k=2)
    D = pairwise_kgram_distances(X, "angular").astype(float)
    for i, j, l in itertools.product(range(25), repeat=3):
        assert D[i, l] <= D[i, j] + D[j, l] + 1e-5


def test_block_computation_matches_single_block(monkeypatch):
    import journey_analyzer.transitions.kgrams as kg
    X, _ = build_kgram_matrix(SEQS * 5, k=2)                              # 30 sessions
    full = pairwise_kgram_distances(X, "cosine")
    monkeypatch.setattr(kg, "_BLOCK_ELEMENTS", 40)                         # forces 1-row blocks
    blocked = kg.pairwise_kgram_distances(X, "cosine")
    assert np.allclose(blocked, full, atol=1e-7)


def test_memory_guard_and_bad_metric():
    X, _ = build_kgram_matrix(SEQS, k=2)
    with pytest.raises(MemoryError):
        pairwise_kgram_distances(X, max_matrix_gb=1e-12)
    with pytest.raises(ValueError):
        pairwise_kgram_distances(X, "nope")


# ======================== spherical k-means ===============================
def _planted(n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    tpl = [["login", "product_view", "add_to_cart", "purchase", "logout"],
           ["login", "page_view", "click", "click", "page_view", "logout"],
           ["login", "product_view", "product_view", "product_view", "logout"]]
    seqs, truth = [], []
    for t_id, t in enumerate(tpl):
        for _ in range(n_per):
            s = [e for e in t if rng.random() > 0.1]
            if rng.random() < 0.3:
                s.insert(rng.integers(0, len(s) + 1), str(rng.choice(sorted({e for x in tpl for e in x}))))
            seqs.append(s or t[:2]); truth.append(t_id)
    return seqs, np.array(truth)


def test_spherical_kmeans_recovers_planted_groups_and_is_seeded():
    seqs, truth = _planted()
    X, _ = build_kgram_matrix(seqs, k=2)
    labels, centers, inertia, n_iter = spherical_kmeans(X, 3, rng=np.random.default_rng(0))
    assert cluster_purity(labels, truth) >= 0.9
    assert np.allclose(np.linalg.norm(centers, axis=1), 1.0)
    sizes = np.bincount(labels)
    assert list(sizes) == sorted(sizes, reverse=True)
    again = spherical_kmeans(X, 3, rng=np.random.default_rng(0))[0]
    assert np.array_equal(labels, again)
    with pytest.raises(ValueError):
        spherical_kmeans(X, 0)


# ============================== Markov ====================================
S2 = [["a", "b"], ["a", "b"], ["a", "c"], ["b"]]


def test_count_and_normalize_known_example():
    states = build_states(S2)
    assert states == [START, "a", "b", "c", END]
    C = count_transitions(S2, states)
    i = {s: k for k, s in enumerate(states)}
    assert C[i[START], i["a"]] == 3 and C[i[START], i["b"]] == 1
    assert C[i["a"], i["b"]] == 2 and C[i["a"], i["c"]] == 1
    P = normalize_transitions(C, states)
    assert P[i[START], i["a"]] == pytest.approx(0.75)
    assert P[i["a"], i["b"]] == pytest.approx(2 / 3)
    assert np.allclose(P.sum(axis=1)[:-1], 1.0) and P[-1].sum() == 0     # <END> is absorbing
    with pytest.raises(KeyError):
        count_transitions([["zzz"]], states)


def test_smoothing_keeps_rows_normalised_and_forbids_illegal_transitions():
    states = build_states(S2)
    P = normalize_transitions(count_transitions(S2, states), states, smoothing=0.5)
    i = {s: k for k, s in enumerate(states)}
    assert np.allclose(P.sum(axis=1)[:-1], 1.0)
    assert P[i["c"], i["a"]] > 0                                          # unseen but allowed
    assert P[i["a"], i[START]] == 0 and P[i[START], i[END]] == 0


def test_log_likelihood_by_hand():
    states = build_states(S2)
    P = normalize_transitions(count_transitions(S2, states), states)
    expected = np.log(0.75) + np.log(2 / 3) + np.log(P[states.index("b"), states.index(END)])
    assert sequence_log_likelihood(["a", "b"], P, states) == pytest.approx(expected)
    assert sequence_log_likelihood(["c", "c"], P, states) == float("-inf")


def _brute_force_best(P, states, length):
    events = [s for s in states if s not in (START, END)]
    idx = {s: i for i, s in enumerate(states)}
    best, arg = -np.inf, None
    for path in itertools.product(events, repeat=length):
        ll = sequence_log_likelihood(list(path), P, states)
        if ll > best:
            best, arg = ll, list(path)
    return arg, best


def test_representative_path_matches_brute_force_and_respects_min_len():
    rng = np.random.default_rng(1)
    seqs = [list(rng.choice(list("abc"), rng.integers(1, 6), p=[0.5, 0.3, 0.2])) for _ in range(60)]
    states = build_states(seqs)
    P = normalize_transitions(count_transitions(seqs, states), states, smoothing=0.01)
    for length in (1, 2, 3, 4):
        path, ll = representative_path(P, states, min_len=length, max_len=length)
        ref_path, ref_ll = _brute_force_best(P, states, length)
        assert len(path) == length and ll == pytest.approx(ref_ll)
    path, _ = representative_path(P, states, min_len=3, max_len=6)
    assert 3 <= len(path) <= 6
    short, _ = representative_path(P, states, min_len=1, max_len=6)
    assert len(short) == 1                                                # the short-sequence bias
    with pytest.raises(ValueError):
        representative_path(P, states, 0, 3)


def test_conversion_probability_closed_forms():
    # START -> a (1); a -> buy 0.5, a -> END 0.5   =>  P(reach buy) = 0.5
    seqs = [["a", "buy"], ["a"]]
    states = build_states(seqs)
    P = normalize_transitions(count_transitions(seqs, states), states)
    h = conversion_probability(P, states, "buy")
    assert h["a"] == pytest.approx(0.5) and h[START] == pytest.approx(0.5)
    assert h["buy"] == 1.0 and h[END] == 0.0
    # with a self loop: a -> a 0.5, a -> buy 0.25, a -> END 0.25  =>  h = 0.25/(1-0.5) = 0.5
    P2 = np.zeros((4, 4)); st = [START, "a", "buy", END]
    P2[0, 1] = 1; P2[1, 1] = 0.5; P2[1, 2] = 0.25; P2[1, 3] = 0.25; P2[2, 3] = 1
    assert conversion_probability(P2, st, "buy")["a"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        conversion_probability(P, states, "nope")


# ======================= evaluation metrics ===============================
def test_ari_and_purity():
    a = [0, 0, 1, 1, 2, 2]
    assert adjusted_rand_index(a, [5, 5, 3, 3, 9, 9]) == pytest.approx(1.0)   # relabelling does not matter
    assert adjusted_rand_index(a, [0, 1, 0, 1, 0, 1]) < 0.1
    assert cluster_purity(a, [0, 0, 1, 1, 2, 1]) == pytest.approx(5 / 6)
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    x, y = rng.integers(0, 4, 200), rng.integers(0, 3, 200)
    assert adjusted_rand_index(x, y) == pytest.approx(metrics.adjusted_rand_score(x, y))


# ======================== TransitionModel =================================
def test_model_full_pipeline_on_planted_data():
    seqs, truth = _planted()
    tm = TransitionModel(k=2).fit_kgrams(seqs)
    res = tm.cluster(3, "kmedoids")
    assert cluster_purity(tm.labels_, truth) >= 0.9 and res["converged"]
    assert tm.silhouette_score() > 0.3 and len(tm.prototypes) == 3
    tm.fit_markov_chain()
    for c in (None, 0, 1, 2):
        P = tm.transition_matrix(c)
        row = P.sum(axis=1)[:-1]                                           # events absent from a cluster have an all-zero row
        assert np.all(np.isclose(row, 1.0) | np.isclose(row, 0.0))
        assert sum(tm.entry_probabilities(c).values()) == pytest.approx(1.0)
        assert all(0 <= v <= 1 for v in tm.dropoff_rates(c).values())
    # model-implied P(session reaches 'purchase') should track the empirical share per cluster
    for c in range(3):
        members = [x for x, l in zip(seqs, tm.labels_) if l == c]
        empirical = np.mean(["purchase" in x for x in members])
        assert tm.conversion_probability("purchase", c)[START] == pytest.approx(empirical, abs=0.1)
    path, ll = tm.get_representative_path(0)
    assert path[0] in tm.states and np.isfinite(ll)
    # relaxing the length bound searches a superset of paths, so the best score can only improve
    relaxed, ll_relaxed = tm.get_representative_path(0, min_len=1)
    assert relaxed and ll_relaxed >= ll - 1e-9


def test_model_kmeans_route_needs_no_distance_matrix():
    seqs, truth = _planted()
    tm = TransitionModel(k=2).fit_kgrams(seqs)
    tm.cluster(3, "kmeans", random_state=0)
    assert tm.distance_matrix is None and tm.medoids_ is None and tm.prototypes is None
    assert cluster_purity(tm.labels_, truth) >= 0.9
    tm.fit_markov_chain()
    assert tm.get_representative_path(1)[0]
    with pytest.raises(ValueError):
        tm.plot_clusters()


def test_silhouette_after_kmeans_keeps_the_clustering():
    """Regression: silhouette_score() built the distance matrix lazily, and fit_distance_matrix()
    wiped labels_, so cluster(method="kmeans") + silhouette_score() raised a TypeError."""
    seqs = [["a", "b", "c"]] * 15 + [["x", "y", "z"]] * 15
    tm = TransitionModel(k=2).fit_kgrams(seqs)
    tm.cluster(2, "kmeans", random_state=0)
    before = tm.labels_.copy()
    assert tm.silhouette_score() > 0.9
    assert np.array_equal(tm.labels_, before) and tm.cluster_method_ == "kmeans"
    assert tm.distance_matrix is not None


def test_model_errors_and_resets():
    tm = TransitionModel()
    for call in (lambda: tm.cluster(2), lambda: tm.fit_distance_matrix(), lambda: tm.fit_markov_chain(),
                 lambda: tm.dropoff_rates(), lambda: tm.silhouette_score()):
        with pytest.raises(ValueError):
            call()
    seqs, _ = _planted(10)
    tm.fit_kgrams(seqs)
    with pytest.raises(ValueError):
        tm.dropoff_rates()                                             # chain not fitted yet
    tm.cluster(2)
    tm.fit_markov_chain()
    with pytest.raises(ValueError):
        tm.transition_matrix(cluster=7)
    tm.cluster(3)                                                      # new clustering invalidates chains
    with pytest.raises(ValueError):
        tm.transition_matrix(cluster=0)
    with pytest.raises(ValueError):
        tm.cluster(2, "nope")
    with pytest.raises(ValueError):
        TransitionModel(k=0)
    with pytest.raises(ValueError):
        TransitionModel(metric="x")


def test_model_without_clusters_has_a_global_chain_only():
    seqs, _ = _planted(10)
    tm = TransitionModel(k=2).fit_kgrams(seqs).fit_markov_chain()
    assert tm.get_representative_path()[0]
    assert abs(sum(tm.entry_probabilities().values()) - 1) < 1e-9
    assert tm.log_likelihood(seqs[0]) == float("-inf") or np.isfinite(tm.log_likelihood(seqs[0]))


def test_plots_run():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from journey_analyzer.visualization import plot_dropoff_rates, plot_transition_matrix
    seqs, _ = _planted(15)
    tm = TransitionModel(k=2).fit_kgrams(seqs)
    tm.cluster(3)
    tm.fit_markov_chain()
    ax = plot_transition_matrix(tm.transition_matrix(0), tm.states, title="t")
    plt.close(ax.figure)
    # a fixed layout gives every cluster the same rows/columns, so panels are comparable
    rows = [s for s in tm.states if s != END]
    cols = [s for s in tm.states if s != START]
    shapes = set()
    for c in range(3):
        ax = plot_transition_matrix(tm.transition_matrix(c), tm.states, rows=rows, cols=cols)
        shapes.add(ax.images[0].get_array().shape)
        plt.close(ax.figure)
    assert shapes == {(len(rows), len(cols))}
    ax = plot_dropoff_rates({"all": tm.dropoff_rates(), "c0": tm.dropoff_rates(0)}, title="d")
    plt.close(ax.figure)
    fig, _ = tm.plot_clusters()
    plt.close(fig)
