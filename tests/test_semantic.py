"""
Tests for Stage 3 (semantic/). Every test states the property it protects.

Nothing here needs the internet, a GPU or PyTorch: SBERT is replaced by
HashingEncoder, a deterministic bag-of-words stand-in. What these tests CANNOT
show is the quality of real SBERT embeddings; that is checked in the notebooks.
"""

import sys

import numpy as np
import pytest
from scipy.special import gammaln

from journey_analyzer import SemanticNLPModel, TransitionModel
from journey_analyzer.evaluation import adjusted_rand_index
from journey_analyzer.semantic import (GSDMM, HashingEncoder, LDATopicModel, SBERTEncoder,
                                       SessionRenderer, build_term_matrix, ctfidf_scores,
                                       distinctive_terms, exemplar_indices, make_terms)

sklearn = pytest.importorskip("sklearn")


# ---------------------------------------------------------------------------
# Fixtures: small labelled controls
# ---------------------------------------------------------------------------
def composition_control(n=40):
    """Two groups with disjoint events: any method must separate them."""
    seqs = [["login", "page_view", "click"]] * n + [["product_view", "add_to_cart", "purchase"]] * n
    return seqs, np.array([0] * n + [1] * n)


def order_control(n=60, seed=0):
    """Same events, opposite order: only order-aware representations can separate them."""
    rng = np.random.default_rng(seed)
    fwd = ["page_view", "click", "product_view", "add_to_cart", "purchase"]
    seqs, y = [], []
    for label, t in enumerate((fwd, fwd[::-1])):
        for _ in range(n):
            s = [e for e in t if rng.random() > 0.1]
            seqs.append(s if len(s) >= 2 else list(t)); y.append(label)
    return seqs, np.array(y)


# ---------------------------------------------------------------------------
# rendering.py
# ---------------------------------------------------------------------------
class TestRendering:
    S = ["view_electronics.smartphone", "view_electronics.smartphone",
         "cart_electronics.smartphone", "purchase_electronics.smartphone"]

    def test_natural_text(self):
        assert SessionRenderer("natural").render(self.S) == (
            "viewed electronics smartphone, viewed electronics smartphone, "
            "added electronics smartphone to cart, purchased electronics smartphone")

    def test_symbols_mode_keeps_exact_tokens(self):
        assert SessionRenderer("symbols").render(self.S[:2]) == "view_electronics.smartphone view_electronics.smartphone"

    def test_words_mode_drops_punctuation(self):
        assert SessionRenderer("words").render(["view_personal_care.hair_care"]) == "view personal care hair care"

    def test_rle_collapses_runs_but_not_separated_repeats(self):
        r = SessionRenderer("natural", rle=True)
        assert r.render(self.S) == ("viewed electronics smartphone x2, added electronics smartphone to cart, "
                                    "purchased electronics smartphone")
        assert r.render(["view", "cart", "view"]) == "viewed item, added item to cart, viewed item"

    def test_window_keeps_the_last_events_by_default(self):
        assert SessionRenderer("symbols", window=2).render(self.S) == \
               "cart_electronics.smartphone purchase_electronics.smartphone"
        assert SessionRenderer("symbols", window=1, window_side="first").render(self.S) == \
               "view_electronics.smartphone"

    def test_window_longer_than_session_is_harmless(self):
        assert SessionRenderer("symbols", window=99).tokens(self.S) == [s for s in self.S]

    def test_category_depth_coarsens_text_and_tokens(self):
        r = SessionRenderer("natural", category_depth=1)
        assert r.render(["view_electronics.audio.headphone"]) == "viewed electronics"
        assert r.tokens(["view_electronics.audio.headphone", "login"]) == ["view_electronics", "login"]

    def test_numeric_or_missing_category_becomes_unknown_item(self):
        r = SessionRenderer("natural")
        assert r.render(["view_2053013555631882655", "transaction", "addtocart"]) == \
               "viewed item, purchased item, added item to cart"

    def test_tokens_without_a_known_action_are_plain_words(self):
        assert SessionRenderer("natural").render(["login", "page_view", "add_to_cart"]) == \
               "login, page view, added item to cart"

    def test_empty_session_and_validation(self):
        assert SessionRenderer().render([]) == ""
        for bad in (dict(mode="x"), dict(window=0), dict(window_side="mid"), dict(category_depth=0)):
            with pytest.raises(ValueError):
                SessionRenderer(**bad)


# ---------------------------------------------------------------------------
# terms.py
# ---------------------------------------------------------------------------
class TestTerms:
    def test_modes(self):
        assert make_terms(["a", "b"], "unigram") == ["a", "b"]
        assert make_terms(["a", "b"], "bigram", boundaries=False) == ["a > b"]
        assert make_terms(["a", "b"], "bigram") == ["<START> > a", "a > b", "b > <END>"]
        both = make_terms(["a", "b"], "both")
        assert both[:2] == ["a", "b"] and len(both) == 5

    def test_unigrams_never_contain_boundaries(self):
        assert all("<" not in t for t in make_terms(["a", "b"], "unigram"))

    def test_bigrams_are_the_2grams_of_stage_2(self):
        from journey_analyzer.transitions import build_kgram_matrix
        seqs = [["a", "b", "b"], ["b", "a"]]
        _, grams = build_kgram_matrix(seqs, 2)
        want = {" > ".join(g) for g in grams}
        got = {t for s in seqs for t in make_terms(s, "bigram")}
        assert got == want

    def test_term_matrix_counts_and_min_df(self):
        X, vocab = build_term_matrix([["a", "a", "b"], ["b"], ["c"]])
        assert X.shape == (3, 3) and X[0, vocab.index("a")] == 2
        X2, vocab2 = build_term_matrix([["a", "a", "b"], ["b"], ["c"]], min_df=2)
        assert vocab2 == ["b"] and X2.shape == (3, 1)

    def test_fixed_vocabulary_drops_unseen_terms(self):
        X, _ = build_term_matrix([["a", "zzz"]], vocabulary=["a", "b"])
        assert X.shape == (1, 2) and X.toarray().tolist() == [[1.0, 0.0]]


# ---------------------------------------------------------------------------
# gsdmm.py
# ---------------------------------------------------------------------------
class TestGSDMM:
    def test_gammaln_form_equals_the_explicit_product(self):
        """The implementation shortcut prod_{j<c}(x+j) = Gamma(x+c)/Gamma(x) must be exact."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            x, c = rng.uniform(0.1, 30), int(rng.integers(1, 7))
            explicit = np.sum(np.log(x + np.arange(c)))
            assert gammaln(x + c) - gammaln(x) == pytest.approx(explicit)

    def test_recovers_disjoint_groups_and_orders_clusters_by_size(self):
        seqs, y = composition_control()
        seqs = seqs + [["login", "page_view", "click"]] * 10          # make group 0 larger
        y = np.append(y, [0] * 10)
        g = GSDMM(n_clusters=6, random_state=0).fit([make_terms(s) for s in seqs])
        assert adjusted_rand_index(g.labels_, y) == pytest.approx(1.0)
        assert g.n_clusters_found_ == 2
        assert list(g.cluster_sizes_) == sorted(g.cluster_sizes_, reverse=True)
        assert g.labels_.min() == 0 and g.labels_.max() == g.n_clusters_found_ - 1

    def test_n_clusters_is_an_upper_bound(self):
        seqs, _ = composition_control()
        g = GSDMM(n_clusters=15, random_state=1).fit([make_terms(s) for s in seqs])
        assert g.n_clusters_found_ <= 15 and g.n_clusters_found_ < 15

    def test_same_seed_same_result_different_seed_may_differ(self):
        docs = [make_terms(s) for s in order_control()[0]]
        a = GSDMM(8, random_state=3).fit(docs).labels_
        b = GSDMM(8, random_state=3).fit(docs).labels_
        assert np.array_equal(a, b)

    def test_bigram_terms_see_order_unigram_terms_do_not(self):
        """The bag-of-words limitation that motivates terms.py (ARI 0 vs ~1 in the scratch experiment)."""
        seqs, y = order_control()
        uni = GSDMM(6, random_state=0).fit([make_terms(s, "unigram") for s in seqs])
        bi = GSDMM(6, random_state=0).fit([make_terms(s, "bigram") for s in seqs])
        ari_uni, ari_bi = adjusted_rand_index(uni.labels_, y), adjusted_rand_index(bi.labels_, y)
        # The property is "bigrams see order", not "GSDMM is perfect": it over-fragments (it
        # finds more clusters than the 2 true groups), which caps ARI below 1.
        assert ari_uni < 0.2 and ari_bi > 0.5 and ari_bi - ari_uni > 0.4

    def test_predict_matches_training_labels_and_ignores_unseen_terms(self):
        seqs, y = composition_control()
        docs = [make_terms(s) for s in seqs]
        g = GSDMM(6, random_state=0).fit(docs)
        assert (g.predict(docs) == g.labels_).mean() > 0.97
        probs = np.exp(g.predict_log_proba([["login", "never_seen_before"]]))
        assert probs.shape == (1, g.n_clusters_found_) and probs.sum() == pytest.approx(1.0)
        # a session made only of unseen terms is assigned by the prior (no crash)
        assert g.predict([["unseen_1", "unseen_2"]]).shape == (1,)

    def test_top_terms_describe_each_cluster(self):
        seqs, y = composition_control()
        g = GSDMM(4, random_state=0).fit([make_terms(s, "unigram") for s in seqs])
        tops = [set(t[:3]) for t in g.top_terms(3)]
        assert {"login", "page_view", "click"} in tops and {"product_view", "add_to_cart", "purchase"} in tops

    def test_history_is_recorded_and_validation(self):
        g = GSDMM(5, n_iter=7, tol=0.0, random_state=0).fit([make_terms(s) for s in composition_control()[0]])
        assert len(g.history_["moved"]) <= 7 and len(g.history_["n_clusters"]) == len(g.history_["moved"])
        with pytest.raises(ValueError):
            GSDMM(0)
        with pytest.raises(ValueError):
            GSDMM(3, alpha=0)
        with pytest.raises(ValueError):
            GSDMM(3).predict([["a"]])


# ---------------------------------------------------------------------------
# lda.py
# ---------------------------------------------------------------------------
class TestLDA:
    def test_mixtures_are_distributions_and_dominant_topic_separates_groups(self):
        seqs, y = composition_control()
        lda = LDATopicModel(2, random_state=0).fit([make_terms(s) for s in seqs])
        assert lda.doc_topic_.shape == (len(seqs), 2)
        assert np.allclose(lda.doc_topic_.sum(axis=1), 1.0)
        assert adjusted_rand_index(lda.labels_, y) == pytest.approx(1.0)

    def test_transform_new_sessions_and_top_terms(self):
        seqs, y = composition_control()
        lda = LDATopicModel(2, random_state=0).fit([make_terms(s, "unigram") for s in seqs])
        theta = lda.transform([["login", "click"], ["purchase", "unseen"]])
        assert theta.shape == (2, 2) and theta.argmax(axis=1)[0] != theta.argmax(axis=1)[1]
        assert len(lda.top_terms(3)) == 2

    def test_must_be_fitted(self):
        with pytest.raises(ValueError):
            LDATopicModel(2).transform([["a"]])


# ---------------------------------------------------------------------------
# sbert.py
# ---------------------------------------------------------------------------
class TestEncoders:
    def test_hashing_encoder_is_deterministic_unit_length_and_lexical(self):
        e1, e2 = HashingEncoder(64), HashingEncoder(64)
        a = e1.encode(["viewed phone", "viewed phone", "purchased laptop"])
        assert np.allclose(a, e2.encode(["viewed phone", "viewed phone", "purchased laptop"]))
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
        assert np.allclose(a[0], a[1])
        assert a[0] @ a[2] < 0.5                                   # no shared words -> not similar
        assert HashingEncoder(64).encode(["a b"]).dtype == np.float32

    def test_hashing_encoder_cannot_bridge_synonyms_or_order(self):
        e = HashingEncoder(256)
        v = e.encode(["login", "sign in", "a b c", "c b a"])
        assert abs(v[0] @ v[1]) < 0.5                              # synonyms are unrelated
        assert v[2] @ v[3] == pytest.approx(1.0)                   # order is invisible

    def test_sbert_encoder_dedups_and_restores_order(self):
        calls = []

        class Spy(HashingEncoder):
            def encode(self, texts, **kw):
                calls.append(list(texts)); return super().encode(texts, **kw)

        enc = SBERTEncoder(encoder=Spy(32))
        texts = ["a b", "c d", "a b", "a b", "c d"]
        emb = enc.encode(texts)
        assert calls[0] == ["a b", "c d"]                          # each distinct text encoded once
        assert enc.n_texts_ == 5 and enc.n_unique_ == 2
        assert np.allclose(emb[0], emb[2]) and np.allclose(emb[1], emb[4]) and not np.allclose(emb[0], emb[1])

        enc2 = SBERTEncoder(encoder=Spy(32), dedup=False)
        enc2.encode(texts)
        assert len(calls[-1]) == 5 and enc2.n_unique_ == 5          # uncached cost can be measured

    def test_importing_the_library_does_not_import_torch(self):
        assert "torch" not in sys.modules and "sentence_transformers" not in sys.modules
        SBERTEncoder()                                              # constructing must not load the model

    def test_missing_sentence_transformers_gives_a_clear_error(self):
        pytest.importorskip("sklearn")
        try:
            import sentence_transformers  # noqa: F401
            pytest.skip("sentence-transformers is installed")
        except ImportError:
            with pytest.raises(ImportError, match="sentence-transformers"):
                SBERTEncoder().encode(["a"])

    def test_truncation_report_fallback_counts_words(self):
        rep = SBERTEncoder(encoder=HashingEncoder()).truncation_report(["a b c", "a " * 300], max_tokens=256)
        assert rep["share_over_limit"] == 0.5 and rep["max"] == 300 and "lower bound" in rep["unit"]


# ---------------------------------------------------------------------------
# describe.py
# ---------------------------------------------------------------------------
class TestDescribe:
    def setup_method(self):
        seqs, self.y = composition_control(10)
        self.docs = [make_terms(s, "unigram") for s in seqs]
        self.X, self.vocab = build_term_matrix(self.docs)

    def test_ctfidf_prefers_terms_unique_to_the_cluster(self):
        S = ctfidf_scores(self.X, self.y, 2)
        j = self.vocab.index("login"); k = self.vocab.index("purchase")
        assert S[0, j] > 0 and S[1, j] == 0 and S[1, k] > 0 and S[0, k] == 0

    def test_a_term_present_everywhere_scores_lower_than_a_unique_one(self):
        docs = [["common", "x"]] * 5 + [["common", "y"]] * 5
        X, vocab = build_term_matrix(docs)
        S = ctfidf_scores(X, np.array([0] * 5 + [1] * 5), 2)
        assert S[0, vocab.index("x")] > S[0, vocab.index("common")]

    def test_distinctive_terms_report_presence_shares(self):
        top = distinctive_terms(self.X, self.vocab, self.y, top_n=3)
        first = top[0][0]
        assert first["term"] in {"login", "page_view", "click"}
        assert first["in_cluster"] == pytest.approx(1.0) and first["overall"] == pytest.approx(0.5)

    def test_exemplars_are_members_closest_to_the_centre(self):
        F = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.5, 0.5]], dtype=float)
        labels = np.array([0, 0, 1, 1, 0])
        ex = exemplar_indices(F, labels, n=1)
        assert ex[0][0] in (0, 1, 4) and ex[1][0] in (2, 3)
        assert set(exemplar_indices(F, labels, n=5)[0]) == {0, 1, 4}
        assert len(exemplar_indices(self.X, self.y, n=2)[0]) == 2          # sparse input works too


# ---------------------------------------------------------------------------
# model.py
# ---------------------------------------------------------------------------
class TestSemanticNLPModel:
    def make(self, seqs, **kw):
        return SemanticNLPModel(encoder=HashingEncoder(128), **kw).fit_texts(seqs)

    @pytest.mark.parametrize("method", ["sbert", "gsdmm", "lda", "hdbscan"])
    def test_every_method_separates_the_composition_control(self, method):
        seqs, y = composition_control(30)
        sm = self.make(seqs)
        res = sm.cluster(2, method, random_state=0)
        assert adjusted_rand_index(sm.labels_, y) == pytest.approx(1.0)
        assert res["n_clusters_found"] == 2 and sm.method_ == method
        assert list(res["sizes"]) == sorted(res["sizes"], reverse=True) or method == "hdbscan"
        assert set(sm.timings_) >= {"render", "cluster"}

    def test_order_blind_embedding_vs_order_aware_terms(self):
        """A bag-of-words encoder cannot see order; bigram terms can. The central Stage 3 caveat."""
        seqs, y = order_control()
        sm = self.make(seqs)
        sm.cluster(2, "sbert", random_state=0)
        assert adjusted_rand_index(sm.labels_, y) < 0.2
        sm.cluster(4, "gsdmm", random_state=0)
        assert adjusted_rand_index(sm.labels_, y) > 0.5            # default terms = unigrams + bigrams

    def test_clustering_requires_fitted_texts_and_valid_arguments(self):
        sm = SemanticNLPModel(encoder=HashingEncoder())
        with pytest.raises(ValueError, match="fit_texts"):
            sm.cluster(2)
        sm.fit_texts(composition_control(5)[0])
        with pytest.raises(ValueError):
            sm.cluster(2, "nope")
        with pytest.raises(ValueError):
            sm.cluster(None, "sbert")
        with pytest.raises(ValueError, match="No clustering"):
            sm.describe()

    def test_embeddings_are_computed_on_demand_and_cached(self):
        sm = self.make(composition_control(5)[0])
        assert sm.embeddings is None
        sm.cluster(2, "sbert", random_state=0)
        assert sm.embeddings.shape[0] == 10 and "embed" in sm.timings_
        e = sm.embeddings
        sm.cluster(2, "sbert", random_state=1)
        assert sm.embeddings is e                                   # not recomputed

    def test_refitting_texts_resets_state(self):
        sm = self.make(composition_control(5)[0])
        sm.cluster(2, "sbert", random_state=0)
        sm.fit_texts(composition_control(6)[0])
        assert sm.labels_ is None and sm.embeddings is None

    def test_describe_gives_terms_and_real_example_sessions(self):
        seqs, y = composition_control(20)
        sm = self.make(seqs); sm.cluster(2, "gsdmm", random_state=0)
        d = sm.describe(top_n=3, n_examples=2)
        assert [c["size"] for c in d] == [20, 20] and sum(c["share"] for c in d) == pytest.approx(1.0)
        for c in d:
            assert len(c["examples"]) == 2 and all(e in seqs for e in c["examples"])
            assert c["top_terms"][0]["in_cluster"] == pytest.approx(1.0)
            assert isinstance(c["example_texts"][0], str)
        terms0 = {t["term"] for t in d[0]["top_terms"]}
        assert terms0 <= {"login", "page_view", "click", "<START> > login", "click > <END>", "login > page_view", "page_view > click"} \
            or terms0 <= {"product_view", "add_to_cart", "purchase", "<START> > product_view",
                          "purchase > <END>", "product_view > add_to_cart", "add_to_cart > purchase"}

    def test_silhouette_in_own_space_is_high_for_clean_clusters(self):
        seqs, _ = composition_control(30)
        for method in ("sbert", "gsdmm", "lda"):
            sm = self.make(seqs); sm.cluster(2, method, random_state=0)
            assert sm.silhouette_score(sample_size=40) > 0.5, method

    def test_silhouette_needs_two_clusters(self):
        sm = self.make(composition_control(5)[0]); sm.cluster(1, "sbert", random_state=0)
        with pytest.raises(ValueError, match="single cluster"):
            sm.silhouette_score()

    @pytest.mark.parametrize("method", ["sbert", "gsdmm", "lda"])
    def test_predict_reproduces_training_labels_on_clean_data(self, method):
        seqs, y = composition_control(30)
        sm = self.make(seqs); sm.cluster(2, method, random_state=0)
        assert (sm.predict(seqs) == sm.labels_).mean() > 0.97

    def test_predict_lda_with_kmeans_assignment_and_hdbscan_limit(self):
        seqs, y = composition_control(30)
        sm = self.make(seqs); sm.cluster(2, "lda", random_state=0, assignment="kmeans")
        assert (sm.predict(seqs) == sm.labels_).mean() > 0.97
        sm.cluster(method="hdbscan")
        with pytest.raises(NotImplementedError):
            sm.predict(seqs)

    def test_hdbscan_marks_outliers_as_noise_and_keeps_them_as_the_last_cluster(self):
        rng = np.random.default_rng(0)
        vocab = [f"e{i}" for i in range(40)]
        seqs, y = composition_control(40)
        seqs = seqs + [list(rng.choice(vocab, 4)) for _ in range(6)]          # six unrelated sessions
        sm = self.make(seqs)
        res = sm.cluster(method="hdbscan", min_cluster_size=15)
        assert res["n_noise"] >= 1 and sm.noise_mask_.sum() == res["n_noise"]
        assert (sm.labels_[sm.noise_mask_] == sm.n_clusters_found_ - 1).all()
        assert sm.describe()[-1]["is_noise"] is True

    def test_hdbscan_umap_is_optional_with_a_clear_error(self):
        try:
            import umap  # noqa: F401
            pytest.skip("umap-learn is installed")
        except ImportError:
            sm = self.make(composition_control(10)[0])
            with pytest.raises(ImportError, match="umap-learn"):
                sm.cluster(method="hdbscan", reducer="umap")

    def test_profile_applies_the_stage_2_markov_profile_to_the_clusters(self):
        seqs, y = composition_control(20)
        sm = self.make(seqs); sm.cluster(2, "sbert", random_state=0)
        tm = sm.profile()
        assert isinstance(tm, TransitionModel) and tm.cluster_method_ == "external"
        assert np.array_equal(tm.labels_, sm.labels_)
        # one of the two clusters is the buyer group: conversion from START must be 1 there, 0 in the other
        conv = sorted(tm.conversion_probability("purchase", c)["<START>"] for c in (0, 1))
        assert conv == pytest.approx([0.0, 1.0])
        assert tm.get_representative_path(0)[0]

    def test_profile_event_map_coarsens_states_without_changing_clusters(self):
        """Category-level tokens spread 'reach a cart' over many states; mapping to actions fixes it."""
        seqs = ([["view_a.x", "cart_a.x"]] * 10 + [["view_b.y", "cart_b.y"]] * 10 + [["view_c.z"] * 3] * 10)
        sm = self.make(seqs); sm.cluster(3, "sbert", random_state=0)
        fine = sm.profile()
        assert "cart" not in fine.states and "cart_a.x" in fine.states
        coarse = sm.profile(event_map=lambda s: [e.split("_")[0] for e in s])
        assert set(coarse.states) == {"<START>", "view", "cart", "<END>"}
        assert np.array_equal(coarse.labels_, sm.labels_)
        reach = sorted(coarse.conversion_probability("cart", c)["<START>"] for c in range(3))
        assert reach == pytest.approx([0.0, 1.0, 1.0])            # two clusters add to cart, one never does

    def test_get_params_documents_the_pipeline(self):
        sm = SemanticNLPModel(SessionRenderer("words", rle=True, window=3), HashingEncoder(), term_mode="bigram")
        sm.fit_texts(composition_control(5)[0]); sm.cluster(2, "gsdmm", random_state=0)
        p = sm.get_params()
        assert p["method"] == "gsdmm" and p["term_mode"] == "bigram" and p["render_window"] == 3
        assert p["encoder"] == "HashingEncoder"


# ---------------------------------------------------------------------------
# transitions.TransitionModel.assign_labels (added for the shared profile)
# ---------------------------------------------------------------------------
class TestAssignLabels:
    def test_external_labels_feed_the_markov_layer(self):
        seqs, y = composition_control(10)
        tm = TransitionModel(k=2).fit_kgrams(seqs).assign_labels(y).fit_markov_chain()
        assert tm.cluster_method_ == "external" and tm.medoids_ is None
        assert tm.dropoff_rates(0)["click"] == 1.0
        assert tm.silhouette_score() > 0.9                      # clean groups on the k-gram distance

    def test_validation(self):
        tm = TransitionModel(k=2).fit_kgrams(composition_control(3)[0])
        with pytest.raises(ValueError, match="one label per session"):
            tm.assign_labels([0, 1])
        with pytest.raises(ValueError, match=">= 0"):
            tm.assign_labels([-1] * 6)
        with pytest.raises(ValueError):
            TransitionModel(k=2).assign_labels([0])

    def test_a_new_clustering_invalidates_old_chains(self):
        seqs, y = composition_control(5)
        tm = TransitionModel(k=2).fit_kgrams(seqs).assign_labels(y).fit_markov_chain()
        tm.assign_labels(np.zeros(len(seqs), dtype=int))
        with pytest.raises(ValueError, match="fit_markov_chain"):
            tm.dropoff_rates(0)
