import itertools
import random

import numpy as np
import pytest

from journey_analyzer import BaselineAligner
from journey_analyzer.alignment import (
    assign_stages,
    levenshtein_distance as lev,
    normalized_levenshtein_distance as nlev,
    weighted_levenshtein_distance as wlev,
)


# ---- levenshtein_distance --------------------------------------------------
def test_known_value_kitten_sitting():
    assert lev(list("kitten"), list("sitting")) == 3


def test_identity_and_empty():
    assert lev(["a", "b"], ["a", "b"]) == 0
    assert lev([], ["a", "b"]) == 2
    assert lev(["a", "b", "c"], []) == 3
    assert lev([], []) == 0


def test_single_insertion_example_from_docstring():
    assert lev(["login", "buy"], ["login", "browse", "buy"]) == 1


def test_symmetry_and_triangle_inequality_random():
    rnd = random.Random(0)
    seqs = [[rnd.choice("abc") for _ in range(rnd.randint(0, 7))] for _ in range(25)]
    for x, y, z in itertools.product(seqs[:10], repeat=3):
        assert lev(x, y) == lev(y, x)
        assert lev(x, z) <= lev(x, y) + lev(y, z)


def test_asymmetric_costs_direction():
    # ["a"] -> ["a","b"] needs one insertion; the reverse needs one deletion.
    assert lev(["a"], ["a", "b"], insert_cost=2, delete_cost=5) == 2
    assert lev(["a", "b"], ["a"], insert_cost=2, delete_cost=5) == 5


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        lev(["a"], ["b"], substitute_cost=-1)


# ---- normalized ------------------------------------------------------------
def test_normalized_range_and_empty():
    assert nlev([], []) == 0.0
    assert nlev(["a"], ["b"]) == 1.0
    assert 0.0 <= nlev(list("abc"), list("abd")) <= 1.0


# ---- stages / weighted -----------------------------------------------------
STAGES = {"login": "pre", "browse": "pre", "purchase": "purchase", "logout": "post"}


def test_assign_stages_and_unknown_event():
    assert assign_stages(["login", "purchase"], STAGES) == [("login", "pre"), ("purchase", "purchase")]
    with pytest.raises(KeyError):
        assign_stages(["refund"], STAGES)
    assert assign_stages(["refund"], STAGES, default_stage="post") == [("refund", "post")]


def test_weighted_scales_with_stage_weight():
    a = assign_stages(["login", "purchase", "logout"], STAGES)
    b = assign_stages(["login", "logout"], STAGES)            # missing purchase
    assert wlev(a, b) == 1.0
    assert wlev(a, b, {"purchase": 3.0}) == 3.0
    assert wlev(a, b, {"pre": 10.0}) == 1.0                    # difference not in 'pre'


def test_weighted_stage_only_in_one_journey():
    a = assign_stages(["login"], STAGES)
    b = assign_stages(["login", "logout", "logout"], STAGES)
    assert wlev(a, b) == 2.0


# ---- BaselineAligner -------------------------------------------------------
SEQS = [["login", "browse", "purchase"], ["login", "purchase"], ["login", "browse", "browse", "logout"]]


def test_matrix_properties():
    D = BaselineAligner().fit_distance_matrix(SEQS)
    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.all(np.diag(D) == 0)
    assert D[0, 1] == 1


def test_weighted_matrix_uses_stages():
    al = BaselineAligner("weighted", stage_weights={"purchase": 3.0}, event_to_stage=STAGES)
    D = al.fit_distance_matrix(SEQS)
    assert D[0, 1] == 1.0   # only "browse" (stage pre, weight 1) is missing
    assert D[0, 2] > 0


def test_weighted_requires_mapping_and_bad_metric():
    with pytest.raises(ValueError):
        BaselineAligner("weighted")
    with pytest.raises(ValueError):
        BaselineAligner("nope")


def test_memory_guard_and_readonly_matrix():
    with pytest.raises(MemoryError):
        BaselineAligner(max_matrix_gb=1e-9).fit_distance_matrix(SEQS)
    al = BaselineAligner()
    al.fit_distance_matrix(SEQS)
    with pytest.raises(ValueError):
        al.distance_matrix[0, 1] = 99


def test_custom_callable_and_prototypes_not_ready():
    al = BaselineAligner(metric=lambda a, b: abs(len(a) - len(b)))
    D = al.fit_distance_matrix(SEQS)
    assert D[0, 2] == 1
    with pytest.raises(NotImplementedError):
        al.extract_prototypes(2)
