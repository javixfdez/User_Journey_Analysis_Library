import numpy as np
import pytest

from journey_analyzer import BaselineAligner
from journey_analyzer.visualization import classical_mds


def _euclid(points):
    return np.linalg.norm(points[:, None] - points[None], axis=2)


def test_mds_recovers_planar_points_up_to_rotation():
    pts = np.random.default_rng(0).random((25, 2))
    D = _euclid(pts)
    X, explained, stress = classical_mds(D)
    assert X.shape == (25, 2)
    assert stress < 1e-6 and explained == pytest.approx(1.0)
    assert np.allclose(_euclid(X), D, atol=1e-6)         # distances preserved


def test_mds_high_dim_points_have_nonzero_stress():
    D = _euclid(np.random.default_rng(1).random((30, 6)))
    _, explained, stress = classical_mds(D)
    assert 0 < explained < 1 and 0 < stress < 1


def test_mds_rejects_non_square():
    with pytest.raises(ValueError):
        classical_mds(np.zeros((3, 4)))


def test_plot_clusters_runs_and_requires_clustering():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seqs = [["login", "buy"], ["login", "browse", "buy"], ["logout", "click"],
            ["click", "logout"], ["login", "buy", "logout"], ["click"]] * 5
    al = BaselineAligner()
    al.fit_distance_matrix(seqs)
    with pytest.raises(ValueError):
        al.plot_clusters()
    al.extract_prototypes(2)
    fig, X = al.plot_clusters(title="test", links=False)
    assert X.shape == (len(seqs), 2) and len(fig.axes) >= 2
    plt.close(fig)
    # a single cluster only draws the left panel
    al.extract_prototypes(1)
    fig, _ = al.plot_clusters()
    plt.close(fig)
