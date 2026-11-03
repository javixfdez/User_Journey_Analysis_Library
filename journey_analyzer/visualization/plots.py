"""
plots.py - Visual tools for journey clusters.

THE PROBLEM
-----------
Sessions are compared with a distance matrix D (N x N). A scatter plot needs
2D coordinates, so we have to PROJECT the sessions onto a plane such that
sessions that are close in D are close on the plot. That is Multidimensional
Scaling (MDS).

WHAT IS HERE
------------
classical_mds : numpy-only MDS (Torgerson). Deterministic, no extra packages.
plot_clusters : two-panel figure
                 left  = sessions coloured by cluster, medoids as stars,
                         optional thin lines from each session to its medoid
                 right = the same points coloured by each session's own
                         silhouette (green = well placed, red = misplaced)

HOW TO READ THE PLOT (honest limits)
------------------------------------
* A 2D picture of N-dimensional distances is always distorted. The title shows
  how good the fit is (see ``stress`` below). Use the plot to build intuition;
  trust the numbers (inertia, silhouette) for conclusions.
* Many sessions are IDENTICAL journeys (distance 0), so they land on exactly the
  same point. Points are jittered slightly for display so that overlapping
  sessions remain visible. The jitter never changes the data.
"""

from typing import Optional, Tuple

import numpy as np

from ..alignment.kmedoids import silhouette_samples_from_distances

__all__ = ["classical_mds", "plot_clusters"]


# ---------------------------------------------------------------------------
# Classical (Torgerson) MDS
# ---------------------------------------------------------------------------
def classical_mds(D: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, float, float]:
    """
    Project a distance matrix to ``n_components`` coordinates.

    Steps (all linear algebra, no randomness):
      1. Square the distances.
      2. "Double-centre" them: subtract row means and column means, add the
         grand mean. This turns distances into inner products B between the
         (unknown) centred coordinates.
      3. Eigen-decompose B. The coordinates are the top eigenvectors scaled by
         the square roots of their eigenvalues.

    Returns
    -------
    X         : (N, n_components) coordinates.
    explained : share of the (positive) eigenvalue mass kept by the chosen axes.
                Edit distance is not exactly Euclidean, so some eigenvalues are
                negative and are ignored here; this is normal.
    stress    : Kruskal stress-1 = sqrt( sum (D - D_2d)^2 / sum D^2 ).
                0 = perfect picture; below ~0.2 is usually acceptable, above
                ~0.4 the picture is a rough cartoon.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix.")

    D2 = D ** 2
    B = -0.5 * (D2 - D2.mean(axis=0, keepdims=True) - D2.mean(axis=1, keepdims=True) + D2.mean())

    eigvals, eigvecs = np.linalg.eigh(B)              # ascending order
    order = np.argsort(eigvals)[::-1][:n_components]  # largest first
    lam = np.clip(eigvals[order], 0.0, None)          # drop negative directions
    X = eigvecs[:, order] * np.sqrt(lam)

    positive_mass = eigvals[eigvals > 0].sum()
    explained = float(lam.sum() / positive_mass) if positive_mass > 0 else 0.0

    D_2d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    denom = float((D ** 2).sum())
    stress = float(np.sqrt(((D - D_2d) ** 2).sum() / denom)) if denom > 0 else 0.0
    return X, explained, stress


# ---------------------------------------------------------------------------
# Cluster scatter plot
# ---------------------------------------------------------------------------
def plot_clusters(
    D: np.ndarray,
    labels: np.ndarray,
    medoids: np.ndarray,
    title: Optional[str] = None,
    links: bool = True,
    jitter: float = 0.012,
    random_state: int = 0,
    figsize: Tuple[float, float] = (14, 6),
):
    """
    Scatter plot of sessions around their medoids.

    Parameters
    ----------
    D        : (N, N) distance matrix the clustering was computed on.
    labels   : (N,) cluster id of every session.
    medoids  : (k,) session index of each cluster's medoid (position c = cluster c).
    links    : draw a faint line from every session to its medoid.
    jitter   : display-only noise, as a fraction of the plot range, so that
               identical sessions do not hide each other. 0 disables it.

    Returns
    -------
    (fig, X) : the matplotlib figure and the (N, 2) MDS coordinates used.
    """
    import matplotlib.pyplot as plt                      # lazy: matplotlib stays optional
    from matplotlib.collections import LineCollection

    D = np.asarray(D)
    labels = np.asarray(labels)
    medoids = np.asarray(medoids, dtype=int)
    k = len(medoids)

    X, explained, stress = classical_mds(D, 2)

    # Display coordinates: jitter everything except the medoids themselves.
    rng = np.random.default_rng(random_state)
    span = np.ptp(X, axis=0).max() or 1.0
    P = X + rng.normal(0.0, jitter * span, size=X.shape)
    P[medoids] = X[medoids]

    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    colors = [cmap(c % cmap.N) for c in range(k)]

    show_silhouette = len(np.unique(labels)) >= 2
    sil = silhouette_samples_from_distances(D, labels) if show_silhouette else None
    fig, axes = plt.subplots(1, 2 if show_silhouette else 1, figsize=figsize, squeeze=False)
    axes = axes[0]

    # ---- left panel: clusters + medoids ---------------------------------
    ax = axes[0]
    if links:
        segments = np.stack([P, X[medoids[labels]]], axis=1)       # (N, 2, 2): session -> its medoid
        ax.add_collection(LineCollection(
            segments, colors=[colors[c] for c in labels], linewidths=0.4, alpha=0.12, zorder=1))
    for c in range(k):
        members = labels == c
        ax.scatter(P[members, 0], P[members, 1], s=22, color=colors[c], alpha=0.75,
                   edgecolor="none", zorder=2, label=f"cluster {c} ({members.sum()} sessions)")
        ax.scatter(*X[medoids[c]], marker="*", s=420, color=colors[c],
                   edgecolor="black", linewidth=1.3, zorder=4)
        ax.annotate(f"M{c}", X[medoids[c]], xytext=(7, 7), textcoords="offset points",
                    fontsize=9, fontweight="bold", zorder=5)
    ax.set_title("Sessions around their medoids (star = medoid)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    # ---- right panel: silhouette of every session ------------------------
    if show_silhouette:
        ax2 = axes[1]
        order = np.argsort(-sil)                         # best first, worst drawn last (on top)
        sc = ax2.scatter(P[order, 0], P[order, 1], c=sil[order], cmap="RdYlGn",
                         vmin=-1, vmax=1, s=22, edgecolor="none", zorder=2)
        ax2.scatter(X[medoids, 0], X[medoids, 1], marker="*", s=420, facecolor="none",
                    edgecolor="black", linewidth=1.3, zorder=4)
        fig.colorbar(sc, ax=ax2, label="silhouette of each session")
        ax2.set_title(f"Same points coloured by silhouette (mean {sil.mean():.2f})")

    for a in axes:
        a.set_xlabel("MDS axis 1")
        a.set_ylabel("MDS axis 2")
        a.set_aspect("equal", adjustable="datalim")
        a.grid(alpha=0.2)

    cost = float(D[medoids][labels, np.arange(len(labels))].sum())
    fig.suptitle(
        (title + "\n" if title else "")
        + f"k={k} | inertia={cost:.0f} | 2D fit: {explained:.0%} of variance kept, "
          f"stress={stress:.2f} (points jittered for display)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig, X