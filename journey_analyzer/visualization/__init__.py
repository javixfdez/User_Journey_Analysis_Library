# journey_analyzer/visualization/__init__.py
# matplotlib is imported lazily inside the plotting functions, so importing
# this package never fails when matplotlib is missing.
from .plots import classical_mds, plot_clusters

__all__ = ["classical_mds", "plot_clusters"]
