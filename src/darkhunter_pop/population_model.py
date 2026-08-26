"""Stage: ``population_model``.

Hierarchical multiplicity -> type mixture (BH / NS / WD / other / outlier) over a non-parametric
compact-object mass function (ARCHITECTURE.md §4). Not yet implemented.

Downstream of ``sensitivity_analysis``: consume per-class covariate recommendations from that
stage's HDF5 artifact when choosing rate covariates — do not hard-code extras beyond mass
without that artifact, and do not treat those recommendations as silently applied defaults.
"""
