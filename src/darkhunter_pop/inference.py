"""Stage: ``inference``.

Staged-but-connected inhomogeneous Poisson point-process likelihood sampled with ``dynesty``
(ARCHITECTURE.md §4). Not yet implemented.

Downstream of ``sensitivity_analysis``: read dimensionality / binned-vs-unbinned
recommendations from that stage's HDF5 artifact when configuring the likelihood. Those
records are opt-in; this module's defaults must not be silently rewritten by the
sensitivity stage.
"""
