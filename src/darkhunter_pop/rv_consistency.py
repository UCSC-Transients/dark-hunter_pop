"""Stages: ``rv_astrometry_gate`` and ``joint_orbit_fit``.

Consumes ``dark-hunter_rv`` (The Joker) output (ARCHITECTURE.md §4). Gate failures skip
``joint_orbit_fit`` with reason ``rv_astrometry_gate_failed``. Not yet implemented.
"""
