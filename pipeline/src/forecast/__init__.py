"""Model layer of the rerun (protocol Section 4): registry, search spaces, frozen
configurations, fitting from a frozen configuration, tuning and evaluation.

Nothing here produces a forecast that a model did not produce: there is no fallback
forecaster, no substitute value and no per-task adjustment.
"""
