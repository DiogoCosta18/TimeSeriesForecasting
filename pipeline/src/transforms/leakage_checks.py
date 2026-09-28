from __future__ import annotations


def assert_stl_fit_uses_training_only(transform, train_len: int) -> None:
    if getattr(transform, "n_train_", None) != train_len:
        raise AssertionError("STL transform was not fitted on the requested training slice length")

