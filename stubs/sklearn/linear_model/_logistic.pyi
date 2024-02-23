from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.linear_model._base import (  # type: ignore[import-untyped]
    LinearClassifierMixin,
    SparseCoefMixin,
)

class LogisticRegression(LinearClassifierMixin, SparseCoefMixin, BaseEstimator):  # type: ignore
    def __init__(
        self,
        penalty: str = ...,
        *,
        dual: bool = ...,
        tol: float = ...,
        C: float = ...,
        fit_intercept: bool = ...,
        intercept_scaling: int = ...,
        class_weight: None | dict[Any, float] | str = ...,
        random_state: None | int = ...,
        solver: str = ...,
        max_iter: int = ...,
        multi_class: str = ...,
        verbose: int = ...,
        warm_start: bool = ...,
        n_jobs: None | int = ...,
        l1_ratio: float = ...,
    ) -> None: ...
