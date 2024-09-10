from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import numpy as np
import numpy.typing as npt
from sklearn.base import RegressorMixin  # type: ignore[import-untyped]
from sklearn.svm._base import BaseLibSVM, BaseSVC  # type: ignore[import-untyped]

Number = int | float

Array = (
    npt.ArrayLike
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int_]
    | List[List[Number]]
    | List[Number]
)

class SVC(BaseSVC):  # type: ignore[misc]
    if TYPE_CHECKING:
        from _typeshed import Self
    else:
        Self = Any

    def __init__(
        self,
        *,
        C: float = ...,
        kernel: str = ...,
        degree: int = ...,
        gamma: str = ...,
        coef0: float = ...,
        shrinking: bool = ...,
        probability: bool = ...,
        tol: float = ...,
        cache_size: int = ...,
        class_weights: None | int = ...,
        verbose: bool = ...,
        max_iter: int = ...,
        decision_function_shape: str = ...,
        break_ties: bool = ...,
        random_state: None | int = ...,
    ) -> None: ...
    def fit(
        self: Self,
        X: list[list[Any]] | Array,
        y: Array,
        sample_weight: None | Array = ...,
    ) -> Self: ...
    def score(
        self,
        X: list[list[Any]] | Array,
        y: Array,
        sample_weight: None | Array = ...,
    ) -> float: ...

class SVR(RegressorMixin, BaseLibSVM):  # type: ignore[misc]
    if TYPE_CHECKING:
        from _typeshed import Self
    else:
        Self = Any

    def __init__(
        self,
        *,
        kernel: str = ...,
        degree: int = ...,
        gamma: str = ...,
        coef0: float = ...,
        tol: float = ...,
        C: float = ...,
        epsilon: float = ...,
        shrinking: bool = ...,
        cache_size: int = ...,
        verbose: bool = ...,
        max_iter: int = ...,
    ) -> None: ...
    def fit(
        self: Self,
        X: list[list[Any]] | Array,
        y: Array,
        sample_weight: None | Array = ...,
    ) -> Self: ...
