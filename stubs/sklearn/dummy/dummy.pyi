from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.base import ClassifierMixin, MultiOutputMixin

Number = int | float

Array = (
    npt.ArrayLike
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int_]
    | List[List[Number]]
    | List[Number]
)

class DummyClassifier(MultiOutputMixin, ClassifierMixin, BaseEstimator):  # type: ignore
    if TYPE_CHECKING:
        from _typeshed import Self
    else:
        Self = Any

    def __init__(
        self,
        strategy: str = ...,
        random_state: None | int = ...,
        constant: None | int | str | Array = ...,
    ): ...
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
