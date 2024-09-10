from __future__ import annotations

from typing import Any, List

import numpy as np
import numpy.typing as npt
from numpy._typing import _NestedSequence, _SupportsArray
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore[import-untyped]

Number = int | float

Array = (
    npt.ArrayLike
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int_]
    | List[List[Number]]
    | List[Number]
)

class KBinsDiscretizer(TransformerMixin, BaseEstimator):  # type: ignore
    def __init__(
        self,
        n_bins: int = ...,
        *,
        encode: str = ...,
        strategy: str,
        dtype: None | np.float32 | np.float64 = ...,
        subsample: str = ...,
        random_state: None | int = ...,
    ) -> None: ...
    def fit_transform(
        self,
        X: list[list[Any]] | Array,
        y: None | Array = ...,
        **fit_params: Any,
    ) -> (
        _SupportsArray[np.dtype[Any]] | _NestedSequence[_SupportsArray[np.dtype[Any]]]
    ): ...
