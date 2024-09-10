from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import numpy as np
import numpy.typing as npt

Number = int | float

Array = (
    npt.ArrayLike
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int_]
    | List[List[Number]]
    | List[Number]
)

class Pipeline:
    if TYPE_CHECKING:
        from _typeshed import Self
    else:
        Self = Any

    def __init__(
        self,
        steps: list[tuple[Any, ...]],
        *,
        memory: Any = ...,
        verbose: None | bool = ...,
    ) -> None: ...
    def fit(
        self: Self,
        X: list[list[Any]] | Array,
        y: None | Array = ...,
        **fit_params: Any,
    ) -> Self: ...
    def predict_proba(
        self,
        X: list[list[Any]] | Array,
        **predict_proba_params: str,
    ) -> npt.NDArray[Any]: ...
