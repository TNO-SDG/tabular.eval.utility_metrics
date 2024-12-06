from __future__ import annotations

import numpy as np
import numpy.typing as npt

Number = int | float

Array = (
    npt.ArrayLike
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int_]
    | list[list[Number]]
    | list[Number]
)

def chi2_contingency(
    observed: Array,
    correction: None | bool = ...,
    lambda_: None | float | str = ...,
) -> tuple[float, float, int, Array]: ...
