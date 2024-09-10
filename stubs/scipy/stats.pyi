from __future__ import annotations

from typing import List

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

def chi2_contingency(
    observed: Array,
    correction: None | bool = ...,
    lambda_: None | float | str = ...,
) -> tuple[float, float, int, Array]: ...
