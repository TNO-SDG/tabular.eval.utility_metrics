from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

Number = int | float

Array = (
    npt.ArrayLike
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int_]
    | List[List[Number]]
    | List[Number]
)

Color = str | Tuple[float, float, float] | Tuple[float, float, float, float]

def heatmap(
    data: Array | pd.DataFrame,
    *,
    vmin: None | float = ...,
    vmax: None | float = ...,
    cmap: None | Color | Sequence[Color] = ...,
    center: None | float = ...,
    robust: bool = ...,
    annot: None | bool | Array | pd.DataFrame = ...,
    fmt: str = ...,
    annot_kws: None | dict[Any, Any] = ...,
    linewidths: float | int = ...,
    linecolor: Color = ...,
    cbar: bool = ...,
    cbar_kws: None | dict[Any, Any] = ...,
    cbar_ax: Any = ...,
    square: bool = ...,
    xticklabels: str | Array | list[Any] | int | bool = ...,
    yticklabels: str | Array | list[Any] | int | bool = ...,
    mask: None | Array | pd.DataFrame = ...,
    ax: Any = ...,
    **kwargs: Any,
) -> Any: ...
