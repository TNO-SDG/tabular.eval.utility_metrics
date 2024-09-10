from __future__ import annotations

from typing import Any, Callable, List, Mapping, Sequence, Tuple

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

def kdeplot(
    data: None | Array | pd.DataFrame | Sequence[Any] | Mapping[Any, Any] = ...,
    *,
    x: None | Array = ...,
    y: None | Array = ...,
    hue: None | Array = ...,
    weights: None | Array = ...,
    palette: None | str | list[Any] | dict[Any, Any] | Color = ...,
    hue_order: None | Array = ...,
    hue_norm: None | tuple[Any, ...] = ...,
    color: Any = ...,
    fill: None | bool = ...,
    multiple: str = ...,
    common_norm: bool = ...,
    common_grid: bool = ...,
    cumulative: bool = ...,
    bw_method: str | int | float | Callable[[Any], Any] = ...,
    bw_adjust: Number = ...,
    warn_singular: bool = ...,
    log_scale: (
        None
        | bool
        | int
        | float
        | list[Number | bool]
        | tuple[Number | bool, Number | bool]
    ) = ...,
    levels: int | Array = ...,
    thresh: int = ...,
    gridsize: int = ...,
    cut: Number = ...,
    clip: (
        None
        | Array
        | tuple[Number, Number]
        | tuple[tuple[Number, Number], tuple[Number, Number]]
    ) = ...,
    legend: bool = ...,
    cbar: bool = ...,
    cbar_ax: Any = ...,
    cbar_kws: None | dict[Any, Any] = ...,
    ax: Any = ...,
    **kwargs: Any,
) -> Any: ...
