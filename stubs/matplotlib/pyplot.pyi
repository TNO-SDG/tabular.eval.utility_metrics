from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

Number = float | int

Array = (
    npt.ArrayLike
    | npt.NDArray[np.float64]
    | npt.NDArray[np.int_]
    | List[List[Number]]
    | List[Number]
)

color = str | Tuple[float, float, float] | Tuple[float, float, float, float]

def axline(
    xy1: Sequence[float],
    xy2: Sequence[float] | None = ...,
    *,
    slope: float | None = ...,
    **kwargs: Any,
) -> Line2D: ...
def clf() -> None: ...
def close(fig: Figure | str | int | None = ...) -> None: ...
def figure(
    num: None | str | int = ...,
    figsize: None | tuple[float, float] = ...,
    dpi: None | int = ...,
    facecolor: None | str = ...,
    edgecolor: None | str = ...,
    frameon: bool = ...,
    FigureClass: Any = ...,
    clear: bool = ...,
    **kwargs: Any,
) -> Any: ...
def grid(
    visible: None | Any = ..., which: str = ..., axis: str = ..., **kwargs: Any
) -> None: ...
def hist(
    x: Array | Sequence[Array],
    bins: None | int | str | Sequence[Any] = ...,
    range: None | tuple[Any, ...] = ...,
    density: bool = ...,
    weights: None | Array = ...,
    cumulative: bool | int = ...,
    bottom: None | Array | Number = ...,
    histtype: str = ...,
    align: str = ...,
    orientation: str = ...,
    rwidth: None | float = ...,
    log: bool = ...,
    color: None | color | Sequence[color] = ...,
    label: None | str | list[str] = ...,
    stacked: bool = ...,
    *,
    data: None | Array = ...,
    **kwargs: Any,
) -> tuple[Array | list[Array], Array, Any]: ...
def legend(*args: Any, **kwargs: Any) -> Any: ...
def plot(
    *args: Any, scalex: bool = ..., scaley: bool = ..., data: Any = ..., **kwargs: Any
) -> None: ...
def savefig(*args: Any, **kwargs: Any) -> None: ...
def show(*args: Any, **kwargs: Any) -> Any: ...
def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    sharex: bool | str = ...,
    sharey: bool | str = ...,
    squeeze: bool = ...,
    width_ratios: None | Array = ...,
    height_ratios: None | Array = ...,
    subplot_kw: None | dict[Any, Any] = ...,
    gridspec_kw: None | dict[Any, Any] = ...,
    **fig_kw: Any,
) -> Any: ...
def text(x: Any, y: Any, s: Any, fontdict: Any = None, **kwargs: Any) -> Any: ...
def title(
    label: str,
    fontdict: None | Any = ...,
    loc: None | str = ...,
    pad: None | float = ...,
    *,
    y: None | float = ...,
    **kwargs: Any,
) -> None: ...
def xticks(
    ticks: None | Array = ..., *, minor: bool = ..., **kwargs: Any
) -> tuple[list[float], list[str]]: ...
def ylabel(
    ylabel: str,
    fontdict: None | dict[str, str | int] = ...,
    labelpad: None | float = ...,
    *,
    loc: None | Any = ...,
    **kwargs: Any,
) -> str: ...
def ylim(*args: Any, **kwargs: Any) -> tuple[float, float]: ...
