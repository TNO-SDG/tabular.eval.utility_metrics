from __future__ import annotations

from typing import Any

from plotly.basedatatypes import BaseFigure

class Figure(BaseFigure):
    def __init__(
        self,
        data: Any = ...,
        layout: Any = ...,
        frames: Any = ...,
        skip_invalid: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def add_trace(
        self,
        trace: Any,
        row: Any = ...,
        col: Any = ...,
        secondary_y: Any = ...,
        exclude_empty_subplots: bool = ...,
    ) -> Any: ...
    def update_layout(
        self,
        dict1: None | dict[Any, Any] = ...,
        overwrite: bool = ...,
        **kwargs: Any,
    ) -> Any: ...
