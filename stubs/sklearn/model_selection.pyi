from __future__ import annotations

from typing import Any

import numpy.typing as npt

def train_test_split(
    *arrays: npt.ArrayLike,
    test_size: float | None = ...,
    train_size: float | None = ...,
    random_state: int | None = ...,
    shuffle: bool = ...,
    stratify: npt.ArrayLike | None = None,
) -> list[npt.NDArray[Any]]: ...
