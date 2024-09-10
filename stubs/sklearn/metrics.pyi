from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

def roc_curve(
    y_true: npt.NDArray[Any],
    y_score: npt.NDArray[Any],
    *,
    pos_label: int | str | None = None,
    sample_weight: npt.ArrayLike | None = ...,
    drop_intermediate: bool = ...,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]: ...
def roc_auc_score(
    y_true: npt.NDArray[Any],
    y_score: npt.NDArray[Any],
    *,
    average: str | None = ...,
    sample_weight: npt.ArrayLike | None = ...,
    max_fpr: float | None = ...,
    multi_class: str = ...,
    labels: npt.ArrayLike | None = ...,
) -> float: ...
