from typing import Any, Literal

import numpy as np
import numpy.typing as npt

class RegressionResults:
    @property
    def fittedvalues(self) -> npt.NDArray[np.float64]: ...

class RegressionModel:
    def fit(
        self,
        method: Literal["pinv", "qr"] = ...,
        cov_type: Literal[
            "nonrobust",
            "fixed scale",
            "HC0",
            "HC1",
            "HC2",
            "HC3",
            "HAC",
            "hac-panel",
            "hac-groupsum",
            "cluster",
        ] = ...,
        cov_kwds: list[Any] | None = ...,
        use_t: bool | None = ...,
        **kwargs: Any,
    ) -> RegressionResults: ...

class OLS(RegressionModel):
    def __init__(
        self,
        endog: Any,
        exog: Any | None = ...,
        missing: str = ...,
        hasconst: Any | None = ...,
        **kwargs: Any,
    ) -> None: ...
