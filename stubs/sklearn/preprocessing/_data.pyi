from sklearn.base import BaseEstimator  # type: ignore[import-untyped]
from sklearn.base import OneToOneFeatureMixin, TransformerMixin

class StandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):  # type: ignore
    def __init__(
        self,
        *,
        copy: bool = ...,
        with_mean: bool = ...,
        with_std: bool = ...,
    ) -> None: ...
