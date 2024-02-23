"""
Module that tests the functionality of multivariate_predictions_acc_regr.py
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import pytest

from tno.sdg.tabular.eval.utility_metrics.multivariate_predictions_acc_regr import (
    avg_abs_prediction_differences_svm,
    calculate_prediction_scores,
)
from tno.sdg.tabular.eval.utility_metrics.test.utils import (
    CAT_COLS,
    CORRELATED_SINGLE_DFS_CAT,
    CORRELATED_SINGLE_DFS_NUM,
    RANDOM_CAT_DFS,
    RANDOM_NUM_DFS,
    add_arguments,
)


@pytest.mark.parametrize(
    "dataframe, cols_cat",
    add_arguments(RANDOM_CAT_DFS, CAT_COLS) + add_arguments(RANDOM_NUM_DFS, []),
)
def test_calculate_prediction_scores(
    dataframe: pd.DataFrame,
    cols_cat: Sequence[str],
) -> None:
    """
    Test whether calculate_prediction_scores returns a prediction score for each column and
    whether the prediction score is between 0 and 1.

    :param cols_cat: list of categorical columns in the dataframe.
    :param dataframe: Dataframe with data
    """
    train = dataframe.head(int(0.65 * len(dataframe)))
    test = dataframe.drop(train.index)
    test = test[train.columns]
    scores = calculate_prediction_scores(train, test, cols_cat)
    assert len(scores) == len(dataframe.columns)
    for score in scores:
        assert 0 <= score <= 1


# tuples of a dataframe and a copy thereof with permuted rows
IDD_DFS_NUM = [(df, df.sample(frac=1)) for df in CORRELATED_SINGLE_DFS_NUM]
IDD_DFS_CAT = [(df, df.sample(frac=1)) for df in CORRELATED_SINGLE_DFS_CAT]


@pytest.mark.parametrize(
    "original_dataframe, synthetic_dataframe, cols_cat",
    add_arguments(IDD_DFS_CAT, CAT_COLS[:2]) + add_arguments(IDD_DFS_NUM, []),
)
def test_avg_abs_prediction_differences_svm(
    original_dataframe: pd.DataFrame,
    synthetic_dataframe: pd.DataFrame,
    cols_cat: Sequence[str],
) -> None:
    """
    Test whether identically distributed dataframes, one being a permutation of the elements in the
    other, get an average absolute prediction difference of about 0.

    :param original_dataframe: First dataframe.
    :param synthetic_dataframe: Second dataframe.
    :param cols_cat: categorical column names in the dataframes.
    """
    score = avg_abs_prediction_differences_svm(
        original_dataframe, synthetic_dataframe, cols_cat=cols_cat
    )
    assert score == pytest.approx(0.0, abs=1e-3, rel=1e-3)
