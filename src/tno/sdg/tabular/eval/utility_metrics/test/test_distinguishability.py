"""
Module that tests the functionality of distinguishability.py
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import pytest

from tno.sdg.tabular.eval.utility_metrics.distinguishability import (
    logistical_regression_auc,
    mean_propensity_difference_logistical_regression,
)
from tno.sdg.tabular.eval.utility_metrics.test.utils import (
    CAT_COLS,
    CORRELATED_SINGLE_DFS_CAT,
    CORRELATED_SINGLE_DFS_NUM,
    add_arguments,
)

# tuples of a dataframe and a copy thereof with permuted rows
IDD_DFS_NUM = [(df, df.sample(frac=1)) for df in CORRELATED_SINGLE_DFS_NUM]
IDD_DFS_CAT = [(df, df.sample(frac=1)) for df in CORRELATED_SINGLE_DFS_CAT]


@pytest.mark.parametrize(
    "original_dataframe, synthetic_dataframe, cols_cat",
    add_arguments(IDD_DFS_CAT, CAT_COLS[:2]) + add_arguments(IDD_DFS_NUM, []),
)
def test_mean_propensity_difference_logistical_regression(
    original_dataframe: pd.DataFrame,
    synthetic_dataframe: pd.DataFrame,
    cols_cat: Sequence[str],
) -> None:
    """
    Test whether identically distributed dataframes, one being a permutation of the elements in the
    other, get an average absolute mean propensity difference of about 0.

    :param original_dataframe: Part of the input set used for training.
    :param synthetic_dataframe: Part of the input set used for testing.
    :param cols_cat: categorical column names in the dataframes.
    """
    mean_propensity_difference, _ = mean_propensity_difference_logistical_regression(
        original_dataframe,
        synthetic_dataframe,
        cols_cat=cols_cat,
    )
    assert mean_propensity_difference == pytest.approx(0.0)


@pytest.mark.parametrize(
    "original_dataframe, synthetic_dataframe, cols_cat",
    add_arguments(IDD_DFS_CAT, CAT_COLS[:2]) + add_arguments(IDD_DFS_NUM, []),
)
def test_logistical_regression_auc(
    original_dataframe: pd.DataFrame,
    synthetic_dataframe: pd.DataFrame,
    cols_cat: Sequence[str],
) -> None:
    """
    Test whether identically distributed dataframes, one being a permutation of the elements in the
    other, get an average absolute mean propensity difference of about 0.

    :param original_dataframe: Part of the input set used for training.
    :param synthetic_dataframe: Part of the input set used for testing.
    :param cols_cat: categorical column names in the dataframes.
    """
    _, utility_disth = logistical_regression_auc(
        original_dataframe,
        synthetic_dataframe,
        cols_cat=cols_cat,
    )
    assert 0 <= utility_disth <= 1
