"""
Module that tests the functionality of spiderplot.py
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import pytest

from tno.sdg.tabular.eval.utility_metrics.spiderplot import compute_all_metrics
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
def test_compute_all_metrics(
    original_dataframe: pd.DataFrame,
    synthetic_dataframe: pd.DataFrame,
    cols_cat: Sequence[str],
) -> None:
    """
    Test whether the resulting dictionary of metrics from compute_all_metrics contains data for
    all metrics.

    :param original_dataframe: First dataframe.
    :param synthetic_dataframe: Second dataframe.
    :param cols_cat: categorical column names in the dataframes.
    """
    dataset_name = "synthetic dataframe"
    all_metrics = compute_all_metrics(
        original_dataframe, {dataset_name: synthetic_dataframe}, cols_cat
    )
    assert dataset_name in all_metrics
    dataset_entry = all_metrics[dataset_name]
    for metric_name in [
        "univariate_distributions",
        "bivariate_distributions",
        "multivariate_distributions",
        "distinguishability",
    ]:
        assert metric_name in dataset_entry


@pytest.mark.parametrize(
    "original_dataframe, synthetic_dataframe, cols_cat",
    add_arguments(IDD_DFS_CAT, CAT_COLS[:2]) + add_arguments(IDD_DFS_NUM, []),
)
def test_compute_all_metrics_minmax(
    original_dataframe: pd.DataFrame,
    synthetic_dataframe: pd.DataFrame,
    cols_cat: Sequence[str],
) -> None:
    """
    Test whether the resulting dictionary of metrics from compute_all_metrics contains data for
    all metrics between 0 and 1.

    :param original_dataframe: First dataframe.
    :param synthetic_dataframe: Second dataframe.
    :param cols_cat: categorical column names in the dataframes.
    """
    dataset_name = "synthetic dataframe"
    all_metrics = compute_all_metrics(
        original_dataframe, {dataset_name: synthetic_dataframe}, cols_cat
    )
    assert dataset_name in all_metrics
    dataset_entry = all_metrics[dataset_name]
    for metric_name in [
        "univariate_distributions",
        "bivariate_distributions",
        "multivariate_distributions",
        "distinguishability",
    ]:
        assert 0 <= dataset_entry[metric_name] <= 1
