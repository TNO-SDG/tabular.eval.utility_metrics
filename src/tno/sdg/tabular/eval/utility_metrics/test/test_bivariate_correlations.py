"""
Module that tests the functionality of bivariate_correlations.py.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tno.sdg.tabular.eval.utility_metrics.bivariate_correlations import (
    avg_abs_correlation_difference,
    avg_abs_correlation_difference_cramers,
    compute_cramers_correlation_matrix,
    compute_cramers_v,
    discretize_matrix,
)
from tno.sdg.tabular.eval.utility_metrics.test.utils import (
    BINS,
    CAT_COLS,
    CORRELATED_MULTIPLE_DFS_CAT,
    CORRELATED_MULTIPLE_DFS_NUM,
    CORRELATED_SINGLE_DFS_CAT,
    CORRELATED_SINGLE_DFS_NUM,
    CORRELATED_VECTORS_CAT,
    UNCORRELATED_SINGLE_DFS_CAT,
    UNCORRELATED_VECTORS_CAT,
    add_arguments,
)

MANUAL_CRAMERS_V_EXAMPLE: list[Any] = [
    (
        np.array([1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0]),
        np.array([1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
        0.169,
    )
]


@pytest.mark.parametrize(
    "vector_1, vector_2, expected_val",
    add_arguments(CORRELATED_VECTORS_CAT, 1.0)
    + add_arguments(UNCORRELATED_VECTORS_CAT, 0.0)
    + MANUAL_CRAMERS_V_EXAMPLE,
)
def test_correlation_cramersv(
    vector_1: npt.NDArray[Any], vector_2: npt.NDArray[Any], expected_val: float
) -> None:
    """
    Test whether the Cramer's v method returns the correct answers for perfectly correlated and
    perfectly uncorrelated vectors.

    :param vector_1: First vector.
    :param vector_2: Second vector.
    :param expected_val: Expected correlation between provided vectors.
    """
    cramers_v = compute_cramers_v(vector_1, vector_2)
    assert cramers_v == pytest.approx(expected_val, abs=0.0001)


@pytest.mark.parametrize(
    "matrix, expected_val",
    add_arguments(CORRELATED_SINGLE_DFS_CAT, 1.0)
    + add_arguments(UNCORRELATED_SINGLE_DFS_CAT, 0.0),
)
def test_correlation_matrix(matrix: pd.DataFrame, expected_val: float) -> None:
    """
    Test to make sure that the cross-correlation of all columns yields the expected result for
    fully correlated matrices and fully uncorrelated matrices.

    :param matrix: Pandas Dataframe with (un)correlated columns.
    :param expected_val: Expected value for correlation.
    """
    correlation_matrix = compute_cramers_correlation_matrix(matrix)
    cols = list(matrix.columns)
    # for all non-diagonal entries, check the expected value
    for col1, col2 in combinations(cols, 2):
        assert correlation_matrix.loc[col1, col2] == pytest.approx(
            expected_val, abs=0.05
        )


@pytest.mark.parametrize(
    "matrix",
    CORRELATED_SINGLE_DFS_CAT + CORRELATED_SINGLE_DFS_NUM,
)
def test_discretize_matrix(matrix: pd.DataFrame) -> None:
    """
    Tests to ensure that the number of unique options after discretizing a dataframe is
    less than or equal to the number of bins used.

    :param matrix: Pandas Dataframe.
    """
    disc_matrix = discretize_matrix(matrix, matrix.columns, nr_of_bins=BINS)
    value_counts = disc_matrix.nunique()
    assert all(col_value_counts <= BINS for col_value_counts in value_counts)


@pytest.mark.parametrize(
    "dataframe_1, dataframe_2, cols_cat, expected_value",
    add_arguments(CORRELATED_MULTIPLE_DFS_CAT, (CAT_COLS, 0.0))
    + add_arguments(CORRELATED_MULTIPLE_DFS_NUM, ([], 0.0)),
)
def test_bivariate_avg_abs_difference_cramers(
    dataframe_1: pd.DataFrame,
    dataframe_2: pd.DataFrame,
    cols_cat: list[str],
    expected_value: float,
) -> None:
    """
    Test to check that identically distributed dataframes have indeed an absolute average
    correlation difference of about 0.

    :param dataframe_1: Pandas Dataframe.
    :param dataframe_2: Pandas Dataframe.
    :param cols_cat: List of categorical columns.
    :param expected_value: Expected correlation between the provided dataframes.
    """
    abs_corr_diff = avg_abs_correlation_difference_cramers(
        dataframe_1, dataframe_2, cols_cat, BINS
    )
    assert abs_corr_diff == pytest.approx(expected_value, abs=0.005)


@pytest.mark.parametrize(
    "dataframe_1, dataframe_2, cols_cat, expected_value",
    add_arguments(CORRELATED_MULTIPLE_DFS_CAT, (CAT_COLS, 0.0))
    + add_arguments(CORRELATED_MULTIPLE_DFS_NUM, ([], 0.0)),
)
def test_bivariate_avg_abs_difference(
    dataframe_1: pd.DataFrame,
    dataframe_2: pd.DataFrame,
    cols_cat: list[str],
    expected_value: float,
) -> None:
    """
    Test to check that identically distributed dataframes have indeed an absolute average
    correlation difference of about 0.

    :param dataframe_1: Pandas Dataframe.
    :param dataframe_2: Pandas Dataframe.
    :param cols_cat: List of categorical columns.
    :param expected_value: Expected correlation between the provided dataframes.
    """
    abs_corr_diff = avg_abs_correlation_difference(
        dataframe_1, dataframe_2, cols_cat, BINS
    )
    assert abs_corr_diff == pytest.approx(expected_value, abs=0.005)
