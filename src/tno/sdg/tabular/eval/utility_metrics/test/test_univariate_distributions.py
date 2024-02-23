"""
Module that tests the functionality of univariate_distributions.py.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from tno.sdg.tabular.eval.utility_metrics.test.utils import (
    BINS,
    CAT_COLS,
    CATS_HALF,
    CORRELATED_MULTIPLE_DFS_CAT,
    CORRELATED_MULTIPLE_DFS_NUM,
    IDENTICAL_DISTRIBUTIONS,
    ORTHOGONAL_DISTRIBUTIONS,
    RANDOM_CAT_VECTORS,
    RANDOM_NUM_VECTORS,
    UNCORRELATED_MULTIPLE_DFS_CAT,
    UNCORRELATED_MULTIPLE_DFS_NUM,
    add_arguments,
)
from tno.sdg.tabular.eval.utility_metrics.univariate_distributions import (
    average_hellinger_distance,
    compute_hellinger_distance,
    discrete_vector_to_distribution,
    discretize_vector,
)


@pytest.mark.parametrize(
    "distribution_1, distribution_2, expected_val",
    add_arguments(ORTHOGONAL_DISTRIBUTIONS, 1.0)
    + add_arguments(IDENTICAL_DISTRIBUTIONS, 0.0),
)
def test_hellinger_distance_function(
    distribution_1: npt.NDArray[Any],
    distribution_2: npt.NDArray[Any],
    expected_val: float,
) -> None:
    """
    Test if the output has the expected properties of the Hellinger distance. Orthogonal
    distributions should have value 1, identical distributions should have value 0 and for all
    distributions, the value should be between 0 and 1.

    :param distribution_1: First distribution.
    :param distribution_2: Second distribution.
    :param expected_val: Expected Hellinger distance.
    """
    hellinger_distance = compute_hellinger_distance(distribution_1, distribution_2)
    assert hellinger_distance == pytest.approx(expected_val)


@pytest.mark.parametrize("vector", RANDOM_CAT_VECTORS)
def test_discrete_vector_to_distribution(vector: npt.NDArray[Any]) -> None:
    """
    Test if vector_to_distribution function correctly produces distributions by checking
    the properties of the result.

    :param vector: Input vector.
    """
    vals, distribution = discrete_vector_to_distribution(vector, categories=CATS_HALF)
    assert all(0 <= i <= 1 for i in distribution)
    assert sum(distribution) == pytest.approx(1)
    assert len(distribution) == len(CATS_HALF)
    assert list(vals) == CATS_HALF


@pytest.mark.parametrize("vector", RANDOM_NUM_VECTORS)
def test_discretize_vector(vector: npt.NDArray[Any]) -> None:
    """
    Test if the discretization function works properly by checking for the expected properties.

    :param vector: Input vector.
    """
    discretized_vector = discretize_vector(vector, nr_of_bins=BINS)
    vector_unique = np.unique(vector)
    disc_vector_unique = np.unique(discretized_vector)
    # The number of unique values in the discretised vector should be fewer or equal
    assert len(disc_vector_unique) <= len(vector_unique)

    min_val_index: int = int(np.argmin(vector))
    max_val_index: int = int(np.argmax(vector))

    # the smallest value should have been assigned to the first bin
    assert discretized_vector[min_val_index] == 0
    # the largest value should have been assigned to the last bin
    assert discretized_vector[max_val_index] == BINS - 1


@pytest.mark.parametrize(
    "dataframe_1, dataframe_2, cat_cols",
    add_arguments(CORRELATED_MULTIPLE_DFS_NUM, {})
    + add_arguments(UNCORRELATED_MULTIPLE_DFS_NUM, {})
    + add_arguments(CORRELATED_MULTIPLE_DFS_CAT, CAT_COLS)
    + add_arguments(UNCORRELATED_MULTIPLE_DFS_CAT, CAT_COLS),
)
def test_average_hellinger_distance(
    dataframe_1: pd.DataFrame,
    dataframe_2: pd.DataFrame,
    cat_cols: Sequence[Any],
) -> None:
    """
    Test whether the average Hellinger distance between of the columns of
    identically distributed dataframes is 0.

    :param dataframe_1: Pandas Dataframe.
    :param dataframe_2: Pandas Dataframe.
    :param cat_cols: A sequence of categorical columns in the dataframes.
    """
    avg_distance = average_hellinger_distance(
        dataframe_1, dataframe_2, cat_cols, n_bins=BINS
    )
    assert avg_distance == pytest.approx(0)
