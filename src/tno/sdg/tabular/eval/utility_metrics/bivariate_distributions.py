"""
Computation of the utility metric that compares the bivariate distributions of the original and
synthetic data.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Container, Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
import seaborn as sns

from tno.sdg.tabular.eval.utility_metrics.univariate_distributions import (
    discretize_vector,
)


def avg_abs_correlation_difference(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Container[str],
    n_bins: int = 50,
) -> npt.NDArray[Any]:
    """
    Compute of the average absolute difference between the correlations in the original dataframe
    and the correlations in the synthetic dataframe.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: The average absolute difference between the correlations in the original dataframe and
        the correlations in the synthetic dataframe.
    """

    numerical_cols = [col for col in original_data.columns if col not in cols_cat]
    # Compute the correlation matrices
    corr_original = compute_cramers_correlation_matrix(
        discretize_matrix(original_data, numerical_cols, n_bins)
    )
    corr_synthetic = compute_cramers_correlation_matrix(
        discretize_matrix(synthetic_data, numerical_cols, n_bins)
    )

    # Compute the average absolute difference of the correlation indices
    abs_difference_corr: Any = abs(corr_original - corr_synthetic)
    absdiff_corr_average: npt.NDArray[Any] = np.mean(abs_difference_corr.values)

    return absdiff_corr_average


def discretize_matrix(
    data_frame: pd.DataFrame, cols_to_discretize: Iterable[Any], nr_of_bins: int
) -> pd.DataFrame:
    """
    Function that discretizes columns from a selection of columns

    :param data_frame: Data.
    :param cols_to_discretize: Columns to discretize.
    :param nr_of_bins: Number of bins to use for discretization.
    :return: The original dataframe with the columns to discretize discretized.
    """
    new_dataframe = data_frame.copy()
    for col in cols_to_discretize:
        new_dataframe[col] = discretize_vector(
            data_frame[col].to_numpy(), nr_of_bins=nr_of_bins
        )
    return new_dataframe


def compute_cramers_correlation_matrix(
    data_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute of the correlation matrix of a dataframe with discrete columns.

    :param data_frame: A discrete pandas dataframe that contains the data.
    :return: The correlation matrix of the dataframe.
    """
    # Create empty dataframe to save the correlations
    cols = data_frame.columns
    df_corr = pd.DataFrame(
        data=[(1 for _ in range(len(cols))) for i in range(len(cols))],
        columns=cols,
    )
    df_corr.set_index(pd.Index(cols), inplace=True)

    # Calculate the correlations for each pair of features
    for col1, col2 in combinations(cols, 2):
        correlation = compute_cramers_v(
            data_frame[col1].to_numpy(), data_frame[col2].to_numpy()
        )
        df_corr.loc[col1, col2] = correlation
        df_corr.loc[col2, col1] = correlation

    return df_corr


def compute_cramers_v(
    vector_1: npt.NDArray[Any],
    vector_2: npt.NDArray[Any],
) -> float:
    """
    Compute the Cramer's V correlation between two vectors.

    :param vector_1: First column from the dataframe.
    :param vector_2: Second column from the dataframe.
    :return: The Cramer's V correlation between vec1 and vec2.
    """

    # Create the contingency table and compute test statistic chi2
    crosstab = np.array(pd.crosstab(vector_1, vector_2))
    chi_squared = scipy.stats.chi2_contingency(crosstab, correction=False)[0]

    # Get the minimum value between cols and rows
    smallest_tab_dimension = min(crosstab.shape) - 1
    # Get the number of observations n
    crosstab_sum = np.sum(crosstab)

    # Compute correlation
    correlation: float = np.sqrt(chi_squared / (crosstab_sum * smallest_tab_dimension))
    return correlation


def visualise_cramers_correlation_matrices(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Container[str],
    n_bins: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Plot the correlations matrices of the original and synthetic data.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: The correlation matrices for the plots and the plot.
    """

    # Compute the correlation matrices
    disc_cols = [col for col in original_data.columns if col not in cols_cat]
    corr_original = compute_cramers_correlation_matrix(
        discretize_matrix(original_data, disc_cols, n_bins)
    )
    corr_synthetic = compute_cramers_correlation_matrix(
        discretize_matrix(synthetic_data, disc_cols, n_bins)
    )

    # Plot heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        corr_original,
        cmap="rocket_r",
        annot=True,
        annot_kws={"size": 6},
        fmt=".2f",
        vmin=-1,
        vmax=1,
        mask=corr_original.isnull(),
        ax=axes[0],
    ).set(title="Correlations original dataset")
    sns.heatmap(
        corr_synthetic,
        cmap="rocket_r",
        annot=True,
        annot_kws={"size": 6},
        fmt=".2f",
        vmin=-1,
        vmax=1,
        mask=corr_synthetic.isnull(),
        ax=axes[1],
    ).set(title="Correlations synthetic dataset")

    return corr_original, corr_synthetic, fig
