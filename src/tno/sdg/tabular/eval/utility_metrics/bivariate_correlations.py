"""
Computation of the utility metric that compares the bivariate correlations of the original and
synthetic data. For visualisation of correlations only Cramer's V is used. For spiderplot a
combination of numerical correlations Pearson's r, categorical correlations Cramer's V and numerical
to categorical correlation ANOVA are calculated and their difference between synthetic and
original data averaged.
"""

from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Container, Iterable, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm

from tno.sdg.tabular.eval.utility_metrics.univariate_distributions import (
    discretize_vector,
)


# pylint: disable=invalid-name
def avg_abs_correlation_difference(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Sequence[str],
    n_bins: int = 50,
) -> npt.NDArray[np.float64]:
    """
    Compute of the total weighted average absolute difference between the correlations in the original
    dataframe and the correlations in the synthetic dataframe. Using Pearson's R, Cramer's V and ANOVA methods.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: The names of the categorical columns.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: The average absolute difference between the correlations in the original dataframe and
        the correlations in the synthetic dataframe.
    """
    # For categorical columns Cramer's V correlation method to calculate average correlation difference:
    if cols_cat:
        og_cat = original_data[list(cols_cat)]
        syn_cat = synthetic_data[list(cols_cat)]
        U_cramer = avg_abs_correlation_difference_cramers(
            og_cat, syn_cat, cols_cat, n_bins
        )
    else:
        U_cramer = 0

    # For numerical columns Pearson's r correlation method to calculate average correlation difference:
    cols_num = [col for col in original_data.columns if col not in cols_cat]
    if cols_num:
        og_num = original_data[cols_num]
        syn_num = synthetic_data[cols_num]
        U_pearson = avg_abs_correlation_difference_pearson(og_num, syn_num)
    else:
        U_pearson = 0

    # For numerical and categorical columns ANOVA correlation method to calculate average correlation difference:
    if cols_cat and cols_num:
        U_anova = avg_abs_correlation_difference_anova(
            original_data, synthetic_data, cols_cat
        )
    else:
        U_anova = 0

    # Amount of numerical, categorical and combination correlations
    weight_pearson = (1 / 2) * (len(cols_num) * (len(cols_num) - 1))
    weight_cramer = (1 / 2) * (len(cols_cat) * (len(cols_cat) - 1))
    weight_anova = len(cols_cat) * len(cols_num)

    # Weighted contributions of correlation differences
    U_weighted_pearson = weight_pearson * U_pearson
    U_weighted_cramer = weight_cramer * U_cramer
    U_weighted_anova = weight_anova * U_anova

    # Weighted total correlation difference
    U_weighted = (U_weighted_pearson + U_weighted_cramer + U_weighted_anova) / (
        weight_pearson + weight_cramer + weight_anova
    )
    return cast(npt.NDArray[np.float64], U_weighted)


def avg_abs_correlation_difference_cramers(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Container[str],
    n_bins: int = 50,
) -> float:
    """
    Compute of the average absolute difference between the correlations in the original dataframe
    and the correlations in the synthetic dataframe with Cramer's V method for categorical columns.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: The names of the categorical columns.
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
    abs_difference_corr = (corr_original - corr_synthetic).abs()
    absdiff_corr_cramers_average = cast(float, np.mean(abs_difference_corr.values))
    absdiff_corr_cramers_average = (
        absdiff_corr_cramers_average * 2
    )  # Only lower triangular elements

    return absdiff_corr_cramers_average


def discretize_matrix(
    data_frame: pd.DataFrame, cols_to_discretize: Iterable[Any], nr_of_bins: int
) -> pd.DataFrame:
    """
    Function that discretises columns from a selection of columns

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

    # In case there is only one category in the vector, set the correlation to 0.
    if len(np.unique(vector_1)) == 1:
        correlation = 0
        logging.warning("All the same categories in vector. Correlation set to zero.")
    if len(np.unique(vector_2)) == 1:
        correlation = 0
        logging.warning("All the same categories in vector. Correlation set to zero.")

    return correlation


def avg_abs_correlation_difference_pearson(
    original_data_num: pd.DataFrame,
    synthetic_data_num: pd.DataFrame,
) -> float:
    """
    Compute of the average absolute difference between the correlations in the original dataframe
    and the correlations in the synthetic dataframe with the Pearson's r coefficient for numerical
    columns.

    :param original_data_num: A pandas dataframe that contains the numerical columns of original data.
    :param synthetic_data_num: A pandas dataframe that contains the numerical columns of synthetic data.
    :return: The average absolute difference between the correlations in the original dataframe and
        the correlations in the synthetic dataframe.
    """
    # Compute the correlation matrices
    corr_original = compute_pearson_correlation_matrix(original_data_num)
    corr_synthetic = compute_pearson_correlation_matrix(synthetic_data_num)

    # Compute the average absolute difference of the correlation indices
    abs_difference_corr = (corr_original - corr_synthetic).abs()
    absdiff_corr_pearson_average = cast(float, np.mean(abs_difference_corr.values))

    absdiff_corr_pearson_average = (
        absdiff_corr_pearson_average * 2
    )  # Only lower triangular elements
    absdiff_corr_pearson_average = (
        absdiff_corr_pearson_average * 1 / 2
    )  # Absolute difference can be 2 maximum, therefore divide by 2

    return absdiff_corr_pearson_average


def compute_pearson_correlation_matrix(
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
        correlation = compute_pearson_rho(
            data_frame[col1].to_numpy(), data_frame[col2].to_numpy()
        )
        df_corr.loc[col1, col2] = correlation
        df_corr.loc[col2, col1] = correlation

    return df_corr


def compute_pearson_rho(
    vector_1: npt.NDArray[Any],
    vector_2: npt.NDArray[Any],
) -> float:
    """
    Compute the Pearson's r correlation between two vectors.

    :param vector_1: First column from the dataframe.
    :param vector_2: Second column from the dataframe.
    :return: The Pearson's r correlation between vec1 and vec2.
    """
    # Compute correlation between vectors
    correlation_matrix_pearson = np.corrcoef(vector_1, vector_2)

    # Coefficient is in lower left and upper right
    return cast(float, correlation_matrix_pearson[0, 1])


def avg_abs_correlation_difference_anova(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, cols_cat: Iterable[str]
) -> float:
    """
    Compute of the average absolute difference between the correlations in the original dataframe
    and the correlations in the synthetic dataframe with ANOVA method between unordered /categorical
    and ordered/numerical columns.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: The names of the categorical columns.
    :return: The average absolute difference between the correlations in the original dataframe and
            the correlations in the synthetic dataframe.
    """
    # Calculate ANOVA correlation coefficient
    anova_original = compute_anova_correlation(original_data, cols_cat)
    anova_synthetic = compute_anova_correlation(synthetic_data, cols_cat)

    # Calculate difference between correlations
    abs_difference_corr: npt.NDArray[np.float64] = abs(
        np.array(anova_original) - np.array(anova_synthetic)
    )
    absdiff_corr_anova_average = cast(float, np.mean(abs_difference_corr))
    return absdiff_corr_anova_average


def compute_anova_correlation(
    data_frame: pd.DataFrame,
    cols_cat: Iterable[str],
) -> list[float]:
    """
    Compute ANOVA correlation of a dataframe for the unordered/categorical feature to the
    ordered/numerical feature.

    :param data_frame: A pandas dataframe that contains the data.
    :param cols_cat: The names of the categorical columns.
    :return: The results of the ANOVA correlation test of the dataframe.
    """
    cols_num = [col for col in data_frame.columns if col not in cols_cat]

    # Calculate correlations between first ordered and then unordered columns
    dict_anova = []
    ANOVA_association_tot = 0
    for num_col in cols_num:  # Ordered feature predicted by the unordered feature
        for cat_col in cols_cat:  # Unordered feature with k categories
            y = data_frame[num_col]
            z_cat = data_frame[cat_col]
            # Create dummy variable per category to perform linear regression fit on
            z_dummies = pd.get_dummies(z_cat, drop_first=True, dtype=float)

            for dummy_name in z_dummies.columns:
                z = z_dummies[dummy_name]
                model = sm.OLS(y, z).fit()
                sse = np.sum((model.fittedvalues - y) ** 2)  # Sum of Squares Regression
                ssr = np.sum(
                    (model.fittedvalues - y.mean()) ** 2
                )  # Sum of Squares Error
                ANOVA_association = ssr / (ssr + sse)  # Association Score

                # In case there is no variation in the numerical or categorical column,
                # set correlation to zero.
                if len(np.unique(z)) == 1:
                    ANOVA_association = 0
                    logging.warning(
                        "Only one category in categorical column. Correlation is set to zero."
                    )
                if len(np.unique(y)) == 1:
                    ANOVA_association = 0
                    logging.warning(
                        "Only one unique value in numerical column. Correlation is set to zero."
                    )
                ANOVA_association_tot += ANOVA_association
            # Add average ANOVA score per column pare
            dict_anova.append(ANOVA_association_tot / len(z_dummies))
    return dict_anova


def visualise_cramers_correlation_matrices(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Container[str],
    n_bins: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """
    Plot the correlations matrices with Cramer's V of the original and synthetic data.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: The names of the categorical columns.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: The correlation matrices for the plots and the plot.
    """
    # Compute the correlation matrices with Cramer's V
    disc_cols = [col for col in original_data.columns if col not in cols_cat]
    corr_original = compute_cramers_correlation_matrix(
        discretize_matrix(original_data, disc_cols, n_bins)
    )
    corr_synthetic = compute_cramers_correlation_matrix(
        discretize_matrix(synthetic_data, disc_cols, n_bins)
    )

    # Plot the heatmaps for the correlations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        corr_original,
        cmap="rocket_r",
        annot=True,
        annot_kws={"size": 35 / np.sqrt(len(corr_original))},
        fmt=".2f",
        vmin=0,
        vmax=1,
        mask=corr_original.isnull(),
        ax=axes[0],
    ).set(title="Correlations original dataset")
    sns.heatmap(
        corr_synthetic,
        cmap="rocket_r",
        annot=True,
        annot_kws={"size": 35 / np.sqrt(len(corr_synthetic))},
        fmt=".2f",
        vmin=0,
        vmax=1,
        mask=corr_synthetic.isnull(),
        ax=axes[1],
    ).set(title="Correlations synthetic dataset")

    return corr_original, corr_synthetic, fig
