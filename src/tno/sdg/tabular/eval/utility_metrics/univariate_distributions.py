"""
Computation of the utility metric that compares the univariate distributions of the original and
synthetic data.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.preprocessing import KBinsDiscretizer


def average_hellinger_distance(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Iterable[Any],
    n_bins: int = 50,
) -> float:
    """
    Compute the average Hellinger distance between an original and synthetic dataframe.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A sequence of columns in the dataframes that are categorical.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: The average Hellinger distance.
    """
    hellinger_distance_sum = 0.0

    # Compute the Hellinger distance for each column in the dataframe
    for col in original_data.columns:
        if col in cols_cat:
            categories = np.unique(original_data[col])
            hellinger_distance = _process_and_compare_categorical(
                original_data[col].to_numpy(),
                synthetic_data[col].to_numpy(),
                categories=categories,
            )
        else:
            hellinger_distance = _process_and_compare_numerical(
                original_data[col].to_numpy(),
                synthetic_data[col].to_numpy(),
                nr_of_bins=n_bins,
            )
        hellinger_distance_sum += hellinger_distance

    hellinger_distance_average = hellinger_distance_sum / len(original_data.columns)

    return hellinger_distance_average


def discretize_vector(
    vector: npt.NDArray[Any],
    nr_of_bins: int,
) -> npt.NDArray[Any]:
    """
    Discretize a vector based on a number of bins.

    :param vector: Input vector.
    :param nr_of_bins: Number of bins to use.
    :return: The discretized vector.
    """
    discretizer = KBinsDiscretizer(
        n_bins=nr_of_bins, encode="ordinal", strategy="uniform"
    )
    return discretizer.fit_transform(vector.reshape(-1, 1)).reshape(1, -1)[0]  # type: ignore[no-any-return,union-attr]


def discrete_vector_to_distribution(
    vector: npt.NDArray[Any], categories: Sequence[Any] | None = None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """
    Return the distribution of discrete values in the vector. If no list of categories is provided,
    the set of unique values in the vector is taken as the list of categories.

    :param categories: List of categories that can appear in the vector.
    :param vector: Input vector.
    :return: An array of all the unique values and an array of the respective probabilities.
    """
    unique_vals, counts = np.unique(vector, return_counts=True)
    if categories is None:
        return unique_vals, counts / counts.sum()

    counts_filtered = np.array([0] * len(categories))
    for i, cat in enumerate(categories):
        counts_filtered[i] = counts[unique_vals == cat].sum()
    return np.array(categories), counts_filtered / counts_filtered.sum()


def compute_hellinger_distance(
    distribution_1: npt.NDArray[Any],
    distribution_2: npt.NDArray[Any],
) -> float:
    """
    Function that computes the Hellinger distance between two distributions.

    :param distribution_1: a vector of probabilities that sums to 1.
    :param distribution_2: a vector of probabilities that sums to 1.
    :return: The Hellinger Distance between the distributions.
    """
    return float(
        (1 / np.sqrt(2))
        * np.linalg.norm(np.sqrt(distribution_1) - np.sqrt(distribution_2))
    )


def _process_and_compare_numerical(
    vector_1: npt.NDArray[Any],
    vector_2: npt.NDArray[Any],
    nr_of_bins: int,
) -> float:
    """
    Compute the Hellinger distance between the discretized vectors.

    :param vector_1: Vector with numerical data.
    :param vector_2: Vector with numerical data.
    :param nr_of_bins: Number of bins to use for discretization. Use None if no discretization is
         needed.
    :return: Hellinger distance between the processed vectors.
    """
    final_vectors = discretize_vector(
        np.concatenate((vector_1, vector_2)), nr_of_bins=nr_of_bins
    )
    final_vector_1, final_vector_2 = np.split(final_vectors, 2)  # pylint: disable=W0632

    _, distribution_1 = discrete_vector_to_distribution(
        final_vector_1, categories=list(range(nr_of_bins))
    )
    _, distribution_2 = discrete_vector_to_distribution(
        final_vector_2, categories=list(range(nr_of_bins))
    )

    hellinger_distance = compute_hellinger_distance(distribution_1, distribution_2)
    return float(hellinger_distance)


def _process_and_compare_categorical(
    vector_1: npt.NDArray[Any], vector_2: npt.NDArray[Any], categories: Sequence[Any]
) -> Any:
    """
    Compute the Hellinger distance: a distance metric to quantify the similarity between two
    probability distributions. Distance between distributions will be a number in [0,1], where 0 is
    minimum distance (maximum similarity) and 1 is maximum distance (minimum similarity).

    :param vector_1: Column from the original dataframe.
    :param vector_2: Column from the synthetic dataframe.
    :param categories: The categories available in the data.
    :return: Hellinger distance between the processed vectors.
    """
    _, distribution_1 = discrete_vector_to_distribution(vector_1, categories=categories)
    _, distribution_2 = discrete_vector_to_distribution(vector_2, categories=categories)

    hellinger_distance = compute_hellinger_distance(distribution_1, distribution_2)

    return hellinger_distance


def visualise_distributions(  # pylint: disable=too-many-arguments,too-many-locals
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols: Sequence[str],
    cat_col_names: Sequence[str],
    n_bins: int = 50,
) -> dict[str, Figure]:
    """
    Plot the probability distributions of a column for the original and synthetic data.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols: List of the selected columns for which the distributions is plotted.
    :param cat_col_names: names of categorical columns
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: A dictionary mapping each column to an overview of the data in that column and
        a dictionary mapping each column to a respective plot.
    """
    # Distribution plot for the columns given in cols
    figures = {}

    for col in cols:
        vec1 = original_data[col]
        vec2 = synthetic_data[col]

        # If numerical/continues column:
        if col not in cat_col_names:
            # Discretize continuous variables
            discretizer = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="uniform"
            )

            vecs_discretized = discretizer.fit_transform(
                np.concatenate(
                    (vec1.values.reshape(-1, 1), vec2.values.reshape(-1, 1))  # type: ignore[union-attr]
                )
            )

            (  # pylint: disable=unbalanced-tuple-unpacking
                vec1_discretized,
                vec2_discretized,
            ) = np.array_split(vecs_discretized, 2)

            _, fig = plot_numerical_bar(vec1_discretized, vec2_discretized, col, n_bins)

        else:  # Else if categorical column
            _, fig = plot_categorical_bar(
                vec1.to_numpy(), vec2.to_numpy(), col, original_data, synthetic_data
            )

        figures[col] = fig

    return figures


def plot_numerical_bar(
    vec1_discretized: npt.NDArray[Any],
    vec2_discretized: npt.NDArray[Any],
    col: str,
    n_bins: int = 50,
) -> tuple[list[list[int]], Figure]:
    """
    Plot the distributions of a numerical column for the original and synthetic data.

    :param vec1_discretized: An array for discretised original data column.
    :param vec2_discretized: An array for discretised synthetic data column.
    :param col: Selected columns for which the distributions is plotted.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: The counts per dataset and a plot.
    """
    fig, ax = plt.subplots()

    counts, _, _ = ax.hist(
        [vec1_discretized.reshape(-1), vec2_discretized.reshape(-1)],
        bins=n_bins,
        density=False,  # Display counts
        color=["b", "r"],
        alpha=0.3,
        label=["Original dataset", "Synthetic dataset"],
    )
    # Add ext for labels, title and custom x-axis tick labels.
    ax.set_title("Distribution original dataset vs synthetic dataset for " + col)
    ax.set_xlabel(str(col))
    ax.set_ylabel("Counts")
    ax.legend(loc="upper right", ncols=2)
    ax.legend()
    return counts, fig


def plot_categorical_bar(
    vec1: npt.NDArray[Any],
    vec2: npt.NDArray[Any],
    col: str,
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> tuple[dict[str, float], Figure]:
    """
    Plot the distributions of a categorical column for the original and synthetic data.

    :param vec1: An array for discretized original data column.
    :param vec2: An array for discretized synthetic data column.
    :param col: Selected column for which the distributions is plotted.
    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :return: A dictionary mapping each counts per dataset and
        a plot.
    """
    fig_counts, ax = plt.subplots()
    # Used to calculate counts in equal order
    n_categories = original_data[col].nunique()
    counts, _, _ = ax.hist(
        [vec1, vec2],
        bins=n_categories,
        density=False,
        label=["Original dataset", "Synthetic dataset"],
    )
    plt.close(fig_counts)

    categories = original_data[col].unique()
    counts_categories: dict[str, float] = {
        "Original": counts[0] / len(original_data),
        "Synthetic": counts[1] / len(synthetic_data),
    }

    x = np.arange(len(categories))  # the label locations
    width = 0.33  # the width of the bars set for two datasets
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for attribute, measurement in counts_categories.items():
        offset = width * multiplier
        color = ["k"]
        if attribute == "Original":
            color = ["b"]
        if attribute == "Synthetic":
            color = ["r"]
        rects = ax.bar(
            x + offset, measurement, width, color=color, label=attribute, alpha=0.3
        )
        ax.bar_label(rects, padding=1)

        multiplier += 1

    # Add ext for labels, title and custom x-axis tick labels.
    ax.set_ylabel("Density")
    ax.set_title("Distribution original dataset vs synthetic dataset for " + col)
    ax.set_xlabel("Categories")
    ax.set_xticks(x + width, categories)
    ax.legend(loc="upper right", ncols=2)
    ax.set_ylim(0, 1)
    ax.set_title("Distribution original dataset vs synthetic dataset for " + col)

    return counts, fig
