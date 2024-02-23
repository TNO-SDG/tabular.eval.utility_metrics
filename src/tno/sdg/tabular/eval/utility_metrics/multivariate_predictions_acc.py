"""
Computation of the utility metric that compares the multivariate scores of the original and
synthetic data for classification tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.svm import SVC


def avg_abs_classification_accuracy_difference_svc(  # pylint: disable=too-many-arguments
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Sequence[str],
    n_bins: int = 50,
    frac_training: float = 0.7,
    random_state: int = 1,
) -> float:
    """
    Compute the average absolute difference between of the accuracies of classifications tasks
    performed with the original dataframe and accuracies of classifications tasks performed with
    the synthetic dataframe.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :param frac_training: The fraction of the dataset that is used for training the model. The rest
        is used for calculating the accuracy.
    :param random_state: The random seed that is used to split the dataset into training and
        testing.
    :return: The average absolute difference between the prediction accuracies.
    """

    original_data_train = original_data.sample(
        frac=frac_training, random_state=random_state
    )
    original_data_test = original_data.drop(original_data_train.index)
    accuracies_original = calculate_accuracies_scores(
        original_data_train, original_data_test, cols_cat, n_bins
    )

    synthetic_data_train = synthetic_data.sample(
        frac=frac_training, random_state=random_state
    )

    accuracies_synthetic = calculate_accuracies_scores(
        synthetic_data_train, original_data_test, cols_cat, n_bins
    )

    # Compute the average absolute difference of the accuracy scores
    absdiff_accuracy_average = np.mean(
        abs(np.subtract(accuracies_original, accuracies_synthetic))
    )

    return float(absdiff_accuracy_average)


def calculate_accuracies_scores(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols_cat: Iterable[str],
    n_bins: int,
) -> list[float]:
    """
    This function uses an SVM to predict each column based on the other columns in the dataframe.
    The accuracies for this classification task are then returned.
    df_train train represents the part of the dataframe used for training and df_test represents
    the part of the dataframe used for testing.

    :param df_train: A pandas dataframe that contains the training data.
    :param df_test: A pandas dataframe that contains the test data.
    :param cols_cat: An iterable containing the names of the categorical columns.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: A list with the accuracies of the SVM classification task for each column in the
        dataframe.
    """
    classification_scores = []

    # Obtain the prediction accuracies
    for col in df_train.columns:
        score = _compute_accuracy_score_for_column(
            df_train, df_test, col, cols_cat, n_bins
        )
        classification_scores.append(score)

    return classification_scores


def _compute_accuracy_score_for_column(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    cols_cat: Iterable[str],
    n_bins: int,
) -> Any:
    """
    Compute the prediction accuracy of a classification task for a specific column using an SVM.

    :param df_train: A pandas dataframe that contains the training data.
    :param df_test: A pandas dataframe that contains the test data.
    :param target_col: The name of the target column of the classification task.
    :param cols_cat: A list containing the names of the categorical columns.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :return: A list with the accuracies of classification tasks for each column in the dataframe.
    """
    x_train = pd.get_dummies(
        df_train.drop(target_col, axis=1),
        prefix_sep="__",
        columns=[col for col in cols_cat if col != target_col],
    )
    x_test = pd.get_dummies(
        df_test.drop(target_col, axis=1),
        prefix_sep="__",
        columns=[col for col in cols_cat if col != target_col],
    )
    # Add category column if it misses in X_train and X_test
    # (in case the category was not present in the test set)
    for col in x_train.columns:
        if col not in x_test.columns:
            x_test[col] = 0

    for col in x_test.columns:
        if col not in x_train.columns:
            x_train[col] = 0

    # Select target variable and discretize in case of a numerical column
    if target_col not in cols_cat:
        discretizer = KBinsDiscretizer(
            n_bins=n_bins, encode="ordinal", strategy="uniform"
        )

        df_train_vals: npt.NDArray[Any] = df_train[target_col].values  # type: ignore[assignment]
        y_train = pd.Series(
            np.concatenate(discretizer.fit_transform(df_train_vals.reshape(-1, 1))),
            index=x_train.index,
        )
        df_test_vals: npt.NDArray[Any] = df_test[target_col].values  # type: ignore[assignment]

        y_test = pd.Series(
            np.concatenate(discretizer.transform(df_test_vals.reshape(-1, 1))),
            index=x_test.index,
        )
    else:
        y_train = df_train[target_col]
        y_test = df_test[target_col]

    if y_train.nunique() == 1:
        # If the target variable is constant in the training set, a dummy classifier is used
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(x_train, y_train)
        logging.warning("There is only one class to predict.")
        return np.mean(dummy_clf.score(x_test, y_test))

    # Train a support vector classification (SVC)
    clf = SVC(kernel="poly")
    clf.fit(np.array(x_train), np.array(y_train))
    return np.mean(clf.score(np.array(x_test), np.array(y_test)))


def visualise_accuracies_scores(  # pylint: disable=too-many-arguments
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Sequence[str],
    n_bins: int = 50,
    frac_training: float = 0.7,
    random_state: int = 1,
) -> tuple[list[float], list[float], Any]:
    """
    Plot the accuracies scores of multivariate prediction models for the original and synthetic
    data.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :param n_bins: The number of bins that is used to discretise the numerical columns.
    :param frac_training: The fraction of the dataset that is used for training the model. The rest
        is used for calculating the accuracy.
    :param random_state: The random seed that is used to split the dataset into training and
        testing.
    :return: The prediction accuracies for the original and synthetic data and the plot.
    """
    # Split the original data in a train and test set
    original_data_train = original_data.sample(
        frac=frac_training, random_state=random_state
    )
    original_data_test = original_data.drop(original_data_train.index)
    # Sample from the synthetic dataset such that the training set is the same size
    synthetic_data_train = synthetic_data.sample(
        frac=frac_training, random_state=random_state
    )

    accuracies_original = calculate_accuracies_scores(
        original_data_train, original_data_test, cols_cat, n_bins
    )
    accuracies_synthetic = calculate_accuracies_scores(
        synthetic_data_train, original_data_test, cols_cat, n_bins
    )

    fig, ax = plt.subplots()
    ax.plot(
        original_data.columns,
        accuracies_original,
        label="Original",
        color="b",
        alpha=0.3,
    )
    ax.plot(
        original_data.columns,
        accuracies_synthetic,
        label="Synthetic",
        color="r",
        alpha=0.3,
    )
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.grid()
    plt.title("SVM classification accuracy per column")
    plt.ylabel("Accuracy")

    # Plot values in figure
    for a, b in zip(original_data.columns, accuracies_original):
        plt.text(a, b, str(np.round(b * 100, 0) / 100))
    for c, d in zip(original_data.columns, accuracies_synthetic):
        plt.text(c, d, str(np.round(d * 100, 0) / 100))

    plt.legend()
    plt.show()

    return accuracies_original, accuracies_synthetic, fig
