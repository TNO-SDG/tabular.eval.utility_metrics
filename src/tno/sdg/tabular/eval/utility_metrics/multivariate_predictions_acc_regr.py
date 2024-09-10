"""
Computation of the utility metric that compares the multivariate scores of the original and
synthetic data for regression and classification tasks.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, SVR


def avg_abs_prediction_differences_svm(  # pylint: disable=too-many-arguments
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Sequence[str],
    frac_training: float = 0.7,
    random_state: int = 1,
) -> float:
    """
    Compute the average absolute difference between of the accuracies of classifications and regression
    tasks performed with the original dataframe and accuracies of classifications and regression tasks
    performed with the synthetic dataframe.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :param frac_training: The fraction of the dataset that is used for training the model. The rest
        is used for calculating the accuracy and R-squared scores.
    :param random_state: The random seed that is used to split the dataset into training and
        testing.
    :return: The average absolute difference between the prediction accuracies.
    """

    original_data_train = original_data.sample(
        frac=frac_training, random_state=random_state
    )
    original_data_test = original_data.drop(original_data_train.index)
    predictions_original = calculate_prediction_scores(
        original_data_train, original_data_test, cols_cat
    )

    synthetic_data_train = synthetic_data.sample(
        frac=frac_training, random_state=random_state
    )

    predictions_synthetic = calculate_prediction_scores(
        synthetic_data_train, original_data_test, cols_cat
    )

    # Compute the average absolute difference of the prediction scores
    absdiff_prediction_average = np.mean(
        abs(np.subtract(predictions_original, predictions_synthetic))
    )

    return float(absdiff_prediction_average)


def calculate_prediction_scores(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    cols_cat: Iterable[str],
) -> list[float]:
    """
    This function uses an SVM to predict each column based on the other columns in the dataframe.
    The accuracies for this classification and regression task are then returned.
    df_train train represents the part of the dataframe used for training and df_test represents
    the part of the dataframe used for testing.

    :param df_train: A pandas dataframe that contains the training data.
    :param df_test: A pandas dataframe that contains the test data.
    :param cols_cat: An iterable containing the names of the categorical columns.
    :return: A list with the accuracies of the SVM prediction tasks for each column in the
        dataframe.
    """
    prediction_scores = []

    # Obtain the prediction accuracies for classification and regression tasks.
    for col in df_train.columns:
        score = _compute_prediction_score_for_column(df_train, df_test, col, cols_cat)
        # If the R2 score for a regression task is below zero, set to zero to indicate
        # no predictive power and in coherence with other utility measures between 0 and 1.
        if score < 0:
            score = 0
            logging.warning(
                "The R-squared score is below zero and therefore set to zero."
            )
        # Generate list of all scores
        prediction_scores.append(score)

    return prediction_scores


def _compute_prediction_score_for_column(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    cols_cat: Iterable[str],
) -> Any:
    """
    Compute the prediction accuracy of a classification or regression task for a specific column
    using an SVM.

    :param df_train: A pandas dataframe that contains the training data.
    :param df_test: A pandas dataframe that contains the test data.
    :param target_col: The name of the target column of the classification task.
    :param cols_cat: A list containing the names of the categorical columns.
    :return: A list with the accuracies of classification and regression tasks for each column in
    the dataframe.
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
    if target_col in cols_cat:
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
    else:
        y_train = df_train[target_col]
        y_test = df_test[target_col]

        # Train a support vector regression (SVR)
        clf = SVR(kernel="rbf")
        clf.fit(np.array(x_train), np.array(y_train))

    return np.mean(clf.score(np.array(x_test), np.array(y_test)))


def visualise_prediction_scores(  # pylint: disable=too-many-arguments
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Sequence[str],
    frac_training: float = 0.7,
    random_state: int = 1,
) -> tuple[list[float], list[float], Any]:
    """
    Plot the accuracies scores of multivariate prediction models for the original and synthetic
    data. Showing R-squared and accuracies in one plot.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :param frac_training: The fraction of the dataset that is used for training the model. The rest
        is used for calculating the accuracy and R-squared scores.
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

    accuracies_original = calculate_prediction_scores(
        original_data_train, original_data_test, cols_cat
    )
    accuracies_synthetic = calculate_prediction_scores(
        synthetic_data_train, original_data_test, cols_cat
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
    plt.title("SVC and SVR for categorical and numerical variables")
    plt.ylabel("Accuracy and R-squared")

    # Plot values in figure
    for a, b in zip(original_data.columns, accuracies_original):
        plt.text(a, b, str(np.round(b * 100, 0) / 100))
    for c, d in zip(original_data.columns, accuracies_synthetic):
        plt.text(c, d, str(np.round(d * 100, 0) / 100))

    plt.legend()
    plt.show()

    return accuracies_original, accuracies_synthetic, fig


def visualise_regression_scores(  # pylint: disable=too-many-arguments
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Sequence[str],
    frac_training: float = 0.7,
    random_state: int = 1,
) -> tuple[list[float], list[float], Any]:
    """
    Plot the scores of multivariate regression models for the original and synthetic data.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :param frac_training: The fraction of the dataset that is used for training the model. The rest
        is used for calculating the accuracy and R-squared scores.
    :param random_state: The random seed that is used to split the dataset into training and
        testing.
    :return: The regression scores for the original and synthetic data and the plot.
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

    regression_original = calculate_prediction_scores(
        original_data_train, original_data_test, cols_cat
    )
    regression_synthetic = calculate_prediction_scores(
        synthetic_data_train, original_data_test, cols_cat
    )

    cols_num = set(original_data.columns) - set(cols_cat)
    index_num = np.array(
        [original_data.columns.get_loc(c) for c in cols_num if c in cols_num]
    )

    regression_original_num = pd.DataFrame(regression_original).iloc[index_num]
    regression_synthetic_num = pd.DataFrame(regression_synthetic).iloc[index_num]

    fig, ax = plt.subplots()
    ax.plot(
        original_data.columns[index_num],
        regression_original_num,
        label="Original",
        color="b",
        alpha=0.3,
    )
    ax.plot(
        original_data.columns[index_num],
        regression_synthetic_num,
        label="Synthetic",
        color="r",
        alpha=0.3,
    )
    plt.grid()
    plt.title("SVM regression score per numerical variable")
    plt.ylabel("R-squared")

    # Plot values in figure
    for a, b in zip(original_data.columns[index_num], regression_original_num):
        plt.text(a, b, f"{b:0%}")
    for c, d in zip(original_data.columns[index_num], regression_synthetic_num):
        plt.text(c, d, f"{d:0%}")

    plt.legend()
    plt.show()
    return regression_original, regression_synthetic, fig
