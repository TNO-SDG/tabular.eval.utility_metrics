"""
Computation of the utility metric that distinguishes the synthetic data from the original
data with a classification task.
"""

from __future__ import annotations

from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _logistical_regression_auc(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, cols_cat: Iterable[str]
) -> tuple[npt.NDArray[Any], float, npt.NDArray[Any]]:
    """
    Compute the Area Under the Curve for the false positive (FPR) and true positive (TPR) rates
    for the original and the synthetic data based on a classification task (logistic regression)
    to distinguish the two classes.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :return: Area Under the Curve of FPR, TPR, and test_y
    """
    # Create y variable (0 for original data and 1 for synthetic data)
    y_original = np.zeros(original_data.shape[0])
    y_synthetic = np.ones(synthetic_data.shape[0])
    y_coor = np.concatenate((y_original, y_synthetic))

    data_frame = pd.concat([original_data, synthetic_data])
    x_one_hot = pd.get_dummies(data_frame, prefix_sep="__", columns=list(cols_cat))

    # Split into train/test sets
    train_x, test_x, train_y, test_y = train_test_split(
        x_one_hot, y_coor, test_size=0.3, random_state=42
    )

    # Fit the model and predict the scores
    pipe = Pipeline([("scaler", StandardScaler()), ("logistic_classifier", lr())])
    pipe.fit(train_x, train_y)
    predictions = pipe.predict_proba(test_x)

    # Keep only positive outcome
    pred_pos = predictions[:, 1]

    # Calculate Utility distinguishability with AUC
    tot_auc = roc_auc_score(test_y, pred_pos)

    # When the classifier performs worse than a random classifier and the AUC is below 0.5
    # this indicates no distinguishability --> utility = 1. Therefor set to maximum 1.
    utility_disth = min(2 * (1 - tot_auc), 1)
    return predictions, utility_disth, test_y


def logistical_regression_auc(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, cols_cat: Iterable[str]
) -> tuple[npt.NDArray[Any], float]:
    """
    Compute the Area Under the Curve for the false positive (FPR) and true positive (TPR) rates
    for the original and the synthetic data based on a classification task (logistic regression)
    to distinguish the two classes.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :return: Area Under the Curve of FPR and TPR
    """
    predictions, utility_disth, _ = _logistical_regression_auc(
        original_data=original_data, synthetic_data=synthetic_data, cols_cat=cols_cat
    )
    return predictions, utility_disth


def visualise_logistical_regression_auc(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, cols_cat: Iterable[str]
) -> tuple[npt.NDArray[Any], float, Figure]:
    """
    Visualise the Area Under the Curve for the false positive (FPR) and true positive (TPR) rates
    for the original and the synthetic data based on a classification task (logistic regression)
    to distinguish the two classes.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :return: Area Under the Curve of FPR and TPR
    """
    predictions, utility_disth, test_y = _logistical_regression_auc(
        original_data=original_data, synthetic_data=synthetic_data, cols_cat=cols_cat
    )
    pred_pos = predictions[:, 1]
    tot_auc = roc_auc_score(test_y, pred_pos)

    # Define false positives and true positives
    pred_fpr, pred_tpr, _ = roc_curve(test_y, pred_pos)

    # Plot AUC
    fig, ax = plt.subplots()
    ax.plot(
        pred_fpr, pred_tpr, linestyle="-", label=(f"AUC={tot_auc:.3f}")
    )  # add auc number
    # Define sxis labels
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    # Add diagonal with slope 1 as a baseline for no distinguishability
    plt.axline(
        (0, 0), slope=1, linestyle="--", color="red", label="No distinguishability"
    )

    # Show the legend and add grid
    plt.legend()
    plt.grid()
    # show the plot
    plt.show()

    return predictions, utility_disth, fig


def mean_propensity_difference_logistical_regression(
    original_data: pd.DataFrame, synthetic_data: pd.DataFrame, cols_cat: Iterable[str]
) -> tuple[Any, Any]:
    """
    Compute the difference between the mean propensity score for the original dataframe and
    the mean propensity score for the synthetic dataframe based on a classification task (logistic
    regression) to distinguish the two classes.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :return: The difference between the mean propensity scores.
    """
    # Create y variable (0 for 'original' data and 1 for synthetic data)
    y_original = np.zeros(original_data.shape[0])
    y_synthetic = np.ones(synthetic_data.shape[0])
    y_coor = np.concatenate((y_original, y_synthetic))

    data_frame = pd.concat([original_data, synthetic_data])
    x_one_hot = pd.get_dummies(data_frame, prefix_sep="__", columns=list(cols_cat))

    # Fit the model and predict the scores
    pipe = Pipeline([("scaler", StandardScaler()), ("logistic_classifier", lr())])
    pipe.fit(x_one_hot, y_coor)
    predictions = pipe.predict_proba(x_one_hot)

    # Calculate the absolute difference between the mean propensity scores
    split = original_data.shape[0]

    predictions_orig = predictions[:, 1][:split]
    predictions_synth = predictions[:, 1][data_frame.shape[0] - split :]
    difference_mean_propensity_score = abs(
        np.mean(predictions_synth) - np.mean(predictions_orig)
    )

    return difference_mean_propensity_score, predictions


def visualise_propensity_scores(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    cols_cat: Iterable[str],
) -> tuple[npt.NDArray[Any], Any]:
    """
    Plot the distribution of the propensity scores for the original dataframe and for the
    synthetic dataframe based on a classification task to distinguish the two classes.

    :param original_data: A pandas dataframe that contains the original data.
    :param synthetic_data: A pandas dataframe that contains the synthetic data.
    :param cols_cat: A list containing the names of the categorical columns.
    :return: The predictions for the original and synthetic data and the plot.
    """
    # Create y variable (0 for 'original' data and 1 for synthetic data)
    y_original = np.zeros(original_data.shape[0])
    y_synthetic = np.ones(synthetic_data.shape[0])
    y_coor = np.concatenate((y_original, y_synthetic))

    _, predictions = mean_propensity_difference_logistical_regression(
        original_data, synthetic_data, cols_cat
    )

    # Plot the propensity scores
    fig, ax = plt.subplots()
    sns.kdeplot(x=predictions[:, 1], hue=y_coor, ax=ax).set(
        title="Propensity scores original and synthetic dataset"
    )
    ax.legend(labels=["Original dataset", "Synthetic dataset"])
    ax.set_xlim(0, 1)
    ax.set_xlim(0, 1)
    plt.grid()
    return predictions, fig
