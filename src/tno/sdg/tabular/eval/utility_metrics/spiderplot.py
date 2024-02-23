"""
Module that contains the main interface for users to test the quality of synthetic datasets.
Computation of all four metrics to visualize in one spiderplot.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go

from tno.sdg.tabular.eval.utility_metrics.bivariate_correlations import (
    avg_abs_correlation_difference,
)
from tno.sdg.tabular.eval.utility_metrics.distinguishability import (
    logistical_regression_auc,
)
from tno.sdg.tabular.eval.utility_metrics.multivariate_predictions_acc import (
    avg_abs_classification_accuracy_difference_svc,
)
from tno.sdg.tabular.eval.utility_metrics.univariate_distributions import (
    average_hellinger_distance,
)


def compute_all_metrics(
    original_data: pd.DataFrame,
    synthetic_datasets: dict[Any, Any],
    cols_cat: Sequence[str],
    n_bins: int = 50,
    frac_training: float = 0.7,
    random_state: int = 1,
) -> dict[Any, Any]:
    """
    Computation of the utility metrics.

    :param original_data: A pandas dataframe that contains the original data
    :param synthetic_datasets: A dictionary containing the synthetic data pandas dataframes. The keys specify
                                the names of the datasets. The order and names of the columns should be the
                                same as of the original dataframe
    :param cols_cat: A list containing the names of the categorical columns
    :param n_bins: The number of bins that is used to discretise the numerical columns
    :param frac_training: The fraction of the dataset that is used for training the model. The rest is used for
                                calculating the accuracy and R-squared scores
    :param random_state: The random seed that is used to split the dataset into training and testing
    :return: Dictionary containing the values for the utility metrics
    """

    metrics_dict = {}
    for key, synthetic_data in synthetic_datasets.items():
        # Compute average Hellinger Distance
        univariate_metric = average_hellinger_distance(
            original_data, synthetic_data, cols_cat
        )
        logging.info("Univariate done")

        # Compute average absolute difference between the correlations
        bivariate_metric = avg_abs_correlation_difference(
            original_data, synthetic_data, cols_cat
        )
        logging.info("Bivariate done")

        # Compute average absolute difference between the prediction accuracies
        multivariate_metric = avg_abs_classification_accuracy_difference_svc(
            original_data,
            synthetic_data,
            cols_cat,
            n_bins,
            frac_training,
            random_state,
        )

        logging.info("Multivariate done")

        # Compute difference between the AUC of the FPR and TPR
        _, distinguishability_metric = logistical_regression_auc(
            original_data, synthetic_data, cols_cat
        )
        logging.info("Distinguishability done")

        metrics_dict[key] = {
            "univariate_distributions": univariate_metric,
            "bivariate_distributions": bivariate_metric,
            "multivariate_distributions": multivariate_metric,
            "distinguishability": distinguishability_metric,
        }

    return metrics_dict


def compute_spider_plot(metrics_results: dict[Any, Any]) -> Any:
    """
    Function that visualizes the four main metrics of how good synthetic data is using a
    spider plot.

    :param metrics_results: A dictionary that maps synthetic data set names to an overview of metric
        values. Each overview of metrics contains an entry for each key in METRIC_NAMES.
    :return: A Figure object representing the spider plot.
    """
    # Initiate plot
    fig = go.Figure()

    metric_names = [
        "Bivariate correlations",
        "Univariate distributions",
        "Distinguishability",
        "Multivariate predictions",
        "Bivariate correlations",
    ]

    for key, metrics in metrics_results.items():
        fig.add_trace(
            go.Scatterpolar(
                r=[
                    1 - metrics["bivariate_distributions"],
                    1 - metrics["univariate_distributions"],
                    metrics["distinguishability"],
                    1 - metrics["multivariate_distributions"],
                    1 - metrics["bivariate_distributions"],
                ],
                theta=metric_names,
                name=key,
            )
        )

    # Layout plot
    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 1.05],
            },
            "angularaxis": {
                "tickfont": {
                    "size": 20,
                },
            },
        },
        showlegend=True,
        title_text="Spiderplot",
        title_x=0.48,
        title_font_size=30,
    )
    return fig
