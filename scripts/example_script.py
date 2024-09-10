"""
This script evaluates utility of synthetic data. We evaluate univariate,
bivariate, multivariate utility and distinguishability (record-level utility).
All four dimensions are depicted in a radar chart/spiderplot. All the required computations
are performed with the function compute_all. Multiple synthetic data sets can
be used as input.
"""

import os
from pathlib import Path

import pandas as pd

from tno.sdg.tabular.eval.utility_metrics import (
    compute_all_metrics,
    compute_spider_plot,
    visualise_accuracies_scores,
    visualise_correlation_matrices,
    visualise_distributions,
    visualise_logistical_regression_auc,
    visualise_prediction_scores,
    visualise_propensity_scores,
)

if __name__ == "__main__":
    # Load example data
    path_home = Path(os.path.dirname(os.path.realpath(__file__)))
    path_data = path_home / "datasets"
    original_data = pd.read_csv(path_data / "original_dataset.csv", index_col=0)
    synthetic_data_1 = pd.read_csv(path_data / "synthetic_dataset_1.csv", index_col=0)
    synthetic_data_2 = pd.read_csv(path_data / "synthetic_dataset_2.csv", index_col=0)

    # Specify numerical and categorical columns
    numerical_column_names = ["col1", "col2"]
    categorical_column_names = ["col3", "col4"]

    # Generate folder to save figures to
    path_plots = Path("./plots")
    if not (os.path.exists(path_plots)):
        os.mkdir(path_plots)

    #################################Spiderplot#############################################################
    ### Evaluation using four utility metrics generating a spiderplot
    computed_metrics = compute_all_metrics(
        original_data,
        {"Dataset 1": synthetic_data_1, "Dataset 2": synthetic_data_2},
        categorical_column_names,
    )

    spider_plot = compute_spider_plot(computed_metrics)
    spider_plot.write_html(file=path_plots / "spiderplot.html", auto_open=True)

    #################################Univariate##############################################################
    ### Plotting univariate distributions
    figures = visualise_distributions(
        original_data,
        synthetic_data_1,
        ["col1", "col2", "col3", "col4"],
        cat_col_names=categorical_column_names,
    )
    for col, fig in figures.items():
        fig.savefig(
            path_plots / f"distribution_{col}.png", dpi=300, bbox_inches="tight"
        )

    #################################Bivariate###############################################################
    ### Plotting correlations between columns
    (
        correlation_matrix_original,
        correlation_matrix_synthetic,
        cor_fig,
    ) = visualise_correlation_matrices(
        original_data, synthetic_data_1, categorical_column_names
    )
    cor_fig.savefig(
        path_plots / "correlation_matrices.png", dpi=300, bbox_inches="tight"
    )

    #################################Multivariate##########################################################
    ### Plotting classification tasks for all columns, converting numerical columns to categories
    (
        accuracies_scores_original,
        accuracies_score_synthetic,
        pred_fig,
    ) = visualise_accuracies_scores(
        original_data, synthetic_data_1, categorical_column_names
    )
    pred_fig.savefig(
        path_plots / "svm_classification.png", dpi=300, bbox_inches="tight"
    )

    ### Plotting regression and classification task for numerical and categorical columns
    (
        prediction_scores_original,
        prediction_score_synthetic,
        pred_fig_regr,
    ) = visualise_prediction_scores(
        original_data, synthetic_data_1, categorical_column_names
    )
    pred_fig_regr.savefig(
        path_plots / "svm_regression_classification.png", dpi=300, bbox_inches="tight"
    )

    #################################Distinguishability#####################################################
    ### Plotting distinguishability by classifying Synthetic and Original samples with propensity score

    propensity_scores, prop_fig = visualise_propensity_scores(
        original_data, synthetic_data_1, categorical_column_names
    )
    prop_fig.savefig(path_plots / "propensity_scores.png", dpi=300, bbox_inches="tight")

    # Plotting Area Under Curve for False Positive and True Positive rates for classifying synthetic and
    # original samples
    predictions, utility_disth, auc_fig = visualise_logistical_regression_auc(
        original_data, synthetic_data_1, categorical_column_names
    )
    auc_fig.savefig(path_plots / "AUC_curves.png", dpi=300, bbox_inches="tight")

    #################################The_END###############################################################
