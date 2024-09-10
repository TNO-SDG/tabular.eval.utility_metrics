"""
Root imports for the tno.sdg.tabular.eval.utility_metrics package.
"""

# Explicit re-export of all functionalities, such that they can be imported properly. Following
# https://www.python.org/dev/peps/pep-0484/#stub-files and
# https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport

from .bivariate_correlations import (
    visualise_cramers_correlation_matrices as visualise_correlation_matrices,
)
from .distinguishability import (
    visualise_logistical_regression_auc as visualise_logistical_regression_auc,
)
from .distinguishability import (
    visualise_propensity_scores as visualise_propensity_scores,
)
from .multivariate_predictions_acc import (
    visualise_accuracies_scores as visualise_accuracies_scores,
)
from .multivariate_predictions_acc_regr import (
    visualise_prediction_scores as visualise_prediction_scores,
)
from .spiderplot import compute_all_metrics as compute_all_metrics
from .spiderplot import compute_spider_plot as compute_spider_plot
from .univariate_distributions import visualise_distributions as visualise_distributions

__version__ = "0.3.2"
