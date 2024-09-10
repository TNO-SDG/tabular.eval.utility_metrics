# TNO PET Lab - Synthetic Data Generation (SDG) - Tabular - Evaluation - Utility Metrics

Extensive evaluation of the utility of synthetic data sets. The original and synthetic data are compared on distinguishability and on a univariate, bivariate and multivariate level. All four metrics are visualized in one plot with a spiderplot. Where one equals 'complete overlap' and zero equals 'no overlap' between original and synthetic data. This plot can depict multiple synthetic data sets. Therefore it can be used to evaluate different levels of privacy protection in synthetic data sets, varying parameter settings in synthetic data generators, or completely different synthetic data generators.

All individual metrics depicted in the spiderplot can be visualized as well. The example_script.py shows you step by step how to generate all visualizations. The main functionalities of the scripts are:
- Univariate distributions: shows the distributions of one variable for the original and synthetic data.
- Bivariate correlations: visualizes a Pearson-r correlation matrix for all variables.
- Multivariate predictions: shows an SVM classifier predicts accuracies for each variable training on either original or synthetic data tested on original data.
- Distinguishability: shows the AUC of a logistic classifier that classifies samples as either original or synthetic.
- Spiderplot: generates spiderplot for these four metrics.

Note that any required pre-processing of the (synthetic) data sets should be done prior. Take into account addressing NANs, missing values, outliers and scaling the data.

For more information on the selected metrics, please refer to the paper (link will be added upon publication) or contact madelon.molhoek@tno.nl. As we aim to keep developing our code feedback and tips are welcome.

![Utility depicted in spider plot for adult data set, for different values of epsilon. Data are generated with CTGAN and can be found in scripts/datasets.](https://raw.githubusercontent.com/TNO-SDG/tabular.eval.utility_metrics/main/scripts/spider_plot.png)

### PET Lab

The TNO PET Lab consists of generic software components, procedures, and functionalities developed and maintained on a regular basis to facilitate and aid in the development of PET solutions. The lab is a cross-project initiative allowing us to integrate and reuse previously developed PET functionalities to boost the development of new protocols and solutions.

The package `tno.sdg.tabular.eval.utility_metrics` is part of the [TNO Python Toolbox](https://github.com/TNO-PET).

_Limitations in (end-)use: the content of this software package may solely be used for applications that comply with international export control laws._  
_This implementation of cryptographic software has not been audited. Use at your own risk._

## Documentation

Documentation of the `tno.sdg.tabular.eval.utility_metrics` package can be found
[here](https://docs.pet.tno.nl/sdg/tabular/eval/utility_metrics/0.3.2).

## Install

Easily install the `tno.sdg.tabular.eval.utility_metrics` package using `pip`:

```console
$ python -m pip install tno.sdg.tabular.eval.utility_metrics
```

_Note:_ If you are cloning the repository and wish to edit the source code, be
sure to install the package in editable mode:

```console
$ python -m pip install -e 'tno.sdg.tabular.eval.utility_metrics'
```

If you wish to run the tests you can use:

```console
$ python -m pip install 'tno.sdg.tabular.eval.utility_metrics[tests]'
```

## Usage

See the script in the `scripts` directory.
