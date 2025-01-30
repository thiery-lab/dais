# dais

Repository with code used for the paper "Doubly Adaptive Importance Sampling"
by Willem van den Boom, Andrea Cremaschi and Alexandre H. Thiery
(2024, [arXiv:2404.18556](https://arxiv.org/abs/2404.18556)).


## Description of files

* [`dais/`](dais/) is a Python module that provides an implementation of doubly
  adaptive importance sampling (DAIS) as well as other methods used for
  comparison. Other Python scripts import it.

* [notebooks/control_variate.ipynb](notebooks/control_variate.ipynb) contains
  code for Figure 1 on the advantage of using Stein's identity.

* [notebooks/banana_mixture.ipynb](notebooks/banana_mixture.ipynb) produces the
  figures for the two-dimensional synthetic examples (Figure 2, S2 and S3).

* [`notebooks/regression.ipynb`](notebooks/regression.ipynb) produces the
  figures for the logistic regression examples (Figure 3, S4 to S7, and S9).
  It reads in data from [`data_regression/`](`data_regression/`). It takes a
  bit longer to run, especially when the amount of RAM is less than 16GB.

* [notebooks/convergence_banana.ipynb](notebooks/convergence_banana.ipynb) and
  [notebooks/convergence_gaussian.ipynb]
  (notebooks/convergence_gaussian.ipynb) contain code on monitoring of
  convergence (Figure S1).


* [`environment.yml`](environment.yml) can be used to [create a conda
  environment] similar to the one with which the code was tested on Ubuntu.
  Also, a suitable conda environment can be created as follows:
```
conda create -n dais -c conda-forge blackjax jupyter pandas papermill rpy2
```

Then, the results in the paper can be reproduced by running
```
conda activate dais
cd notebooks/
papermill control_variate.ipynb control_variate_output.ipynb --log-output
papermill banana_mixture.ipynb banana_mixture_output.ipynb --log-output
papermill regression.ipynb regression_output.ipynb --log-output
papermill convergence_banana.ipynb convergence_banana_output.ipynb --log-output
papermill convergence_gaussian.ipynb convergence_gaussian_output.ipynb --log-output
```

The code for the synthetic inverse problem (Figure S8) is available from the
[GitHub repository] for
[arXiv:2404.18556v1](https://arxiv.org/abs/2404.18556v1).


[create a conda environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
[GitHub repository]: https://github.com/thiery-lab/dais/tree/arXiv_2404.18556v1
