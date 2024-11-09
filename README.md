# dais

Repository with code used for the paper "Doubly Adaptive Importance Sampling"
by Willem van den Boom, Andrea Cremaschi and Alexandre H. Thiery
(2024, [arXiv:2404.18556](https://arxiv.org/abs/2404.18556)).


## Description of files

* [`dais.py`](dais.py) is a Python module that provides an implementation of
doubly adaptive importance sampling (DAIS). Other Python scripts import it.

* [`banana_and_mixture.py`](banana_and_mixture.py) produces the figure for the
two-dimensional synthetic examples.

* [`inverse_problem.py`](inverse_problem.py) produces the figure for the
synthetic inverse problem.

* [`regression.py`](regression.py) produces the figure for the logistic
regression examples. The results are saved which
[`regression_VI.py`](regression_VI.py) and
[`regression_no_Stein.py`](regression_no_Stein.py) load to produce the
comparison figures with VI and DAIS without updates based on Stein's identity,
respectively.

* The folder [illustrations](illustrations/) contains code for the figures used
in Section 2 to discuss the advantage of using Stein's identity and the
monitoring of convergence.

* [`environment.yml`](environment.yml) can be used to
[create a conda environment] with which the code was tested. Note that some of
the package versions are not available for Apple Silicon / osx-arm64. Instead,
an osx-64 environment can be created on Apple Silicon Macs using
`conda env create --platform osx-64 -f environment.yml`.


[create a conda environment]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file
