"""
This script produces the figure for the logistic regression examples without
Stein updates in van den Boom, Cremaschi and Thiery (2024, arXiv:2404.18556).
"""
import pickle
import time
import urllib.request

import IPython.core.magics.execution as IPy_exec
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
import pandas as pd
import rpy2.robjects.numpy2ri
import rpy2.robjects.packages as rpackages
import scipy.optimize

import dais


# We use a function from the R package `discretization`.
utils = rpackages.importr("utils")

if not rpackages.isinstalled("discretization"):
    utils.install_packages(
        "discretization", repos="https://cloud.r-project.org"
    )

discretization = rpackages.importr("discretization")
rpy2.robjects.numpy2ri.activate() # Enable passing NumPy arrays to R.


def time_init():
    """Initialize CPU and wall times."""
    return dict(t_CPU = time.process_time(), t_wall = time.perf_counter())


def time_print(times):
    """Print elapsed CPU and wall times since `times`."""
    print("CPU time :", IPy_exec._format_time(
        time.process_time() - times["t_CPU"]
    ))

    print("Wall time:", IPy_exec._format_time(
        time.perf_counter() - times["t_wall"]
    ))


@jax.vmap
def log1pexp(arg):
    """
    log1pexp, also known as the softplus function

    Evaluate log(1 + exp(arg)) per Equation 10 from
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
    """
    # We don't use if/else as JAX cannot vectorize that.
    # This code is suboptimal in that it calculates both possible return values
    # regardless of whether `arg < 33.3`.
    return np.where(
        arg < 33.3,
        np.log1p(np.exp(arg)),
        arg
    )


data_sets = pickle.load(open("regression.p", "rb"))  # Load HMC results.

for data_name in data_sets.keys():
    print("\nProcessing the", data_name, "data...")
    
    tmp = onp.genfromtxt(
        urllib.request.urlopen(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                + data_sets[data_name]["url"] + ".data"
        ),
        dtype=str, delimiter=","
    )

    if data_name == "spam":
        y = tmp[:, -1] == "0"
        tmp = tmp[:, :-1].astype(float)

        tmp = onp.asarray(discretization.mdlp(onp.hstack(
            (tmp, y[:, onp.newaxis])
        ))[1])[:, :-1].astype(int)
    elif data_name == "krkp":
        y = tmp[:, -1] == "won"
        tmp = tmp[:, :-1]
    elif data_name == "ionosphere":
        y = tmp[:, -1] == "g"
        tmp = tmp[:, :-1].astype(float)

        tmp = onp.hstack((
            tmp[:, :2], # The first two columns are already discretized.
            onp.asarray(discretization.mdlp(onp.hstack(
                (tmp[:, 2:], y[:, onp.newaxis])
            ))[1])[:, :-1]
        )).astype(int)
    elif data_name == "mushroom":
        y = tmp[:, 0] == "e"
        tmp = tmp[:, 1:]
    else:
        raise ValueError(
            "`data_name` is not one of 'spam', 'krkp', 'ionosphere' and " \
             + "'mushroom'."
        )
    
    # Transform y to a jax.numpy.array of -1s and 1s.
    y = np.asarray(2*y - 1)
    d_tmp = tmp.shape[1]
    
    # Ensure that the first category is the most common for dummy variable
    # encoding.
    for j in range(d_tmp):
        counts = onp.unique(tmp[:, j], return_counts=True)
        tmp[tmp[:, j] == counts[0][onp.argmax(counts[1])], j] = 0
    
    A = np.hstack([pd.get_dummies(
        tmp[:, j], drop_first=True
    ) for j in range(d_tmp)])

    n = len(y)
    A = np.hstack((np.ones((n, 1), dtype=int), A)) # Add intercept.
    d = A.shape[1]
    print("Number of covariates:", d)
    
    
    def log_likelihood(x):
        """Log-likelihood of logistic regression"""
        return -np.sum(log1pexp(- A@x * y))
    
    
    def U_scalar(x):
        """Log posterior with a N(0, 10I) prior up to a constant"""
        return log_likelihood(x) - np.sum(x * x) / 20.0
    

    @jax.jit
    def neg_log_dens(x):
        """Negative log posterior using `jax.numpy`."""
        return -U_scalar(x)


    grad_neg_log_dens = jax.jit(jax.grad(neg_log_dens))
    
    print("\nComputing initial values...")
    times = time_init()
    
    MAP = scipy.optimize.minimize(
        fun=neg_log_dens, x0=np.zeros(d), jac=grad_neg_log_dens
    )['x']

    Σ_init = np.linalg.inv(jax.hessian(neg_log_dens)(MAP))
    time_print(times)
    
    print("\nRunning doubly adaptive importance sampling...")
    times = time_init()
    
    data_sets[data_name]["µ_DAIS"], data_sets[data_name]["Σ_DAIS"] = dais.DAIS(
        U_scalar=U_scalar, S=data_sets[data_name]["S"], S_eff_target=10**3,
        max_iter=50, µ=MAP, Σ=Σ_init, Stein_update=False
    )
    
    time_print(times)


# Creating the figure:
fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)

for i, data_name in enumerate(data_sets.keys()):
    def plot_results(HMC_vals, DAIS_vals, label, ax):
        ax.set_title(data_name.title() + " data")
        ax.set_aspect('equal')

        ax.plot(
            [0, 1], [0, 1], transform=ax.transAxes, color="red", linewidth=0.5
        )

        ax.scatter(
            HMC_vals, DAIS_vals, s=20, facecolors="none", edgecolors="black"
        )

        ax.set(
            xlabel="HMC posterior " + label, ylabel="DAIS posterior " + label
        )
    

    plot_results(
        HMC_vals=data_sets[data_name]["µ_MCMC"],
        DAIS_vals=data_sets[data_name]["µ_DAIS"], label="mean",
        ax=axes[int(i >= 2), 2*i % 4]
    )
    
    plot_results(
        HMC_vals=np.sqrt(np.diag(data_sets[data_name]["Σ_MCMC"])),
        DAIS_vals=np.sqrt(np.diag(data_sets[data_name]["Σ_DAIS"])),
        label="standard deviation", ax=axes[int(i >= 2), (2*i + 1) % 4]
    )

fig.savefig("regression_no_Stein.pdf")
