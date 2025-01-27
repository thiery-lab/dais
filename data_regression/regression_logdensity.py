import jax
import jax.numpy as jnp
import os


def make_regression_logdensity(data_name):
    """ define the log density for the logistic regression problem
    
    input:
    =====
    data_name: str, name of the data set \in {"spam", "krkp", "ionosphere", "mushroom"}
    
    output:
    ======
    log_dens: function, log density of the logistic regression problem
    dim_covariates: int, dimension of the covariates
    """
    error_msg = "data_name not in the list of data sets"
    assert data_name in ["spam", "krkp", "ionosphere", "mushroom"], error_msg

    # folder of this module
    __thisdir__ = os.path.dirname(os.path.abspath(__file__))
    data_folder = __thisdir__
    data_file = os.path.join(data_folder, data_name + ".npz")
    regression_data = jnp.load(data_file)
    A = regression_data["X"]
    y = regression_data["y"]
    dim_covariates = A.shape[1]

    # define the softplus function
    @jax.vmap
    def log1pexp(arg):
        """
        log1pexp, also known as the softplus function
        Evaluate log(1 + exp(arg)) per Equation 10 from
        https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
        """
        return jnp.where(arg < 33.3, jnp.log1p(jnp.exp(arg)), arg)

    # define the log likelihood
    def log_likelihood(x):
        """Log-likelihood of logistic regression"""
        return -jnp.sum(log1pexp(- A@x * y))

    # define the log posterior up to a constant
    def log_dens(x):
        """Log posterior with a N(0, 10I) prior up to a constant"""
        variance_x = 10.
        log_prior = -0.5*jnp.sum(x * x) / variance_x
        return log_likelihood(x) + log_prior

    return log_dens, dim_covariates