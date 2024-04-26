import numpy as np
import scipy.linalg

# local imports
from util import psd


def sample_gaussian(size,           # int:       number of samples
                    mu,             # (dim,):    mean vector
                    Gamma,          # (dim,dim): covariance matrix
                    chol=np.nan,    # (dim,dim): lower-cholesky decomp of Gamma
                    ):
    """
    samples from a multivariate gaussian N(mu, Sigma)
    """
    assert mu.shape[0] == Gamma.shape[0]
    assert Gamma.shape[0] == Gamma.shape[1]

    # make sure chol is lower triangular if given
    if not np.all(np.isnan(chol)):
        assert np.allclose(chol, np.tril(chol)), "chol is not lower triangular"

    dim = len(mu)
    chol = np.where(np.isnan(chol),
                    scipy.linalg.cholesky(Gamma, lower=True),
                    chol)
    xi = np.random.normal(0, 1, size=(size, dim))
    xi = xi - np.mean(xi, axis=0)[None, :]
    return mu[None, :] + xi @ chol.T


def log_gauss_density_batch(
                        x,              # (S,dim):   samples
                        mu,             # (dim,):    mean vector
                        Gamma,          # (dim,dim): covariance matrix
                        Inv=np.nan,     # (dim,dim): inverse covariance matrix
                        ):
    """
    computes the log density of a multivariate gaussian N(mu, Sigma) 
    for a batch of samples x

    **remark**: the normalization constant is not included
    """
    assert mu.shape[0] == Gamma.shape[0]
    assert Gamma.shape[0] == Gamma.shape[1]

    # if Inv is not provided, compute it:
    Inv = np.where(np.isnan(Inv), psd.invert_psd_matrix(Gamma), Inv)
    return -0.5*np.sum((x-mu[None, :]) * ((x-mu[None, :]) @ Inv), axis=1)


def grad_log_gauss_density_batch(
                            x,              # (S,dim):   samples
                            mu,             # (dim,):    mean vector
                            Gamma,          # (dim,dim): covariance matrix
                            Inv=np.nan,     # (dim,dim): inverse covariance matrix
                            ):
    """
    computes the gradient (with respect to `x`) of the log density 
    of a multivariate gaussian N(x; mu, Sigma) for a batch of samples x
    """
    assert mu.shape[0] == Gamma.shape[0]
    assert Gamma.shape[0] == Gamma.shape[1]

    # if Inv is not provided, compute it:
    Inv = np.where(np.isnan(Inv), psd.invert_psd_matrix(Gamma), Inv)
    return -(x-mu[None, :]) @ Inv.T
