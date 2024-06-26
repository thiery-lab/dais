import numpy as np
import jax
import effective_sample_size as ESS
import scipy.linalg

# local imports
import util.psd as psd
import util.gauss as gauss


def ELBO(
        log_target_batch,   # (S,dim) -> (S,): log of the target density
        samples,                    # (S,dim):      samples from N(mu, Sigma)
        mu,                         # (dim,):       mean vector
        Gamma,                      # (dim,dim):    covariance matrix
        Inv=np.nan,                 # (dim,dim):    inverse covariance matrix
        ):
    """
    estimates the evidence lower bound (ELBO) of the variational inference problem
        ELBO = -E_q[ log( q(X) / p_unormalized(X) ) ]
    """
    assert mu.shape[0] == Gamma.shape[0]
    assert Gamma.shape[0] == Gamma.shape[1]

    # if Inv is not provided, compute it:    
    Inv = np.where(np.isnan(Inv), psd.invert_psd_matrix(Gamma), Inv)

    log_det = psd.log_det_psd(Gamma)
    log_gauss = gauss.log_gauss_density_batch(samples, mu, Gamma, Inv)
    log_gauss = log_gauss - 0.5*log_det
    log_pi = log_target_batch(samples)
    return -np.mean(log_gauss-log_pi)


def DAIS(
        *,
        log_target,                 # callable:     log_target(z) -> float
        log_target_grad=None,       # callable:     log_target_grad(z) -> (dim,)
        mu_init,                    # (dim,):       initial mean vector
        Gamma_init,                 # (dim,dim):    initial covariance matrix
        n_samples,                  # int:          number of samples to use
        n_iter,                     # int:          number of iterations
        ESS_threshold=0.5,          # float:        effective sample size threshold
        alpha_damp=1.,              # float:        damping factor for the mean/covariance updates
        save_history=False,         # bool:         whether to save the history of the algorithm
        ):
    """
    runs the DAIS algorithm

    output:
    =======

    dict = {
        "mu":           (dim,):         final mean vector
        "Gamma":        (dim,dim):      final covariance matrix
        "ELBO":         float:          final ELBO
        "mu_traj":      [(dim,)]:       trajectory of the mean vector
        "Gamma_traj":   [(dim,dim)]:    trajectory of the covariance matrix
        "ELBO_traj":    [float]:        trajectory of the ELBO
        "eps_traj":     [float]:        trajectory of the epsilon values
        }
    """
    assert mu_init.shape[0] == Gamma_init.shape[0]
    assert Gamma_init.shape[0] == Gamma_init.shape[1]

    # initialize the jitted target quantitites
    log_target = jax.jit(log_target)
    if log_target_grad is None:
        log_target_grad = jax.grad(log_target)
    log_target_grad = jax.jit(jax.grad(log_target))
    log_target_batch = jax.jit(jax.vmap(log_target))
    log_target_grad_batch = jax.jit(jax.vmap(log_target_grad))

    def grad_phi_batch(
                    xs,             # (S,dim):      samples
                    mu,             # (dim,):       mean vector
                    Gamma,          # (dim,dim):    covariance matrix
                    Inv=np.nan,     # (dim,dim):    inverse covariance matrix
                    ):
        """
        computes: grad_x[ log( pi / q) ]
        where q(x) = N(x; mu, Sigma) is a Gaussian approximation of the the target density pi(x)
        """
        assert mu.shape[0] == Gamma.shape[0]
        assert Gamma.shape[0] == Gamma.shape[1]

        # if Gamma_inv is not provided, compute it:
        Inv = np.where(np.isnan(Inv), psd.invert_psd_matrix(Gamma), Inv)
        Phis = log_target_grad_batch(xs)
        Phis = Phis - gauss.grad_log_gauss_density_batch(xs, mu, Gamma, Inv)
        return Phis

    # initialize
    Gamma = np.copy(Gamma_init)
    mu = np.copy(mu_init)
    chol = scipy.linalg.cholesky(Gamma, lower=True)
    Gamma_inv = psd.invert_psd_matrix(Gamma, chol=chol)

    # list to save history
    ELBO_list = []
    mu_list = []
    Gamma_list = []
    eps_list = []

    for _ in range(n_iter):
        # generate samples from current approx
        xs = gauss.sample_gaussian(n_samples, mu, Gamma, chol=chol)

        # compute Phi
        log_q_unormalized = gauss.log_gauss_density_batch(
                                                        xs,
                                                        mu,
                                                        Gamma,
                                                        Gamma_inv)
        Phi = log_target_batch(xs) - log_q_unormalized
        Phi -= np.max(Phi)

        # find epsilon & compute weights
        eps = ESS.next_temperature(Phi, ESS_threshold)

        # Do a loop to ensure that the updated covariance matrix is psd
        # if that is not the case, the next temperature `eps` is reduced
        Phis = grad_phi_batch(xs, mu, Gamma, Gamma_inv)
        while True:
            # compute importance sampling weights
            log_w = eps*Phi
            log_w = log_w - np.max(log_w)
            w = np.exp(log_w)
            w = w / np.sum(w)

            # update mean
            Gamma_Phis = (Phis @ Gamma.T)
            Mean_Gamma_Phis = np.sum(w[:, None] * Gamma_Phis, axis=0)
            mu_ = mu + eps*Mean_Gamma_Phis

            # update covariance
            Gamma_Phis_centred = Gamma_Phis - Mean_Gamma_Phis[None, :]
            xs_mean = np.sum(w[:, None] * xs, axis=0)
            xs_centred = xs - xs_mean[None, :]
            # compute delta_Gamma = Cov[Gamma @ Phis, xs]
            delta_Gamma = Gamma_Phis_centred.T @ (w[:, None] * xs_centred)
            # make sure that the covariance matrix is symmetric
            delta_Gamma = 0.5*(delta_Gamma+delta_Gamma.T)
            # update Gamma
            Gamma_ = Gamma + alpha_damp * eps * delta_Gamma

            # make sure that the updated covariance matrix is positive definite
            if psd.is_psd(Gamma_):
                break
            else:
                # reduce the temperature
                eps = 0.5*eps

        # do the damped mean / covariance updates
        # and update chol, Gamma_inv
        Gamma = Gamma + alpha_damp * eps * delta_Gamma
        mu = mu + alpha_damp*(mu_-mu)
        chol = scipy.linalg.cholesky(Gamma, lower=True)
        Gamma_inv = psd.invert_psd_matrix(Gamma, chol=chol)

        if save_history:
            mu_list.append(np.copy(mu))
            Gamma_list.append(np.copy(Gamma))

        # update the ELBO and epsilon lists
        eps_list.append(eps)
        current_ELBO = ELBO(
                            log_target_batch,
                            xs,
                            mu, Gamma, Inv=Gamma_inv,
                            )
        ELBO_list.append(current_ELBO)

    output_dict = {
        "mu": mu,
        "Gamma": Gamma,
        "ELBO": ELBO_list[-1],
        "mu_traj": mu_list,
        "Gamma_traj": Gamma_list,
        "ELBO_traj": ELBO_list,
        "eps_traj": eps_list,
    }

    return output_dict
