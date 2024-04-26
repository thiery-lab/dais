import warnings

import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import scipy.optimize
import scipy.stats

jax.config.update("jax_enable_x64", True)  # Enable double precision with JAX.
outer = jax.vmap(np.outer)  # Vectorized outer product


def DAIS(
    U_scalar, S=10**4, verbose=True, max_iter=10, S_eff_target=None, µ=None,
    Σ=None, d=None, Stein_update=True
):
    """
    Doubly adaptive importance sampling for a Gaussian approximation
    
    The following function implements doubly adaptive importance sampling
    (DAIS) for a Gaussian approximation to a target density.

    `U_scalar` is the log-density of the target. It should use jax.numpy or
    autograd.numpy instead of numpy such that JAX can parallelize and
    differentiate it.

    `S` is the importance sample size.
    `verbose` indicates whether iterative diagnostics should be printed.

    `max_iter` is the maximum number number of iterations. The function will
    return the current approximation even if the algorithm has not yet
    converged at that iteration number.

    `S_eff_target` is the guaranteed effective sample size which defaults to
    `S / 3.0`.

    `µ` and `Σ` are the mean and covariance of the initial Gaussian
    approximation. The default is a standard Gaussian.

    `d` is the dimensionality of the support of the target.

    `Stein_update` indicates whether updates based on Stein's lemma are
    considered.
    """
    if d is None:
        if µ is None:
            if Σ is None:
                raise ValueError("At least one of the arguments `µ`, `Σ` and "
                    + "`d` has to be given.")
            else:
                d = len(Σ)
        else:
            d = len(µ)

    if µ is None:
        µ = np.zeros(d)
    
    if Σ is None:
        Σ = np.identity(d)

    if S_eff_target is None:
        # Specify the effective sample size targeted by the tempering.
        S_eff_target = S / 3.0

    U = jax.jit(jax.vmap(U_scalar))
    grad_U = jax.jit(jax.vmap(jax.grad(U_scalar)))
    npr.seed(1)  # Set seed for reproducibility.
    ε_prev = 0.0
    ε_cum_prod = 1.0
    check_convergence = False
    n_check = 5  # How many iterations to based the convergence check on
    diff_arr = onp.empty(n_check)

    for t in range(max_iter):
        # Generate samples from the q_t.
        try:
            q_t = scipy.stats.multivariate_normal(
                mean=µ, cov=Σ, allow_singular=True
            )
        except ValueError:
            warnings.warn(
                "The new Σ is not positive semi-definite. "
                    + "Therefore, resorting to the previous Σ."
            )

            Σ = Σ_old

            q_t = scipy.stats.multivariate_normal(
                mean=µ, cov=Σ, allow_singular=True
            )
        
        x_s = q_t.rvs(size=S)

        # `x_s` might not have the correct shape if `S=1` or `d=1`.
        x_s.reshape(S, d)
        
        # Compute the importance weights.
        log_w = U(x_s) - q_t.logpdf(x_s)
        log_w -= np.max(log_w)
        

        # Check whether the effective sample size is large enough.
        # Otherwise, temper to achieve `S_eff_target`.
        def weights(ε):
            w = np.exp(ε * log_w)
            return w / w.sum()
        

        def S_eff(ε):
            w = weights(ε)
            return 1/np.sum(w * w) - S_eff_target
        

        if S_eff(1) >= 0:
            ε = 1
        else:
            ε = scipy.optimize.root_scalar(S_eff, bracket=[0, 1]).root
        
        if verbose:
            print("ε = % 0.5f" %ε)
        
        w = weights(ε)[:, np.newaxis]
        µ_old, Σ_old = µ, Σ
        

        # Compute the new µ and Σ.
        # We use Stein's lemma if that yields a reduction in Monte Carlo error.
        def MCSS(arg_s):
            """Compute the importance-weighted Monte Carlo sum of squares."""
            empirical_mean = (w * arg_s).sum(axis=0)
            SS = (w * (arg_s - empirical_mean)**2).sum()
            return SS, empirical_mean
        

        if not Stein_update:
            µ = (w * x_s).sum(axis=0)
        else:
            SS_regular, µ_regular = MCSS(x_s)
            ε_Σ_grad_Φ_s = ε * (grad_U(x_s)@Σ + x_s - µ_old)
            SS_Stein, diff_Stein = MCSS(ε_Σ_grad_Φ_s)
            # µ = np.where(SS_regular < SS_Stein, µ_regular, µ + diff_Stein)
            µ += diff_Stein

        m = x_s - µ

        if Stein_update:
            SS_regular, _ = MCSS(m**2)
            tmp = ε_Σ_grad_Φ_s + µ_old - µ
            SS_Stein, _ = MCSS(m * tmp)

        # if not Stein_update or SS_regular < SS_Stein:
        if not Stein_update: #or SS_regular < SS_Stein:
            Σ = (w[:, :, np.newaxis] * outer(m, m)).sum(axis=0)
        else:
            E_x_grad_Φ_Σ = (w[:, :, np.newaxis] * outer(m, tmp)).sum(axis=0)
            # `E_x_grad_Φ_Σ` is symmetric while our importance sampling
            # estimate of it probably is not.
            # We therefore symmetrize `E_x_grad_Φ_Σ`.
            Σ += 0.5 * (E_x_grad_Φ_Σ + E_x_grad_Φ_Σ.T)
        
        if ε == 1.0:
            break

        ε_cum_prod *= 1.0 - ε

        if ε < ε_prev and ε_cum_prod < 0.01:
            check_convergence = True

        if check_convergence:
            # Check convergence based on the mean absolute innovation of µ and
            # the diagonal of Σ.
            diff = np.abs(np.concatenate(
                (µ - µ_old, np.diag(Σ) - np.diag(Σ_old))
            )).mean()

            diff_arr[t % n_check] = diff

            if diff > diff_arr.mean():
                break

        ε_prev = ε
    
    if verbose:
            print("DAIS finished in", t + 1, "iterations with ε =", ε)

    return µ, Σ
