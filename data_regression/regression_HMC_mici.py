################################################################################################
# This script runs Hamiltonian Monte Carlo (HMC) on the logistic regression problem
# and saves the MCMC samples, the mean and covariance of the MCMC samples, and the
# acceptance probability in a numpy file.
# Sampler used: mici
################################################################################################
import numpy as onp
import jax
import jax.numpy as np
import os
import scipy
import mici
import mici._version
import time


# PARAMS for HMC:
MCMC_warmup_iter = 500
MCMC_iter = 5000

# set the random seed
rng = onp.random.RandomState(seed=1234)


# iterate over the datasets
data_list = ["spam", "krkp", "ionosphere", "mushroom"]
for data_name in data_list:
    print(f"Processing the {data_name} data:")
    time_start = time.time()

    # load data
    data_folder = "./"
    data_file = os.path.join(data_folder, data_name + ".npz")
    regression_data = np.load(data_file)
    A = regression_data["X"]
    y = regression_data["y"]
    dim_covariates = A.shape[1]

    ###########################
    # Define the model 
    ###########################
    @jax.vmap
    def log1pexp(arg):
        """
        log1pexp, also known as the softplus function
        Evaluate log(1 + exp(arg)) per Equation 10 from
        https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf.
        """
        return np.where(arg < 33.3, np.log1p(np.exp(arg)), arg)

    # define the log likelihood
    def log_likelihood(x):
        """Log-likelihood of logistic regression"""
        return -np.sum(log1pexp(- A@x * y))

    # define the log posterior up to a constant
    def U_scalar(x):
        """Log posterior with a N(0, 10I) prior up to a constant"""
        variance_x = 10.
        return log_likelihood(x) - 0.5*np.sum(x * x) / variance_x

    @jax.jit
    def neg_log_dens(x):
        """Negative log posterior using `jax.numpy`."""
        return -U_scalar(x)

    # compute the gradient of the negative log posterior
    grad_neg_log_dens = jax.jit(jax.grad(neg_log_dens))
    ###############################
    # End of model definition
    ################################

    print("\t Computing initial values...")
    # compute the MAP estimate
    MAP = scipy.optimize.minimize(
                            fun=neg_log_dens,
                            x0=np.zeros(dim_covariates),
                            jac=grad_neg_log_dens)['x']

    # compute the Hessian at the MAP
    # inverse of the Hessian at the MAP is used as the initial covariance matrix
    Sigma_init = np.linalg.inv(jax.hessian(neg_log_dens)(MAP))
    # add a small diagonal term to the initial covariance matrix
    eps = 1e-3
    Sigma_init += eps * np.eye(dim_covariates)

    print("\t Running Hamiltonian Monte Carlo...")
    print(f"\t Number of warmup iterations: {MCMC_warmup_iter}")
    print(f"\t Number of main iterations: {MCMC_iter}")

    system = mici.systems.EuclideanMetricSystem(
        neg_log_dens=lambda x: onp.asarray(neg_log_dens(x)),
        grad_neg_log_dens=lambda x: onp.asarray(grad_neg_log_dens(x)),
        metric=onp.asarray(Sigma_init),
    )

    # since we adapted the mass matrix, we set the initial step size to 1
    step_size = 1.
    integrator = mici.integrators.LeapfrogIntegrator(
                                        system,
                                        step_size=step_size,)

    # Initialize the HMC sampler
    sampler = mici.samplers.DynamicMultinomialHMC(
                                        system,
                                        integrator,
                                        rng)

    # Run the HMC sampler
    init_state = MAP  # Initialize at the MAP.
    final_state, chains, chain_stats = sampler.sample_chains(
        n_main_iter=MCMC_iter,
        n_warm_up_iter=MCMC_warmup_iter,
        init_states=[init_state],
        display_progress=False,)

    # computes acceptance rate
    mean_accept_prob = chain_stats['accept_stat'][0].mean()
    print(f"\t Mean accept prob: {mean_accept_prob:0.2f}")

    # compute time
    time_end = time.time()
    print(f"\t Time taken: {time_end - time_start:0.2f} seconds")

    # Discard first 10% of samples as burnin.
    MCMC_history = chains['pos'][0][(MCMC_iter // 10):MCMC_iter]
    # compute the mean and covariance of the MCMC samples
    mu_mcmc = MCMC_history.mean(axis=0)
    Sigma_mcmc = onp.cov(MCMC_history, rowvar=False)

    # save the MCMC outputs and mean and covariance 
    # of the MCMC samples in a numpy file
    print("\t Saving the MCMC outputs to disk...")
    folder_name = "mcmc_outputs"
    onp.savez(
        os.path.join(folder_name, data_name + "_MCMC_mici.npz"),
        MCMC_history=MCMC_history,
        mu_mcmc=mu_mcmc,
        Sigma_mcmc=Sigma_mcmc,
        HMC_accept_prob=mean_accept_prob,
        MCMC_iter=MCMC_iter,
        MCMC_warmup_iter=MCMC_warmup_iter,
        MCMC_sampler=f"mici ver:{mici._version.__version__}",
        )
