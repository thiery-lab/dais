################################################################################################
# This script runs Hamiltonian Monte Carlo (HMC) on the logistic regression problem
# and saves the MCMC samples, the mean and covariance of the MCMC samples, and the
# acceptance probability in a numpy file.
# Sampler used: blackjax
################################################################################################
import numpy as np
import jax
import jax.numpy as jnp
import os
import scipy
import pylab as plt
import blackjax
import time
from regression_logdensity import make_regression_logdensity


key = jax.random.key(0)


# PARAMS for HMC:
MCMC_warmup_iter = 1000
MCMC_iter = 100_000

# set the random seed
rng = np.random.RandomState(seed=1234)


# iterate over the datasets
data_list = ["spam", "krkp", "ionosphere", "mushroom"]
for data_name in data_list:
    print(f"Processing the {data_name} data:")
    time_start = time.time()

    logdensity, dim_covariates = make_regression_logdensity(data_name)
    logdensity = jax.jit(logdensity)
    neglogdensity = lambda x: -logdensity(x)
    grad_neglogdensity = jax.jit(jax.grad(neglogdensity))

    print("DIM: ", dim_covariates)

    print("\t Computing initial values...")
    # compute the MAP estimate
    MAP = scipy.optimize.minimize(
                            fun=neglogdensity,
                            x0=np.zeros(dim_covariates),
                            jac=grad_neglogdensity)['x']

    # compute the Hessian at the MAP
    # inverse of the Hessian at the MAP is used as the initial covariance matrix
    Sigma_init = np.linalg.inv(jax.hessian(neglogdensity)(MAP))
    # add a small diagonal term to the initial covariance matrix
    eps = 1e-3
    Sigma_init += eps * np.eye(dim_covariates)

    print("\t Running Hamiltonian Monte Carlo...")
    print(f"\t Number of warmup iterations: {MCMC_warmup_iter}")
    print(f"\t Number of main iterations: {MCMC_iter}")

    inv_mass_matrix = Sigma_init
    num_integration_steps = 10
    step_size = 1. / float(dim_covariates)**0.5

    # create the HMC kernel
    hmc = blackjax.hmc(logdensity, step_size, inv_mass_matrix, num_integration_steps)
    hmc_kernel = jax.jit(hmc.step)

    # initialize the HMC kernel
    initial_position = MAP
    initial_state = hmc.init(initial_position)

    def inference_loop(rng_key, kernel, initial_state, num_samples):
        def one_step(state, rng_key):
            state, stat = kernel(rng_key, state)
            return state, {"state": state, "stats": stat}

        keys = jax.random.split(rng_key, num_samples)
        _, trajectory = jax.lax.scan(one_step, initial_state, keys)
        return trajectory

    # jit the inference loop
    inference_loop = jax.jit(inference_loop, static_argnums=(1,3,))

    # NUTS sampler
    nuts = blackjax.nuts(logdensity, step_size, inv_mass_matrix)
    initial_state = nuts.init(initial_position)

    # warmup adaptation
    print("\t Warming up...")
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    key, warmup_key, sample_key = jax.random.split(key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=MCMC_warmup_iter)

    # run the NUTS sampler
    print("\t Running the NUTS sampler...")
    kernel = blackjax.nuts(logdensity, **parameters).step
    key, sample_key = jax.random.split(key)
    trajectory = inference_loop(sample_key, kernel, state, MCMC_iter)
    MCMC_history = np.array(trajectory["state"].position)

    mean_accept_prob = np.mean(trajectory["stats"].acceptance_rate)
    print("\t Acceptance rate: ", mean_accept_prob)

    mean_integration_steps = np.mean(trajectory["stats"].num_integration_steps)
    print("\t Mean integration steps: ", mean_integration_steps)

    # computes acceptance rate
    print(f"\t Mean accept prob: {mean_accept_prob:0.2f}")
    print(f"\t Mean integration steps: {mean_integration_steps:0.2f}")

    # compute time
    time_end = time.time()
    print(f"\t Time taken: {time_end - time_start:0.2f} seconds")


    # get rid of the first 10% of the samples
    MCMC_history = MCMC_history[int(0.1 * MCMC_history.shape[0]):]

    # compute the mean and covariance of the MCMC samples
    mu_mcmc = MCMC_history.mean(axis=0)
    Sigma_mcmc = np.cov(MCMC_history, rowvar=False)

    # save the MCMC outputs and mean and covariance 
    # of the MCMC samples in a numpy file
    print("\t Saving the MCMC outputs to disk...")
    folder_name = "mcmc_outputs"
    np.savez(
        os.path.join(folder_name, data_name + "_MCMC.npz"),
        mu_mcmc=mu_mcmc,
        Sigma_mcmc=Sigma_mcmc,
        HMC_accept_prob=mean_accept_prob,
        MCMC_iter=MCMC_iter,
        MCMC_warmup_iter=MCMC_warmup_iter,
        MCMC_sampler=f"blackjax ver:{blackjax.__version__}",
        )
